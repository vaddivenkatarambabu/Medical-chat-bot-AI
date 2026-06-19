import logging
import os
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

# This backend uses PyTorch sentence-transformer embeddings. Keep Transformers
# away from TensorFlow/Keras imports so Keras 3 installations do not break /get.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, jsonify, request
from sqlalchemy.exc import SQLAlchemyError

from src.auth import (
    AuthenticationError,
    extract_bearer_token,
    get_request_identity,
    token_sha256,
)
from src.database import check_database, init_database, session_scope
from src.prompt import system_prompt
from src.rate_limit import RateLimiter
from src.repositories import (
    ChatRepository,
    NotFoundError,
    serialize_conversation,
    serialize_message,
    serialize_user,
)
from src.schemas import (
    RequestValidationError,
    parse_chat_payload,
    parse_conversation_payload,
    parse_email_auth_payload,
)
from src.supabase_email import (
    SupabaseEmailConfigurationError,
    SupabaseEmailDeliveryError,
    get_auth_settings,
    send_email_otp,
    send_recovery_email,
)


DEFAULT_INDEX_NAME = "medical-chatbot"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_RETRIEVER_K = 3
MAX_MESSAGE_LENGTH = 2_000
DEFAULT_DEV_CORS_ORIGINS = {
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
}
DEFAULT_DEV_FRONTEND_URL = "http://127.0.0.1:8080"


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)
chat_repository = ChatRepository()
rate_limiter = RateLimiter()


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing."""


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigurationError(f"Missing required environment variable: {name}")
    return value


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if minimum is not None and parsed < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    return parsed


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if minimum is not None and parsed < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    return parsed


def _get_message() -> str:
    try:
        return parse_chat_payload(request).message
    except RequestValidationError:
        return ""


def _cors_allowed_origins() -> set[str]:
    configured = set(DEFAULT_DEV_CORS_ORIGINS)
    configured.update({
        origin.strip().rstrip("/")
        for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    })
    frontend_url = os.getenv("FRONTEND_URL")
    if frontend_url:
        configured.add(frontend_url.strip().rstrip("/"))

    return configured


def _frontend_url() -> str:
    return os.getenv("FRONTEND_URL", DEFAULT_DEV_FRONTEND_URL).strip().rstrip("/")


def _normalized_origin(url: str) -> str | None:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def _auth_redirect_allowed_origins() -> set[str]:
    configured = {
        origin.strip().rstrip("/")
        for origin in os.getenv("AUTH_REDIRECT_ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    }
    if not configured:
        configured = _cors_allowed_origins()

    configured.add(_frontend_url())
    return {
        normalized
        for origin in configured
        if origin != "*"
        for normalized in [_normalized_origin(origin)]
        if normalized
    }


def _is_allowed_auth_redirect(redirect_to: str) -> bool:
    origin = _normalized_origin(redirect_to)
    return bool(origin and origin in _auth_redirect_allowed_origins())


def _client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return request.remote_addr or "unknown"


def _rate_limit_key(scope: str) -> str:
    email = ""
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        if isinstance(payload, dict):
            value = payload.get("email")
            email = value.strip().lower() if isinstance(value, str) else ""

    return f"{scope}:{_client_ip()}:{email}"


def _check_rate_limit(scope: str, *, limit: int, window_seconds: int):
    result = rate_limiter.check(
        _rate_limit_key(scope),
        limit=limit,
        window_seconds=window_seconds,
    )
    if result.allowed:
        return None

    response = jsonify({
        "error": "Too many requests. Please try again later.",
        "retry_after": result.retry_after,
    })
    response.status_code = 429
    response.headers["Retry-After"] = str(result.retry_after)
    return response


@lru_cache(maxsize=1)
def get_rag_chain() -> Any:
    """Build the retrieval augmented generation chain once per process."""
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq
    from langchain_pinecone import PineconeVectorStore
    from src.helper import download_hugging_face_embeddings

    pinecone_api_key = _required_env("PINECONE_API_KEY")
    groq_api_key = _required_env("GROQ_API_KEY")

    os.environ["PINECONE_API_KEY"] = pinecone_api_key

    embeddings = download_hugging_face_embeddings()
    index_name = os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
    retriever_k = _env_int("RETRIEVER_K", DEFAULT_RETRIEVER_K, minimum=1)

    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )
    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": retriever_k},
    )

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        temperature=_env_float("GROQ_TEMPERATURE", 0.2, minimum=0.0),
        max_tokens=_env_int("GROQ_MAX_TOKENS", 1024, minimum=1),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


def create_app() -> Flask:
    init_database()

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = _env_int("MAX_CONTENT_LENGTH_BYTES", 1_000_000, minimum=1)

    @app.after_request
    def add_security_headers(response):
        origin = request.headers.get("Origin")
        allowed_origins = _cors_allowed_origins()
        if origin and ("*" in allowed_origins or origin.rstrip("/") in allowed_origins):
            response.headers["Access-Control-Allow-Origin"] = (
                "*" if "*" in allowed_origins else origin
            )
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PATCH, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization, X-Guest-Session-Id"
            )
            response.headers["Access-Control-Max-Age"] = "600"
            response.headers.add("Vary", "Origin")

        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "script-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; "
            "img-src 'self' data:;",
        )
        return response

    @app.get("/")
    def index() -> str:
        frontend_url = _frontend_url()
        return (
            "<!doctype html>"
            '<html lang="en">'
            "<head>"
            '<meta charset="utf-8" />'
            '<meta name="viewport" content="width=device-width, initial-scale=1" />'
            "<title>MediCore API</title>"
            "</head>"
            '<body style="font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.5;">'
            "<h1>MediCore API is running</h1>"
            "<p>The React frontend is served separately.</p>"
            f'<p><a href="{frontend_url}">Open MediCore frontend</a></p>'
            "<ul>"
            '<li><a href="/health">/health</a></li>'
            "<li>POST /get</li>"
            "</ul>"
            "</body>"
            "</html>"
        )

    @app.get("/health")
    def health():
        payload = {"status": "ok"}

        if request.args.get("deep") == "1":
            try:
                check_database()
                payload["database"] = "ok"
            except Exception:
                logger.exception("Database health check failed")
                return jsonify({"status": "error", "database": "error"}), 503

        return jsonify(payload)

    @app.get("/api/auth/me")
    def auth_me():
        limited = _check_rate_limit("auth:me", limit=120, window_seconds=60)
        if limited:
            return limited

        try:
            identity = get_request_identity(
                request,
                require_auth=True,
                require_verified=True,
            )
            with session_scope() as db:
                user = chat_repository.resolve_user(
                    db,
                    identity,
                    create=True,
                    user_agent=request.headers.get("User-Agent"),
                    ip_address=_client_ip(),
                )
                if user is None:
                    raise AuthenticationError("Authentication is required")
                return jsonify({"user": serialize_user(user)})
        except AuthenticationError as exc:
            return jsonify({"error": str(exc)}), 401
        except SQLAlchemyError:
            logger.exception("Failed to load authenticated user")
            return jsonify({"error": "Database query failed"}), 503

    @app.post("/api/auth/session")
    def sync_auth_session():
        limited = _check_rate_limit("auth:session", limit=30, window_seconds=60)
        if limited:
            return limited

        try:
            identity = get_request_identity(
                request,
                require_auth=True,
                require_verified=True,
            )
            with session_scope() as db:
                user = chat_repository.resolve_user(
                    db,
                    identity,
                    create=True,
                    user_agent=request.headers.get("User-Agent"),
                    ip_address=_client_ip(),
                )
                if user is None:
                    raise AuthenticationError("Authentication is required")
                return jsonify({"user": serialize_user(user)})
        except AuthenticationError as exc:
            return jsonify({"error": str(exc)}), 401
        except SQLAlchemyError:
            logger.exception("Failed to sync authenticated session")
            return jsonify({"error": "Database write failed"}), 503

    @app.post("/api/auth/logout")
    def auth_logout():
        limited = _check_rate_limit("auth:logout", limit=30, window_seconds=60)
        if limited:
            return limited

        try:
            token = extract_bearer_token(request)
            if not token:
                raise AuthenticationError("Authentication is required")

            with session_scope() as db:
                chat_repository.revoke_session(db, token_sha256(token))
            return jsonify({"ok": True})
        except AuthenticationError as exc:
            return jsonify({"error": str(exc)}), 401
        except SQLAlchemyError:
            logger.exception("Failed to revoke session")
            return jsonify({"error": "Database write failed"}), 503

    @app.get("/api/auth/email-settings")
    def auth_email_settings():
        limited = _check_rate_limit("auth:email-settings", limit=60, window_seconds=60)
        if limited:
            return limited

        try:
            settings = get_auth_settings()
            return jsonify({
                "email_provider_enabled": settings.email_provider_enabled,
                "signup_disabled": settings.signup_disabled,
                "mailer_autoconfirm": settings.mailer_autoconfirm,
            })
        except SupabaseEmailConfigurationError as exc:
            logger.warning("Supabase email configuration error: %s", exc)
            return jsonify({"error": str(exc)}), 503
        except SupabaseEmailDeliveryError as exc:
            logger.warning("Could not load Supabase email settings: %s", exc)
            return jsonify({"error": str(exc)}), exc.status_code or 503

    @app.post("/api/auth/send-otp")
    def auth_send_otp():
        limited = _check_rate_limit("auth:send-otp", limit=5, window_seconds=900)
        if limited:
            return limited

        try:
            payload = parse_email_auth_payload(request)
            if not _is_allowed_auth_redirect(payload.redirect_to):
                return jsonify({"error": "redirect_to is not an allowed auth origin"}), 400
            send_email_otp(
                payload.email,
                payload.redirect_to,
                create_user=payload.create_user,
            )
            return jsonify({
                "ok": True,
                "message": "Verification email request accepted by Supabase.",
            })
        except RequestValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except SupabaseEmailConfigurationError as exc:
            logger.warning("Supabase OTP configuration error: %s", exc)
            return jsonify({"error": str(exc)}), 503
        except SupabaseEmailDeliveryError as exc:
            logger.warning("Supabase OTP delivery request failed: %s", exc)
            return jsonify({"error": str(exc)}), exc.status_code or 502

    @app.post("/api/auth/send-recovery")
    def auth_send_recovery():
        limited = _check_rate_limit("auth:send-recovery", limit=5, window_seconds=900)
        if limited:
            return limited

        try:
            payload = parse_email_auth_payload(request)
            if not _is_allowed_auth_redirect(payload.redirect_to):
                return jsonify({"error": "redirect_to is not an allowed auth origin"}), 400
            send_recovery_email(payload.email, payload.redirect_to)
            return jsonify({
                "ok": True,
                "message": "Password recovery email request accepted by Supabase.",
            })
        except RequestValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except SupabaseEmailConfigurationError as exc:
            logger.warning("Supabase recovery configuration error: %s", exc)
            return jsonify({"error": str(exc)}), 503
        except SupabaseEmailDeliveryError as exc:
            logger.warning("Supabase recovery delivery request failed: %s", exc)
            return jsonify({"error": str(exc)}), exc.status_code or 502

    @app.post("/get")
    def chat():
        try:
            payload = parse_chat_payload(request)
        except RequestValidationError as exc:
            return str(exc), 400

        message = payload.message
        if not message:
            return "Please enter a question.", 400
        if len(message) > MAX_MESSAGE_LENGTH:
            return f"Question is too long. Limit is {MAX_MESSAGE_LENGTH} characters.", 413

        try:
            response = get_rag_chain().invoke({"input": message})
        except ConfigurationError as exc:
            logger.warning("Application is not configured: %s", exc)
            return "The assistant is not configured. Please check server environment variables.", 503
        except Exception:
            logger.exception("Failed to generate assistant response")
            return "Sorry, I could not generate a response right now. Please try again.", 500

        answer = response.get("answer") if isinstance(response, dict) else None
        answer_text = str(answer or "I don't know.")

        try:
            identity = get_request_identity(
                request,
                guest_session_id=payload.guest_session_id,
            )
            with session_scope() as db:
                chat_repository.save_chat_turn(
                    db,
                    identity=identity,
                    message=message,
                    answer=answer_text,
                    conversation_id=payload.conversation_id,
                    client_message_id=payload.client_message_id,
                    user_agent=request.headers.get("User-Agent"),
                    ip_address=request.headers.get("X-Forwarded-For", request.remote_addr),
                )
        except AuthenticationError:
            logger.info("Skipping chat persistence because the auth token is not trusted")
        except SQLAlchemyError:
            logger.exception("Failed to persist chat turn")

        return answer_text

    @app.route("/get", methods=["OPTIONS"])
    def chat_preflight():
        return "", 204

    @app.get("/api/conversations")
    def list_conversations():
        try:
            guest_session_id = request.args.get("guest_session_id")
            identity = get_request_identity(
                request,
                guest_session_id=guest_session_id,
                require_auth=guest_session_id is None,
                require_verified=guest_session_id is None,
            )
            with session_scope() as db:
                user = chat_repository.resolve_user(db, identity, create=False)
                conversations = chat_repository.list_conversations(db, user)
                return jsonify([serialize_conversation(item) for item in conversations])
        except AuthenticationError as exc:
            return jsonify({"error": str(exc)}), 401
        except SQLAlchemyError:
            logger.exception("Failed to list conversations")
            return jsonify({"error": "Database query failed"}), 503

    @app.post("/api/conversations")
    def create_conversation():
        try:
            payload = parse_conversation_payload(request)
            identity = get_request_identity(
                request,
                guest_session_id=payload.guest_session_id,
                require_auth=payload.guest_session_id is None,
                require_verified=payload.guest_session_id is None,
            )
            with session_scope() as db:
                user = chat_repository.resolve_user(
                    db,
                    identity,
                    create=True,
                    user_agent=request.headers.get("User-Agent"),
                    ip_address=request.headers.get("X-Forwarded-For", request.remote_addr),
                )
                conversation = chat_repository.create_conversation(
                    db,
                    user,
                    title=payload.title,
                    guest_session_id=identity.guest_session_id,
                )
                return jsonify(serialize_conversation(conversation)), 201
        except RequestValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except AuthenticationError as exc:
            return jsonify({"error": str(exc)}), 401
        except SQLAlchemyError:
            logger.exception("Failed to create conversation")
            return jsonify({"error": "Database write failed"}), 503

    @app.get("/api/conversations/<conversation_id>/messages")
    def list_messages(conversation_id: str):
        try:
            guest_session_id = request.args.get("guest_session_id")
            identity = get_request_identity(
                request,
                guest_session_id=guest_session_id,
                require_auth=guest_session_id is None,
                require_verified=guest_session_id is None,
            )
            with session_scope() as db:
                user = chat_repository.resolve_user(db, identity, create=False)
                conversation = chat_repository.get_conversation_for_user(
                    db,
                    user,
                    conversation_id,
                )
                messages = chat_repository.list_messages(db, conversation)
                return jsonify([serialize_message(item) for item in messages])
        except AuthenticationError as exc:
            return jsonify({"error": str(exc)}), 401
        except NotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except SQLAlchemyError:
            logger.exception("Failed to list messages")
            return jsonify({"error": "Database query failed"}), 503

    @app.patch("/api/conversations/<conversation_id>")
    def rename_conversation(conversation_id: str):
        try:
            payload = parse_conversation_payload(request)
            if not payload.title:
                return jsonify({"error": "title is required"}), 400

            identity = get_request_identity(
                request,
                guest_session_id=payload.guest_session_id,
                require_auth=payload.guest_session_id is None,
                require_verified=payload.guest_session_id is None,
            )
            with session_scope() as db:
                user = chat_repository.resolve_user(db, identity, create=False)
                conversation = chat_repository.get_conversation_for_user(
                    db,
                    user,
                    conversation_id,
                )
                conversation = chat_repository.rename_conversation(
                    db,
                    conversation,
                    payload.title,
                )
                return jsonify(serialize_conversation(conversation))
        except RequestValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except AuthenticationError as exc:
            return jsonify({"error": str(exc)}), 401
        except NotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except SQLAlchemyError:
            logger.exception("Failed to rename conversation")
            return jsonify({"error": "Database write failed"}), 503

    @app.delete("/api/conversations/<conversation_id>")
    def delete_conversation(conversation_id: str):
        try:
            guest_session_id = request.args.get("guest_session_id")
            identity = get_request_identity(
                request,
                guest_session_id=guest_session_id,
                require_auth=guest_session_id is None,
                require_verified=guest_session_id is None,
            )
            with session_scope() as db:
                user = chat_repository.resolve_user(db, identity, create=False)
                conversation = chat_repository.get_conversation_for_user(
                    db,
                    user,
                    conversation_id,
                )
                chat_repository.delete_conversation(db, conversation)
                return jsonify({"ok": True})
        except AuthenticationError as exc:
            return jsonify({"error": str(exc)}), 401
        except NotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except SQLAlchemyError:
            logger.exception("Failed to delete conversation")
            return jsonify({"error": "Database write failed"}), 503

    @app.route("/api/conversations", methods=["OPTIONS"])
    @app.route("/api/conversations/<path:_path>", methods=["OPTIONS"])
    def conversations_preflight(_path: str | None = None):
        return "", 204

    @app.route("/api/auth/<path:_path>", methods=["OPTIONS"])
    def auth_preflight(_path: str | None = None):
        return "", 204

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_RUN_HOST", "0.0.0.0"),
        port=_env_int("PORT", 1819, minimum=1),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
