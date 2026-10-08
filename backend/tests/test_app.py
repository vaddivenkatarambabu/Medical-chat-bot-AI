import os
import tempfile
from pathlib import Path

os.environ["DATABASE_URL"] = (
    f"sqlite:///{(Path(tempfile.gettempdir()) / 'medicore_test.sqlite3').as_posix()}"
)

import pytest
from sqlalchemy import select

import app as app_module
import src.auth as auth_module
from src import models  # noqa: F401
from src.auth import create_guest_session_credential
from src.database import Base, engine, session_scope
from src.models import Conversation, Message, User


class FakeDocument:
    def __init__(self, metadata):
        self.metadata = metadata


class FakeChain:
    calls = []

    def invoke(self, payload):
        self.calls.append(payload)

        return {
            "answer": f"answer for {payload['input']}",
            "context": [
                FakeDocument(
                    {
                        "source": "data/test.pdf",
                        "page": 4,
                        "chunk": 2,
                        "knowledge_base_version": "test",
                    }
                )
            ],
        }


@pytest.fixture(autouse=True)
def reset_database():
    app_module.rate_limiter = app_module.RateLimiter()
    FakeChain.calls = []

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    yield

    Base.metadata.drop_all(bind=engine)


def test_health_route():
    client = app_module.create_app().test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_deep_health_checks_database():
    client = app_module.create_app().test_client()

    response = client.get("/health?deep=1")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "ok",
        "database": "ok",
    }


def test_send_otp_validates_email():
    client = app_module.create_app().test_client()

    response = client.post(
        "/api/auth/send-otp",
        json={
            "email": "not-an-email",
            "redirect_to": ("http://127.0.0.1:8080/verify-otp"),
        },
    )

    assert response.status_code == 400
    assert "valid email" in response.get_json()["error"]


def test_send_otp_calls_supabase_email_service(monkeypatch):
    calls = []

    def fake_send_email_otp(
        email,
        redirect_to,
        *,
        create_user=True,
    ):
        calls.append(
            (
                email,
                redirect_to,
                create_user,
            )
        )

    monkeypatch.setattr(
        app_module,
        "send_email_otp",
        fake_send_email_otp,
    )

    client = app_module.create_app().test_client()

    response = client.post(
        "/api/auth/send-otp",
        json={
            "email": "Person@Example.com",
            "redirect_to": ("http://127.0.0.1:8080/verify-otp"),
        },
    )

    assert response.status_code == 200
    assert response.get_json()["ok"] is True

    assert calls == [
        (
            "person@example.com",
            "http://127.0.0.1:8080/verify-otp",
            True,
        )
    ]


def test_send_otp_can_disable_user_creation(monkeypatch):
    calls = []

    def fake_send_email_otp(
        email,
        redirect_to,
        *,
        create_user=True,
    ):
        calls.append(
            (
                email,
                redirect_to,
                create_user,
            )
        )

    monkeypatch.setattr(
        app_module,
        "send_email_otp",
        fake_send_email_otp,
    )

    client = app_module.create_app().test_client()

    response = client.post(
        "/api/auth/send-otp",
        json={
            "email": "Person@Example.com",
            "redirect_to": ("http://127.0.0.1:8080/verify-otp"),
            "create_user": False,
        },
    )

    assert response.status_code == 200

    assert calls == [
        (
            "person@example.com",
            "http://127.0.0.1:8080/verify-otp",
            False,
        )
    ]


def test_send_otp_rejects_untrusted_redirect(monkeypatch):
    def fake_send_email_otp(
        email,
        redirect_to,
        *,
        create_user=True,
    ):
        raise AssertionError("OTP email should not be requested")

    monkeypatch.setattr(
        app_module,
        "send_email_otp",
        fake_send_email_otp,
    )

    client = app_module.create_app().test_client()

    response = client.post(
        "/api/auth/send-otp",
        json={
            "email": "person@example.com",
            "redirect_to": ("https://example.invalid/verify-otp"),
        },
    )

    assert response.status_code == 400
    assert "redirect_to" in response.get_json()["error"]


def test_index_route_does_not_require_template():
    client = app_module.create_app().test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert "MediCore API is running" in response.get_data(as_text=True)


def test_chat_rejects_empty_message():
    client = app_module.create_app().test_client()

    response = client.post(
        "/get",
        data={"msg": "   "},
    )

    assert response.status_code == 400
    assert "Please enter" in response.get_json()["error"]


def test_chat_rejects_long_message():
    client = app_module.create_app().test_client()

    response = client.post(
        "/get",
        data={"msg": "x" * (app_module.MAX_MESSAGE_LENGTH + 1)},
    )

    assert response.status_code == 413
    assert "too long" in response.get_json()["error"]


def test_chat_returns_chain_answer(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "get_rag_chain",
        lambda: FakeChain(),
    )

    client = app_module.create_app().test_client()

    response = client.post(
        "/get",
        data={"msg": "What is fever?"},
    )

    assert response.status_code == 200
    assert response.get_json()["answer"] == "answer for What is fever?"

    assert response.get_json()["sources"] == [
        {
            "source": "data/test.pdf",
            "page": 4,
            "chunk": 2,
            "knowledge_base_version": "test",
        }
    ]


def test_chat_accepts_json_payload(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "get_rag_chain",
        lambda: FakeChain(),
    )

    client = app_module.create_app().test_client()

    response = client.post(
        "/get",
        json={"message": "What is cough?"},
    )

    assert response.status_code == 200
    assert response.get_json()["answer"] == "answer for What is cough?"


def test_chat_persists_guest_turn(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "get_rag_chain",
        lambda: FakeChain(),
    )

    client = app_module.create_app().test_client()

    response = client.post(
        "/get",
        json={
            "message": "What is cough?",
            "conversation_id": "guest",
            "client_message_id": "client-message-1",
            "guest_session_id": "guest-session-1",
        },
    )

    assert response.status_code == 200

    with session_scope() as db:
        users = list(db.scalars(select(User)))

        conversations = list(db.scalars(select(Conversation)))

        messages = list(db.scalars(select(Message).order_by(Message.created_at)))

    assert len(users) == 1
    assert users[0].auth_provider == "guest"
    assert users[0].external_id == "guest-session-1"

    assert len(conversations) == 1
    assert conversations[0].external_id == "guest:guest-session-1"

    assert [message.role for message in messages] == [
        "user",
        "assistant",
    ]

    assert messages[0].content == "What is cough?"

    assert messages[1].content == "answer for What is cough?"


def test_chat_saves_turn_to_created_conversation(
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "get_rag_chain",
        lambda: FakeChain(),
    )

    client = app_module.create_app().test_client()

    created = client.post(
        "/api/conversations",
        json={
            "guest_session_id": "guest-session-1",
        },
    )

    assert created.status_code == 201

    conversation_id = created.get_json()["id"]

    response = client.post(
        "/get",
        json={
            "message": "What is cough?",
            "conversation_id": conversation_id,
            "client_message_id": "client-message-1",
            "guest_session_id": "guest-session-1",
        },
    )

    assert response.status_code == 200

    messages = client.get(
        f"/api/conversations/{conversation_id}/messages",
        query_string={
            "guest_session_id": "guest-session-1",
        },
    )

    conversations = client.get(
        "/api/conversations",
        query_string={
            "guest_session_id": "guest-session-1",
        },
    )

    assert messages.status_code == 200

    assert [message["role"] for message in messages.get_json()] == [
        "user",
        "assistant",
    ]

    assert conversations.status_code == 200

    assert conversations.get_json()[0]["title"] == "What is cough?"


def test_chat_returns_configuration_error(
    monkeypatch,
):
    def raise_configuration_error():
        raise app_module.ConfigurationError("missing config")

    monkeypatch.setattr(
        app_module,
        "get_rag_chain",
        raise_configuration_error,
    )

    client = app_module.create_app().test_client()

    response = client.post(
        "/get",
        data={"msg": "What is fever?"},
    )

    assert response.status_code == 503
    assert "not configured" in response.get_json()["error"]


def test_chat_is_rate_limited(monkeypatch):
    monkeypatch.setenv(
        "CHAT_RATE_LIMIT",
        "1",
    )

    monkeypatch.setenv(
        "CHAT_RATE_LIMIT_WINDOW_SECONDS",
        "60",
    )

    monkeypatch.setattr(
        app_module,
        "get_rag_chain",
        lambda: FakeChain(),
    )

    client = app_module.create_app().test_client()

    first = client.post(
        "/get",
        json={"message": "What is fever?"},
    )

    second = client.post(
        "/get",
        json={"message": "What is cough?"},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["Retry-After"]


def test_chat_includes_recent_conversation_history(
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "get_rag_chain",
        lambda: FakeChain(),
    )

    client = app_module.create_app().test_client()

    created = client.post(
        "/api/conversations",
        json={"guest_session_id": "guest-session-1"},
    )

    conversation_id = created.get_json()["id"]

    first = client.post(
        "/get",
        json={
            "message": "What is fever?",
            "conversation_id": conversation_id,
            "client_message_id": "client-message-1",
            "guest_session_id": "guest-session-1",
        },
    )

    second = client.post(
        "/get",
        json={
            "message": "What did I ask before?",
            "conversation_id": conversation_id,
            "client_message_id": "client-message-2",
            "guest_session_id": "guest-session-1",
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200

    assert "User: What is fever?" in FakeChain.calls[-1]["conversation_history"]

    assert "Assistant: answer for What is fever?" in FakeChain.calls[-1]["conversation_history"]


def test_duplicate_client_message_returns_saved_answer(
    monkeypatch,
):
    chain = FakeChain()

    monkeypatch.setattr(
        app_module,
        "get_rag_chain",
        lambda: chain,
    )

    client = app_module.create_app().test_client()

    payload = {
        "message": "What is fever?",
        "conversation_id": "guest",
        "client_message_id": "client-message-1",
        "guest_session_id": "guest-session-1",
    }

    first = client.post(
        "/get",
        json=payload,
    )

    second = client.post(
        "/get",
        json=payload,
    )

    assert first.status_code == 200
    assert second.status_code == 200

    assert second.get_json() == {
        "answer": "answer for What is fever?",
        "sources": [],
    }

    assert len(FakeChain.calls) == 1


def test_guest_session_endpoint_issues_signed_credential(
    monkeypatch,
):
    monkeypatch.setenv(
        "GUEST_SESSION_SECRET",
        "test-secret",
    )

    client = app_module.create_app().test_client()

    response = client.post("/api/guest-session")

    assert response.status_code == 200

    credential = response.get_json()["guest_session_id"]

    assert credential.startswith("gst1.")


def test_signed_guest_session_persists_under_stable_identity(
    monkeypatch,
):
    monkeypatch.setenv(
        "GUEST_SESSION_SECRET",
        "test-secret",
    )

    monkeypatch.setattr(
        app_module,
        "get_rag_chain",
        lambda: FakeChain(),
    )

    credential, _expires_at = create_guest_session_credential()

    client = app_module.create_app().test_client()

    response = client.post(
        "/get",
        json={
            "message": "What is cough?",
            "conversation_id": "guest",
            "client_message_id": "client-message-1",
            "guest_session_id": credential,
        },
    )

    assert response.status_code == 200

    with session_scope() as db:
        user = db.scalar(select(User))

    assert user.auth_provider == "guest"
    assert user.external_id != credential


def test_env_int_rejects_invalid_range(
    monkeypatch,
):
    monkeypatch.setenv(
        "RETRIEVER_K",
        "0",
    )

    try:
        app_module._env_int(
            "RETRIEVER_K",
            3,
            minimum=1,
        )
    except app_module.ConfigurationError as exc:
        assert "at least 1" in str(exc)
    else:
        raise AssertionError("ConfigurationError was not raised")


def test_supabase_api_fallback_uses_authoritative_user_identity(
    monkeypatch,
):
    monkeypatch.setenv(
        "APP_ENV",
        "test",
    )

    monkeypatch.delenv(
        "SUPABASE_JWT_SECRET",
        raising=False,
    )

    monkeypatch.setenv(
        "ALLOW_SUPABASE_API_AUTH_FALLBACK",
        "1",
    )

    monkeypatch.setattr(
        auth_module,
        "_fetch_supabase_user",
        lambda token: {
            "id": "trusted-user-id",
            "email": "trusted@example.com",
            "user_metadata": {
                "full_name": "Trusted User",
            },
            "email_confirmed_at": ("2026-01-01T00:00:00Z"),
        },
    )

    def unexpected_jwt_decode(
        *args,
        **kwargs,
    ):
        raise AssertionError("JWT claims must not be decoded when using Supabase API fallback")

    monkeypatch.setattr(
        auth_module.jwt,
        "decode",
        unexpected_jwt_decode,
    )

    user = auth_module.authenticated_user_from_token("fake-token")

    assert user.external_id == "trusted-user-id"

    assert user.email == "trusted@example.com"

    assert user.display_name == "Trusted User"

    assert user.email_verified is True


def test_supabase_api_fallback_rejects_missing_user(
    monkeypatch,
):
    monkeypatch.setenv(
        "APP_ENV",
        "test",
    )

    monkeypatch.delenv(
        "SUPABASE_JWT_SECRET",
        raising=False,
    )

    monkeypatch.setenv(
        "ALLOW_SUPABASE_API_AUTH_FALLBACK",
        "1",
    )

    monkeypatch.setattr(
        auth_module,
        "_fetch_supabase_user",
        lambda token: None,
    )

    try:
        auth_module.authenticated_user_from_token("fake-token")

    except auth_module.AuthenticationError as exc:
        assert str(exc) == "Invalid authentication token"

    else:
        raise AssertionError("AuthenticationError was not raised")


def test_verified_jwt_identity_must_match_supabase_user(
    monkeypatch,
):
    monkeypatch.setenv(
        "SUPABASE_JWT_SECRET",
        "test-secret",
    )

    monkeypatch.setattr(
        auth_module,
        "_fetch_supabase_user",
        lambda token: {
            "id": "supabase-user-id",
            "email": "user@example.com",
        },
    )

    monkeypatch.setattr(
        auth_module,
        "_decode_supabase_token",
        lambda token: {
            "sub": "different-user-id",
            "exp": 2_000_000_000,
            "email": "user@example.com",
        },
    )

    try:
        auth_module.authenticated_user_from_token("verified-token")

    except auth_module.AuthenticationError as exc:
        assert str(exc) == "Authentication identity mismatch"

    else:
        raise AssertionError("AuthenticationError was not raised")

def test_production_auth_requires_jwt_secret(monkeypatch):
    monkeypatch.setenv(
        "APP_ENV",
        "production",
    )

    monkeypatch.delenv(
        "SUPABASE_JWT_SECRET",
        raising=False,
    )

    monkeypatch.setenv(
        "ALLOW_SUPABASE_API_AUTH_FALLBACK",
        "0",
    )

    with pytest.raises(
        auth_module.AuthenticationError,
        match="SUPABASE_JWT_SECRET is required",
    ):
        auth_module.validate_auth_configuration()


def test_production_auth_rejects_api_fallback(monkeypatch):
    monkeypatch.setenv(
        "APP_ENV",
        "production",
    )

    monkeypatch.setenv(
        "SUPABASE_JWT_SECRET",
        "test-secret",
    )

    monkeypatch.setenv(
        "ALLOW_SUPABASE_API_AUTH_FALLBACK",
        "1",
    )

    with pytest.raises(
        auth_module.AuthenticationError,
        match="ALLOW_SUPABASE_API_AUTH_FALLBACK",
    ):
        auth_module.validate_auth_configuration()


def test_production_auth_accepts_secure_configuration(monkeypatch):
    monkeypatch.setenv(
        "APP_ENV",
        "production",
    )

    monkeypatch.setenv(
        "SUPABASE_JWT_SECRET",
        "test-secret",
    )

    monkeypatch.setenv(
        "ALLOW_SUPABASE_API_AUTH_FALLBACK",
        "0",
    )

    auth_module.validate_auth_configuration()


def test_development_auth_requires_explicit_fallback(
    monkeypatch,
):
    monkeypatch.setenv(
        "APP_ENV",
        "development",
    )

    monkeypatch.delenv(
        "SUPABASE_JWT_SECRET",
        raising=False,
    )

    monkeypatch.setenv(
        "ALLOW_SUPABASE_API_AUTH_FALLBACK",
        "0",
    )

    with pytest.raises(
        auth_module.AuthenticationError,
        match="Backend JWT verification is not configured",
    ):
        auth_module.authenticated_user_from_token("fake-token")
