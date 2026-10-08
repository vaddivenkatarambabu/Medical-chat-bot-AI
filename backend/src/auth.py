import hashlib
import hmac
import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

import jwt
from flask import Request
from jwt import InvalidTokenError


class AuthenticationError(RuntimeError):
    pass
def _runtime_environment() -> str:
    value = os.getenv("APP_ENV", "development").strip().lower()

    allowed = {
        "development",
        "test",
        "production",
    }

    if value not in allowed:
        raise AuthenticationError("APP_ENV must be one of: development, test, production")

    return value


def _allow_supabase_api_auth_fallback() -> bool:
    value = (
        os.getenv(
            "ALLOW_SUPABASE_API_AUTH_FALLBACK",
            "0",
        )
        .strip()
        .lower()
    )

    truthy = {
        "1",
        "true",
        "yes",
        "on",
    }

    falsy = {
        "0",
        "false",
        "no",
        "off",
    }

    if value in truthy:
        return True

    if value in falsy:
        return False

    raise AuthenticationError("ALLOW_SUPABASE_API_AUTH_FALLBACK must be a boolean")


def validate_auth_configuration() -> None:
    environment = _runtime_environment()

    secret = os.getenv(
        "SUPABASE_JWT_SECRET",
        "",
    ).strip()

    allow_api_fallback = _allow_supabase_api_auth_fallback()

    if environment == "production":
        if not secret:
            raise AuthenticationError("SUPABASE_JWT_SECRET is required when APP_ENV=production")

        if allow_api_fallback:
            raise AuthenticationError(
                "ALLOW_SUPABASE_API_AUTH_FALLBACK must be disabled when APP_ENV=production"
            )


GUEST_SESSION_PREFIX = "gst1"
GUEST_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
DEFAULT_GUEST_SESSION_TTL_DAYS = 30
MAX_GUEST_SESSION_TTL_DAYS = 90


@dataclass(frozen=True)
class AuthenticatedUser:
    provider: str
    external_id: str
    email: str | None
    display_name: str | None
    avatar_url: str | None
    token_hash: str | None
    expires_at: datetime | None
    email_verified_at: datetime | None

    @property
    def email_verified(self) -> bool:
        return self.email_verified_at is not None


@dataclass(frozen=True)
class RequestIdentity:
    user: AuthenticatedUser | None
    guest_session_id: str | None

    @property
    def is_authenticated(self) -> bool:
        return self.user is not None


def token_sha256(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _env_int(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = os.getenv(name)

    if value is None:
        return default

    try:
        parsed = int(value)
    except ValueError as exc:
        raise AuthenticationError(f"{name} must be an integer") from exc

    if parsed < minimum or parsed > maximum:
        raise AuthenticationError(f"{name} must be between {minimum} and {maximum}")

    return parsed


def _guest_session_secret() -> str:
    return os.getenv("GUEST_SESSION_SECRET", "").strip()


def _guest_session_ttl() -> timedelta:
    days = _env_int(
        "GUEST_SESSION_TTL_DAYS",
        DEFAULT_GUEST_SESSION_TTL_DAYS,
        minimum=1,
        maximum=MAX_GUEST_SESSION_TTL_DAYS,
    )

    return timedelta(days=days)


def _allow_unsigned_guest_sessions() -> bool:
    return (
        os.getenv(
            "ALLOW_UNSIGNED_GUEST_SESSIONS",
            "1",
        ).strip()
        != "0"
    )


def _validate_unsigned_guest_session_id(value: str) -> str:
    if not GUEST_SESSION_ID_PATTERN.match(value):
        raise AuthenticationError("Invalid guest session")

    return value


def _guest_signature(
    body: str,
    secret: str,
) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        body.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def create_guest_session_credential() -> tuple[str, datetime]:
    session_id = secrets.token_urlsafe(24)

    expires_at = datetime.now(timezone.utc) + _guest_session_ttl()

    secret = _guest_session_secret()

    if not secret:
        return session_id, expires_at

    body = f"{GUEST_SESSION_PREFIX}.{session_id}.{int(expires_at.timestamp())}"

    signature = _guest_signature(
        body,
        secret,
    )

    return (
        f"{body}.{signature}",
        expires_at,
    )


def normalize_guest_session_id(
    value: str | None,
) -> str | None:
    if not value:
        return None

    credential = value.strip()

    if not credential:
        return None

    if not credential.startswith(f"{GUEST_SESSION_PREFIX}."):
        if _allow_unsigned_guest_sessions():
            return _validate_unsigned_guest_session_id(credential)

        raise AuthenticationError("Signed guest session is required")

    secret = _guest_session_secret()

    if not secret:
        raise AuthenticationError("GUEST_SESSION_SECRET is required for signed guest sessions")

    parts = credential.split(".")

    if len(parts) != 4:
        raise AuthenticationError("Invalid guest session")

    (
        prefix,
        session_id,
        expires_at_raw,
        signature,
    ) = parts

    if prefix != GUEST_SESSION_PREFIX:
        raise AuthenticationError("Invalid guest session")

    session_id = _validate_unsigned_guest_session_id(session_id)

    try:
        expires_at = int(expires_at_raw)
    except ValueError as exc:
        raise AuthenticationError("Invalid guest session") from exc

    if datetime.now(timezone.utc).timestamp() >= expires_at:
        raise AuthenticationError("Guest session expired")

    body = f"{prefix}.{session_id}.{expires_at_raw}"

    expected = _guest_signature(
        body,
        secret,
    )

    if not hmac.compare_digest(
        signature,
        expected,
    ):
        raise AuthenticationError("Invalid guest session")

    return session_id


def extract_bearer_token(
    request: Request,
) -> str | None:
    auth_header = request.headers.get(
        "Authorization",
        "",
    ).strip()

    if not auth_header:
        return None

    scheme, _, token = auth_header.partition(" ")

    if scheme.lower() != "bearer" or not token.strip():
        raise AuthenticationError("Only Bearer authentication is supported")

    return token.strip()


def _as_datetime(
    timestamp: Any,
) -> datetime | None:
    if timestamp is None:
        return None

    try:
        return datetime.fromtimestamp(
            int(timestamp),
            tz=timezone.utc,
        )
    except (
        TypeError,
        ValueError,
        OSError,
    ):
        return None


def _parse_datetime(
    value: Any,
) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None

    try:
        normalized = value.replace(
            "Z",
            "+00:00",
        )

        return datetime.fromisoformat(normalized)

    except ValueError:
        return None


def _fetch_supabase_user(
    token: str,
) -> dict[str, Any] | None:
    supabase_url = (
        os.getenv(
            "SUPABASE_URL",
            "",
        )
        .strip()
        .rstrip("/")
    )

    supabase_key = (
        os.getenv(
            "SUPABASE_SERVICE_ROLE_KEY",
            "",
        ).strip()
        or os.getenv(
            "SUPABASE_PUBLISHABLE_KEY",
            "",
        ).strip()
        or os.getenv(
            "SUPABASE_ANON_KEY",
            "",
        ).strip()
    )

    if not supabase_url or not supabase_key:
        return None

    request = UrlRequest(
        f"{supabase_url}/auth/v1/user",
        headers={
            "apikey": supabase_key,
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
        method="GET",
    )

    try:
        with urlopen(
            request,
            timeout=5,
        ) as response:
            payload = response.read().decode("utf-8")

            data = json.loads(payload)

            return data if isinstance(data, dict) else None

    except (
        HTTPError,
        URLError,
        TimeoutError,
        json.JSONDecodeError,
    ):
        return None


def _decode_supabase_token(
    token: str,
) -> dict[str, Any]:
    secret = os.getenv(
        "SUPABASE_JWT_SECRET",
        "",
    ).strip()

    if not secret:
        raise AuthenticationError("SUPABASE_JWT_SECRET is required to trust backend Bearer tokens")

    algorithms = [
        item.strip()
        for item in os.getenv(
            "SUPABASE_JWT_ALGORITHMS",
            "HS256",
        ).split(",")
        if item.strip()
    ]

    audience = (
        os.getenv(
            "SUPABASE_JWT_AUDIENCE",
            "authenticated",
        ).strip()
        or None
    )

    try:
        return jwt.decode(
            token,
            secret,
            algorithms=algorithms,
            audience=audience,
            options={
                "require": [
                    "exp",
                    "sub",
                ]
            },
        )

    except InvalidTokenError as exc:
        raise AuthenticationError("Invalid authentication token") from exc


def authenticated_user_from_token(
    token: str,
) -> AuthenticatedUser:
    """
    Authenticate a Supabase bearer token.

    With SUPABASE_JWT_SECRET configured:
    - Cryptographically verify the JWT.
    - Use verified JWT `sub` as the identity.
    - When Supabase /auth/v1/user also returns
      a user, require its ID to match the
      verified JWT subject.

    Without SUPABASE_JWT_SECRET:
    - Require the authoritative Supabase
      /auth/v1/user response.
    - Use only that API response for identity.
    - Never decode the JWT in fallback mode.
    """

    supabase_user = _fetch_supabase_user(token)

    secret = os.getenv(
        "SUPABASE_JWT_SECRET",
        "",
    ).strip()

    if secret:
        # Cryptographically verified JWT.
        claims = _decode_supabase_token(token)

        if supabase_user is not None:
            supabase_user_id = supabase_user.get("id")

            verified_jwt_subject = claims.get("sub")

            if (
                supabase_user_id
                and verified_jwt_subject
                and supabase_user_id != verified_jwt_subject
            ):
                raise AuthenticationError("Authentication identity mismatch")

        external_id = claims.get("sub")

    else:
        if not _allow_supabase_api_auth_fallback():
            raise AuthenticationError(
            "Backend JWT verification is not configured. "
            "Set SUPABASE_JWT_SECRET or explicitly enable "
            "ALLOW_SUPABASE_API_AUTH_FALLBACK for non-production development."
            )

        if supabase_user is None:
            raise AuthenticationError(
            "Invalid authentication token"
                )

        external_id = supabase_user.get("id")

    if not isinstance(external_id, str) or not external_id:
        raise AuthenticationError("Authenticated user does not contain a valid subject")

    metadata: dict[str, Any] = {}

    if isinstance(
        supabase_user,
        dict,
    ):
        metadata = supabase_user.get("user_metadata") or {}

    if not isinstance(metadata, dict):
        metadata = {}

    if secret:
        metadata = metadata or claims.get("user_metadata") or {}

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

    if secret:
        email = (
            supabase_user.get("email")
            if isinstance(
                supabase_user,
                dict,
            )
            else None
        ) or claims.get("email")
    else:
        email = supabase_user.get("email")

    display_name = (
        metadata.get("full_name")
        or metadata.get("name")
        or (claims.get("name") if secret else None)
        or email
    )

    avatar_url = metadata.get("avatar_url") or (claims.get("picture") if secret else None)

    email_verified_at = None

    if isinstance(
        supabase_user,
        dict,
    ):
        email_verified_at = _parse_datetime(
            supabase_user.get("email_confirmed_at")
        ) or _parse_datetime(supabase_user.get("confirmed_at"))

    if email_verified_at is None and secret:
        email_verified_at = _parse_datetime(claims.get("email_confirmed_at"))

    return AuthenticatedUser(
        provider="supabase",
        external_id=external_id,
        email=(
            email
            if isinstance(
                email,
                str,
            )
            else None
        ),
        display_name=(
            display_name
            if isinstance(
                display_name,
                str,
            )
            else None
        ),
        avatar_url=(
            avatar_url
            if isinstance(
                avatar_url,
                str,
            )
            else None
        ),
        token_hash=token_sha256(token),
        expires_at=(_as_datetime(claims.get("exp")) if secret else None),
        email_verified_at=email_verified_at,
    )


def get_request_identity(
    request: Request,
    guest_session_id: str | None = None,
    require_auth: bool = False,
    require_verified: bool = False,
) -> RequestIdentity:
    token = extract_bearer_token(request)

    if token:
        try:
            user = authenticated_user_from_token(token)

            if require_verified and not user.email_verified:
                raise AuthenticationError("Email verification is required")

            return RequestIdentity(
                user=user,
                guest_session_id=guest_session_id,
            )

        except AuthenticationError:
            if require_auth:
                raise

    if require_auth:
        raise AuthenticationError("Authentication is required")

    header_guest_session_id = request.headers.get("X-Guest-Session-Id")

    return RequestIdentity(
        user=None,
        guest_session_id=normalize_guest_session_id(guest_session_id or header_guest_session_id),
    )
