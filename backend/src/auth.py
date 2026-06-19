import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, urlopen

import jwt
from flask import Request
from jwt import InvalidTokenError


class AuthenticationError(RuntimeError):
    pass


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


def extract_bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get("Authorization", "").strip()
    if not auth_header:
        return None

    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthenticationError("Only Bearer authentication is supported")

    return token.strip()


def _as_datetime(timestamp: Any) -> datetime | None:
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None

    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _fetch_supabase_user(token: str) -> dict[str, Any] | None:
    supabase_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    supabase_key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        or os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip()
        or os.getenv("SUPABASE_ANON_KEY", "").strip()
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
        with urlopen(request, timeout=5) as response:
            payload = response.read().decode("utf-8")
            data = json.loads(payload)
            return data if isinstance(data, dict) else None
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None


def _decode_supabase_token(token: str) -> dict[str, Any]:
    secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()
    if not secret:
        raise AuthenticationError(
            "SUPABASE_JWT_SECRET is required to trust backend Bearer tokens"
        )

    algorithms = [
        item.strip()
        for item in os.getenv("SUPABASE_JWT_ALGORITHMS", "HS256").split(",")
        if item.strip()
    ]
    audience = os.getenv("SUPABASE_JWT_AUDIENCE", "authenticated").strip() or None

    try:
        return jwt.decode(
            token,
            secret,
            algorithms=algorithms,
            audience=audience,
            options={"require": ["exp", "sub"]},
        )
    except InvalidTokenError as exc:
        raise AuthenticationError("Invalid authentication token") from exc


def _decode_unverified_supabase_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            options={
                "verify_signature": False,
                "verify_exp": False,
                "verify_aud": False,
            },
        )
    except InvalidTokenError:
        return {}

    return payload if isinstance(payload, dict) else {}


def authenticated_user_from_token(token: str) -> AuthenticatedUser:
    supabase_user = _fetch_supabase_user(token)
    secret = os.getenv("SUPABASE_JWT_SECRET", "").strip()

    if secret:
        claims = _decode_supabase_token(token)
        if supabase_user is None:
            supabase_user = {}
    else:
        if supabase_user is None:
            raise AuthenticationError("Invalid authentication token")
        claims = _decode_unverified_supabase_token(token)

    external_id = claims.get("sub") or supabase_user.get("id")

    if not isinstance(external_id, str) or not external_id:
        raise AuthenticationError("Token does not contain a valid subject")

    metadata = supabase_user.get("user_metadata") or claims.get("user_metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    email = supabase_user.get("email") or claims.get("email")
    display_name = (
        metadata.get("full_name")
        or metadata.get("name")
        or claims.get("name")
        or email
    )
    avatar_url = metadata.get("avatar_url") or claims.get("picture")
    email_verified_at = (
        _parse_datetime(supabase_user.get("email_confirmed_at"))
        or _parse_datetime(supabase_user.get("confirmed_at"))
        or _parse_datetime(claims.get("email_confirmed_at"))
    )

    return AuthenticatedUser(
        provider="supabase",
        external_id=external_id,
        email=email if isinstance(email, str) else None,
        display_name=display_name if isinstance(display_name, str) else None,
        avatar_url=avatar_url if isinstance(avatar_url, str) else None,
        token_hash=token_sha256(token),
        expires_at=_as_datetime(claims.get("exp")),
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
        guest_session_id=guest_session_id or header_guest_session_id,
    )
