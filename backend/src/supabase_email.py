import json
import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


logger = logging.getLogger(__name__)


class SupabaseEmailConfigurationError(RuntimeError):
    pass


class SupabaseEmailDeliveryError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class SupabaseAuthSettings:
    email_provider_enabled: bool
    signup_disabled: bool
    mailer_autoconfirm: bool


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    if not local or not domain:
        return "***"

    visible = local[:2]
    return f"{visible}{'*' * max(3, len(local) - len(visible))}@{domain}"


def _supabase_auth_config() -> tuple[str, str]:
    supabase_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    supabase_key = (
        os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip()
        or os.getenv("SUPABASE_ANON_KEY", "").strip()
    )

    if not supabase_url:
        raise SupabaseEmailConfigurationError("Missing SUPABASE_URL")
    if not supabase_key:
        raise SupabaseEmailConfigurationError(
            "Missing SUPABASE_PUBLISHABLE_KEY or SUPABASE_ANON_KEY"
        )

    return supabase_url, supabase_key


def _json_request(
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    query: dict[str, str] | None = None,
) -> dict[str, Any]:
    supabase_url, supabase_key = _supabase_auth_config()
    encoded_query = f"?{urlencode(query)}" if query else ""
    url = f"{supabase_url}/auth/v1{path}{encoded_query}"
    payload = json.dumps(body or {}).encode("utf-8") if body is not None else None

    request = Request(
        url,
        data=payload,
        method=method,
        headers={
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=10) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        message = _extract_error_message(raw) or f"Supabase Auth returned {exc.code}"
        raise SupabaseEmailDeliveryError(message, status_code=exc.code) from exc
    except (URLError, TimeoutError) as exc:
        raise SupabaseEmailDeliveryError(
            "Could not reach Supabase Auth email service"
        ) from exc
    except json.JSONDecodeError:
        return {}


def _extract_error_message(raw: str) -> str | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip() or None

    for key in ("msg", "message", "error_description", "error"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def get_auth_settings() -> SupabaseAuthSettings:
    payload = _json_request("GET", "/settings")
    external = payload.get("external")
    if not isinstance(external, dict):
        external = {}

    return SupabaseAuthSettings(
        email_provider_enabled=external.get("email") is True,
        signup_disabled=payload.get("disable_signup") is True,
        mailer_autoconfirm=payload.get("mailer_autoconfirm") is True,
    )


def send_email_otp(email: str, redirect_to: str, *, create_user: bool = True) -> None:
    settings = get_auth_settings()
    if not settings.email_provider_enabled:
        raise SupabaseEmailConfigurationError(
            "Supabase Email Auth is disabled for this project"
        )
    if settings.signup_disabled:
        raise SupabaseEmailConfigurationError(
            "Supabase signups are disabled for this project"
        )

    _json_request(
        "POST",
        "/otp",
        query={"redirect_to": redirect_to},
        body={
            "email": email,
            "data": {},
            "create_user": create_user,
            "gotrue_meta_security": {},
            "code_challenge": None,
            "code_challenge_method": None,
        },
    )
    logger.info("Supabase OTP request accepted for %s", mask_email(email))


def send_recovery_email(email: str, redirect_to: str) -> None:
    settings = get_auth_settings()
    if not settings.email_provider_enabled:
        raise SupabaseEmailConfigurationError(
            "Supabase Email Auth is disabled for this project"
        )

    _json_request(
        "POST",
        "/recover",
        query={"redirect_to": redirect_to},
        body={
            "email": email,
            "gotrue_meta_security": {},
            "code_challenge": None,
            "code_challenge_method": None,
        },
    )
    logger.info("Supabase recovery email request accepted for %s", mask_email(email))
