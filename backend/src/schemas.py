from dataclasses import dataclass
import re
from typing import Any
from urllib.parse import urlparse

from flask import Request


MAX_TITLE_LENGTH = 120
MAX_EXTERNAL_ID_LENGTH = 255
MAX_CLIENT_MESSAGE_ID_LENGTH = 128
MAX_GUEST_SESSION_ID_LENGTH = 128
MAX_REDIRECT_URL_LENGTH = 500
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class RequestValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ChatPayload:
    message: str
    conversation_id: str | None
    client_message_id: str | None
    guest_session_id: str | None


@dataclass(frozen=True)
class ConversationPayload:
    title: str | None
    guest_session_id: str | None


@dataclass(frozen=True)
class EmailAuthPayload:
    email: str
    redirect_to: str
    create_user: bool


def _clean_string(value: Any, max_length: int, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RequestValidationError(f"{field_name} must be a string")

    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) > max_length:
        raise RequestValidationError(f"{field_name} is too long")
    return cleaned


def _request_data(request: Request) -> dict[str, Any]:
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            raise RequestValidationError("JSON body must be an object")
        return payload

    return {
        key: request.values.get(key)
        for key in (
            "msg",
            "message",
            "input",
            "conversation_id",
            "conversationId",
            "client_message_id",
            "clientMessageId",
            "guest_session_id",
            "guestSessionId",
            "title",
        )
    }


def parse_chat_payload(request: Request) -> ChatPayload:
    payload = _request_data(request)
    message = payload.get("msg") or payload.get("message") or payload.get("input")

    if not isinstance(message, str):
        message = ""

    return ChatPayload(
        message=message.strip(),
        conversation_id=_clean_string(
            payload.get("conversation_id") or payload.get("conversationId"),
            MAX_EXTERNAL_ID_LENGTH,
            "conversation_id",
        ),
        client_message_id=_clean_string(
            payload.get("client_message_id") or payload.get("clientMessageId"),
            MAX_CLIENT_MESSAGE_ID_LENGTH,
            "client_message_id",
        ),
        guest_session_id=_clean_string(
            payload.get("guest_session_id")
            or payload.get("guestSessionId")
            or request.headers.get("X-Guest-Session-Id"),
            MAX_GUEST_SESSION_ID_LENGTH,
            "guest_session_id",
        ),
    )


def parse_conversation_payload(request: Request) -> ConversationPayload:
    payload = _request_data(request)
    return ConversationPayload(
        title=_clean_string(payload.get("title"), MAX_TITLE_LENGTH, "title"),
        guest_session_id=_clean_string(
            payload.get("guest_session_id")
            or payload.get("guestSessionId")
            or request.headers.get("X-Guest-Session-Id"),
            MAX_GUEST_SESSION_ID_LENGTH,
            "guest_session_id",
        ),
    )


def parse_email_auth_payload(request: Request) -> EmailAuthPayload:
    payload = _request_data(request)
    email = _clean_string(payload.get("email"), 320, "email")
    redirect_to = _clean_string(
        payload.get("redirect_to") or payload.get("redirectTo"),
        MAX_REDIRECT_URL_LENGTH,
        "redirect_to",
    )

    if not email or not EMAIL_PATTERN.match(email):
        raise RequestValidationError("Enter a valid email address")

    if not redirect_to:
        raise RequestValidationError("redirect_to is required")

    redirect_url = urlparse(redirect_to)
    if redirect_url.scheme not in {"http", "https"} or not redirect_url.netloc:
        raise RequestValidationError("redirect_to must be a valid HTTP(S) URL")

    create_user = payload.get("create_user")
    if create_user is None:
        create_user = payload.get("createUser")
    if create_user is None:
        create_user = True
    if not isinstance(create_user, bool):
        raise RequestValidationError("create_user must be a boolean")

    return EmailAuthPayload(
        email=email.lower(),
        redirect_to=redirect_to,
        create_user=create_user,
    )
