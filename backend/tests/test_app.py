import os
import tempfile
from pathlib import Path

os.environ["DATABASE_URL"] = (
    f"sqlite:///{(Path(tempfile.gettempdir()) / 'medicore_test.sqlite3').as_posix()}"
)

import app as app_module
import pytest
from sqlalchemy import select

from src import models  # noqa: F401
from src.database import Base, engine, session_scope
from src.models import Conversation, Message, User


class FakeChain:
    def invoke(self, payload):
        return {"answer": f"answer for {payload['input']}"}


@pytest.fixture(autouse=True)
def reset_database():
    app_module.rate_limiter = app_module.RateLimiter()
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
    assert response.get_json() == {"status": "ok", "database": "ok"}


def test_send_otp_validates_email():
    client = app_module.create_app().test_client()

    response = client.post(
        "/api/auth/send-otp",
        json={
            "email": "not-an-email",
            "redirect_to": "http://127.0.0.1:8080/verify-otp",
        },
    )

    assert response.status_code == 400
    assert "valid email" in response.get_json()["error"]


def test_send_otp_calls_supabase_email_service(monkeypatch):
    calls = []

    def fake_send_email_otp(email, redirect_to, *, create_user=True):
        calls.append((email, redirect_to, create_user))

    monkeypatch.setattr(app_module, "send_email_otp", fake_send_email_otp)
    client = app_module.create_app().test_client()

    response = client.post(
        "/api/auth/send-otp",
        json={
            "email": "Person@Example.com",
            "redirect_to": "http://127.0.0.1:8080/verify-otp",
        },
    )

    assert response.status_code == 200
    assert response.get_json()["ok"] is True
    assert calls == [
        ("person@example.com", "http://127.0.0.1:8080/verify-otp", True)
    ]


def test_send_otp_can_disable_user_creation(monkeypatch):
    calls = []

    def fake_send_email_otp(email, redirect_to, *, create_user=True):
        calls.append((email, redirect_to, create_user))

    monkeypatch.setattr(app_module, "send_email_otp", fake_send_email_otp)
    client = app_module.create_app().test_client()

    response = client.post(
        "/api/auth/send-otp",
        json={
            "email": "Person@Example.com",
            "redirect_to": "http://127.0.0.1:8080/verify-otp",
            "create_user": False,
        },
    )

    assert response.status_code == 200
    assert calls == [
        ("person@example.com", "http://127.0.0.1:8080/verify-otp", False)
    ]


def test_send_otp_rejects_untrusted_redirect(monkeypatch):
    def fake_send_email_otp(email, redirect_to, *, create_user=True):
        raise AssertionError("OTP email should not be requested")

    monkeypatch.setattr(app_module, "send_email_otp", fake_send_email_otp)
    client = app_module.create_app().test_client()

    response = client.post(
        "/api/auth/send-otp",
        json={
            "email": "person@example.com",
            "redirect_to": "https://example.invalid/verify-otp",
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

    response = client.post("/get", data={"msg": "   "})

    assert response.status_code == 400
    assert "Please enter" in response.get_data(as_text=True)


def test_chat_rejects_long_message():
    client = app_module.create_app().test_client()

    response = client.post("/get", data={"msg": "x" * (app_module.MAX_MESSAGE_LENGTH + 1)})

    assert response.status_code == 413
    assert "too long" in response.get_data(as_text=True)


def test_chat_returns_chain_answer(monkeypatch):
    monkeypatch.setattr(app_module, "get_rag_chain", lambda: FakeChain())
    client = app_module.create_app().test_client()

    response = client.post("/get", data={"msg": "What is fever?"})

    assert response.status_code == 200
    assert response.get_data(as_text=True) == "answer for What is fever?"


def test_chat_accepts_json_payload(monkeypatch):
    monkeypatch.setattr(app_module, "get_rag_chain", lambda: FakeChain())
    client = app_module.create_app().test_client()

    response = client.post("/get", json={"message": "What is cough?"})

    assert response.status_code == 200
    assert response.get_data(as_text=True) == "answer for What is cough?"


def test_chat_persists_guest_turn(monkeypatch):
    monkeypatch.setattr(app_module, "get_rag_chain", lambda: FakeChain())
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
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "What is cough?"
    assert messages[1].content == "answer for What is cough?"


def test_chat_saves_turn_to_created_conversation(monkeypatch):
    monkeypatch.setattr(app_module, "get_rag_chain", lambda: FakeChain())
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


def test_chat_returns_configuration_error(monkeypatch):
    def raise_configuration_error():
        raise app_module.ConfigurationError("missing config")

    monkeypatch.setattr(app_module, "get_rag_chain", raise_configuration_error)
    client = app_module.create_app().test_client()

    response = client.post("/get", data={"msg": "What is fever?"})

    assert response.status_code == 503
    assert "not configured" in response.get_data(as_text=True)


def test_env_int_rejects_invalid_range(monkeypatch):
    monkeypatch.setenv("RETRIEVER_K", "0")

    try:
        app_module._env_int("RETRIEVER_K", 3, minimum=1)
    except app_module.ConfigurationError as exc:
        assert "at least 1" in str(exc)
    else:
        raise AssertionError("ConfigurationError was not raised")
