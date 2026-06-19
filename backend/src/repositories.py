from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy import or_
from sqlalchemy.orm import Session

from src.auth import AuthenticatedUser, RequestIdentity
from src.models import Conversation, Message, User, UserSession, new_uuid


class RepositoryError(RuntimeError):
    pass


class NotFoundError(RepositoryError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _message_parts(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def _title_from_message(message: str) -> str:
    normalized = " ".join(message.split())
    if not normalized:
        return "New consultation"
    return normalized[:117] + "..." if len(normalized) > 120 else normalized


def _external_conversation_id(
    conversation_id: str | None,
    guest_session_id: str | None,
) -> str | None:
    if conversation_id and conversation_id != "guest":
        return conversation_id
    if guest_session_id:
        return f"guest:{guest_session_id}"
    return None


def serialize_conversation(conversation: Conversation) -> dict[str, Any]:
    return {
        "id": conversation.id,
        "external_id": conversation.external_id,
        "title": conversation.title,
        "summary": conversation.summary,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
    }


def serialize_user(user: User) -> dict[str, Any]:
    return {
        "id": user.id,
        "auth_provider": user.auth_provider,
        "external_id": user.external_id,
        "email": user.email,
        "display_name": user.display_name,
        "avatar_url": user.avatar_url,
        "is_guest": user.is_guest,
        "email_verified": user.email_verified,
        "email_verified_at": (
            user.email_verified_at.isoformat() if user.email_verified_at else None
        ),
        "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
        "created_at": user.created_at.isoformat(),
        "updated_at": user.updated_at.isoformat(),
    }


def serialize_message(message: Message) -> dict[str, Any]:
    return {
        "id": message.id,
        "conversation_id": message.conversation_id,
        "role": message.role,
        "content": message.content,
        "parts": message.parts,
        "client_message_id": message.client_message_id,
        "created_at": message.created_at.isoformat(),
    }


class ChatRepository:
    def resolve_user(
        self,
        db: Session,
        identity: RequestIdentity,
        *,
        create: bool,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> User | None:
        if identity.user:
            return self._upsert_authenticated_user(
                db,
                identity.user,
                user_agent=user_agent,
                ip_address=ip_address,
            )

        if identity.guest_session_id:
            return self._upsert_guest_user(db, identity.guest_session_id)

        if create:
            return None
        return None

    def _upsert_authenticated_user(
        self,
        db: Session,
        auth_user: AuthenticatedUser,
        *,
        user_agent: str | None,
        ip_address: str | None,
    ) -> User:
        user = db.scalar(
            select(User).where(
                User.auth_provider == auth_user.provider,
                User.external_id == auth_user.external_id,
            )
        )

        if user is None:
            user = User(
                auth_provider=auth_user.provider,
                external_id=auth_user.external_id,
                email=auth_user.email,
                display_name=auth_user.display_name,
                avatar_url=auth_user.avatar_url,
                is_guest=False,
                email_verified=auth_user.email_verified,
                email_verified_at=auth_user.email_verified_at,
                last_login_at=_now(),
            )
            db.add(user)
            db.flush()
        else:
            user.email = auth_user.email
            user.display_name = auth_user.display_name
            user.avatar_url = auth_user.avatar_url
            user.email_verified = auth_user.email_verified
            user.email_verified_at = auth_user.email_verified_at
            user.last_login_at = _now()
            user.updated_at = _now()

        user.email_verified = auth_user.email_verified
        user.email_verified_at = auth_user.email_verified_at
        user.last_login_at = _now()

        if auth_user.token_hash:
            session = db.scalar(
                select(UserSession).where(UserSession.token_hash == auth_user.token_hash)
            )
            if session is None:
                db.add(
                    UserSession(
                        user_id=user.id,
                        token_hash=auth_user.token_hash,
                        auth_provider=auth_user.provider,
                        user_agent=user_agent,
                        ip_address=ip_address,
                        expires_at=auth_user.expires_at,
                    )
                )
            else:
                session.last_seen_at = _now()
                session.user_agent = user_agent or session.user_agent
                session.ip_address = ip_address or session.ip_address

        return user

    def revoke_session(self, db: Session, token_hash: str) -> bool:
        session = db.scalar(
            select(UserSession).where(UserSession.token_hash == token_hash)
        )

        if session is None:
            return False

        session.revoked_at = _now()
        db.flush()
        return True

    def _upsert_guest_user(self, db: Session, guest_session_id: str) -> User:
        user = db.scalar(
            select(User).where(
                User.auth_provider == "guest",
                User.external_id == guest_session_id,
            )
        )

        if user is None:
            user = User(
                auth_provider="guest",
                external_id=guest_session_id,
                display_name="Guest user",
                is_guest=True,
            )
            db.add(user)
            db.flush()
        else:
            user.updated_at = _now()

        return user

    def create_conversation(
        self,
        db: Session,
        user: User | None,
        *,
        title: str | None = None,
        external_id: str | None = None,
        guest_session_id: str | None = None,
    ) -> Conversation:
        conversation = Conversation(
            id=new_uuid(),
            user_id=user.id if user else None,
            external_id=external_id,
            guest_session_id=guest_session_id,
            title=title or "New consultation",
        )
        db.add(conversation)
        db.flush()
        return conversation

    def get_or_create_conversation(
        self,
        db: Session,
        *,
        user: User | None,
        conversation_id: str | None,
        guest_session_id: str | None,
        first_message: str,
    ) -> Conversation:
        external_id = _external_conversation_id(conversation_id, guest_session_id)

        conversation = None
        if conversation_id and conversation_id != "guest":
            filters = [Conversation.id == conversation_id]
            if external_id:
                filters.append(Conversation.external_id == external_id)

            conversation = db.scalar(
                select(Conversation).where(
                    or_(*filters),
                    Conversation.user_id == (user.id if user else None),
                )
            )
        elif external_id:
            conversation = db.scalar(
                select(Conversation).where(
                    Conversation.external_id == external_id,
                    Conversation.user_id == (user.id if user else None),
                )
            )

        if conversation is not None:
            return conversation

        return self.create_conversation(
            db,
            user,
            title=_title_from_message(first_message),
            external_id=external_id,
            guest_session_id=guest_session_id,
        )

    def save_chat_turn(
        self,
        db: Session,
        *,
        identity: RequestIdentity,
        message: str,
        answer: str,
        conversation_id: str | None,
        client_message_id: str | None,
        user_agent: str | None,
        ip_address: str | None,
    ) -> Conversation:
        user = self.resolve_user(
            db,
            identity,
            create=True,
            user_agent=user_agent,
            ip_address=ip_address,
        )
        conversation = self.get_or_create_conversation(
            db,
            user=user,
            conversation_id=conversation_id,
            guest_session_id=identity.guest_session_id,
            first_message=message,
        )

        db.add_all(
            [
                Message(
                    conversation_id=conversation.id,
                    user_id=user.id if user else None,
                    role="user",
                    content=message,
                    parts=_message_parts(message),
                    client_message_id=client_message_id,
                ),
            Message(
                conversation_id=conversation.id,
                user_id=user.id if user else None,
                role="assistant",
                content=answer,
                    parts=_message_parts(answer),
            ),
        ]
        )
        if conversation.title == "New consultation":
            conversation.title = _title_from_message(message)
        conversation.updated_at = _now()
        db.flush()
        return conversation

    def list_conversations(
        self,
        db: Session,
        user: User | None,
        *,
        limit: int = 50,
    ) -> list[Conversation]:
        if user is None:
            return []

        return list(
            db.scalars(
                select(Conversation)
                .where(Conversation.user_id == user.id)
                .order_by(Conversation.updated_at.desc())
                .limit(limit)
            )
        )

    def get_conversation_for_user(
        self,
        db: Session,
        user: User | None,
        conversation_id: str,
    ) -> Conversation:
        if user is None:
            raise NotFoundError("Conversation not found")

        conversation = db.scalar(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user.id,
            )
        )
        if conversation is None:
            raise NotFoundError("Conversation not found")
        return conversation

    def list_messages(
        self,
        db: Session,
        conversation: Conversation,
    ) -> list[Message]:
        return list(
            db.scalars(
                select(Message)
                .where(Message.conversation_id == conversation.id)
                .order_by(Message.created_at.asc())
            )
        )

    def rename_conversation(
        self,
        db: Session,
        conversation: Conversation,
        title: str,
    ) -> Conversation:
        conversation.title = title
        conversation.updated_at = _now()
        db.flush()
        return conversation

    def delete_conversation(self, db: Session, conversation: Conversation) -> None:
        db.delete(conversation)
        db.flush()
