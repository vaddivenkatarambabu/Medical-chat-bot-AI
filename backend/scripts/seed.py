from src.auth import RequestIdentity
from src.database import init_database, session_scope
from src.repositories import ChatRepository


DEMO_GUEST_SESSION_ID = "demo-guest-session"


def main() -> None:
    init_database()
    repository = ChatRepository()

    identity = RequestIdentity(user=None, guest_session_id=DEMO_GUEST_SESSION_ID)

    with session_scope() as db:
        repository.save_chat_turn(
            db,
            identity=identity,
            message="What should I do for a mild fever?",
            answer=(
                "For a mild fever, rest, drink fluids, and monitor your symptoms. "
                "Seek medical care if the fever is high, persistent, worsening, or "
                "comes with severe symptoms."
            ),
            conversation_id="seed-demo",
            client_message_id="seed-user-message-1",
            user_agent="seed-script",
            ip_address="127.0.0.1",
        )

    print("Seeded demo guest conversation.")


if __name__ == "__main__":
    main()
