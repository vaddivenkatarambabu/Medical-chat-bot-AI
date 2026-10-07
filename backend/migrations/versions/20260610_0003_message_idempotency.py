"""enforce chat message idempotency

Revision ID: 20260610_0003
Revises: 20260610_0002
Create Date: 2026-06-10 00:00:02
"""

from alembic import op

revision = "20260610_0003"
down_revision = "20260610_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_chat_messages_conversation_role_client_message",
        "chat_messages",
        ["conversation_id", "role", "client_message_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_chat_messages_conversation_role_client_message",
        "chat_messages",
        type_="unique",
    )
