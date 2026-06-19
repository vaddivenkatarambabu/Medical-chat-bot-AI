"""add auth user metadata

Revision ID: 20260610_0002
Revises: 20260610_0001
Create Date: 2026-06-10 00:00:01
"""
from alembic import op
import sqlalchemy as sa


revision = "20260610_0002"
down_revision = "20260610_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "app_users",
        sa.Column(
            "email_verified",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "app_users",
        sa.Column("email_verified_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "app_users",
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("app_users", "last_login_at")
    op.drop_column("app_users", "email_verified_at")
    op.drop_column("app_users", "email_verified")
