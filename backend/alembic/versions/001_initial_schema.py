"""initial schema — users, policies, predictions, reserving_runs

Revision ID: 001_initial
Revises:
Create Date: 2026-09-02
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── users ───────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("username", sa.String(length=50), nullable=False),
        sa.Column("email", sa.String(length=100), nullable=False),
        sa.Column("hashed_password", sa.String(length=200), nullable=False),
        sa.Column("full_name", sa.String(length=100), nullable=True),
        sa.Column("role", sa.String(length=20), nullable=False, server_default="viewer"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # ── policies ────────────────────────────────────────────────
    op.create_table(
        "policies",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("veh_power", sa.Integer(), nullable=False),
        sa.Column("veh_age", sa.Integer(), nullable=False),
        sa.Column("veh_brand", sa.String(length=10), nullable=False),
        sa.Column("veh_gas", sa.String(length=10), nullable=False),
        sa.Column("driv_age", sa.Integer(), nullable=False),
        sa.Column("bonus_malus", sa.Integer(), nullable=False),
        sa.Column("region", sa.String(length=50), nullable=False),
        sa.Column("area", sa.String(length=5), nullable=False),
        sa.Column("density", sa.Float(), nullable=False),
        sa.Column("exposure", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── predictions ─────────────────────────────────────────────
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("module", sa.String(length=20), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=True),
        sa.Column("request_json", sa.JSON(), nullable=False),
        sa.Column("response_json", sa.JSON(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_predictions_module", "predictions", ["module"])

    # ── reserving_runs ──────────────────────────────────────────
    op.create_table(
        "reserving_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("grcode", sa.Integer(), nullable=False),
        sa.Column("evaluation_year", sa.Integer(), nullable=False),
        sa.Column("ibnr_estimate", sa.Float(), nullable=False),
        sa.Column("mack_lower", sa.Float(), nullable=True),
        sa.Column("mack_upper", sa.Float(), nullable=True),
        sa.Column("conformal_lower", sa.Float(), nullable=True),
        sa.Column("conformal_upper", sa.Float(), nullable=True),
        sa.Column("triangle_json", sa.JSON(), nullable=True),
        sa.Column("ldfs_json", sa.JSON(), nullable=True),
        sa.Column("model_version", sa.String(length=50), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_reserving_runs_grcode", "reserving_runs", ["grcode"])


def downgrade() -> None:
    op.drop_table("reserving_runs")
    op.drop_table("predictions")
    op.drop_table("policies")
    op.drop_table("users")
