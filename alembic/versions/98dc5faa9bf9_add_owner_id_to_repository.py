"""add_owner_id_to_repository

Revision ID: 98dc5faa9bf9
Revises: b3d7afa09adb
Create Date: 2025-12-14 19:04:07.530051

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = '98dc5faa9bf9'
down_revision: Union[str, Sequence[str], None] = 'b3d7afa09adb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('repository', schema=None) as batch_op:
        # NOTE:
        # We add this column as nullable to avoid breaking existing installations.
        # If you already have repository rows, backfill owner_id values and then
        # create a follow-up migration to make it NOT NULL.
        batch_op.add_column(sa.Column('owner_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        batch_op.create_index(batch_op.f('ix_repository_owner_id'), ['owner_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('repository', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_repository_owner_id'))
        batch_op.drop_column('owner_id')
