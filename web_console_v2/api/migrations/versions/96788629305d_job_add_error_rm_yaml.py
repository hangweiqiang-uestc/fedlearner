"""empty message

Revision ID: 96788629305d
Revises: 20d2e65b1fd2
Create Date: 2021-04-11 14:18:20.590194

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '96788629305d'
down_revision = '20d2e65b1fd2'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('job_v2', sa.Column('error_message', sa.Text(), nullable=True, comment='error message'))
    op.drop_column('job_v2', 'yaml_template')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('job_v2', sa.Column('yaml_template', mysql.TEXT(), nullable=True, comment='yaml_template'))
    op.drop_column('job_v2', 'error_message')
    # ### end Alembic commands ###