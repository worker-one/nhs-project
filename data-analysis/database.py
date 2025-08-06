import os
from sqlalchemy import create_engine

db_path = os.path.join(os.path.dirname(__file__), '../nhs_db.db')
engine = create_engine(f'sqlite:///{db_path}')
