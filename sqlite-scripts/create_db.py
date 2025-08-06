import os
from sqlalchemy import create_engine
from models import Base

def main():
    db_path = os.path.join(os.path.dirname(__file__), '../nhs_db.db')
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)
    print("Database created at", db_path)

if __name__ == "__main__":
    main()
