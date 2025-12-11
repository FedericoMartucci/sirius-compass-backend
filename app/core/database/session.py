from sqlmodel import SQLModel, create_engine, Session

sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

engine = create_engine(sqlite_url, echo=False)

def create_db_and_tables():
    """Initializes the database schema."""
    SQLModel.metadata.create_all(engine)

def get_session():
    """Dependency for FastAPI to get DB session."""
    with Session(engine) as session:
        yield session