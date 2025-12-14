import os
from sqlmodel import SQLModel, create_engine, Session

# Default to SQLite for local dev if env vars are missing, but prefer Postgres
POSTGRES_USER = os.getenv("POSTGRES_USER", "sirius_compass_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "sirius_compass_password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "sirius_compass_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5332")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Fallback to SQLite if explicitly requested or if we want a default
# DATABASE_URL = "sqlite:///database.db"

engine = create_engine(DATABASE_URL, echo=False)

def create_db_and_tables():
    """Initializes the database schema."""
    # Ensure all SQLModel tables are registered before creating the schema.
    import app.core.database.models  # noqa: F401
    SQLModel.metadata.create_all(engine)

def get_session():
    """Dependency for FastAPI to get DB session."""
    with Session(engine) as session:
        yield session
