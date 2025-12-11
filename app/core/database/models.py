from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, JSON, Text

class Repository(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    url: str = Field(index=True, unique=True)
    name: str
    last_synced_at: Optional[datetime] = None # Key for incremental updates
    
    activities: List["Activity"] = Relationship(back_populates="repository")

class Activity(SQLModel, table=True):
    """
    Stores individual Commits, PRs, or Ticket updates.
    This allows us to run SQL queries for graphs without asking the AI.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    repository_id: int = Field(foreign_key="repository.id", index=True)
    
    # Core Data
    source_id: str = Field(index=True) # Commit Hash or PR ID
    type: str = Field(index=True) # COMMIT, PR_MERGE, PR_OPEN
    author: str = Field(index=True)
    timestamp: datetime = Field(index=True)
    
    # Content (Stored to avoid re-fetching from GitHub)
    title: str
    content: str = Field(sa_column=Column(Text)) # The Diff or Body
    
    # Metadata for graphs
    files_changed_count: int = 0
    additions: int = 0
    deletions: int = 0
    
    repository: Optional[Repository] = Relationship(back_populates="activities")