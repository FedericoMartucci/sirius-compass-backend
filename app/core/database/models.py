from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, Text

class Repository(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    url: str = Field(index=True, unique=True)
    name: str
    last_synced_at: Optional[datetime] = None # Key for incremental updates
    
    activities: List["Activity"] = Relationship(back_populates="repository")

class Activity(SQLModel, table=True):
    """
    Stores individual Commits, PRs, or Ticket updates.
    This enables SQL-based metric aggregation (e.g., Velocity).
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    repository_id: int = Field(foreign_key="repository.id", index=True)
    
    # Source Discriminators
    source_platform: str = Field(index=True) # "github" | "linear"
    source_id: str = Field(index=True)       # Commit SHA or Issue ID
    type: str = Field(index=True)            # "COMMIT", "PR_MERGE", "TICKET"
    
    # Core Data
    author: str = Field(index=True)
    timestamp: datetime = Field(index=True)
    title: str
    content: str = Field(sa_column=Column(Text)) # Diff body or Ticket description
    
    # Metrics (Critical for Dashboard)
    story_points: int = 0          # <-- FOR VELOCITY CHART (from Linear)
    files_changed_count: int = 0
    status_label: Optional[str] = None # e.g. "Done"
    
    repository: Optional[Repository] = Relationship(back_populates="activities")