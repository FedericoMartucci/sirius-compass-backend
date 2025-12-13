from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, Text

class Repository(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    
    id: Optional[int] = Field(default=None, primary_key=True)
    url: str = Field(index=True, unique=True)
    name: str
    last_synced_at: Optional[datetime] = None
    
    activities: List["Activity"] = Relationship(back_populates="repository")
    reports: List["AnalysisReport"] = Relationship(back_populates="repository")

class Activity(SQLModel, table=True):
    """
    Stores individual Commits, PRs, or Ticket updates.
    Acts as a Data Lake for metrics and analysis.
    """
    __table_args__ = {"extend_existing": True}

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
    
    # Metrics
    story_points: int = 0          # Critical for Velocity Charts
    files_changed_count: int = 0
    status_label: Optional[str] = None # e.g. "Done", "In Progress"
    
    repository: Optional[Repository] = Relationship(back_populates="activities")

class AnalysisReport(SQLModel, table=True):
    """
    Stores the result of the Analyst Graph.
    The Chat Graph reads from this table to answer user questions.
    """
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    repository_id: int = Field(foreign_key="repository.id")
    developer_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # KPIs
    quality_score: int
    prs_merged: int
    
    # The Summary used by the Chatbot
    feedback_summary: str = Field(sa_column=Column(Text))
    
    repository: Optional[Repository] = Relationship(back_populates="reports")