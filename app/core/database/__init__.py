from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, JSON

class Repository(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    url: str = Field(index=True, unique=True)
    name: str
    last_analyzed: Optional[datetime] = None
    
    reports: List["AnalysisReport"] = Relationship(back_populates="repository")

class AnalysisReport(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    repository_id: int = Field(foreign_key="repository.id")
    developer_name: str = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Metrics (For graphing)
    quality_score: int
    prs_merged: int
    commits_count: int
    
    # Detailed Data (JSON)
    # We store lists/dicts as JSON using SA Column
    detected_skills: List[str] = Field(default=[], sa_column=Column(JSON))
    feedback_summary: str # Markdown text
    security_alerts: bool
    
    repository: Optional[Repository] = Relationship(back_populates="reports")
    
    # Future: Link to Linear Ticket ID
    linear_ticket_id: Optional[str] = None