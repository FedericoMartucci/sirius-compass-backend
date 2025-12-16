from enum import Enum
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class UnifiedStatus(str, Enum):
    """
    Universal Sirius statuses.
    """
    TODO = "TODO"
    IN_PROGRESS = "IN_PROGRESS"
    REVIEW = "REVIEW"
    BLOCKED = "BLOCKED"
    DONE = "DONE"

class UnifiedTask(BaseModel):
    source_id: str = Field(..., description="Original ID")
    source_platform: str = Field(..., description="Platform name")
    title: str
    description: Optional[str] = None
    status: UnifiedStatus
    assignee: Optional[str] = None
    url: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        use_enum_values = True

class ActivityType(str, Enum):
    COMMIT = "COMMIT"
    PR_OPEN = "PR_OPEN"
    PR_MERGE = "PR_MERGE"
    PR_COMMENT = "PR_COMMENT"
    TICKET = "TICKET"  # <--- NEW: Required for Linear integration

class UnifiedActivity(BaseModel):
    """
    Agnostic representation of technical activity.
    """
    source_id: str = Field(..., description="Commit Hash or ID")
    source_platform: str = "github" # Default to github, overwrite for linear
    type: ActivityType
    author: str
    title: Optional[str] = None
    content: str
    related_task_id: Optional[str] = None
    timestamp: datetime
    url: Optional[str] = None
    additions: int = 0
    deletions: int = 0
    files_changed: List[str] = Field(default_factory=list)
    story_points: int = 0
    status_label: Optional[str] = None
    external_key: Optional[str] = None

    # Optional ticket/provider metadata (mainly used for Linear ingestion)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assignee: Optional[str] = None
    status_type: Optional[str] = None
    cycle_name: Optional[str] = None
    cycle_number: Optional[int] = None
    cycle_starts_at: Optional[datetime] = None
    cycle_ends_at: Optional[datetime] = None

class DeveloperReport(BaseModel):
    developer_name: str
    period_start: datetime
    period_end: datetime
    tasks_completed: int
    prs_merged: int
    quality_score: int
    feedback_summary: str
    detected_skills: List[str]
