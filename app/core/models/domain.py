from enum import Enum
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field # type: ignore

class UnifiedStatus(str, Enum):
    """
    Universal Sirius statuses.
    Any status from Trello/Jira must be mapped to one of these to maintain consistency.
    """
    TODO = "TODO"             # Backlog / To Do
    IN_PROGRESS = "IN_PROGRESS" # Doing / In Dev
    REVIEW = "REVIEW"         # QA / Code Review / PR Open
    BLOCKED = "BLOCKED"       # Waiting / Blocked
    DONE = "DONE"             # Completed / Merged

class UnifiedTask(BaseModel):
    """
    Agnostic representation of a management task (ticket).
    Used to normalize data coming from Trello, Jira, Asana, etc.
    """
    source_id: str = Field(..., description="Original ID in the source platform (e.g., Trello Card ID)")
    source_platform: str = Field(..., description="Name of the source platform (e.g., 'trello', 'jira')")
    title: str = Field(..., description="Title or summary of the task")
    description: Optional[str] = Field(None, description="Detailed body or description")
    status: UnifiedStatus = Field(..., description="Status normalized to the Sirius standard")
    assignee: Optional[str] = Field(None, description="Identifier of the assigned developer")
    url: str = Field(..., description="Direct link to the resource")
    created_at: datetime
    updated_at: datetime

    class Config:
        use_enum_values = True

class ActivityType(str, Enum):
    COMMIT = "COMMIT"
    PR_OPEN = "PR_OPEN"
    PR_MERGE = "PR_MERGE"
    PR_COMMENT = "PR_COMMENT"

class UnifiedActivity(BaseModel):
    """
    Agnostic representation of technical activity in repositories.
    Normalizes data from GitHub, GitLab, Bitbucket, etc.
    """
    source_id: str = Field(..., description="Commit Hash or Pull Request ID")
    author: str = Field(..., description="User who performed the action")
    type: ActivityType = Field(..., description="Type of activity performed")
    content: str = Field(..., description="Commit message, PR body, or summary diff")
    related_task_id: Optional[str] = Field(None, description="ID of the related task (if detected)")
    timestamp: datetime
    url: Optional[str] = Field(None, description="Direct link to the commit or PR")
    additions: int = 0
    deletions: int = 0
    files_changed: List[str] = Field(default_factory=list, description="List of modified files")

class DeveloperReport(BaseModel):
    """
    Output model for the performance and quality report.
    """
    developer_name: str
    period_start: datetime
    period_end: datetime
    tasks_completed: int
    prs_merged: int
    quality_score: int = Field(..., ge=0, le=10, description="Score 0-10 calculated by AI")
    feedback_summary: str = Field(..., description="Generated text with improvement suggestions")
    detected_skills: List[str] = Field(default_factory=list, description="Inferred technical skills")