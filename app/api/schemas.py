from datetime import datetime
from pydantic import BaseModel, Field, AliasChoices, ConfigDict
from typing import Any, Dict, List, Optional

class AnalyzeRequest(BaseModel):
    repo_url: str
    developer_name: str = "Unknown"
    lookback_days: int = 720 
    project_name: Optional[str] = None
    linear_team_key: Optional[str] = None
    user_id: Optional[str] = None

class AnalyzeResponse(BaseModel):
    status: str
    report: Dict[str, Any]
    metadata: Dict[str, Any]
    message: Optional[str] = None
    report_summary: Optional[str] = None

class ChatRequest(BaseModel):
    thread_id: str
    message: str
    repo_name: str
    user_id: Optional[str] = None
    project_name: Optional[str] = None


class ProjectDTO(BaseModel):
    id: str
    name: str


class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class GuestDTO(BaseModel):
    id: int
    email: str


class ProjectGuestDTO(BaseModel):
    project_id: int
    guest: GuestDTO
    role: str
    created_at: datetime


class InviteGuestRequest(BaseModel):
    project_id: Optional[int] = None
    project_name: Optional[str] = None
    email: str = Field(..., min_length=3, max_length=320)
    role: str = Field(default="viewer", min_length=3, max_length=50)


class ConnectionDTO(BaseModel):
    id: int
    type: str
    name: str
    project: str
    status: str
    lastSync: str
    last_error: Optional[str] = None


class CreateConnectionRequest(BaseModel):
    type: str
    project_name: str

    repo_url: Optional[str] = None
    github_token: Optional[str] = None

    linear_api_key: Optional[str] = None
    linear_team_key: Optional[str] = None

    user_id: Optional[str] = None


class ChatThreadDTO(BaseModel):
    thread_id: str
    title: str
    updated_at: datetime


class ChatMessageDTO(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReportDTO(BaseModel):
    id: int
    week: str
    project: str
    repository: str
    status: str
    summary: str
    created_at: datetime


class SyncRequest(BaseModel):
    project_name: str = Field(..., min_length=1, max_length=200)
    repo_name: Optional[str] = None

    providers: List[str] = Field(default_factory=lambda: ["github", "linear"])
    full_history: bool = False

    max_commits: Optional[int] = Field(default=300, ge=1, le=10000)
    max_prs: Optional[int] = Field(default=200, ge=1, le=10000)
    max_tickets: Optional[int] = Field(default=200, ge=1, le=10000)


class SyncRunDTO(BaseModel):
    id: int
    status: str
    provider: str
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    progress_current: int = 0
    progress_total: Optional[int] = None
    message: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class UserSettingsDTO(BaseModel):
    """User-scoped UI settings."""

    default_project_id: Optional[str] = None
    default_time_range: str = "30d"


class SaveUserSettingsPayload(BaseModel):
    """Payload for PUT /user-settings.

    Accepts both snake_case and camelCase inputs.
    """

    model_config = ConfigDict(extra="ignore")

    default_project_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("default_project_id", "defaultProjectId"),
    )
    default_time_range: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("default_time_range", "defaultTimeRange"),
    )
