from datetime import datetime
from pydantic import BaseModel, Field
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


class ConnectionDTO(BaseModel):
    id: int
    type: str
    name: str
    project: str
    status: str
    lastSync: str


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
