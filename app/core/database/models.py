from typing import Any, Dict, List, Optional
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship, UniqueConstraint
from sqlalchemy import Column, Text, JSON

class Repository(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("owner_id", "url", name="unique_user_repo"),
        {"extend_existing": True}
    )
    
    id: Optional[int] = Field(default=None, primary_key=True)
    url: str = Field(index=True) # unique=False to allow multi-tenancy
    name: str
    owner_id: str = Field(index=True)  # Auth0 user ID for multi-tenancy
    last_analyzed: Optional[datetime] = None
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
    commits_count: int = 0
    security_alerts: bool = False
    
    # The Summary used by the Chatbot
    detected_skills: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    feedback_summary: str = Field(sa_column=Column(Text))

    # Future: Link to Linear Ticket ID
    linear_ticket_id: Optional[str] = None
    
    repository: Optional[Repository] = Relationship(back_populates="reports")


class Project(SQLModel, table=True):
    """
    Internal project entity (can map to multiple code repositories and ticket providers).

    This decouples "Project" (product/team context) from a single Git repository.
    """

    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class ProjectOwner(SQLModel, table=True):
    """Maps a Project to its owning user (multi-tenancy).

    We keep ownership in a separate table to avoid relying on ProjectRepository links,
    so projects can exist before any repository is connected.
    """

    __table_args__ = (
        UniqueConstraint("owner_id", "project_id", name="unique_owner_project"),
        {"extend_existing": True},
    )

    project_id: int = Field(foreign_key="project.id", primary_key=True)
    owner_id: str = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class ProjectRepository(SQLModel, table=True):
    """
    Many-to-many relationship between Project and Repository.
    """

    __table_args__ = {"extend_existing": True}

    project_id: int = Field(foreign_key="project.id", primary_key=True)
    repository_id: int = Field(foreign_key="repository.id", primary_key=True)
    is_primary: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class Guest(SQLModel, table=True):
    """A guest user invited to access one or more projects.

    Guests are primarily identified by email (invite flow). If/when the invited person
    authenticates, we can also link their Auth0 subject to `external_user_id`.
    """

    __table_args__ = (
        UniqueConstraint("email", name="unique_guest_email"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True)

    external_user_id: Optional[str] = Field(default=None, index=True)
    invited_by_owner_id: Optional[str] = Field(default=None, index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    accepted_at: Optional[datetime] = Field(default=None, index=True)


class ProjectGuest(SQLModel, table=True):
    """Many-to-many relationship between Project and Guest."""

    __table_args__ = (
        UniqueConstraint("project_id", "guest_id", name="unique_project_guest"),
        {"extend_existing": True},
    )

    project_id: int = Field(foreign_key="project.id", primary_key=True)
    guest_id: int = Field(foreign_key="guest.id", primary_key=True)
    role: str = Field(default="viewer", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class UserSettings(SQLModel, table=True):
    """Per-user settings keyed by Auth0 `sub` (multi-tenancy)."""

    __table_args__ = (
        UniqueConstraint("user_id", name="unique_user_settings_user_id"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)

    # Stored as FK to internal Project.id; returned to clients as string.
    default_project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)
    default_time_range: str = Field(default="30d", index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class Ticket(SQLModel, table=True):
    """
    Canonical ticket entity (Linear/Trello/etc.).

    The current state is stored here; changes over time are stored in TicketEvent.
    """

    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)

    source_platform: str = Field(index=True)  # "linear" | "trello" | ...
    source_id: str = Field(index=True, unique=True)  # Provider-specific ticket ID
    key: Optional[str] = Field(default=None, index=True)  # e.g. "TRI-229"

    title: str
    description: Optional[str] = Field(default=None, sa_column=Column(Text))

    story_points: int = 0
    status_label: Optional[str] = Field(default=None, index=True)

    created_at: datetime = Field(index=True)
    updated_at: Optional[datetime] = Field(default=None, index=True)
    completed_at: Optional[datetime] = None
    url: Optional[str] = None

    raw_payload: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class TicketEvent(SQLModel, table=True):
    """
    Immutable ticket timeline events (status changes, points changes, etc.).

    For MVP we generate these events by diffing snapshots on each sync run.
    """

    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    ticket_id: int = Field(foreign_key="ticket.id", index=True)

    occurred_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    event_type: str = Field(index=True)  # e.g. "created", "status_changed", "points_changed"
    actor: Optional[str] = Field(default=None, index=True)

    from_value: Optional[str] = None
    to_value: Optional[str] = None

    payload: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class ChatThread(SQLModel, table=True):
    """
    Persistent chat thread owned by a PM/user.

    We store the conversation in SQL so LangGraph can be rehydrated across restarts.
    """

    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    external_thread_id: str = Field(index=True, unique=True)
    owner_id: Optional[str] = Field(default=None, index=True)

    title: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class ChatMessage(SQLModel, table=True):
    """
    Individual message inside a ChatThread.

    `role` is aligned with LangChain roles: "system" | "user" | "assistant" | "tool".
    """

    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    chat_thread_id: int = Field(foreign_key="chatthread.id", index=True)

    role: str = Field(index=True)
    content: str = Field(sa_column=Column(Text))

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    message_metadata: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class IntegrationCredential(SQLModel, table=True):
    """
    Encrypted API credentials for external providers (GitHub/Linear/etc.).

    Secrets are encrypted at rest using an application-level master key.
    """

    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    owner_id: Optional[str] = Field(default=None, index=True)

    provider: str = Field(index=True)  # "github" | "linear" | ...
    name: Optional[str] = None

    encrypted_secret: str = Field(sa_column=Column(Text))
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class ProjectIntegration(SQLModel, table=True):
    """
    Project-scoped configuration for an external provider (Linear/GitHub/etc.).

    This allows a single project to map to multiple repositories and a single ticket
    provider configuration (e.g., Linear team key + API credential).
    """

    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)

    provider: str = Field(index=True)  # "github" | "linear" | ...
    credential_id: Optional[int] = Field(default=None, foreign_key="integrationcredential.id", index=True)

    settings: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class SyncRun(SQLModel, table=True):
    """
    Tracks background sync runs for GitHub/Linear ingestion.

    This enables progress reporting in the UI and allows the chat agent to
    determine whether the local DB is complete or needs more data.
    """

    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    owner_id: str = Field(index=True)

    project_id: Optional[int] = Field(default=None, foreign_key="project.id", index=True)
    repository_id: Optional[int] = Field(default=None, foreign_key="repository.id", index=True)

    provider: str = Field(index=True)  # "github" | "linear" | "all"
    status: str = Field(index=True)  # "queued" | "running" | "completed" | "failed"

    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    started_at: Optional[datetime] = Field(default=None, index=True)
    finished_at: Optional[datetime] = Field(default=None, index=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    progress_current: int = 0
    progress_total: Optional[int] = None

    message: Optional[str] = Field(default=None, sa_column=Column(Text))
    details: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class DataCoverage(SQLModel, table=True):
    """
    Stores high-level coverage metadata about what is currently available in the DB.

    For example:
    - provider="github", scope_type="repository", scope_id=<repo_id>
    - provider="linear", scope_type="project", scope_id=<project_id>
    """

    __table_args__ = (
        UniqueConstraint(
            "owner_id",
            "scope_type",
            "scope_id",
            "provider",
            name="unique_owner_scope_provider",
        ),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    owner_id: str = Field(index=True)

    scope_type: str = Field(index=True)  # "repository" | "project"
    scope_id: int = Field(index=True)
    provider: str = Field(index=True)  # "github" | "linear"

    earliest_at: Optional[datetime] = Field(default=None, index=True)
    latest_at: Optional[datetime] = Field(default=None, index=True)
    is_complete: bool = Field(default=False, index=True)

    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    last_run_id: Optional[int] = Field(default=None, foreign_key="syncrun.id", index=True)
    notes: Optional[str] = Field(default=None, sa_column=Column(Text))
