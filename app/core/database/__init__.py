"""
Database package public API.

Keep models defined in a single module (`app.core.database.models`) to avoid
SQLAlchemy registry conflicts caused by duplicate class definitions.
"""

from .models import (
    Activity,
    AnalysisReport,
    ChatMessage,
    ChatThread,
    IntegrationCredential,
    Project,
    ProjectRepository,
    Repository,
    Ticket,
    TicketEvent,
)

__all__ = [
    "Activity",
    "AnalysisReport",
    "ChatMessage",
    "ChatThread",
    "IntegrationCredential",
    "Project",
    "ProjectRepository",
    "Repository",
    "Ticket",
    "TicketEvent",
]
