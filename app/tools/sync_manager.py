from langchain_core.tools import tool
from sqlmodel import Session, select
from app.core.database.session import engine
from app.core.database.models import Repository
from app.services.sync import SyncService

@tool
def refresh_repository_data(repo_name: str):
    """
    Triggers a real-time sync with GitHub/Linear for the specified repository.
    Use this when the user asks for the "latest" data, "newest commits", or if the information seems outdated.
    """
    with Session(engine) as session:
        # 1. Resolve Repo
        repo = session.exec(select(Repository).where(Repository.name.contains(repo_name))).first()
        if not repo:
            return "Repository not found. Please specify the correct repository name."

        # 2. Sync
        service = SyncService(session)
        service.ensure_repository_sync(repo.url, days_lookback=2)
        
        return f"Successfully refreshed data for {repo.name}. You can now query for the latest activities."
