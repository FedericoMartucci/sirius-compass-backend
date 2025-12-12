import os
from datetime import datetime, timezone
from sqlmodel import Session, select
from app.adapters.linear.adapter import LinearAdapter
from app.adapters.github.adapter import GitHubAdapter
from app.core.database.models import Repository, Activity
from app.core.logger import get_logger

logger = get_logger(__name__)

class SyncService:
    def __init__(self, session: Session):
        self.session = session
        self.github = GitHubAdapter()
        self.linear = LinearAdapter()

    def sync_linear_issues(self, repo_db: Repository):
        """
        Ingests tickets from Linear.
        """
        if not self.linear._enabled: return

        # Optional: Filter by team key if defined in env
        team_key_filter = os.getenv("LINEAR_TEAM_KEY") 
        logger.info(f"🔄 Syncing Linear Tickets...")
        
        issues = self.linear.fetch_recent_issues(team_key=team_key_filter)

        new_count = 0
        for issue in issues:
            exists = self.session.exec(
                select(Activity).where(
                    Activity.source_id == issue.id, 
                    Activity.source_platform == "linear"
                )
            ).first()

            if not exists:
                try:
                    dt = datetime.fromisoformat(issue.createdAt.replace('Z', '+00:00'))
                except ValueError:
                    dt = datetime.now(timezone.utc)

                db_activity = Activity(
                    repository_id=repo_db.id,
                    source_platform="linear",
                    source_id=issue.id,
                    type="TICKET",
                    author="LinearUser",
                    timestamp=dt,
                    title=issue.title,
                    content=issue.description or "",
                    story_points=issue.estimate, # Storing the metric!
                    status_label=issue.state
                )
                self.session.add(db_activity)
                new_count += 1
        
        self.session.commit()
        logger.info(f"✅ Synced {new_count} new tickets from Linear.")

    def ensure_repository_sync(self, repo_url: str, days_lookback: int = 7):
        """
        Smart Sync Logic:
        1. Check when we last synced this repo.
        2. If never synced, fetch 'days_lookback'.
        3. If synced recently, only fetch the DELTA (from last_sync to now).
        """
        # 1. Find or Create Repository Entry
        repo = self.session.exec(select(Repository).where(Repository.url == repo_url)).first()
        
        if not repo:
            logger.info(f"🆕 New Repository detected: {repo_url}")
            repo_name = repo_url.replace("https://github.com/", "").strip("/")
            repo = Repository(url=repo_url, name=repo_name, last_synced_at=None)
            self.session.add(repo)
            self.session.commit()
            self.session.refresh(repo)
            pass

        self.sync_linear_issues(repo)

        # 2. Determine Time Window
        now = datetime.now(timezone.utc)
        
        if repo.last_synced_at:
            # Incremental Update: Fetch only what's new since last sync
            # Add a small buffer (e.g., 1 min) to avoid missing overlapping commits
            since_date = repo.last_synced_at
            logger.info(f"🔄 Incremental Sync for {repo.name} since {since_date}")
            
            # Optimization: If last sync was < 5 minutes ago, skip GitHub call entirely
            time_diff = (now - repo.last_synced_at).total_seconds()
            if time_diff < 300: 
                logger.info("⚡ Repo is fresh (synced < 5 mins ago). Skipping GitHub.")
                return repo
        else:
            # Full Initial Sync
            logger.info(f"📥 Initial Sync for {repo.name} ({days_lookback} days)")
            # Calculate date manually for adapter API
            from datetime import timedelta
            since_date = now - timedelta(days=days_lookback)

        # 3. Fetch from Adapter
        # Note: We need to update the adapter to accept a specific 'since_date' datetime object
        # instead of just 'int days'. (We will assume adapter handles this logic).
        # Calculating days for the current adapter signature:
        days_delta = (now - since_date).days + 1
        
        raw_activities = self.github.fetch_recent_activity(repo.name, days=days_delta)

        # 4. Save to DB (Upsert - Avoid Duplicates)
        new_count = 0
        for act in raw_activities:
            # Check if exists by source_id (SHA)
            exists = self.session.exec(
                select(Activity).where(Activity.source_id == act.source_id)
            ).first()
            
            if not exists:
                db_activity = Activity(
                    repository_id=repo.id,
                    source_id=act.source_id,
                    type=act.type.value,
                    author=act.author,
                    timestamp=act.timestamp,
                    title=act.content.split('\n')[0][:200], # First line as title
                    content=act.content, # The Diff
                    files_changed_count=len(act.files_changed)
                )
                self.session.add(db_activity)
                new_count += 1
        
        # 5. Update Metadata
        repo.last_synced_at = now
        self.session.add(repo)
        self.session.commit()
        
        logger.info(f"✅ Sync Complete. Added {new_count} new activities.")
        return repo