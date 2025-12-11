# app/adapters/github/adapter.py
import os
import concurrent.futures
from typing import List, Any
from datetime import datetime, timedelta, timezone
from github import Github, GithubException
from app.ports.code_provider import CodeProvider
from app.core.models.domain import UnifiedActivity, ActivityType
from app.core.logger import get_logger

logger = get_logger(__name__)

class GitHubAdapter(CodeProvider):
    """
    Concrete implementation of CodeProvider for GitHub.
    Optimized with ThreadPoolExecutor for parallel diff fetching.
    """

    def __init__(self):
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            logger.error("Missing GITHUB_TOKEN")
            raise ValueError("❌ Missing GITHUB_TOKEN in environment variables.")
        self.client = Github(token)

    def _get_file_patch(self, file: Any) -> str:
        """
        Helper function to fetch a single file's patch.
        Designed to be run in a thread.
        """
        try:
            # Skip binary/lock files to reduce noise and latency
            if file.filename.endswith(('.lock', '.png', '.jpg', 'package-lock.json', '.pyc')):
                return ""
            
            patch = file.patch if file.patch else "[Binary or Large File]"
            
            # Truncate large patches to avoid token overflow
            if len(patch) > 2000:
                patch = patch[:2000] + "\n... (truncated)"
            
            return f"File: {file.filename}\nChanges:\n{patch}"
        except Exception as e:
            logger.warning(f"Failed to fetch patch for {file.filename}: {e}")
            return ""

    def _extract_diff_parallel(self, files_object, max_workers: int = 5) -> str:
        """
        Fetches file patches in PARALLEL to reduce latency.
        """
        # Convert paginated list to a standard list (limit to first 10 files for performance)
        files_list = list(files_object[:10])
        
        diff_summary = []
        
        # Parallel Execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_file = {executor.submit(self._get_file_patch, f): f for f in files_list}
            
            for future in concurrent.futures.as_completed(future_to_file):
                result = future.result()
                if result:
                    diff_summary.append(result)
                    
        return "\n\n".join(diff_summary)

    def fetch_recent_activity(self, repo_name: str, days: int = 7) -> List[UnifiedActivity]:
        logger.info(f"📡 Connecting to GitHub Repo: {repo_name} (Parallel Mode)...")
        activities = []
        try:
            repo = self.client.get_repo(repo_name)
            since_date = datetime.now(timezone.utc) - timedelta(days=days)

            # --- 1. Fetch Commits ---
            # Note: Pagination in PyGithub is lazy. 
            # We iterate and break manually to avoid fetching full history.
            commits = repo.get_commits(since=since_date)
            
            for commit in commits:
                # Optimized Parallel Diff Extraction
                diff_content = self._extract_diff_parallel(commit.files)

                activity = UnifiedActivity(
                    source_id=commit.sha,
                    author=commit.commit.author.name or "Unknown",
                    type=ActivityType.COMMIT,
                    content=f"Message: {commit.commit.message}\n\nCode Diff:\n{diff_content}",
                    timestamp=commit.commit.author.date.replace(tzinfo=timezone.utc),
                    url=commit.html_url,
                    files_changed=[f.filename for f in commit.files[:5]]
                )
                activities.append(activity)
                
                # Safety break for MVP: Analyze max 15 latest commits to ensure speed
                if len(activities) >= 15:
                    break

            # --- 2. Fetch Pull Requests ---
            prs = repo.get_pulls(state='all', sort='updated', direction='desc')
            for pr in prs:
                if pr.updated_at.replace(tzinfo=timezone.utc) < since_date:
                    break 
                
                type_pr = ActivityType.PR_MERGE if pr.merged else ActivityType.PR_OPEN
                
                # Optimized Parallel Diff Extraction for PRs
                diff_content = self._extract_diff_parallel(pr.get_files())

                activity = UnifiedActivity(
                    source_id=str(pr.number),
                    author=pr.user.login,
                    type=type_pr,
                    content=f"Title: {pr.title}\nBody: {pr.body}\n\nCode Diff:\n{diff_content}",
                    timestamp=pr.created_at.replace(tzinfo=timezone.utc),
                    url=pr.html_url,
                    additions=pr.additions,
                    deletions=pr.deletions,
                    files_changed=[f.filename for f in pr.get_files()[:5]]
                )
                activities.append(activity)

            logger.info(f"✅ Fetched {len(activities)} activities from {repo_name}")
            return activities

        except GithubException as e:
            logger.error(f"GitHub API Error: {e.data.get('message', e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected Error: {e}")
            return []