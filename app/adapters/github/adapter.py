import os
import concurrent.futures
from typing import Any, List, Optional
from datetime import datetime, timedelta, timezone
from github import Github, GithubException
from app.core.models.domain import UnifiedActivity, ActivityType
from app.core.logger import get_logger

logger = get_logger(__name__)

class GitHubAdapter:
    """
    Concrete implementation for GitHub.
    Fetches ALL activity within the lookback window to populate the Data Lake (DB).
    """

    def __init__(self, token: Optional[str] = None):
        token = token or os.getenv("GITHUB_TOKEN")
        if not token:
            logger.error("Missing GITHUB_TOKEN")
            raise ValueError("❌ Missing GITHUB_TOKEN in environment variables.")
        self.client = Github(token)

    def _get_file_patch(self, file: Any) -> str:
        """
        Helper to fetch a single file's patch safely.
        """
        try:
            if file.filename.endswith(('.lock', '.png', '.jpg', 'package-lock.json', '.pyc')):
                return ""
            
            patch = file.patch if file.patch else "[Binary/Large]"
            if len(patch) > 2000:
                patch = patch[:2000] + "\n... (truncated)"
            
            return f"File: {file.filename}\nChanges:\n{patch}"
        except Exception:
            return ""

    def _extract_diff_parallel(self, files_object, max_workers: int = 5) -> str:
        """
        Fetches file patches in PARALLEL.
        """
        # Limit to first 5 files to avoid huge payloads
        files_list = list(files_object[:5])
        diff_summary = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self._get_file_patch, f): f for f in files_list}
            for future in concurrent.futures.as_completed(future_to_file):
                if result := future.result():
                    diff_summary.append(result)
                    
        return "\n\n".join(diff_summary)

    def fetch_recent_activity(self, repo_name: str, days: int = 90) -> List[UnifiedActivity]:
        """
        Fetches commits and PRs. 
        NOTE: 'days' should be large (e.g., 720 or 1100) for legacy projects.
        """
        logger.info(f"📡 Connecting to GitHub Repo: {repo_name} (Lookback: {days} days)...")
        activities = []
        
        try:
            repo = self.client.get_repo(repo_name)
            
            # Ensure timezone-aware datetime
            since_date = datetime.now(timezone.utc) - timedelta(days=days)
            logger.info(f"📅 Fetching items since: {since_date.date()}")

            # --- 1. Fetch Commits ---
            # Explicitly use the default branch (main/master) to avoid detached head issues
            default_branch = repo.default_branch
            commits = repo.get_commits(since=since_date, sha=default_branch)
            
            commit_count = 0
            # We iterate without a hard limit on count, only date (controlled by API)
            # But we add a safety ceiling (e.g. 500) to prevent infinite loops in massive repos
            for commit in commits:
                commit_count += 1
                if commit_count > 300: # Safety ceiling for MVP
                    logger.warning("⚠️ Hit safety limit of 300 commits. Stopping fetch.")
                    break

                # For very old commits, sometimes diffs are expensive. 
                # We skip detailed diffs for commits older than 90 days to speed up legacy import
                # unless it's critical. Here we fetch all.
                diff_content = self._extract_diff_parallel(commit.files)

                activity = UnifiedActivity(
                    source_id=commit.sha,
                    source_platform="github",
                    type=ActivityType.COMMIT,
                    author=commit.commit.author.name or "Unknown",
                    content=f"Message: {commit.commit.message}\n\nCode Diff:\n{diff_content}",
                    timestamp=commit.commit.author.date.replace(tzinfo=timezone.utc),
                    url=commit.html_url,
                    files_changed=[f.filename for f in commit.files[:5]]
                )
                activities.append(activity)

            # --- 2. Fetch Pull Requests ---
            prs = repo.get_pulls(state='all', sort='updated', direction='desc')
            pr_count = 0
            
            for pr in prs:
                # Manual date check as PR API doesn't support 'since' well
                pr_date = pr.updated_at.replace(tzinfo=timezone.utc)
                if pr_date < since_date:
                    if pr_count > 10: # Only break if we've seen a few old ones (buffer)
                        break
                    continue
                
                pr_count += 1
                if pr_count > 100: break # Safety ceiling

                type_pr = ActivityType.PR_MERGE if pr.merged else ActivityType.PR_OPEN
                diff_content = self._extract_diff_parallel(pr.get_files())

                activity = UnifiedActivity(
                    source_id=str(pr.number),
                    source_platform="github",
                    type=type_pr,
                    author=pr.user.login,
                    content=f"Title: {pr.title}\nBody: {pr.body}\n\nDiff:\n{diff_content}",
                    timestamp=pr.created_at.replace(tzinfo=timezone.utc),
                    url=pr.html_url,
                    additions=pr.additions,
                    deletions=pr.deletions,
                    files_changed=[f.filename for f in pr.get_files()[:5]]
                )
                activities.append(activity)

            logger.info(f"✅ Fetched {len(activities)} activities ({commit_count} commits, {pr_count} PRs)")
            return activities

        except GithubException as e:
            logger.error(f"GitHub API Error: {e.data.get('message', e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected Error: {e}")
            return []
