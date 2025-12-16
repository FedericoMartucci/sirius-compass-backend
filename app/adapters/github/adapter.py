import os
import concurrent.futures
from dataclasses import dataclass
from typing import Any, List, Optional
from datetime import datetime, timedelta, timezone
from github import Github, GithubException
from app.core.models.domain import UnifiedActivity, ActivityType
from app.core.logger import get_logger

logger = get_logger(__name__)

@dataclass(frozen=True)
class GitHubFetchResult:
    activities: List[UnifiedActivity]
    commits_fetched: int
    prs_fetched: int
    commits_truncated: bool
    prs_truncated: bool
    earliest_commit_at: Optional[datetime]
    latest_commit_at: Optional[datetime]

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
        return self.fetch_recent_activity_with_meta(repo_name=repo_name, days=days).activities

    def fetch_recent_activity_with_meta(
        self,
        repo_name: str,
        days: int = 90,
        *,
        max_commits: Optional[int] = 300,
        max_prs: Optional[int] = 100,
        include_prs: bool = True,
    ) -> GitHubFetchResult:
        """
        Fetches commits and PRs. 
        NOTE: 'days' should be large (e.g., 720 or 1100) for legacy projects.
        """
        logger.info(f"📡 Connecting to GitHub Repo: {repo_name} (Lookback: {days} days)...")
        activities: List[UnifiedActivity] = []
        
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
            commits_truncated = False
            earliest_commit_at: Optional[datetime] = None
            latest_commit_at: Optional[datetime] = None
            for commit in commits:
                if max_commits is not None and commit_count >= max_commits:
                    commits_truncated = True
                    logger.warning(f"⚠️ Hit safety limit of {max_commits} commits. Stopping fetch.")
                    break
                commit_count += 1

                # For very old commits, sometimes diffs are expensive. 
                # We skip detailed diffs for commits older than 90 days to speed up legacy import
                # unless it's critical. Here we fetch all.
                diff_content = self._extract_diff_parallel(commit.files)

                ts = commit.commit.author.date.replace(tzinfo=timezone.utc)
                if earliest_commit_at is None or ts < earliest_commit_at:
                    earliest_commit_at = ts
                if latest_commit_at is None or ts > latest_commit_at:
                    latest_commit_at = ts

                activity = UnifiedActivity(
                    source_id=commit.sha,
                    source_platform="github",
                    type=ActivityType.COMMIT,
                    author=commit.commit.author.name or "Unknown",
                    content=f"Message: {commit.commit.message}\n\nCode Diff:\n{diff_content}",
                    timestamp=ts,
                    url=commit.html_url,
                    files_changed=[f.filename for f in commit.files[:5]]
                )
                activities.append(activity)

            # --- 2. Fetch Pull Requests ---
            pr_count = 0
            prs_truncated = False
            if not include_prs:
                logger.info("Skipping PR sync (include_prs=False)")
                logger.info(f"✅ Fetched {len(activities)} activities ({commit_count} commits, {pr_count} PRs)")
                return GitHubFetchResult(
                    activities=activities,
                    commits_fetched=commit_count,
                    prs_fetched=pr_count,
                    commits_truncated=commits_truncated,
                    prs_truncated=prs_truncated,
                    earliest_commit_at=earliest_commit_at,
                    latest_commit_at=latest_commit_at,
                )

            prs = repo.get_pulls(state='all', sort='updated', direction='desc')
            pr_count = 0
            for pr in prs:
                # Manual date check as PR API doesn't support 'since' well
                pr_date = pr.updated_at.replace(tzinfo=timezone.utc)
                if pr_date < since_date:
                    if pr_count > 10: # Only break if we've seen a few old ones (buffer)
                        break
                    continue
                
                if max_prs is not None and pr_count >= max_prs:
                    prs_truncated = True
                    logger.warning(f"⚠️ Hit safety limit of {max_prs} PRs. Stopping fetch.")
                    break
                pr_count += 1

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
            return GitHubFetchResult(
                activities=activities,
                commits_fetched=commit_count,
                prs_fetched=pr_count,
                commits_truncated=commits_truncated,
                prs_truncated=prs_truncated,
                earliest_commit_at=earliest_commit_at,
                latest_commit_at=latest_commit_at,
            )

        except GithubException as e:
            logger.error(f"GitHub API Error: {e.data.get('message', e)}")
            return GitHubFetchResult(
                activities=[],
                commits_fetched=0,
                prs_fetched=0,
                commits_truncated=False,
                prs_truncated=False,
                earliest_commit_at=None,
                latest_commit_at=None,
            )
        except Exception as e:
            logger.error(f"Unexpected Error: {e}")
            return GitHubFetchResult(
                activities=[],
                commits_fetched=0,
                prs_fetched=0,
                commits_truncated=False,
                prs_truncated=False,
                earliest_commit_at=None,
                latest_commit_at=None,
            )
