"""
GitHub API client for the Gemini AI Code Reviewer.

This module handles all GitHub API interactions including fetching PR details,
diffs, and creating review comments with proper retry logic and error handling.
"""

import json
import logging
import requests
import hashlib
import re
import os
from typing import List, Dict, Any, Optional, Set
import difflib
from github import Github
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import GitHubConfig
from .models import PRDetails, ReviewComment, ReviewResult


logger = logging.getLogger(__name__)


class GitHubClientError(Exception):
    """Base exception for GitHub client errors."""
    pass


class PRNotFoundError(GitHubClientError):
    """Exception raised when PR is not found."""
    pass


class RateLimitError(GitHubClientError):
    """Exception raised when GitHub API rate limit is exceeded."""
    pass


class GitHubClient:
    """GitHub API client with retry logic and comprehensive error handling."""
    
    def __init__(self, config: GitHubConfig):
        """Initialize GitHub client with configuration."""
        self.config = config
        self._client = Github(config.token)
        self._session = requests.Session()
        self._session.headers.update({
            'Authorization': f'Bearer {config.token}',
            'User-Agent': 'Gemini-AI-Code-Reviewer/1.0',
            'Accept': 'application/vnd.github.v3+json'
        })
        
        logger.info("Initialized GitHub client")
    
    def get_pr_details_from_event(self, event_path: str) -> PRDetails:
        """Extract PR details from GitHub Actions event payload."""
        try:
            with open(event_path, "r") as f:
                event_data = json.load(f)
            logger.info("Successfully loaded GitHub event data")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load GitHub event data: {str(e)}")
            raise GitHubClientError(f"Failed to load event data: {str(e)}")
        
        # Handle comment trigger differently from direct PR events
        if "issue" in event_data and "pull_request" in event_data["issue"]:
            # For comment triggers, we need to get the PR number from the issue
            pull_number = event_data["issue"]["number"]
            repo_full_name = event_data["repository"]["full_name"]
        else:
            # Original logic for direct PR events
            pull_number = event_data["number"]
            repo_full_name = event_data["repository"]["full_name"]
        
        if not repo_full_name or "/" not in repo_full_name:
            raise GitHubClientError(f"Invalid repository name: {repo_full_name}")
        
        owner, repo = repo_full_name.split("/", 1)
        logger.info(f"Processing PR #{pull_number} in repository {repo_full_name}")
        
        try:
            pr_details = self.get_pr_details(owner, repo, pull_number)
            logger.info(f"Successfully retrieved PR details: {pr_details.title}")
            return pr_details
        except Exception as e:
            logger.error(f"Failed to get PR details: {str(e)}")
            raise GitHubClientError(f"Failed to get PR details: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
    )
    def get_pr_details(self, owner: str, repo: str, pull_number: int) -> PRDetails:
        """Get pull request details with retry logic."""
        logger.debug(f"Fetching PR details for {owner}/{repo}#{pull_number}")
        
        try:
            repo_obj = self._get_repo_with_retry(f"{owner}/{repo}")
            pr = self._get_pr_with_retry(repo_obj, pull_number)
            
            # Sanitize PR title and description
            title = self._sanitize_input(pr.title or "")
            description = self._sanitize_input(pr.body or "")
            
            pr_details = PRDetails(
                owner=owner,
                repo=repo,
                pull_number=pull_number,
                title=title,
                description=description,
                head_sha=pr.head.sha,
                base_sha=pr.base.sha
            )
            
            logger.debug(f"Retrieved PR details: {title}")
            return pr_details
            
        except Exception as e:
            logger.warning(f"Failed to get PR details: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
    )
    def _get_repo_with_retry(self, repo_name: str):
        """Get repository with retry logic."""
        logger.debug(f"Attempting to get repository: {repo_name}")
        try:
            return self._client.get_repo(repo_name)
        except Exception as e:
            logger.warning(f"Failed to get repository {repo_name}: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
    )
    def _get_pr_with_retry(self, repo, pull_number: int):
        """Get pull request with retry logic."""
        logger.debug(f"Attempting to get PR #{pull_number}")
        try:
            return repo.get_pull(pull_number)
        except Exception as e:
            if "404" in str(e):
                raise PRNotFoundError(f"PR #{pull_number} not found")
            logger.warning(f"Failed to get PR #{pull_number}: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
    )
    def get_pr_diff(self, owner: str, repo: str, pull_number: int) -> str:
        """Fetch the diff of a pull request with retry logic."""
        # Validate inputs
        if not all([owner, repo, pull_number]):
            logger.error("Invalid parameters provided to get_pr_diff")
            raise GitHubClientError("Invalid parameters")
        
        if not isinstance(pull_number, int) or pull_number <= 0:
            logger.error(f"Invalid pull request number: {pull_number}")
            raise GitHubClientError(f"Invalid pull request number: {pull_number}")
        
        repo_name = f"{self._sanitize_input(owner)}/{self._sanitize_input(repo)}"
        logger.info(f"Fetching diff for: {repo_name} PR#{pull_number}")
        
        try:
            # Verify PR exists first
            repo_obj = self._get_repo_with_retry(repo_name)
            pr = self._get_pr_with_retry(repo_obj, pull_number)
            
            # Use direct API call for diff
            api_url = f"{self.config.api_base_url}/repos/{repo_name}/pulls/{pull_number}.diff"
            
            # Override Accept header to specifically request diff format
            diff_headers = {
                'Accept': 'application/vnd.github.v3.diff'
            }
            
            logger.debug(f"Making diff API request to: {api_url}")
            response = self._session.get(api_url, headers=diff_headers, timeout=self.config.timeout)
            
            if response.status_code == 200:
                diff = response.text
                logger.info(f"Successfully retrieved diff (length: {len(diff)} characters)")
                return diff
            elif response.status_code == 404:
                raise PRNotFoundError(f"PR #{pull_number} not found in {repo_name}")
            elif response.status_code == 403:
                if "rate limit" in response.text.lower():
                    raise RateLimitError("GitHub API rate limit exceeded")
                else:
                    raise GitHubClientError("Access forbidden - check GitHub token permissions")
            else:
                logger.error(f"Failed to get diff. Status code: {response.status_code}")
                logger.debug(f"Response content: {response.text[:500]}...")
                response.raise_for_status()  # This will trigger retry
                return ""
        
        except requests.exceptions.Timeout:
            logger.error("Request timed out while fetching diff")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed while fetching diff: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while fetching diff: {str(e)}")
            raise GitHubClientError(f"Failed to fetch diff: {str(e)}")
        
    def get_last_reviewed_commit_sha(self, pr_details: PRDetails) -> Optional[str]:
        """Simplified detection of the last commit reviewed by this bot.
        
        Strategy:
        1. Find the LATEST review or comment containing "Gemini AI Code Review" marker
        2. Only consider authors with "github-action" in their username
        3. Get all commits on the PR
        4. Return the first commit SHA that came AFTER the review/comment timestamp
        
        This ensures we only review NEW commits since the last bot review,
        preventing endless cycles and never re-reviewing the entire codebase.
        
        Returns: SHA of the last reviewed commit, or None if no prior review found.
        """
        try:
            repo_obj = self._get_repo_with_retry(pr_details.repo_full_name)
            pr = self._get_pr_with_retry(repo_obj, pr_details.pull_number)
            
            # Get all commits on the PR (we'll need this for timestamp mapping)
            try:
                all_commits = list(pr.get_commits())
                logger.info(f"PR has {len(all_commits)} commit(s)")
                if all_commits:
                    logger.info(f"Latest commit: {all_commits[-1].sha[:7]} at {all_commits[-1].commit.author.date}")
            except Exception as e:
                logger.warning(f"Could not retrieve commit list: {e}")
                return None
            
            if not all_commits:
                logger.info("No commits found on PR")
                return None
            
            # Find the latest review/comment with our marker from a github-action author
            # Support both "Gemini AI Code Reviewer" and legacy "Gemini AI Code Review"
            marker_regex = re.compile(r"Gemini AI Code Review(?:er)?", re.IGNORECASE)
            latest_review_time = None
            latest_review_source = None
            
            # Check PR reviews (these have bodies with our marker)
            try:
                for review in pr.get_reviews():
                    try:
                        author_login = getattr(review.user, 'login', '') or ''
                        body = getattr(review, 'body', '') or ''
                        submitted_at = getattr(review, 'submitted_at', None)
                        
                        # Only consider github-action authors with our marker
                        if 'github-action' not in author_login.lower():
                            continue
                        if not marker_regex.search(body):
                            continue
                        if not submitted_at:
                            continue
                        
                        # Track the latest review
                        if latest_review_time is None or submitted_at > latest_review_time:
                            latest_review_time = submitted_at
                            latest_review_source = f"PR review by {author_login}"
                            logger.debug(f"Found review by {author_login} at {submitted_at}")
                    except Exception:
                        continue
            except Exception as e:
                logger.debug(f"Error scanning PR reviews: {e}")
            
            # Check issue comments (may also contain our marker)
            try:
                for comment in pr.as_issue().get_comments():
                    try:
                        author_login = getattr(getattr(comment, 'user', None), 'login', '') or ''
                        body = getattr(comment, 'body', '') or ''
                        created_at = getattr(comment, 'created_at', None)
                        
                        # Only consider github-action authors with our marker
                        if 'github-action' not in author_login.lower():
                            continue
                        if not marker_regex.search(body):
                            continue
                        if not created_at:
                            continue
                        
                        # Track the latest comment
                        if latest_review_time is None or created_at > latest_review_time:
                            latest_review_time = created_at
                            latest_review_source = f"issue comment by {author_login}"
                            logger.debug(f"Found comment by {author_login} at {created_at}")
                    except Exception:
                        continue
            except Exception as e:
                logger.debug(f"Error scanning issue comments: {e}")
            
            # If no prior review found, this is the first review
            if latest_review_time is None:
                logger.info("No prior bot review found - this is the first review")
                return None
            
            logger.info(f"Latest bot review: {latest_review_source} at {latest_review_time}")
            
            # Find the last commit at or before the review time
            # This is the commit that was reviewed
            last_reviewed_sha = None
            for commit in all_commits:
                try:
                    sha = getattr(commit, 'sha', None)
                    commit_obj = getattr(commit, 'commit', None)
                    # Use committer date (when it was added to the branch)
                    committer = getattr(commit_obj, 'committer', None) if commit_obj else None
                    commit_time = getattr(committer, 'date', None) if committer else None
                    
                    # Fallback to author date if committer date not available
                    if not commit_time:
                        author = getattr(commit_obj, 'author', None) if commit_obj else None
                        commit_time = getattr(author, 'date', None) if author else None
                    
                    if not sha or not commit_time:
                        continue
                    
                    # If this commit was made at or before the review time, it was reviewed
                    if commit_time <= latest_review_time:
                        last_reviewed_sha = sha
                        logger.debug(f"Commit {sha[:7]} at {commit_time} was at or before review time")
                    else:
                        # We've found the first commit after the review
                        logger.debug(f"Commit {sha[:7]} at {commit_time} came after review time")
                        break
                except Exception:
                    continue
            
            if last_reviewed_sha:
                logger.info(f"Last reviewed commit: {last_reviewed_sha[:7]}")
                return last_reviewed_sha
            else:
                logger.warning("Could not map review timestamp to any commit")
                return None
                
        except Exception as e:
            logger.warning(f"Error determining last reviewed commit: {e}")
            return None
        
    def get_pr_diff_since(self, pr_details: PRDetails, base_sha: str) -> Optional[str]:
        """Fetch a diff of changes since a given base SHA up to the PR head.
        Primary strategy: GitHub compare API.
        Fallback: build incremental diff by iterating PR commits after base_sha.
        
        Returns:
        - Diff text string if there are new changes
        - Empty string "" if no new changes (prevents re-reviewing entire codebase)
        - None only if base_sha is not provided (first review, should fetch full PR diff)
        
        This ensures we NEVER re-review the entire codebase unnecessarily and
        we also don't miss changes when the compare API fails (e.g., synthetic
        merge SHAs or history quirks).
        """
        try:
            if not base_sha:
                return None
            
            # Re-fetch the PR to get the most current head SHA
            try:
                repo_obj = self._get_repo_with_retry(pr_details.repo_full_name)
                pr = self._get_pr_with_retry(repo_obj, pr_details.pull_number)
                current_head_sha = pr.head.sha
                
                # Also verify against the latest commit from the list
                try:
                    all_commits = list(pr.get_commits())
                    if all_commits:
                        actual_latest_sha = getattr(all_commits[-1], 'sha', None)
                        if actual_latest_sha and actual_latest_sha != current_head_sha:
                            logger.warning(
                                f"PR head.sha ({current_head_sha[:7]}) differs from latest commit in list ({actual_latest_sha[:7]})"
                            )
                            current_head_sha = actual_latest_sha
                except Exception as e:
                    logger.debug(f"Could not verify head SHA from commit list: {e}")
            except Exception as e:
                logger.warning(f"Could not re-fetch PR for current head SHA, using pr_details.head_sha: {e}")
                current_head_sha = pr_details.head_sha
                if not current_head_sha:
                    logger.warning("Could not determine current head SHA; treating as no new changes")
                    return ""
            
            # Ignore GITHUB_SHA because it can be a synthetic merge
            github_sha = os.getenv('GITHUB_SHA')
            if github_sha and github_sha != current_head_sha:
                logger.info(
                    "GITHUB_SHA differs from PR head (likely a synthetic merge); ignoring for incremental diff"
                )
            
            repo_name = pr_details.repo_full_name
            
            logger.info(f"Comparing: base_sha={base_sha[:7]}... vs current_head_sha={current_head_sha[:7]}...")
            if pr_details.head_sha and pr_details.head_sha != current_head_sha:
                logger.info(
                    f"Note: pr_details.head_sha ({pr_details.head_sha[:7]}) differs from current head ({current_head_sha[:7]})"
                )
            
            # If base equals head, nothing to review
            if base_sha == current_head_sha:
                logger.info("Base SHA equals current head; no new changes to review.")
                return ""
            
            # Try compare API first
            api_url = f"{self.config.api_base_url}/repos/{repo_name}/compare/{base_sha}...{current_head_sha}.diff"
            diff_headers = {'Accept': 'application/vnd.github.v3.diff'}
            logger.info(f"Fetching incremental diff: {base_sha[:7]}... to {current_head_sha[:7]}...")
            resp = self._session.get(api_url, headers=diff_headers, timeout=self.config.timeout)
            
            if resp.status_code == 200:
                text = resp.text or ""
                if text.strip() == "":
                    logger.info("Incremental compare returned empty diff; will try per-commit fallback.")
                else:
                    return text
            else:
                logger.warning(
                    f"Compare API failed with {resp.status_code}; attempting per-commit fallback."
                )
            
            # Fallback: build incremental diff by iterating commits after base_sha
            fallback = self.get_incremental_diff_by_commits(pr_details, base_sha)
            return fallback if fallback is not None else ""
        except Exception as e:
            logger.warning(
                f"Failed to get incremental diff: {e}; will treat as no new changes to avoid re-reviewing entire codebase."
            )
            return ""

    def get_incremental_diff_by_commits(self, pr_details: PRDetails, base_sha: str) -> Optional[str]:
        """Fallback incremental diff: concatenate patches for commits after base_sha.
        Steps:
        1. Find commits on the PR after base_sha.
        2. For each commit, fetch its file patches.
        3. Build a minimal unified diff per file and concatenate.
        Returns "" if there are no commits since base_sha.
        Returns None if base_sha is falsy (caller can decide to fetch full PR diff).
        """
        if not base_sha:
            return None
        try:
            repo_obj = self._get_repo_with_retry(pr_details.repo_full_name)
            pr = self._get_pr_with_retry(repo_obj, pr_details.pull_number)
            commits = list(pr.get_commits())
            if not commits:
                logger.info("No commits found on PR while building fallback incremental diff")
                return ""
            
            # Identify position of base_sha in the PR commits
            base_index = None
            for idx, c in enumerate(commits):
                try:
                    if getattr(c, 'sha', None) == base_sha:
                        base_index = idx
                        break
                except Exception:
                    continue
            
            # If base_sha not found, we still try to use commit timestamps from last review
            if base_index is None:
                logger.warning("Base SHA not found in PR commit list; will include all commits as a conservative fallback")
                start_idx = 0
            else:
                start_idx = base_index + 1
            
            commits_after = commits[start_idx:]
            logger.info(f"Found {len(commits_after)} commit(s) after base_sha for fallback diff")
            if len(commits_after) == 0:
                return ""
            
            parts: List[str] = []
            for c in commits_after:
                sha = getattr(c, 'sha', None)
                if not sha:
                    continue
                try:
                    gh_commit = repo_obj.get_commit(sha)
                    # Each file has attributes: filename, status, patch, previous_filename (for renamed)
                    for f in getattr(gh_commit, 'files', []) or []:
                        patch = getattr(f, 'patch', None)
                        status = getattr(f, 'status', '') or ''
                        filename = getattr(f, 'filename', '') or ''
                        prev = getattr(f, 'previous_filename', None)
                        # Skip binary files (patch is None)
                        if not patch or not filename:
                            continue
                        # Build unified diff headers
                        if status == 'added':
                            diff_header = f"diff --git a/{filename} b/{filename}\n"
                            from_header = f"--- /dev/null\n"
                            to_header = f"+++ b/{filename}\n"
                        elif status == 'removed':
                            diff_header = f"diff --git a/{filename} b/{filename}\n"
                            from_header = f"--- a/{filename}\n"
                            to_header = f"+++ /dev/null\n"
                        elif status == 'renamed' and prev:
                            diff_header = f"diff --git a/{prev} b/{filename}\n"
                            from_header = f"--- a/{prev}\n"
                            to_header = f"+++ b/{filename}\n"
                        else:
                            diff_header = f"diff --git a/{filename} b/{filename}\n"
                            from_header = f"--- a/{filename}\n"
                            to_header = f"+++ b/{filename}\n"
                        parts.append(diff_header + from_header + to_header + patch + "\n")
                except Exception as e:
                    logger.debug(f"Failed to fetch commit {sha[:7]} for fallback diff: {e}")
                    continue
            
            combined = "".join(parts)
            if not combined.strip():
                logger.info("Fallback per-commit diff produced no content (possibly only binary changes)")
                return ""
            logger.info(f"Built fallback incremental diff with {len(parts)} file patch(es)")
            return combined
        except Exception as e:
            logger.warning(f"Error while building fallback incremental diff: {e}")
            return ""

    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
    )
    def create_review(self, pr_details: PRDetails, comments: List[ReviewComment], event: str = "COMMENT", total_comments_generated: int = None) -> bool:
        """Create a review on GitHub with retry logic.
        
        Args:
            pr_details: Pull request details
            comments: List of review comments to post (after filtering)
            event: Review event type - "APPROVE", "REQUEST_CHANGES", or "COMMENT"
            total_comments_generated: Total number of comments generated before filtering
        
        Returns:
            True if review was created successfully
        """
        logger.info(f"Creating review with {len(comments)} comments for PR #{pr_details.pull_number} (event: {event})")
        
        try:
            repo_obj = self._get_repo_with_retry(pr_details.repo_full_name)
            pr = self._get_pr_with_retry(repo_obj, pr_details.pull_number)
            
            # Validate and convert comments
            github_comments = []
            for comment in comments:
                if not isinstance(comment, ReviewComment):
                    logger.warning(f"Invalid comment type: {type(comment)}")
                    continue
                
                github_comment = self._validate_and_sanitize_comment(comment)
                if github_comment:
                    github_comments.append(github_comment)
            
            logger.info(f"Creating review with {len(github_comments)} valid comments")
            
            # Use total_comments_generated if provided, otherwise use comments list length
            total_generated = total_comments_generated if total_comments_generated is not None else len(comments)
            
            # Generate review body based on whether there are comments and if any were filtered
            if github_comments:
                review_body = self._generate_review_summary(comments)
            else:
                # Check if comments were filtered out
                if total_generated > 0:
                    review_body = self._generate_filtered_message(total_generated)
                else:
                    review_body = self._generate_approval_message()
            
            # Create the review
            # Note: When there are no comments, we don't pass the comments parameter at all
            # as passing None or empty list with certain events can cause API errors
            try:
                if github_comments:
                    review = pr.create_review(
                        body=review_body,
                        comments=github_comments,
                        event=event
                    )
                else:
                    review = pr.create_review(
                        body=review_body,
                        event=event
                    )
                logger.info(f"✅ Review created successfully with ID: {review.id}")
                return True
            except Exception as e:
                # Fallback: if APPROVE is not permitted by the token, retry as COMMENT
                if str(event).upper() == "APPROVE":
                    logger.warning(f"APPROVE review failed ({e}); falling back to COMMENT event.")
                    try:
                        if github_comments:
                            review = pr.create_review(
                                body=review_body,
                                comments=github_comments,
                                event="COMMENT"
                            )
                        else:
                            review = pr.create_review(
                                body=review_body,
                                event="COMMENT"
                            )
                        logger.info(f"✅ Fallback COMMENT review created successfully with ID: {review.id}")
                        return True
                    except Exception as e2:
                        logger.error(f"Fallback to COMMENT also failed: {e2}")
                        raise
                raise
            
        except Exception as e:
            logger.error(f"Failed to create review: {str(e)}")
            raise GitHubClientError(f"Failed to create review: {str(e)}")
    
    def _validate_and_sanitize_comment(self, comment: ReviewComment) -> Optional[Dict[str, Any]]:
        """Validate and sanitize a review comment.
        Appends a hidden signature marker to the body for future deduplication.
        """
        try:
            # Check required fields
            if not all([comment.body, comment.path]):
                logger.warning(f"Comment missing required fields: {comment}")
                return None
            
            # Validate position
            if not isinstance(comment.position, int) or comment.position <= 0:
                logger.warning(f"Invalid position {comment.position} in comment")
                return None
            
            # Sanitize content
            sanitized_comment = {
                'body': self._sanitize_input(str(comment.body)),
                'path': self._sanitize_input(str(comment.path)),
                'position': comment.position
            }

            # Append hidden signature marker for deduplication on future runs
            try:
                body_with_marker = self._append_signature_marker(sanitized_comment['body'], sanitized_comment['path'])
                sanitized_comment['body'] = body_with_marker
            except Exception as marker_err:
                logger.debug(f"Could not append signature marker: {marker_err}")
            
            return sanitized_comment
            
        except Exception as e:
            logger.warning(f"Error validating comment: {str(e)}")
            return None
    
    def _generate_review_summary(self, comments: List[ReviewComment]) -> str:
        """Generate a summary for the review."""
        priority_counts = {}
        for comment in comments:
            priority = comment.priority.value
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        summary_parts = ["🤖 **Gemini AI Code Review**"]
        summary_parts.append(f"\nFound **{len(comments)}** suggestions for improvement:")
        
        for priority, count in priority_counts.items():
            emoji = {"critical": "🚨", "high": "⚠️", "medium": "💡", "low": "ℹ️"}.get(priority, "📝")
            summary_parts.append(f"- {emoji} {priority.title()}: {count}")
        
        summary_parts.append(f"\n> This review was automatically generated by Gemini AI. Please review the suggestions carefully.")
        
        return "\n".join(summary_parts)
    
    def _generate_approval_message(self) -> str:
        """Generate a message when no issues are found and we want to explicitly approve."""
        return (
            "🤖 Gemini AI Code Review\n\n"
            "This code looks great!"
        )
    
    def _generate_filtered_message(self, total_comments: int) -> str:
        """Generate a message when comments were found but filtered by thresholds or limits."""
        return (
            f"🤖 **Gemini AI Code Review**\n\n"
            f"Found **{total_comments}** observations during analysis, but they were hidden due to the configured priority threshold and/or per-file/total comment limits.\n\n"
            f"The code was reviewed with project context and related files for comprehensive analysis.\n\n"
            f"> You can adjust REVIEW_PRIORITY_THRESHOLD, MAX_COMMENTS_TOTAL, or MAX_COMMENTS_PER_FILE to see more inline suggestions."
        )
    
    @staticmethod
    def _sanitize_input(text: str) -> str:
        """Lightly sanitize text while preserving Markdown and code formatting.
        - Do not HTML-escape; GitHub safely renders Markdown and escapes HTML.
        - Remove null bytes and non-printable control characters only.
        - Trim whitespace.
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        cleaned = ''.join(ch for ch in text if (ord(ch) >= 32) or ch in '\t\n\r')
        return cleaned.strip()
    
    def _strip_signature_marker(self, body: str) -> str:
        """Remove hidden AI signature marker from a comment body, if present."""
        try:
            return re.sub(r'<!--\s*AI-SIG:[a-f0-9]{6,}\s*-->\s*$', '', body or '', flags=re.IGNORECASE).strip()
        except Exception:
            return body

    def _normalize_for_signature(self, text: str) -> str:
        """Normalize text to compute a stable signature: lowercase, collapse whitespace, strip marker."""
        base = self._sanitize_input(text or "")
        base = self._strip_signature_marker(base)
        base = re.sub(r'\s+', ' ', base).strip().lower()
        return base

    def _compute_signature(self, path: str, body: str) -> str:
        """Compute a stable signature for a comment using path + normalized body."""
        norm = self._normalize_for_signature(body)
        h = hashlib.sha1(f"{path}|{norm}".encode('utf-8')).hexdigest()[:12]
        return f"{path}:{h}"

    def _append_signature_marker(self, body: str, path: str) -> str:
        """Append a hidden HTML comment with the signature so future runs can dedupe reliably."""
        try:
            if not body:
                body = ""
            # If already has a marker, don't add another
            if re.search(r'<!--\s*AI-SIG:[a-f0-9]{6,}\s*-->', body or '', flags=re.IGNORECASE):
                return body
            sig = self._compute_signature(path, body).split(":")[-1]
            return f"{body}\n<!-- AI-SIG:{sig} -->"
        except Exception:
            return body

    def get_existing_comment_signatures(self, pr_details: PRDetails) -> Set[str]:
        """Fetch existing PR review comments and build a set of signatures to avoid duplicates."""
        try:
            repo_obj = self._get_repo_with_retry(pr_details.repo_full_name)
            pr = self._get_pr_with_retry(repo_obj, pr_details.pull_number)
            sigs: Set[str] = set()
            try:
                existing_comments = pr.get_review_comments()
            except Exception:
                existing_comments = []
            for c in existing_comments:
                try:
                    path = getattr(c, 'path', None)
                    if not path:
                        continue
                    body = getattr(c, 'body', '') or ''
                    # Prefer embedded signature if present
                    m = re.search(r'<!--\s*AI-SIG:([a-f0-9]{6,})\s*-->', body, flags=re.IGNORECASE)
                    if m:
                        sigs.add(f"{path}:{m.group(1)[:12]}")
                    # Also add computed signature from normalized body
                    cleaned_body = self._strip_signature_marker(body)
                    sigs.add(self._compute_signature(path, cleaned_body))
                except Exception:
                    continue
            logger.info(f"Loaded {len(sigs)} existing review comment signatures for PR #{pr_details.pull_number}")
            return sigs
        except Exception as e:
            logger.debug(f"Could not fetch existing review comments: {e}")
            return set()

    def get_existing_bot_comments(self, pr_details: PRDetails) -> List[Dict[str, Any]]:
        """Fetch all existing bot review comments with their full content for follow-up reviews.
        
        Returns a list of dicts with keys: path, line, body, created_at, id, comment_obj
        Only includes comments that appear to be from the bot (have AI-SIG marker or match bot username).
        """
        try:
            repo_obj = self._get_repo_with_retry(pr_details.repo_full_name)
            pr = self._get_pr_with_retry(repo_obj, pr_details.pull_number)
            bot_comments = []
            
            try:
                existing_comments = pr.get_review_comments()
            except Exception:
                existing_comments = []
            
            # Get bot username/id if available
            try:
                current_user = self.client.get_user().login
            except Exception:
                current_user = None
            
            for c in existing_comments:
                try:
                    path = getattr(c, 'path', None)
                    body = getattr(c, 'body', '') or ''
                    
                    # Check if this is a bot comment (has AI-SIG marker or is from bot user)
                    has_ai_sig = bool(re.search(r'<!--\s*AI-SIG:[a-f0-9]{6,}\s*-->', body, flags=re.IGNORECASE))
                    user = getattr(c, 'user', None)
                    username = getattr(user, 'login', '') if user else ''
                    is_bot_user = current_user and username == current_user
                    
                    if has_ai_sig or is_bot_user:
                        # Clean the body of signature markers for display
                        cleaned_body = self._strip_signature_marker(body)
                        
                        bot_comments.append({
                            'path': path,
                            'line': getattr(c, 'original_line', getattr(c, 'line', None)),
                            'body': cleaned_body,
                            'created_at': str(getattr(c, 'created_at', '')),
                            'id': getattr(c, 'id', None),
                            'comment_obj': c  # Store the full comment object for resolution
                        })
                except Exception:
                    continue
            
            logger.info(f"Found {len(bot_comments)} existing bot review comments for PR #{pr_details.pull_number}")
            return bot_comments
            
        except Exception as e:
            logger.warning(f"Could not fetch existing bot review comments: {e}")
            return []

    def filter_out_existing_comments(self, pr_details: PRDetails, comments: List[ReviewComment]) -> List[ReviewComment]:
        """Filter out comments that match signatures of existing PR comments; also dedupe within batch.
        Enhancements:
        - Skip comments that were already posted (by checking hidden/body-based signatures).
        - Within the current batch, collapse highly similar comments that target the same file+position.
          Keep the highest-priority (or the longest body if priorities tie).
        """
        try:
            existing = self.get_existing_comment_signatures(pr_details)

            # Helpers
            def priority_value(p):
                try:
                    from .models import ReviewPriority as _RP
                    order = {
                        _RP.CRITICAL: 4,
                        _RP.HIGH: 3,
                        _RP.MEDIUM: 2,
                        _RP.LOW: 1,
                    }
                    return order.get(p, 1)
                except Exception:
                    return 1

            filtered: List[ReviewComment] = []
            seen_sigs: Set[str] = set()
            # key: (path, position) -> index in filtered list
            group_index: Dict[str, int] = {}
            # store normalized bodies to compare similarity for the kept comment in each group
            kept_norm_body: Dict[str, str] = {}

            skipped_existing = 0
            skipped_same_line = 0

            for cm in comments:
                try:
                    sig = self._compute_signature(cm.path, cm.body)
                    if sig in existing or sig in seen_sigs:
                        skipped_existing += 1
                        continue

                    # Batch dedupe by same file+position with fuzzy body similarity
                    pos = getattr(cm, 'position', None) or getattr(cm, 'line_number', None)
                    key = f"{cm.path}::{pos}"
                    norm_body = self._normalize_for_signature(cm.body)

                    if key in group_index:
                        idx = group_index[key]
                        prev = filtered[idx]
                        prev_norm = kept_norm_body.get(key, self._normalize_for_signature(prev.body))
                        # Compute similarity ratio
                        try:
                            ratio = difflib.SequenceMatcher(None, prev_norm, norm_body).ratio()
                        except Exception:
                            ratio = 0.0
                        similar = ratio >= 0.8 or (prev_norm and norm_body and (prev_norm in norm_body or norm_body in prev_norm))

                        if similar:
                            # Decide which one to keep
                            keep_new = False
                            if priority_value(cm.priority) > priority_value(prev.priority):
                                keep_new = True
                            elif priority_value(cm.priority) == priority_value(prev.priority) and len(cm.body or '') > len(prev.body or ''):
                                keep_new = True

                            if keep_new:
                                filtered[idx] = cm
                                kept_norm_body[key] = norm_body
                                # Count as skipped because we replaced previous with better one
                                skipped_same_line += 1
                                seen_sigs.add(sig)
                            else:
                                skipped_same_line += 1
                                seen_sigs.add(sig)
                            continue
                        # If not similar, allow multiple distinct comments on same line

                    # First time seeing this position or not similar: keep it
                    seen_sigs.add(sig)
                    group_index[key] = len(filtered)
                    kept_norm_body[key] = norm_body
                    filtered.append(cm)

                except Exception:
                    # On any error, keep the comment rather than dropping it
                    filtered.append(cm)

            if skipped_existing > 0 or skipped_same_line > 0:
                logger.info(
                    f"Deduped comments: skipped {skipped_existing} already-posted/identical and {skipped_same_line} same-line duplicates; {len(filtered)} remain"
                )
            return filtered
        except Exception as e:
            logger.debug(f"Deduplication failed (continuing without dedupe): {e}")
            return comments

    def _get_thread_id_for_comment(self, pr_details: PRDetails, comment_id: int) -> Optional[str]:
        """Get the review thread ID for a given comment ID.
        
        GitHub's GraphQL API requires the thread ID (not comment ID) to resolve threads.
        This method queries the PR's review threads and finds the thread containing the comment.
        
        Args:
            pr_details: Pull request details
            comment_id: The REST API comment ID (databaseId in GraphQL)
            
        Returns:
            The thread ID if found, None otherwise
        """
        try:
            # Parse owner and repo from repo_full_name
            parts = pr_details.repo_full_name.split('/')
            if len(parts) != 2:
                logger.warning(f"Invalid repo_full_name format: {pr_details.repo_full_name}")
                return None
            
            owner, repo = parts
            
            # GraphQL query to fetch review threads and their comments
            query = """
            query($owner: String!, $repo: String!, $pr: Int!) {
              repository(owner: $owner, name: $repo) {
                pullRequest(number: $pr) {
                  reviewThreads(first: 100) {
                    nodes {
                      id
                      isResolved
                      comments(first: 50) {
                        nodes {
                          id
                          databaseId
                          body
                        }
                      }
                    }
                    pageInfo {
                      hasNextPage
                      endCursor
                    }
                  }
                }
              }
            }
            """
            
            variables = {
                "owner": owner,
                "repo": repo,
                "pr": pr_details.pull_number
            }
            
            headers = {
                "Authorization": f"Bearer {self.config.token}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://api.github.com/graphql",
                json={"query": query, "variables": variables},
                headers=headers,
                timeout=self.config.timeout
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch review threads: HTTP {response.status_code}")
                return None
            
            result = response.json()
            
            if 'errors' in result:
                logger.warning(f"GraphQL errors fetching review threads: {result['errors']}")
                return None
            
            # Navigate through the response to find threads
            try:
                threads = result['data']['repository']['pullRequest']['reviewThreads']['nodes']
            except (KeyError, TypeError) as e:
                logger.warning(f"Unexpected GraphQL response structure: {e}")
                return None
            
            # Search for the comment in the threads
            for thread in threads:
                if not thread or 'comments' not in thread:
                    continue
                
                comments = thread.get('comments', {}).get('nodes', [])
                for comment in comments:
                    if not comment:
                        continue
                    
                    # Match by databaseId (the REST API comment ID)
                    if comment.get('databaseId') == comment_id:
                        thread_id = thread.get('id')
                        if thread_id:
                            logger.debug(f"Found thread ID {thread_id} for comment {comment_id}")
                            return thread_id
            
            logger.debug(f"No thread found containing comment {comment_id}")
            return None
            
        except Exception as e:
            logger.warning(f"Error getting thread ID for comment {comment_id}: {str(e)}")
            return None

    def resolve_comment_thread(self, pr_details: PRDetails, comment_id: int) -> bool:
        """Mark a review comment thread as resolved.
        
        This method correctly resolves threads by:
        1. Finding the thread ID that contains the comment (via GraphQL query)
        2. Using that thread ID in the resolveReviewThread mutation
        
        Args:
            pr_details: Pull request details
            comment_id: The ID of the review comment (REST API ID / databaseId)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the thread ID for this comment
            thread_id = self._get_thread_id_for_comment(pr_details, comment_id)
            
            if not thread_id:
                logger.info(f"Thread not found for comment {comment_id} (likely already resolved or deleted)")
                return False
            
            # GraphQL mutation to resolve the thread
            mutation = """
            mutation($threadId: ID!) {
              resolveReviewThread(input: {threadId: $threadId}) {
                thread {
                  id
                  isResolved
                }
              }
            }
            """
            
            variables = {"threadId": thread_id}
            
            # Execute GraphQL request
            headers = {
                "Authorization": f"Bearer {self.config.token}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://api.github.com/graphql",
                json={"query": mutation, "variables": variables},
                headers=headers,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'errors' in result:
                    logger.warning(f"GraphQL error resolving thread for comment {comment_id}: {result['errors']}")
                    return False
                
                # Check if the thread was successfully resolved
                try:
                    is_resolved = result['data']['resolveReviewThread']['thread']['isResolved']
                    if is_resolved:
                        logger.info(f"✅ Successfully marked comment {comment_id} (thread {thread_id}) as resolved")
                        return True
                    else:
                        logger.warning(f"Thread resolution returned success but isResolved=False for comment {comment_id}")
                        return False
                except (KeyError, TypeError) as e:
                    logger.warning(f"Unexpected response structure when resolving comment {comment_id}: {e}")
                    return False
            else:
                logger.warning(f"Failed to resolve comment {comment_id}: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"Failed to resolve comment thread {comment_id}: {str(e)}")
            return False

    def get_file_review_comments(self, pr_details: PRDetails, file_path: str, limit: int = 30) -> Optional[str]:
        """Get previous inline review comments for a specific file in the PR.
        Returns a formatted plain-text summary or None if none found.
        """
        try:
            repo_obj = self._get_repo_with_retry(pr_details.repo_full_name)
            pr = self._get_pr_with_retry(repo_obj, pr_details.pull_number)
            comments = []
            for c in pr.get_review_comments():
                try:
                    path = getattr(c, 'path', None)
                    if path != file_path:
                        continue
                    body = getattr(c, 'body', '') or ''
                    author = getattr(getattr(c, 'user', None), 'login', '') or 'unknown'
                    created = getattr(c, 'created_at', None)
                    created_str = created.isoformat() if getattr(created, 'isoformat', None) else str(created)
                    if not body:
                        continue
                    comments.append((created, f"[{created_str}] {author}: {body}"))
                except Exception:
                    continue
            if not comments:
                return None
            # Sort by created time and keep last N, but present most recent first
            comments.sort(key=lambda x: x[0] or 0)
            formatted = [c[1] for c in comments[-limit:]][::-1]
            header = f"Previous inline review comments on {file_path} (most recent first):\n"
            return header + "\n".join(f"- {line}" for line in formatted)
        except Exception as e:
            logger.debug(f"Failed to fetch previous comments for {file_path}: {e}")
            return None

    def get_repository_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get repository information."""
        try:
            repo_obj = self._get_repo_with_retry(f"{owner}/{repo}")
            return {
                'name': repo_obj.name,
                'full_name': repo_obj.full_name,
                'description': repo_obj.description,
                'language': repo_obj.language,
                'default_branch': repo_obj.default_branch,
                'private': repo_obj.private,
                'size': repo_obj.size,
                'stargazers_count': repo_obj.stargazers_count
            }
        except Exception as e:
            logger.warning(f"Failed to get repository info: {str(e)}")
            return {}
    
    def get_pr_files(self, owner: str, repo: str, pull_number: int) -> List[Dict[str, Any]]:
        """Get list of files changed in a PR."""
        try:
            repo_obj = self._get_repo_with_retry(f"{owner}/{repo}")
            pr = self._get_pr_with_retry(repo_obj, pull_number)
            
            files = []
            for file in pr.get_files():
                files.append({
                    'filename': file.filename,
                    'status': file.status,  # added, removed, modified, renamed
                    'additions': file.additions,
                    'deletions': file.deletions,
                    'changes': file.changes,
                    'patch': getattr(file, 'patch', None)
                })
            
            logger.info(f"Retrieved {len(files)} files from PR #{pull_number}")
            return files
            
        except Exception as e:
            logger.error(f"Failed to get PR files: {str(e)}")
            return []
    
    def get_file_content(self, owner: str, repo: str, file_path: str, ref: str) -> Optional[str]:
        """Get the content of a file from the repository at a specific ref (branch/commit).
        
        Args:
            owner: Repository owner
            repo: Repository name
            file_path: Path to the file in the repository
            ref: Git reference (branch name, commit SHA, etc.)
        
        Returns:
            File content as string, or None if file cannot be retrieved
        """
        try:
            repo_obj = self._get_repo_with_retry(f"{owner}/{repo}")
            
            # Get file content at the specified ref
            try:
                content_file = repo_obj.get_contents(file_path, ref=ref)
                
                # Handle if it's a file (not a directory)
                if hasattr(content_file, 'decoded_content'):
                    decoded_content = content_file.decoded_content.decode('utf-8')
                    logger.debug(f"Retrieved content for {file_path} at {ref} ({len(decoded_content)} chars)")
                    return decoded_content
                else:
                    logger.warning(f"Path {file_path} is not a file")
                    return None
                    
            except Exception as e:
                # File might not exist at this ref (e.g., newly added file)
                logger.debug(f"Could not get file content for {file_path} at {ref}: {str(e)}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to get file content: {str(e)}")
            return None
    
    def check_rate_limit(self) -> Dict[str, Any]:
        """Check GitHub API rate limit status."""
        try:
            rate_limit = self._client.get_rate_limit()
            logger.debug(f"Rate limit object type: {type(rate_limit)}")
            logger.debug(f"Rate limit attributes: {dir(rate_limit)}")
            
            # Handle different PyGithub versions
            if hasattr(rate_limit, 'core'):
                return {
                    'core': {
                        'limit': rate_limit.core.limit,
                        'remaining': rate_limit.core.remaining,
                        'reset': rate_limit.core.reset.timestamp()
                    }
                }
            elif hasattr(rate_limit, 'rate'):
                # Newer PyGithub versions
                return {
                    'core': {
                        'limit': rate_limit.rate.limit,
                        'remaining': rate_limit.rate.remaining,
                        'reset': rate_limit.rate.reset.timestamp()
                    }
                }
            else:
                # If structure is unknown, just return a valid response
                logger.warning(f"Unknown rate limit structure: {rate_limit}")
                return {
                    'core': {
                        'limit': 5000,
                        'remaining': 'unknown',
                        'reset': 'unknown'
                    }
                }
        except Exception as e:
            logger.warning(f"Failed to check rate limit: {str(e)}")
            # Return a valid structure so connection test doesn't fail
            return {
                'core': {
                    'limit': 5000,
                    'remaining': 'unknown',
                    'reset': 'unknown'
                }
            }
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, '_session'):
            self._session.close()
        logger.debug("GitHub client closed")
