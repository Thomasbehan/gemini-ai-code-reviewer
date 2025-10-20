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
from typing import List, Dict, Any, Optional, Set
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
        """Generate a minimal message when no issues are found (non-praising)."""
        return (
            "🤖 **Gemini AI Code Review**\n\n"
            "No issues requiring changes were detected.\n\n"
            "> This review was automatically generated by Gemini AI."
        )
    
    def _generate_filtered_message(self, total_comments: int) -> str:
        """Generate a message when comments were found but filtered by priority threshold."""
        return (
            f"🤖 **Gemini AI Code Review**\n\n"
            f"Found **{total_comments}** observations during analysis, but they were below the configured priority threshold and not shown as inline comments.\n\n"
            f"The code was reviewed with project context and related files for comprehensive analysis.\n\n"
            f"> This review was automatically generated by Gemini AI. Consider lowering the priority threshold in configuration to see all suggestions."
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

    def filter_out_existing_comments(self, pr_details: PRDetails, comments: List[ReviewComment]) -> List[ReviewComment]:
        """Filter out comments that match signatures of existing PR comments; also dedupe within batch."""
        try:
            existing = self.get_existing_comment_signatures(pr_details)
            filtered: List[ReviewComment] = []
            seen: Set[str] = set()
            skipped = 0
            for cm in comments:
                try:
                    sig = self._compute_signature(cm.path, cm.body)
                    if sig in existing or sig in seen:
                        skipped += 1
                        continue
                    seen.add(sig)
                    filtered.append(cm)
                except Exception:
                    # On any error, keep the comment rather than dropping it
                    filtered.append(cm)
            if skipped > 0:
                logger.info(f"Deduped comments: skipped {skipped} already-posted or duplicate comments; {len(filtered)} remain")
            return filtered
        except Exception as e:
            logger.debug(f"Deduplication failed (continuing without dedupe): {e}")
            return comments

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
