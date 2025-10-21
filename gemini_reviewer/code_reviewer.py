"""
Main code reviewer orchestrator for the Gemini AI Code Reviewer.

This module contains the main CodeReviewer class that coordinates all components
and implements concurrent processing for improved performance.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Set

from .config import Config
from .models import (
    PRDetails, ReviewResult, ReviewComment, DiffFile, HunkInfo,
    AnalysisContext, ProcessingStats, ReviewPriority
)
from .github_client import GitHubClient, GitHubClientError
from .gemini_client import GeminiClient, GeminiClientError
from .diff_parser import DiffParser, DiffParsingError


logger = logging.getLogger(__name__)


class CodeReviewerError(Exception):
    """Base exception for code reviewer errors."""
    pass


class CodeReviewer:
    """Main orchestrator class for the code review process."""
    
    def __init__(self, config: Config):
        """Initialize the code reviewer with configuration."""
        self.config = config
        
        # Initialize components
        self.github_client = GitHubClient(config.github)
        self.gemini_client = GeminiClient(config.gemini)
        self.diff_parser = DiffParser()
        
        # Cache of existing AI comment signatures for this PR/session to avoid re-generating the same comments
        self._existing_comment_signatures: Set[str] = set()
        
        # Statistics tracking
        self.stats = ProcessingStats(start_time=time.time())
        
        logger.info("Initialized CodeReviewer with all components")
    
    async def review_pull_request(self, event_path: str) -> ReviewResult:
        """Main entry point for reviewing a pull request."""
        logger.info("=== Starting Pull Request Review ===")
        
        try:
            # Extract PR details from GitHub event
            pr_details = self.github_client.get_pr_details_from_event(event_path)
            logger.info(f"Reviewing PR #{pr_details.pull_number}: {pr_details.title}")
            
            # Load existing AI comment signatures so we can avoid re-generating the same comments
            try:
                self._existing_comment_signatures = self.github_client.get_existing_comment_signatures(pr_details)
                logger.info(f"Loaded {len(self._existing_comment_signatures)} existing AI comment signatures for duplicate avoidance during analysis")
            except Exception as _e:
                logger.debug(f"Could not load existing comment signatures prior to analysis: {_e}")
                self._existing_comment_signatures = set()
            
            # Create initial result object
            result = ReviewResult(pr_details=pr_details)
            
            # Get and parse diff
            diff_content = await self._get_pr_diff(pr_details)
            if diff_content == "":
                logger.info("No new changes to review; exiting without posting a review.")
                self.stats.end_time = time.time()
                result.processing_time = self.stats.duration
                return result
            if not diff_content:
                result.errors.append("Failed to retrieve PR diff")
                return result
            
            # Parse diff into structured format
            diff_files = await self._parse_diff(diff_content)
            if not diff_files:
                logger.info("No files found in diff; nothing to review.")
                self.stats.end_time = time.time()
                result.processing_time = self.stats.duration
                return result
            
            # Filter files based on configuration
            filtered_files = await self._filter_files(diff_files)
            if not filtered_files:
                result.errors.append("No files remaining after filtering")
                return result
            
            result.processed_files = len(filtered_files)
            logger.info(f"Processing {len(filtered_files)} files for review")
            
            # Analyze files and generate comments
            if self.config.performance.enable_concurrent_processing:
                comments = await self._analyze_files_concurrently(filtered_files, pr_details)
            else:
                comments = await self._analyze_files_sequentially(filtered_files, pr_details)
            
            result.comments = comments
            
            # If there are no comments at all, skip creating a review to avoid noise
            if len(comments) == 0:
                logger.info("No comments generated across all files - skipping review creation to avoid noise.")
            else:
                # Per-file reviews have already been posted; skip aggregated final review to avoid large payloads.
                logger.info("Per-file reviews posted; skipping aggregated final review to avoid large payloads.")
            
            # Finalize statistics
            self.stats.end_time = time.time()
            result.processing_time = self.stats.duration
            
            logger.info(f"✅ Review completed in {result.processing_time:.2f}s with {len(comments)} comments")
            return result
            
        except Exception as e:
            logger.error(f"Error during PR review: {str(e)}")
            result = ReviewResult(pr_details=PRDetails("", "", 0, "", ""))
            result.errors.append(str(e))
            return result
    
    async def _get_pr_diff(self, pr_details: PRDetails) -> str:
        """Get PR diff with error handling. If the PR has previous AI review activity by this bot,
        only fetch and review changes introduced since the last reviewed commit to avoid duplicate or stale comments.
        """
        try:
            logger.info("Fetching PR diff...")
            # Try to limit scope to new commits since last AI review
            since_sha = self.github_client.get_last_reviewed_commit_sha(pr_details)
            incremental_diff = None
            if since_sha:
                incremental_diff = self.github_client.get_pr_diff_since(pr_details, since_sha)
                if incremental_diff is not None:
                    if incremental_diff == "":
                        logger.info("No new changes since last AI review; skipping diff analysis.")
                        return ""
                    logger.info("Using incremental diff since last AI-reviewed commit.")
                    logger.debug(f"Retrieved incremental diff with {len(incremental_diff)} characters")
                    return incremental_diff
            # Fallback to full PR diff
            diff_content = self.github_client.get_pr_diff(
                pr_details.owner, pr_details.repo, pr_details.pull_number
            )
            logger.debug(f"Retrieved diff with {len(diff_content)} characters")
            return diff_content
        except GitHubClientError as e:
            logger.error(f"Failed to get PR diff: {str(e)}")
            return ""
    
    async def _parse_diff(self, diff_content: str) -> List[DiffFile]:
        """Parse diff content with error handling."""
        try:
            logger.info("Parsing diff content...")
            diff_files = self.diff_parser.parse_diff(diff_content)
            
            # Log parsing statistics
            stats = self.diff_parser.get_parsing_statistics()
            logger.info(f"Parsed {stats['parsed_files']} files, "
                       f"skipped {stats['skipped_files']} files")
            
            return diff_files
        except DiffParsingError as e:
            logger.error(f"Failed to parse diff: {str(e)}")
            return []
    
    async def _filter_files(self, diff_files: List[DiffFile]) -> List[DiffFile]:
        """Filter files based on configuration."""
        logger.info("Filtering files based on configuration...")
        
        # Apply basic filtering
        filtered_files = self.diff_parser.filter_files(
            diff_files,
            exclude_patterns=self.config.review.exclude_patterns,
            max_files=self.config.review.max_files_per_review,
            min_changes=self.config.review.min_line_changes
        )
        
        # Filter large hunks to manage token usage
        filtered_files = self.diff_parser.filter_large_hunks(
            filtered_files,
            max_lines_per_hunk=self.config.review.max_lines_per_hunk,
            max_hunks_per_file=self.config.review.max_hunks_per_file
        )
        
        # Additional filtering based on configuration
        final_files = []
        for diff_file in filtered_files:
            if self.config.should_review_file(diff_file.file_info.path):
                final_files.append(diff_file)
            else:
                logger.debug(f"Skipping file due to config: {diff_file.file_info.path}")
        
        logger.info(f"Filtered down to {len(final_files)} files for review")
        return final_files
    
    async def _analyze_files_sequentially(
        self, 
        diff_files: List[DiffFile], 
        pr_details: PRDetails
    ) -> List[ReviewComment]:
        """Analyze files sequentially."""
        logger.info("Analyzing files sequentially...")
        
        all_comments = []
        
        for i, diff_file in enumerate(diff_files):
            logger.info(f"Analyzing file {i+1}/{len(diff_files)}: {diff_file.file_info.path}")
            
            try:
                file_comments = await self._analyze_single_file(diff_file, pr_details)
                all_comments.extend(file_comments)
                self.stats.files_processed += 1

                # Post comments for this file immediately to avoid large aggregated reviews
                if file_comments:
                    logger.info(f"Posting review for file: {diff_file.file_info.path} with {len(file_comments)} comments")
                    await self._create_github_review(pr_details, file_comments, preferred_event="COMMENT")
                else:
                    logger.debug(f"No comments for file: {diff_file.file_info.path}")
                
            except Exception as e:
                logger.error(f"Error analyzing file {diff_file.file_info.path}: {str(e)}")
                self.stats.errors_encountered += 1
                continue
        
        return all_comments
    
    async def _analyze_files_concurrently(
        self, 
        diff_files: List[DiffFile], 
        pr_details: PRDetails
    ) -> List[ReviewComment]:
        """Analyze files concurrently for improved performance."""
        logger.info(f"Analyzing {len(diff_files)} files concurrently "
                   f"(max workers: {self.config.performance.max_concurrent_files})")
        
        all_comments = []
        
        # Process files in chunks to manage resources
        chunk_size = self.config.performance.chunk_size
        max_workers = min(self.config.performance.max_concurrent_files, len(diff_files))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all file analysis tasks
            future_to_file = {
                executor.submit(self._analyze_single_file_sync, diff_file, pr_details): diff_file
                for diff_file in diff_files
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_file):
                diff_file = future_to_file[future]
                
                try:
                    file_comments = future.result()
                    all_comments.extend(file_comments)
                    self.stats.files_processed += 1
                    
                    logger.debug(f"Completed analysis of {diff_file.file_info.path} "
                               f"({len(file_comments)} comments)")

                    # Post comments for this file immediately
                    if file_comments:
                        logger.info(f"Posting review for file: {diff_file.file_info.path} with {len(file_comments)} comments")
                        await self._create_github_review(pr_details, file_comments, preferred_event="COMMENT")
                    else:
                        logger.debug(f"No comments for file: {diff_file.file_info.path}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing file {diff_file.file_info.path}: {str(e)}")
                    self.stats.errors_encountered += 1
        
        logger.info(f"Concurrent analysis completed: {len(all_comments)} total comments")
        return all_comments
    
    def _analyze_single_file_sync(self, diff_file: DiffFile, pr_details: PRDetails) -> List[ReviewComment]:
        """Synchronous wrapper for analyzing a single file (for thread pool)."""
        return asyncio.run(self._analyze_single_file(diff_file, pr_details))
    
    async def _analyze_single_file(self, diff_file: DiffFile, pr_details: PRDetails) -> List[ReviewComment]:
        """Analyze a single file and return review comments."""
        file_path = diff_file.file_info.path
        logger.debug(f"Starting analysis of {file_path}")
        
        file_comments = []
        skipped_pre_existing = 0
        
        # Detect related files and gather project context
        related_files = await self._detect_related_files(diff_file, pr_details)
        project_context = await self._build_project_context(diff_file, related_files, pr_details)
        
        # Create enhanced analysis context with project understanding
        context = AnalysisContext(
            pr_details=pr_details,
            file_info=diff_file.file_info,
            language=self.diff_parser.get_file_language(file_path),
            related_files=related_files,
            project_context=project_context
        )
        
        # Get prompt template based on configuration
        prompt_template = self.config.get_review_prompt_template()
        
        # Analyze each hunk in the file
        for hunk_index, hunk in enumerate(diff_file.hunks):
            try:
                logger.debug(f"Analyzing hunk {hunk_index+1}/{len(diff_file.hunks)} in {file_path}")
                
                # Get AI analysis for this hunk
                ai_responses = self.gemini_client.analyze_code_hunk(
                    hunk, context, prompt_template
                )
                
                self.stats.api_calls_made += 1
                
                # Convert AI responses to review comments
                for ai_response in ai_responses:
                    comment = self._convert_to_review_comment(
                        ai_response, diff_file, hunk, hunk_index
                    )
                    if comment:
                        # Preemptive duplicate avoidance: skip if this comment (path+body) matches an existing signature
                        try:
                            sig = self.github_client._compute_signature(comment.path, comment.body)
                        except Exception:
                            sig = None
                        if sig and sig in getattr(self, '_existing_comment_signatures', set()):
                            skipped_pre_existing += 1
                            logger.debug(f"Skipping generated duplicate comment on {comment.path} at pos {comment.position}")
                        else:
                            file_comments.append(comment)
                            # Update the session cache so we avoid emitting the same suggestion again later in this run
                            try:
                                if sig:
                                    self._existing_comment_signatures.add(sig)
                            except Exception:
                                pass
                
            except GeminiClientError as e:
                logger.warning(f"AI analysis failed for hunk {hunk_index+1} in {file_path}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error analyzing hunk in {file_path}: {str(e)}")
                continue
        
        logger.debug(f"Generated {len(file_comments)} comments for {file_path}")
        return file_comments
    
    async def _detect_related_files(self, diff_file: DiffFile, pr_details: PRDetails) -> List[str]:
        """Detect related files based on imports and dependencies."""
        file_path = diff_file.file_info.path
        related_files = []
        
        try:
            # Extract imports/requires/includes from the file content
            import_patterns = {
                'python': [r'from\s+(\S+)\s+import', r'import\s+(\S+)'],
                'javascript': [r'import\s+.*\s+from\s+["\']([^"\']+)["\']', r'require\(["\']([^"\']+)["\']\)'],
                'typescript': [r'import\s+.*\s+from\s+["\']([^"\']+)["\']', r'require\(["\']([^"\']+)["\']\)'],
                'java': [r'import\s+([^;]+);'],
                'go': [r'import\s+["\']([^"\']+)["\']', r'import\s+\(\s*["\']([^"\']+)["\']'],
                'ruby': [r'require\s+["\']([^"\']+)["\']'],
            }
            
            language = self.diff_parser.get_file_language(file_path)
            
            if language and language.lower() in import_patterns:
                import re
                patterns = import_patterns[language.lower()]
                
                # Analyze the hunk content for imports
                for hunk in diff_file.hunks:
                    for line in hunk.lines:
                        # Only look at added or context lines, not deleted lines
                        if line.startswith('-'):
                            continue
                        
                        clean_line = line[1:] if line else line  # Remove +/- prefix
                        
                        for pattern in patterns:
                            matches = re.findall(pattern, clean_line)
                            for match in matches:
                                # Convert import path to file path
                                related_file = self._import_to_file_path(match, file_path, language)
                                if related_file and related_file not in related_files:
                                    related_files.append(related_file)
            
            # Limit to most relevant files to avoid token bloat
            related_files = related_files[:5]
            
            if related_files:
                logger.info(f"Detected {len(related_files)} related files for {file_path}")
            
        except Exception as e:
            logger.debug(f"Error detecting related files for {file_path}: {str(e)}")
        
        return related_files
    
    def _import_to_file_path(self, import_path: str, current_file: str, language: str) -> Optional[str]:
        """Convert an import path to a file path."""
        try:
            import os
            
            # Handle relative imports
            if import_path.startswith('.'):
                current_dir = os.path.dirname(current_file)
                # Count leading dots
                dots = len(import_path) - len(import_path.lstrip('.'))
                # Go up directories
                for _ in range(dots - 1):
                    current_dir = os.path.dirname(current_dir)
                import_path = import_path.lstrip('.')
                base_path = os.path.join(current_dir, import_path.replace('.', '/'))
            else:
                # For absolute imports, try common patterns
                base_path = import_path.replace('.', '/')
            
            # Add appropriate extensions based on language
            extensions = {
                'python': ['.py', '/__init__.py'],
                'javascript': ['.js', '/index.js', '.jsx'],
                'typescript': ['.ts', '/index.ts', '.tsx'],
                'java': ['.java'],
                'go': ['.go'],
                'ruby': ['.rb']
            }
            
            possible_paths = []
            if language.lower() in extensions:
                for ext in extensions[language.lower()]:
                    possible_paths.append(base_path + ext)
            
            # Return the first plausible path
            return possible_paths[0] if possible_paths else None
            
        except Exception:
            return None
    
    async def _build_project_context(self, diff_file: DiffFile, related_files: List[str], pr_details: PRDetails) -> Optional[str]:
        """Build project context including previous comments and related file contents."""
        context_parts = []
        max_context_size = 8000  # Limit total context to avoid token bloat
        current_size = 0
        
        try:
            # Always include previous inline comments on this file (if any)
            try:
                prev = self.github_client.get_file_review_comments(pr_details, diff_file.file_info.path, limit=30)
                if prev:
                    # Keep this small; it helps verify resolutions
                    snippet = prev
                    if len(snippet) > 2000:
                        snippet = snippet[:2000] + "\n... (truncated)"
                    context_parts.append(f"### Previous review history for {diff_file.file_info.path}\n{snippet}\n")
                    current_size += len(snippet)
            except Exception:
                pass
            
            # Related files content
            for related_file in related_files or []:
                if current_size >= max_context_size:
                    break
                
                # Try to fetch the file content from the base branch
                content = self.github_client.get_file_content(
                    pr_details.owner,
                    pr_details.repo,
                    related_file,
                    pr_details.base_sha or 'main'
                )
                
                if content:
                    # Limit individual file size
                    if len(content) > 2000:
                        content = content[:2000] + "\n... (truncated)"
                    
                    context_parts.append(f"### Related file: {related_file}\n```\n{content}\n```\n")
                    current_size += len(content)
            
            if context_parts:
                logger.info(f"Built project context with {len(context_parts)} block(s) (~{current_size} chars)")
                return "\n".join(context_parts)
                
        except Exception as e:
            logger.debug(f"Error building project context: {str(e)}")
        
        return None
    
    def _convert_to_review_comment(
        self,
        ai_response,
        diff_file: DiffFile,
        hunk: HunkInfo,
        hunk_index: int
    ) -> Optional[ReviewComment]:
        """Convert AI response to GitHub review comment with basic anchoring validation.
        Tries to ensure the comment targets the correct line by matching an anchor snippet
        to the diff lines. If mismatch, attempts to realign; otherwise discards to avoid noise.
        """
        try:
            # Start with the provided position as 1-based index within this hunk
            position = int(ai_response.line_number)

            # Helper to strip diff prefix and get comparable content
            def _line_payload(s: str) -> str:
                if not s:
                    return ""
                # Remove leading diff marker and one space if present
                if s[0] in ['+', '-', ' ']:
                    s = s[1:]
                return s.lstrip('\t ')

            # Guard: position within bounds
            if position < 1 or position > len(hunk.lines):
                logger.warning(f"Line number {position} is outside hunk range (1..{len(hunk.lines)})")
                return None

            anchor = getattr(ai_response, 'anchor_snippet', None)
            anchor = anchor.strip() if isinstance(anchor, str) else None

            # If we have an anchor, validate alignment; else try to infer from comment
            if not anchor:
                # Try to infer from inline code span in the comment
                body = ai_response.review_comment or ""
                if '`' in body:
                    try:
                        import re as _re
                        m = _re.search(r"`([^`\n]+)`", body)
                        if m:
                            candidate = m.group(1).strip()
                            if len(candidate) >= 2:
                                anchor = candidate
                    except Exception:
                        pass

            # If we have an anchor snippet, attempt validation and possible realignment
            if anchor:
                target_line = _line_payload(hunk.lines[position - 1])
                if anchor not in target_line:
                    # Search for best match within the hunk
                    matches = []
                    for idx, raw in enumerate(hunk.lines, start=1):
                        payload = _line_payload(raw)
                        if anchor in payload:
                            matches.append(idx)
                    if len(matches) == 1:
                        # Realign to the unique matching line
                        logger.info(f"Realigning comment from line {position} to {matches[0]} based on anchor match")
                        position = matches[0]
                    elif len(matches) > 1:
                        # Prefer lines with additions ('+')
                        add_matches = [i for i in matches if hunk.lines[i - 1].startswith('+')]
                        if len(add_matches) == 1:
                            position = add_matches[0]
                            logger.info(f"Realigning to added line {position} among multiple matches")
                        else:
                            # Fall back to nearest match to original
                            nearest = min(matches, key=lambda i: abs(i - position))
                            logger.info(f"Multiple matches; choosing nearest line {nearest} to original {position}")
                            position = nearest
                    else:
                        # No match: discard to avoid unrelated comment
                        logger.warning("Anchor snippet not found in hunk; discarding comment to avoid misalignment")
                        return None

            # Prefer commenting on added lines; if current is deletion-only, try to nudge to nearby added/context
            if hunk.lines[position - 1].startswith('-'):
                # Look within a small window for a '+' or ' ' line
                window = range(max(1, position - 2), min(len(hunk.lines), position + 2) + 1)
                preferred = None
                for i in window:
                    if hunk.lines[i - 1].startswith('+'):
                        preferred = i
                        break
                if not preferred:
                    for i in window:
                        if hunk.lines[i - 1].startswith(' '):
                            preferred = i
                            break
                if preferred:
                    logger.info(f"Adjusting position from deletion line {position} to nearby line {preferred}")
                    position = preferred

            # Final bounds check
            if position < 1 or position > len(hunk.lines):
                return None

            comment = ReviewComment(
                body=ai_response.review_comment,
                path=diff_file.file_info.path,
                position=position,
                line_number=position,
                priority=ai_response.priority,
                category=ai_response.category
            )
            return comment

        except Exception as e:
            logger.warning(f"Error converting AI response to comment: {str(e)}")
            return None
    
    async def _create_github_review(self, pr_details: PRDetails, comments: List[ReviewComment], preferred_event: Optional[str] = None) -> bool:
        """Create GitHub review with comments.
        If preferred_event is provided, it will be used instead of auto-deciding.
        """
        try:
            total_comments = len(comments)
            logger.info(f"Creating GitHub review with {total_comments} total comments...")

            # First, deduplicate against existing PR comments and within this batch
            comments = self.github_client.filter_out_existing_comments(pr_details, comments)
            total_comments = len(comments)
            logger.info(f"After deduplication, {total_comments} comments remain to consider.")
            
            # Filter comments by priority if configured
            filtered_comments = self._filter_comments_by_priority(comments)
            
            # If there are no issues at all, skip creating a review to avoid noise
            if total_comments == 0 and not filtered_comments:
                logger.info("No issues found - skipping review creation to avoid noise")
                return True
            
            # Determine review event
            # Note: Using COMMENT instead of APPROVE because GitHub Actions tokens
            # are not permitted to approve pull requests (GitHub API restriction)
            if preferred_event:
                event = preferred_event
                if filtered_comments:
                    logger.info(f"Using preferred event '{event}' for {len(filtered_comments)} comments")
                else:
                    logger.info(f"Using preferred event '{event}' with no comments")
            else:
                if not filtered_comments:
                    if total_comments > 0:
                        logger.info(f"All {total_comments} comments were filtered by priority threshold - posting comment about this")
                        event = "COMMENT"
                    else:
                        logger.info("No comments generated - posting positive review")
                        event = "COMMENT"
                else:
                    logger.info(f"Found {len(filtered_comments)} comments (out of {total_comments} total) - requesting changes")
                    event = "REQUEST_CHANGES"
            
            # Pass both total and filtered comments so summary can be accurate
            success = self.github_client.create_review(
                pr_details, 
                filtered_comments, 
                event,
                total_comments_generated=total_comments
            )
            if success:
                if filtered_comments:
                    logger.info("✅ Successfully created GitHub review")
                else:
                    logger.info("✅ Successfully posted review")
            
            return success
            
        except GitHubClientError as e:
            logger.error(f"Failed to create GitHub review: {str(e)}")
            return False
    
    def _filter_comments_by_priority(self, comments: List[ReviewComment]) -> List[ReviewComment]:
        """Filter comments based on priority threshold."""
        if not comments:
            return []
        
        priority_order = {
            ReviewPriority.CRITICAL: 4,
            ReviewPriority.HIGH: 3,
            ReviewPriority.MEDIUM: 2,
            ReviewPriority.LOW: 1
        }
        
        threshold_value = priority_order.get(self.config.review.priority_threshold, 1)
        
        filtered_comments = []
        for comment in comments:
            comment_value = priority_order.get(comment.priority, 1)
            if comment_value >= threshold_value:
                filtered_comments.append(comment)
        
        if len(filtered_comments) != len(comments):
            logger.info(f"Filtered {len(comments)} comments to {len(filtered_comments)} "
                       f"based on priority threshold ({self.config.review.priority_threshold.value})")
        
        return filtered_comments
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        github_stats = {}
        gemini_stats = self.gemini_client.get_statistics()
        parsing_stats = self.diff_parser.get_parsing_statistics()
        
        try:
            rate_limit = self.github_client.check_rate_limit()
            github_stats = rate_limit.get('core', {})
        except Exception:
            pass
        
        return {
            'processing': {
                'duration': self.stats.duration,
                'files_processed': self.stats.files_processed,
                'files_skipped': self.stats.files_skipped,
                'api_calls_made': self.stats.api_calls_made,
                'errors_encountered': self.stats.errors_encountered,
                'processing_rate': self.stats.processing_rate
            },
            'github': github_stats,
            'gemini': gemini_stats,
            'parsing': parsing_stats
        }
    
    def test_connections(self) -> Dict[str, bool]:
        """Test connections to external services."""
        logger.info("Testing connections to external services...")
        
        results = {}
        
        # Test GitHub connection
        try:
            # Test by getting rate limit info - if this works, connection is OK
            rate_limit = self.github_client.check_rate_limit()
            
            # If we got any rate limit response, connection is working
            if rate_limit and 'core' in rate_limit:
                results['github'] = True
                remaining = rate_limit.get('core', {}).get('remaining', 'unknown')
                
                # Try to get additional user info for better logging
                try:
                    user = self.github_client._client.get_user()
                    github_user = user.login if user else "unknown"
                    logger.info(f"✅ GitHub connection: OK (user: {github_user}, remaining: {remaining})")
                except Exception:
                    # User info failed, but connection is still OK based on rate limit check
                    logger.info(f"✅ GitHub connection: OK (remaining: {remaining})")
            else:
                # Rate limit check didn't return expected structure
                raise Exception("Rate limit check returned unexpected structure")
                    
        except Exception as e:
            results['github'] = False
            logger.error(f"❌ GitHub connection failed: {str(e)}")
        
        # Test Gemini connection
        try:
            results['gemini'] = self.gemini_client.test_connection()
            if results['gemini']:
                logger.info("✅ Gemini connection: OK")
            else:
                logger.error("❌ Gemini connection failed")
        except Exception as e:
            results['gemini'] = False
            logger.error(f"❌ Gemini connection error: {str(e)}")
        
        return results
    
    def close(self):
        """Clean up resources."""
        logger.info("Cleaning up CodeReviewer resources...")
        
        try:
            self.github_client.close()
            self.gemini_client.close()
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
        
        logger.info("CodeReviewer cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
