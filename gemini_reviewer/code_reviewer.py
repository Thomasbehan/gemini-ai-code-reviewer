"""
Main code reviewer orchestrator for the Gemini AI Code Reviewer.

This module contains the main CodeReviewer class that coordinates all components
and implements concurrent processing for improved performance.
"""

import asyncio
import logging
import time
import re
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
from .context_builder import ContextBuilder
from .comment_processor import CommentProcessor
from .utils import get_file_language


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
        self.context_builder = ContextBuilder(self.github_client, self.diff_parser)
        self.comment_processor = CommentProcessor(config.review, self.github_client)
        
        # Cache of existing AI comment signatures for this PR/session to avoid re-generating the same comments
        self._existing_comment_signatures: Set[str] = set()
        
        # Track if this is a follow-up review and store previous comments
        self._is_followup_review: bool = False
        self._previous_comments: str = ""
        self._current_followup_issues: Set[str] = set()  # Track which files/paths still have issues in follow-up
        # Track unresolved previous comment IDs detected during follow-up and files reviewed in this run
        self._unresolved_prior_ids: Set[int] = set()
        self._current_review_file_paths: Set[str] = set()
        
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
            
            # Fetch existing bot comments to determine if this is a follow-up review
            try:
                existing_bot_comments = self.github_client.get_existing_bot_comments(pr_details)
                # Reset tracking for this run
                self._unresolved_prior_ids = set()
                self._current_review_file_paths = set()
                if existing_bot_comments:
                    self._is_followup_review = True
                    # Format previous comments for the AI and store ID mapping
                    formatted_comments = []
                    self._comment_id_mapping = {}  # Map comment index to comment ID
                    for i, comment in enumerate(existing_bot_comments, 1):
                        formatted_comments.append(
                            f"{i}. File: {comment['path']}\n"
                            f"   Line: {comment.get('line', 'N/A')}\n"
                            f"   Comment: {comment['body']}\n"
                            f"   Posted: {comment.get('created_at', 'N/A')}"
                        )
                        # Store mapping of comment index to ID for resolution later
                        if comment.get('id'):
                            self._comment_id_mapping[i] = comment['id']
                    self._previous_comments = "\n\n".join(formatted_comments)
                    self._previous_bot_comments = existing_bot_comments  # Store full comment data
                    logger.info(f"🔄 FOLLOW-UP REVIEW MODE: Found {len(existing_bot_comments)} previous bot comments. AI will ONLY check if they were resolved.")
                else:
                    self._is_followup_review = False
                    self._previous_comments = ""
                    self._comment_id_mapping = {}
                    self._previous_bot_comments = []
                    logger.info("✨ FIRST REVIEW: No previous bot comments found. AI will perform initial comprehensive review.")
            except Exception as _e:
                logger.debug(f"Could not determine review type: {_e}")
                self._is_followup_review = False
                self._previous_comments = ""
                self._comment_id_mapping = {}
                self._previous_bot_comments = []
            
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
                logger.info("No files remaining after filtering; nothing to review.")
                self.stats.end_time = time.time()
                result.processing_time = self.stats.duration
                return result
            
            result.processed_files = len(filtered_files)
            logger.info(f"Processing {len(filtered_files)} files for review")
            
            # Analyze files and generate comments
            if self.config.performance.enable_concurrent_processing:
                comments = await self._analyze_files_concurrently(filtered_files, pr_details)
            else:
                comments = await self._analyze_files_sequentially(filtered_files, pr_details)
            
            result.comments = comments
            
            # If there are no comments at all
            if len(comments) == 0:
                if self.stats.files_processed > 0:
                    if self._is_followup_review:
                        # In follow-up mode, do not post an "everything looks good" summary.
                        # Resolved threads will be acknowledged via per-thread replies below.
                        logger.info("Follow-up review produced no unresolved items; skipping approval summary.")
                    else:
                        logger.info("No comments generated across all files - posting an approval review to acknowledge clean changes.")
                        try:
                            await self._create_github_review(pr_details, [], preferred_event="APPROVE")
                        except Exception as _e:
                            logger.debug(f"Could not create approval review: {_e}")
                else:
                    logger.info("No comments generated and no files were processed - skipping review creation.")
            else:
                # Per-file reviews have already been posted; skip aggregated final review to avoid large payloads.
                logger.info("Per-file reviews posted; skipping aggregated final review to avoid large payloads.")
            
            # If this was a follow-up review, check for resolved comments and mark them
            if self._is_followup_review:
                await self._resolve_completed_comments(pr_details)
            
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
        
        # Ensure we only analyze each file path once
        unique_files: List[DiffFile] = []
        seen_paths: Set[str] = set()
        for df in diff_files:
            p = df.file_info.path
            if p not in seen_paths:
                seen_paths.add(p)
                unique_files.append(df)
            else:
                logger.debug(f"Skipping duplicate file entry: {p}")
        
        all_comments = []
        
        for i, diff_file in enumerate(unique_files):
            logger.info(f"Analyzing file {i+1}/{len(unique_files)}: {diff_file.file_info.path}")
            # Track that we reviewed this file path during this run (used for resolution replies)
            try:
                self._current_review_file_paths.add(diff_file.file_info.path)
            except Exception:
                pass
            
            try:
                file_comments = await self._analyze_single_file(diff_file, pr_details, unique_files)
                all_comments.extend(file_comments)
                self.stats.files_processed += 1

                # Post comments for this file immediately to avoid large aggregated reviews
                if file_comments:
                    if self._is_followup_review:
                        logger.info(
                            f"Posting follow-up replies on previous comments for file: {diff_file.file_info.path} "
                            f"({len(file_comments)} unresolved item(s))"
                        )
                        await self._post_followup_replies(pr_details, diff_file, file_comments)
                        # Track this path as reviewed
                        try:
                            self._current_review_file_paths.add(diff_file.file_info.path)
                        except Exception:
                            pass
                    else:
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
        """Analyze files concurrently for improved performance.
        Guarantees only one worker per unique file path.
        """
        
        # Ensure we only schedule one task per unique file path
        unique_files: List[DiffFile] = []
        seen_paths: Set[str] = set()
        for df in diff_files:
            p = df.file_info.path
            if p not in seen_paths:
                seen_paths.add(p)
                unique_files.append(df)
            else:
                logger.debug(f"Skipping duplicate file entry: {p}")
        
        logger.info(
            f"Analyzing {len(unique_files)} unique files concurrently (up to "
            f"{self.config.performance.max_concurrent_files} files in parallel; "
            f"one worker per file)"
        )
        
        all_comments = []
        
        max_workers = min(self.config.performance.max_concurrent_files, len(unique_files))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit one task per unique file, passing all files for cross-file awareness
            future_to_file = {
                executor.submit(self._analyze_single_file_sync, diff_file, pr_details, unique_files): diff_file
                for diff_file in unique_files
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_file):
                diff_file = future_to_file[future]
                
                try:
                    file_comments = future.result()
                    all_comments.extend(file_comments)
                    self.stats.files_processed += 1
                    
                    # Track this path as reviewed
                    try:
                        self._current_review_file_paths.add(diff_file.file_info.path)
                    except Exception:
                        pass
                    
                    logger.debug(
                        f"Completed analysis of {diff_file.file_info.path} (" 
                        f"{len(file_comments)} comments)"
                    )

                    # Post comments for this file immediately
                    if file_comments:
                        if self._is_followup_review:
                            logger.info(
                                f"Posting follow-up replies on previous comments for file: {diff_file.file_info.path} "
                                f"({len(file_comments)} unresolved item(s))"
                            )
                            await self._post_followup_replies(pr_details, diff_file, file_comments)
                        else:
                            logger.info(
                                f"Posting review for file: {diff_file.file_info.path} with "
                                f"{len(file_comments)} comments"
                            )
                            await self._create_github_review(pr_details, file_comments, preferred_event="COMMENT")
                    else:
                        logger.debug(f"No comments for file: {diff_file.file_info.path}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing file {diff_file.file_info.path}: {str(e)}")
                    self.stats.errors_encountered += 1
        
        logger.info(f"Concurrent analysis completed: {len(all_comments)} total comments")
        return all_comments
    
    def _analyze_single_file_sync(
        self,
        diff_file: DiffFile,
        pr_details: PRDetails,
        all_diff_files: Optional[List[DiffFile]] = None
    ) -> List[ReviewComment]:
        """Synchronous wrapper for analyzing a single file (for thread pool)."""
        return asyncio.run(self._analyze_single_file(diff_file, pr_details, all_diff_files))
    
    async def _analyze_single_file(
        self,
        diff_file: DiffFile,
        pr_details: PRDetails,
        all_diff_files: Optional[List[DiffFile]] = None
    ) -> List[ReviewComment]:
        """Analyze a single file and return review comments."""
        file_path = diff_file.file_info.path
        logger.debug(f"Starting analysis of {file_path}")

        file_comments = []
        skipped_pre_existing = 0

        # Detect related files and gather project context
        related_files = await self.context_builder.detect_related_files(diff_file, pr_details)
        project_context = await self.context_builder.build_project_context(diff_file, related_files, pr_details)

        # Fetch full file content for better context
        full_file_content = None
        try:
            full_file_content = self.github_client.get_file_content(
                pr_details.owner, pr_details.repo, file_path, pr_details.head_sha or 'HEAD'
            )
        except Exception as e:
            logger.debug(f"Could not fetch full file content for {file_path}: {e}")
            # Fallback to local file if available
            try:
                import os
                local_path = os.path.join(os.getcwd(), file_path)
                if os.path.exists(local_path):
                    with open(local_path, 'r', encoding='utf-8', errors='ignore') as f:
                        full_file_content = f.read()
            except Exception:
                pass

        # Gather list of all changed files and create a summary
        all_changed_files = []
        change_summary = None
        if all_diff_files:
            all_changed_files = [df.file_info.path for df in all_diff_files]
            # Create a summary of changes across all files
            change_summary = self._build_change_summary(all_diff_files)

        # Create enhanced analysis context with project understanding
        context = AnalysisContext(
            pr_details=pr_details,
            file_info=diff_file.file_info,
            language=get_file_language(file_path),
            related_files=related_files,
            project_context=project_context,
            full_file_content=full_file_content,
            all_changed_files=all_changed_files,
            change_summary=change_summary
        )
        
        # Get prompt template based on configuration and review type
        # If this is a follow-up review, temporarily override to FOLLOWUP mode
        if self._is_followup_review:
            from .prompts import ReviewMode
            # Save original mode
            original_mode = self.config.review.review_mode
            # Temporarily set to FOLLOWUP mode
            self.config.review.review_mode = ReviewMode.FOLLOWUP
            prompt_template = self.config.get_review_prompt_template(self._previous_comments)
            # Restore original mode
            self.config.review.review_mode = original_mode
        else:
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
                    comment = self.comment_processor.convert_to_review_comment(
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
        
        # Track files with issues during follow-up review for resolution checking
        if self._is_followup_review and file_comments:
            self._current_followup_issues.add(file_path)
        
        logger.debug(f"Generated {len(file_comments)} comments for {file_path}")
        return file_comments
    
    async def _post_followup_replies(self, pr_details: PRDetails, diff_file: DiffFile, file_comments: List[ReviewComment]) -> None:
        """Post follow-up findings as replies to the specific previous bot comments they're referring to.
        Selection priority per new comment:
        1) If the body references a "Previous issue: ...", reply to the prior bot comment whose body best matches that text.
        2) Otherwise, choose the prior bot comment on the same path with nearest diff position (fallback to nearest line number).
        3) As a last resort, reply to the most recent prior bot comment on the same path.
        """
        try:
            if not getattr(self, '_previous_bot_comments', None):
                logger.debug("No previous bot comments available to reply to.")
                return
            path = diff_file.file_info.path
            prior_for_path = [c for c in self._previous_bot_comments if c.get('path') == path and c.get('id')]
            if not prior_for_path:
                logger.debug(f"No previous bot comments found to reply to for file: {path}")
                return

            # Normalize previous entries: ensure numeric helpers and normalized body
            for c in prior_for_path:
                try:
                    c['_line_num'] = int(c.get('line')) if c.get('line') is not None else None
                except Exception:
                    c['_line_num'] = None
                try:
                    c['_position'] = int(c.get('position')) if c.get('position') is not None else int(c.get('original_position')) if c.get('original_position') is not None else None
                except Exception:
                    c['_position'] = None
                try:
                    txt = (c.get('body') or '').strip()
                    # Normalize whitespace and lowercase for robust matching
                    c['_norm_body'] = re.sub(r"\s+", " ", txt).lower()
                except Exception:
                    c['_norm_body'] = (c.get('body') or '').lower()

            # Helper to extract referenced previous issue text from a follow-up body
            def extract_referenced_issue(text: str) -> str:
                try:
                    import re as _re
                    m = _re.search(r"previous issue\s*:\s*(.+?)\s*(?:status\s*:\s*|$)", text, flags=_re.IGNORECASE | _re.DOTALL)
                    if m:
                        return _re.sub(r"\s+", " ", m.group(1)).strip()
                except Exception:
                    pass
                return ""

            # For each new comment, choose the best prior target
            for cm in file_comments:
                body = getattr(cm, 'body', None) or ""
                if not body.strip():
                    continue

                # 1) Try content-based matching using the referenced previous issue text
                referenced = extract_referenced_issue(body)
                referenced_norm = re.sub(r"\s+", " ", referenced).lower() if referenced else ""
                best = None
                best_score = 0.0
                if referenced_norm:
                    for c in prior_for_path:
                        prev_norm = c.get('_norm_body') or ""
                        if not prev_norm:
                            continue
                        # Containment is a strong signal
                        if referenced_norm and (referenced_norm in prev_norm or prev_norm in referenced_norm):
                            score = 1.0 if referenced_norm in prev_norm else 0.9
                        else:
                            try:
                                import difflib as _dl
                                score = _dl.SequenceMatcher(None, referenced_norm, prev_norm).ratio()
                            except Exception:
                                score = 0.0
                        if score > best_score:
                            best = c
                            best_score = score
                
                # 2) Fallback to nearest diff position/line if no good content match
                if best is None or best_score < 0.6:
                    cm_pos = getattr(cm, 'position', None)
                    cm_line = getattr(cm, 'line_number', None)
                    try:
                        cm_pos = int(cm_pos) if cm_pos is not None else None
                    except Exception:
                        cm_pos = None
                    try:
                        cm_line = int(cm_line) if cm_line is not None else None
                    except Exception:
                        cm_line = None

                    best_dist = None
                    for c in prior_for_path:
                        dist = None
                        if cm_pos is not None and c.get('_position') is not None:
                            dist = abs(c.get('_position') - cm_pos)
                        elif cm_line is not None and c.get('_line_num') is not None:
                            dist = abs(c.get('_line_num') - cm_line)
                        if dist is not None and (best is None or dist < best_dist):
                            best = c
                            best_dist = dist

                # 3) Final fallback: most recent prior comment
                if best is None:
                    try:
                        prior_for_path_sorted = sorted(prior_for_path, key=lambda c: c.get('created_at') or '', reverse=True)
                        best = prior_for_path_sorted[0]
                    except Exception:
                        best = prior_for_path[0]

                target_comment_id = best.get('id') if best else None
                if not target_comment_id:
                    continue
                # Post the follow-up reply to the matched previous comment
                self.github_client.reply_to_comment(pr_details, target_comment_id, body)
                # Track unresolved only if the follow-up indicates not resolved
                try:
                    txt = (body or '').lower()
                    if 'not resolved' in txt or 'unresolved' in txt:
                        self._unresolved_prior_ids.add(int(target_comment_id))
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Failed to post follow-up replies for {diff_file.file_info.path}: {e}")
            return
    
    async def _resolve_completed_comments(self, pr_details: PRDetails) -> None:
        """Post a short resolution reply on previous bot comments that appear resolved.
        
        Rules to minimize false positives:
        - Only runs in FOLLOW-UP reviews.
        - Consider only previous bot comments on files we reviewed in this run.
        - If we posted a follow-up reply indicating Not Resolved for a prior comment, skip it.
        - Otherwise, post a single standardized resolution reply on that prior comment.
        - reply_to_comment is duplicate-safe and will not re-post the same message twice.
        """
        try:
            if not self._is_followup_review:
                return
            prior = getattr(self, '_previous_bot_comments', None) or []
            if not prior:
                return
            reviewed_paths = getattr(self, '_current_review_file_paths', None) or set()
            unresolved_ids = getattr(self, '_unresolved_prior_ids', None) or set()

            # Standard resolution message
            resolution_msg = "✅ This has been fixed thank you"

            resolved_count = 0
            considered = 0
            for c in prior:
                try:
                    cid = c.get('id')
                    path = c.get('path')
                    if not cid or not path:
                        continue
                    # Only consider comments on files we actually reviewed this run
                    if reviewed_paths and path not in reviewed_paths:
                        continue
                    considered += 1
                    # Skip if we determined this comment is still unresolved
                    if cid in unresolved_ids or int(cid) in unresolved_ids:
                        continue
                    # Post the resolution reply (duplicate-safe)
                    ok = self.github_client.reply_to_comment(pr_details, int(cid), resolution_msg)
                    if ok:
                        resolved_count += 1
                except Exception:
                    continue
            if considered > 0:
                logger.info(f"Posted resolution replies on {resolved_count}/{considered} previous comment(s) for reviewed files.")
        except Exception as e:
            logger.debug(f"Resolution reply phase failed: {e}")
            return

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
            filtered_comments = self.comment_processor.filter_comments_by_priority(comments)

            # Apply per-file and total caps to reduce noise
            limited_comments = self.comment_processor.apply_comment_limits(filtered_comments)
            
            # If there are no issues, we'll still create a positive review to acknowledge clean changes
            if total_comments == 0 and not limited_comments:
                logger.info("No issues found - posting a positive review to acknowledge that everything looks good.")
            
            # Determine review event
            # Note: Using COMMENT instead of APPROVE because GitHub Actions tokens
            # are not permitted to approve pull requests (GitHub API restriction)
            if preferred_event:
                event = preferred_event
                if limited_comments:
                    logger.info(f"Using preferred event '{event}' for {len(limited_comments)} comments")
                else:
                    logger.info(f"Using preferred event '{event}' with no comments")
            else:
                if not limited_comments:
                    if total_comments > 0:
                        logger.info(f"All {total_comments} comments were filtered by threshold/limits - posting comment about this")
                        event = "COMMENT"
                    else:
                        logger.info("No comments generated - posting positive review")
                        event = "COMMENT"
                else:
                    logger.info(f"Found {len(limited_comments)} comments (out of {total_comments} after dedupe) - requesting changes")
                    event = "REQUEST_CHANGES"
            
            # Pass both total and filtered comments so summary can be accurate
            success = self.github_client.create_review(
                pr_details, 
                limited_comments, 
                event,
                total_comments_generated=total_comments
            )
            if success:
                if limited_comments:
                    logger.info("✅ Successfully created GitHub review")
                else:
                    logger.info("✅ Successfully posted review")
            
            return success
            
        except GitHubClientError as e:
            logger.error(f"Failed to create GitHub review: {str(e)}")
            return False

    def _build_change_summary(self, diff_files: List[DiffFile]) -> str:
        """Build a summary of all changes in the PR for cross-file awareness.

        Args:
            diff_files: List of all diff files in the PR

        Returns:
            Formatted summary of changes
        """
        summary_parts = ["## All Files Changed in This PR:\n"]

        for diff_file in diff_files:
            file_path = diff_file.file_info.path
            additions = diff_file.file_info.additions
            deletions = diff_file.file_info.deletions
            change_type = diff_file.file_info.change_type or "modified"

            # Extract function/class changes from hunks
            changed_symbols = []
            for hunk in diff_file.hunks:
                # Look for function/class definitions in added lines
                for line in hunk.lines:
                    if line.startswith('+'):
                        clean_line = line[1:].strip()
                        # Python function/class
                        if clean_line.startswith('def ') or clean_line.startswith('async def '):
                            func_match = clean_line.split('(')[0].replace('def ', '').replace('async ', '').strip()
                            if func_match:
                                changed_symbols.append(f"def {func_match}")
                        elif clean_line.startswith('class '):
                            class_match = clean_line.split('(')[0].split(':')[0].replace('class ', '').strip()
                            if class_match:
                                changed_symbols.append(f"class {class_match}")
                        # JavaScript/TypeScript function
                        elif 'function ' in clean_line:
                            import re
                            match = re.search(r'function\s+(\w+)', clean_line)
                            if match:
                                changed_symbols.append(f"function {match.group(1)}")

            symbols_str = ""
            if changed_symbols:
                symbols_str = f" (defines: {', '.join(changed_symbols[:5])})"

            summary_parts.append(
                f"- {file_path} [{change_type}] (+{additions}/-{deletions}){symbols_str}"
            )

        return "\n".join(summary_parts[:30])  # Limit to 30 files

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
