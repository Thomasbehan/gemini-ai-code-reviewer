"""
Comment processor for the Gemini AI Code Reviewer.

This module is responsible for processing AI-generated review comments,
including conversion, filtering, and applying limits.
"""

import logging
from typing import List, Dict, Optional

from .models import ReviewComment, DiffFile, HunkInfo, ReviewPriority
from .config import ReviewConfig

logger = logging.getLogger(__name__)


class CommentProcessor:
    """Processes and filters review comments."""
    
    def __init__(self, review_config: ReviewConfig, github_client=None):
        """Initialize comment processor with configuration.
        
        Args:
            review_config: Review configuration
            github_client: GitHub client (optional, for signature computation)
        """
        self.review_config = review_config
        self.github_client = github_client
    
    def convert_to_review_comment(
        self,
        ai_response,
        diff_file: DiffFile,
        hunk: HunkInfo,
        hunk_index: int
    ) -> Optional[ReviewComment]:
        """Convert AI response to GitHub review comment with basic anchoring validation.
        
        Tries to ensure the comment targets the correct line by matching an anchor snippet
        to the diff lines. If mismatch, attempts to realign; otherwise discards to avoid noise.
        
        Args:
            ai_response: AI response containing line number and comment
            diff_file: The diff file being analyzed
            hunk: The hunk within the file
            hunk_index: Index of the hunk
            
        Returns:
            ReviewComment if successful, None otherwise
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

            # Format the body with fix code if available
            body = ai_response.review_comment
            fix_code = getattr(ai_response, 'fix_code', None)
            
            if fix_code:
                # simple language detection from extension
                lang = diff_file.file_info.file_extension
                if lang and lang.startswith('.'):
                    lang = lang[1:]
                if not lang:
                    lang = ""
                
                body += f"\n\n```{lang}\n{fix_code}\n```"

            # Compute global diff position (across all hunks in this file)
            # GitHub expects 'position' to be the index within the file's entire patch
            try:
                prior_hunks_total = 0
                if hasattr(diff_file, 'hunks') and isinstance(diff_file.hunks, list):
                    for idx, h in enumerate(diff_file.hunks):
                        if idx < hunk_index and hasattr(h, 'lines'):
                            # Each previous hunk contributes its header line ('@@ ... @@') plus its lines
                            prior_hunks_total += (1 + len(h.lines))
                # Add 1 for the current hunk header, then add the 0-based line offset within the hunk
                # (position is 1-based, so subtract 1 to get the correct offset)
                file_patch_position = prior_hunks_total + 1 + (position - 1)
            except Exception:
                # Fallback to hunk-local position if anything goes wrong
                file_patch_position = position

            # Compute best-effort target file line number on the "new" side
            # by walking the hunk lines from its header target_start
            file_line_number = None
            try:
                cur_target_line = int(getattr(hunk, 'target_start', 0))
                if cur_target_line <= 0:
                    cur_target_line = None
                if cur_target_line is not None:
                    for idx, raw in enumerate(hunk.lines, start=1):
                        marker = raw[:1]
                        if marker == '+':
                            # this line exists in the new file; count it
                            if idx == position:
                                file_line_number = cur_target_line
                                break
                            cur_target_line += 1
                        elif marker == ' ':
                            # context line exists in both; count it on target
                            if idx == position:
                                file_line_number = cur_target_line
                                break
                            cur_target_line += 1
                        else:
                            # '-' removed from source; does not advance target line
                            if idx == position and file_line_number is None:
                                # If we happen to land on a deletion, use current target line as closest anchor
                                file_line_number = cur_target_line
                                break
                # Fallback to the requested hunk position if we couldn't derive a target line
                if file_line_number is None:
                    file_line_number = position
            except Exception:
                file_line_number = position

            comment = ReviewComment(
                body=body,
                path=diff_file.file_info.path,
                position=file_patch_position,
                line_number=file_line_number,
                priority=ai_response.priority,
                category=ai_response.category,
                suggestion=getattr(ai_response, 'fix_code', None)
            )
            return comment

        except Exception as e:
            logger.warning(f"Error converting AI response to comment: {str(e)}")
            return None
    
    def filter_comments_by_priority(self, comments: List[ReviewComment]) -> List[ReviewComment]:
        """Filter comments based on priority threshold.
        
        Args:
            comments: List of review comments
            
        Returns:
            Filtered list of comments meeting priority threshold
        """
        if not comments:
            return []
        
        priority_order = {
            ReviewPriority.CRITICAL: 4,
            ReviewPriority.HIGH: 3,
            ReviewPriority.MEDIUM: 2,
            ReviewPriority.LOW: 1
        }
        
        threshold_value = priority_order.get(self.review_config.priority_threshold, 1)
        
        filtered_comments = []
        for comment in comments:
            comment_value = priority_order.get(comment.priority, 1)
            if comment_value >= threshold_value:
                filtered_comments.append(comment)
        
        if len(filtered_comments) != len(comments):
            logger.info(f"Filtered {len(comments)} comments to {len(filtered_comments)} "
                       f"based on priority threshold ({self.review_config.priority_threshold.value})")
        
        return filtered_comments
    
    def apply_comment_limits(self, comments: List[ReviewComment]) -> List[ReviewComment]:
        """Apply per-file and total comment caps to reduce noise.
        
        Keeps highest-priority items first and preserves stable ordering among equals.
        
        Args:
            comments: List of review comments
            
        Returns:
            Limited list of comments after applying caps
        """
        if not comments:
            return []
        
        # Map priority to numeric for sorting
        priority_order = {
            ReviewPriority.CRITICAL: 4,
            ReviewPriority.HIGH: 3,
            ReviewPriority.MEDIUM: 2,
            ReviewPriority.LOW: 1
        }
        
        per_file_cap = max(0, int(getattr(self.review_config, 'max_comments_per_file', 0)))
        total_cap = max(0, int(getattr(self.review_config, 'max_comments_total', 0)))
        
        # If no caps configured, return as-is
        if per_file_cap == 0 and total_cap == 0:
            return comments
        
        # Group by file
        by_file: Dict[str, List[ReviewComment]] = {}
        for cm in comments:
            by_file.setdefault(cm.path, []).append(cm)
        
        # For determinism, within each file, sort by priority (desc) but keep stable original order for ties
        def sort_key(cm: ReviewComment):
            return (-priority_order.get(cm.priority, 1))
        
        selected: List[ReviewComment] = []
        dropped_due_to_file_cap = 0
        for path, group in by_file.items():
            if per_file_cap > 0 and len(group) > per_file_cap:
                sorted_group = sorted(group, key=sort_key)
                kept = sorted_group[:per_file_cap]
                dropped_due_to_file_cap += len(group) - len(kept)
                selected.extend(kept)
            else:
                selected.extend(sorted(group, key=sort_key))
        
        if dropped_due_to_file_cap:
            logger.info(f"Applied per-file cap: dropped {dropped_due_to_file_cap} comments exceeding {per_file_cap}/file")
        
        # Apply total cap across all files
        if total_cap > 0 and len(selected) > total_cap:
            # Sort globally by priority, stable among equals by original relative order (already grouped)
            selected_sorted = sorted(selected, key=sort_key)
            limited = selected_sorted[:total_cap]
            logger.info(f"Applied total cap: reduced {len(selected)} to {len(limited)} comments (cap={total_cap})")
            return limited
        
        return selected
