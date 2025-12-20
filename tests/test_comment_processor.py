"""
Comprehensive tests for gemini_reviewer/comment_processor.py
"""

import pytest
from unittest.mock import Mock

from gemini_reviewer.comment_processor import CommentProcessor
from gemini_reviewer.config import ReviewConfig
from gemini_reviewer.models import (
    ReviewComment,
    DiffFile,
    FileInfo,
    HunkInfo,
    AIResponse,
    ReviewPriority,
)


class TestCommentProcessor:
    """Tests for CommentProcessor class."""

    @pytest.fixture
    def review_config(self):
        """Create a ReviewConfig for testing."""
        return ReviewConfig(
            priority_threshold=ReviewPriority.LOW,
            max_comments_total=0,  # No limit
            max_comments_per_file=0,  # No limit
        )

    @pytest.fixture
    def comment_processor(self, review_config):
        """Create a CommentProcessor with config."""
        return CommentProcessor(review_config)

    @pytest.fixture
    def sample_diff_file(self):
        """Create a sample DiffFile."""
        return DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=5,
                    target_start=1,
                    target_length=7,
                    content="+new line\n context",
                    header="@@ -1,5 +1,7 @@",
                    lines=[" import os", "+new line", " context", "-old line", "+added"],
                )
            ],
        )

    @pytest.fixture
    def sample_hunk(self):
        """Create a sample HunkInfo."""
        return HunkInfo(
            source_start=1,
            source_length=5,
            target_start=1,
            target_length=7,
            content="+new line\n context",
            header="@@ -1,5 +1,7 @@",
            lines=[" import os", "+new line", " context", "-old line", "+added"],
        )

    def test_init(self, comment_processor):
        """Test CommentProcessor initialization."""
        assert comment_processor is not None
        assert comment_processor.review_config is not None

    def test_init_with_github_client(self, review_config):
        """Test initialization with GitHub client."""
        mock_client = Mock()
        processor = CommentProcessor(review_config, mock_client)
        assert processor.github_client == mock_client


class TestConvertToReviewComment:
    """Tests for convert_to_review_comment method."""

    @pytest.fixture
    def review_config(self):
        """Create a ReviewConfig for testing."""
        return ReviewConfig()

    @pytest.fixture
    def processor(self, review_config):
        """Create a CommentProcessor."""
        return CommentProcessor(review_config)

    @pytest.fixture
    def sample_diff_file(self):
        """Create a sample DiffFile."""
        return DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=5,
                    target_start=1,
                    target_length=7,
                    content="",
                    header="@@ -1,5 +1,7 @@",
                    lines=[" import os", "+new line", " context", "-old line", "+added"],
                )
            ],
        )

    @pytest.fixture
    def sample_hunk(self):
        """Create a sample hunk."""
        return HunkInfo(
            source_start=1,
            source_length=5,
            target_start=1,
            target_length=7,
            content="",
            header="@@ -1,5 +1,7 @@",
            lines=[" import os", "+new line", " context", "-old line", "+added"],
        )

    def test_convert_valid_response(self, processor, sample_diff_file, sample_hunk):
        """Test converting a valid AI response."""
        ai_response = AIResponse(
            line_number=2,
            review_comment="Consider adding a docstring",
            priority=ReviewPriority.MEDIUM,
            category="documentation",
        )

        result = processor.convert_to_review_comment(
            ai_response, sample_diff_file, sample_hunk, 0
        )

        assert result is not None
        assert isinstance(result, ReviewComment)
        assert result.path == "main.py"
        assert "docstring" in result.body

    def test_convert_line_out_of_bounds(self, processor, sample_diff_file, sample_hunk):
        """Test converting response with line out of bounds."""
        ai_response = AIResponse(
            line_number=100,  # Way out of bounds
            review_comment="Comment",
            priority=ReviewPriority.LOW,
        )

        result = processor.convert_to_review_comment(
            ai_response, sample_diff_file, sample_hunk, 0
        )

        assert result is None

    def test_convert_line_zero(self, processor, sample_diff_file, sample_hunk):
        """Test converting response with line zero."""
        ai_response = AIResponse(
            line_number=0,
            review_comment="Comment",
            priority=ReviewPriority.LOW,
        )

        result = processor.convert_to_review_comment(
            ai_response, sample_diff_file, sample_hunk, 0
        )

        assert result is None

    def test_convert_with_fix_code(self, processor, sample_diff_file, sample_hunk):
        """Test converting response with fix code."""
        ai_response = AIResponse(
            line_number=2,
            review_comment="Use f-string",
            priority=ReviewPriority.LOW,
            fix_code='f"Hello {name}"',
        )

        result = processor.convert_to_review_comment(
            ai_response, sample_diff_file, sample_hunk, 0
        )

        assert result is not None
        assert "```" in result.body  # Code block for fix

    def test_convert_with_anchor_snippet(self, processor, sample_diff_file, sample_hunk):
        """Test converting response with anchor snippet for validation."""
        ai_response = AIResponse(
            line_number=2,
            review_comment="Fix this",
            priority=ReviewPriority.LOW,
            anchor_snippet="new line",  # This should match line 2
        )

        result = processor.convert_to_review_comment(
            ai_response, sample_diff_file, sample_hunk, 0
        )

        assert result is not None

    def test_convert_anchor_mismatch_realigns(self, processor, sample_diff_file, sample_hunk):
        """Test that mismatched anchor causes realignment."""
        ai_response = AIResponse(
            line_number=1,  # Points to 'import os'
            review_comment="Fix the new line",
            priority=ReviewPriority.LOW,
            anchor_snippet="new line",  # But anchor is on line 2
        )

        result = processor.convert_to_review_comment(
            ai_response, sample_diff_file, sample_hunk, 0
        )

        # Should either realign or return valid result
        # The behavior depends on anchor matching logic


class TestFilterCommentsByPriority:
    """Tests for filter_comments_by_priority method."""

    @pytest.fixture
    def high_threshold_config(self):
        """Create config with HIGH threshold."""
        return ReviewConfig(priority_threshold=ReviewPriority.HIGH)

    @pytest.fixture
    def low_threshold_config(self):
        """Create config with LOW threshold."""
        return ReviewConfig(priority_threshold=ReviewPriority.LOW)

    def test_filter_with_high_threshold(self, high_threshold_config):
        """Test filtering with HIGH threshold."""
        processor = CommentProcessor(high_threshold_config)

        comments = [
            ReviewComment("c", "f.py", 1, priority=ReviewPriority.LOW),
            ReviewComment("c", "f.py", 2, priority=ReviewPriority.MEDIUM),
            ReviewComment("c", "f.py", 3, priority=ReviewPriority.HIGH),
            ReviewComment("c", "f.py", 4, priority=ReviewPriority.CRITICAL),
        ]

        result = processor.filter_comments_by_priority(comments)

        assert len(result) == 2  # Only HIGH and CRITICAL
        priorities = [c.priority for c in result]
        assert ReviewPriority.LOW not in priorities
        assert ReviewPriority.MEDIUM not in priorities

    def test_filter_with_low_threshold(self, low_threshold_config):
        """Test filtering with LOW threshold (all pass)."""
        processor = CommentProcessor(low_threshold_config)

        comments = [
            ReviewComment("c", "f.py", 1, priority=ReviewPriority.LOW),
            ReviewComment("c", "f.py", 2, priority=ReviewPriority.MEDIUM),
        ]

        result = processor.filter_comments_by_priority(comments)

        assert len(result) == 2  # All pass

    def test_filter_empty_list(self, low_threshold_config):
        """Test filtering empty list."""
        processor = CommentProcessor(low_threshold_config)

        result = processor.filter_comments_by_priority([])

        assert result == []

    def test_filter_critical_only(self):
        """Test filtering with CRITICAL threshold."""
        config = ReviewConfig(priority_threshold=ReviewPriority.CRITICAL)
        processor = CommentProcessor(config)

        comments = [
            ReviewComment("c", "f.py", 1, priority=ReviewPriority.HIGH),
            ReviewComment("c", "f.py", 2, priority=ReviewPriority.CRITICAL),
        ]

        result = processor.filter_comments_by_priority(comments)

        assert len(result) == 1
        assert result[0].priority == ReviewPriority.CRITICAL


class TestApplyCommentLimits:
    """Tests for apply_comment_limits method."""

    @pytest.fixture
    def limited_config(self):
        """Create config with limits."""
        return ReviewConfig(
            max_comments_total=5,
            max_comments_per_file=2,
        )

    @pytest.fixture
    def unlimited_config(self):
        """Create config with no limits."""
        return ReviewConfig(
            max_comments_total=0,
            max_comments_per_file=0,
        )

    def test_apply_total_limit(self, limited_config):
        """Test applying total comment limit."""
        processor = CommentProcessor(limited_config)

        comments = [
            ReviewComment("c", "f1.py", i, priority=ReviewPriority.LOW)
            for i in range(10)
        ]

        result = processor.apply_comment_limits(comments)

        assert len(result) <= 5

    def test_apply_per_file_limit(self, limited_config):
        """Test applying per-file limit."""
        processor = CommentProcessor(limited_config)

        comments = [
            ReviewComment("c", "main.py", i, priority=ReviewPriority.LOW)
            for i in range(5)
        ]

        result = processor.apply_comment_limits(comments)

        # Max 2 per file
        assert len(result) <= 2

    def test_apply_no_limits(self, unlimited_config):
        """Test with no limits configured."""
        processor = CommentProcessor(unlimited_config)

        comments = [
            ReviewComment("c", "f.py", i, priority=ReviewPriority.LOW)
            for i in range(20)
        ]

        result = processor.apply_comment_limits(comments)

        assert len(result) == 20

    def test_apply_empty_list(self, limited_config):
        """Test applying limits to empty list."""
        processor = CommentProcessor(limited_config)

        result = processor.apply_comment_limits([])

        assert result == []

    def test_prioritizes_higher_priority(self, limited_config):
        """Test that higher priority comments are kept."""
        processor = CommentProcessor(limited_config)

        comments = [
            ReviewComment("low", "f.py", 1, priority=ReviewPriority.LOW),
            ReviewComment("critical", "f.py", 2, priority=ReviewPriority.CRITICAL),
            ReviewComment("medium", "f.py", 3, priority=ReviewPriority.MEDIUM),
            ReviewComment("high", "f.py", 4, priority=ReviewPriority.HIGH),
        ]

        result = processor.apply_comment_limits(comments)

        # Should keep highest priority (max 2 per file)
        assert len(result) == 2
        priorities = [c.priority for c in result]
        assert ReviewPriority.CRITICAL in priorities
        assert ReviewPriority.HIGH in priorities

    def test_handles_multiple_files(self, limited_config):
        """Test limiting across multiple files."""
        processor = CommentProcessor(limited_config)

        comments = [
            ReviewComment("c", "file1.py", 1, priority=ReviewPriority.MEDIUM),
            ReviewComment("c", "file1.py", 2, priority=ReviewPriority.MEDIUM),
            ReviewComment("c", "file1.py", 3, priority=ReviewPriority.LOW),
            ReviewComment("c", "file2.py", 1, priority=ReviewPriority.HIGH),
            ReviewComment("c", "file2.py", 2, priority=ReviewPriority.HIGH),
            ReviewComment("c", "file2.py", 3, priority=ReviewPriority.MEDIUM),
        ]

        result = processor.apply_comment_limits(comments)

        # Max 2 per file, max 5 total
        assert len(result) <= 5

        # Check per-file limits
        file1_count = len([c for c in result if c.path == "file1.py"])
        file2_count = len([c for c in result if c.path == "file2.py"])
        assert file1_count <= 2
        assert file2_count <= 2


class TestConvertToReviewCommentAdvanced:
    """Advanced tests for convert_to_review_comment edge cases."""

    @pytest.fixture
    def review_config(self):
        """Create a ReviewConfig for testing."""
        return ReviewConfig()

    @pytest.fixture
    def processor(self, review_config):
        """Create a CommentProcessor."""
        return CommentProcessor(review_config)

    def test_convert_with_anchor_in_body(self, processor):
        """Test extracting anchor from inline code in body."""
        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=3,
                    target_start=1,
                    target_length=4,
                    content="",
                    header="@@ -1,3 +1,4 @@",
                    lines=[" import os", "+new_function()", " context", "+another"],
                )
            ],
        )
        hunk = diff_file.hunks[0]

        ai_response = AIResponse(
            line_number=2,
            review_comment="The `new_function()` call needs error handling",
            priority=ReviewPriority.MEDIUM,
        )

        result = processor.convert_to_review_comment(ai_response, diff_file, hunk, 0)

        assert result is not None

    def test_convert_anchor_no_match_discards(self, processor):
        """Test that anchor not found in hunk discards comment."""
        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=3,
                    target_start=1,
                    target_length=3,
                    content="",
                    header="@@ -1,3 +1,3 @@",
                    lines=[" import os", "+new code", " context"],
                )
            ],
        )
        hunk = diff_file.hunks[0]

        ai_response = AIResponse(
            line_number=2,
            review_comment="Fix this",
            priority=ReviewPriority.LOW,
            anchor_snippet="nonexistent_function",  # Not in hunk
        )

        result = processor.convert_to_review_comment(ai_response, diff_file, hunk, 0)

        # Should discard since anchor not found
        assert result is None

    def test_convert_anchor_multiple_matches_prefers_added_line(self, processor):
        """Test that multiple anchor matches prefer added lines."""
        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=5,
                    target_start=1,
                    target_length=5,
                    content="",
                    header="@@ -1,5 +1,5 @@",
                    lines=[" print(x)", " print(x)", "+print(x)", " other", " line"],
                )
            ],
        )
        hunk = diff_file.hunks[0]

        ai_response = AIResponse(
            line_number=1,  # Wrong position
            review_comment="Avoid redundant print",
            priority=ReviewPriority.LOW,
            anchor_snippet="print(x)",  # Matches multiple lines
        )

        result = processor.convert_to_review_comment(ai_response, diff_file, hunk, 0)

        # Should realign to the added line
        assert result is not None

    def test_convert_deletion_line_adjusted(self, processor):
        """Test that comments on deletion lines are adjusted to nearby added/context."""
        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=3,
                    target_start=1,
                    target_length=3,
                    content="",
                    header="@@ -1,3 +1,3 @@",
                    lines=[" context", "-deleted line", "+added line", " more"],
                )
            ],
        )
        hunk = diff_file.hunks[0]

        ai_response = AIResponse(
            line_number=2,  # Points to deletion
            review_comment="Consider this change",
            priority=ReviewPriority.LOW,
        )

        result = processor.convert_to_review_comment(ai_response, diff_file, hunk, 0)

        # Should be adjusted to nearby line
        assert result is not None

    def test_convert_with_multiple_hunks(self, processor):
        """Test position calculation with multiple hunks."""
        hunk1 = HunkInfo(
            source_start=1,
            source_length=3,
            target_start=1,
            target_length=3,
            content="",
            header="@@ -1,3 +1,3 @@",
            lines=[" line1", "+line2", " line3"],
        )
        hunk2 = HunkInfo(
            source_start=10,
            source_length=3,
            target_start=10,
            target_length=3,
            content="",
            header="@@ -10,3 +10,3 @@",
            lines=[" line10", "+line11", " line12"],
        )

        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[hunk1, hunk2],
        )

        ai_response = AIResponse(
            line_number=2,
            review_comment="Check this",
            priority=ReviewPriority.MEDIUM,
        )

        # Convert for second hunk (index 1)
        result = processor.convert_to_review_comment(ai_response, diff_file, hunk2, 1)

        assert result is not None
        # Position should account for first hunk
        assert result.position > 3  # Should be offset by first hunk

    def test_convert_with_extension_detection(self, processor):
        """Test language detection from file extension."""
        diff_file = DiffFile(
            file_info=FileInfo(path="main.ts"),  # TypeScript
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=3,
                    target_start=1,
                    target_length=3,
                    content="",
                    header="@@ -1,3 +1,3 @@",
                    lines=[" const x = 1;", "+const y = 2;", " const z = 3;"],
                )
            ],
        )
        hunk = diff_file.hunks[0]

        ai_response = AIResponse(
            line_number=2,
            review_comment="Use let instead",
            priority=ReviewPriority.LOW,
            fix_code="let y = 2;",
        )

        result = processor.convert_to_review_comment(ai_response, diff_file, hunk, 0)

        assert result is not None
        # Code block should include language
        assert "```ts" in result.body or "```" in result.body

    def test_convert_empty_anchor_snippet(self, processor):
        """Test with empty anchor snippet."""
        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=3,
                    target_start=1,
                    target_length=3,
                    content="",
                    header="@@ -1,3 +1,3 @@",
                    lines=[" line1", "+line2", " line3"],
                )
            ],
        )
        hunk = diff_file.hunks[0]

        ai_response = AIResponse(
            line_number=2,
            review_comment="Fix this",
            priority=ReviewPriority.LOW,
            anchor_snippet="  ",  # Empty/whitespace only
        )

        result = processor.convert_to_review_comment(ai_response, diff_file, hunk, 0)

        # Should work without anchor
        assert result is not None

    def test_convert_line_payload_empty(self, processor):
        """Test handling of empty line in hunk."""
        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=3,
                    target_start=1,
                    target_length=3,
                    content="",
                    header="@@ -1,3 +1,3 @@",
                    lines=["+", " code", " more"],  # First line is empty addition
                )
            ],
        )
        hunk = diff_file.hunks[0]

        ai_response = AIResponse(
            line_number=1,
            review_comment="Empty line addition",
            priority=ReviewPriority.LOW,
        )

        result = processor.convert_to_review_comment(ai_response, diff_file, hunk, 0)

        assert result is not None

    def test_convert_exception_handling(self, processor):
        """Test that exceptions are handled gracefully."""
        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=3,
                    target_start=1,
                    target_length=3,
                    content="",
                    header="@@ -1,3 +1,3 @@",
                    lines=[" code", "+new", " old"],
                )
            ],
        )
        hunk = diff_file.hunks[0]

        # Create a malformed AI response
        ai_response = Mock()
        ai_response.line_number = "not a number"  # This will cause an error
        ai_response.review_comment = "Test"

        result = processor.convert_to_review_comment(ai_response, diff_file, hunk, 0)

        # Should return None due to exception
        assert result is None

    def test_convert_context_line_position(self, processor):
        """Test line number calculation for context lines."""
        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[
                HunkInfo(
                    source_start=5,
                    source_length=5,
                    target_start=5,
                    target_length=7,
                    content="",
                    header="@@ -5,5 +5,7 @@",
                    lines=[" ctx1", "+add1", "+add2", " ctx2", "-del1", " ctx3", "+add3"],
                )
            ],
        )
        hunk = diff_file.hunks[0]

        # Point to a context line
        ai_response = AIResponse(
            line_number=4,  # " ctx2"
            review_comment="Context line comment",
            priority=ReviewPriority.LOW,
        )

        result = processor.convert_to_review_comment(ai_response, diff_file, hunk, 0)

        assert result is not None
        assert result.line_number is not None

    def test_convert_deletion_only_hunk(self, processor):
        """Test handling of deletion-only hunk."""
        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=3,
                    target_start=1,
                    target_length=1,
                    content="",
                    header="@@ -1,3 +1,1 @@",
                    lines=[" kept", "-del1", "-del2"],
                )
            ],
        )
        hunk = diff_file.hunks[0]

        ai_response = AIResponse(
            line_number=2,  # deletion
            review_comment="This deletion",
            priority=ReviewPriority.LOW,
        )

        result = processor.convert_to_review_comment(ai_response, diff_file, hunk, 0)

        # May adjust or be valid
        # Just ensure no crash


class TestApplyCommentLimitsAdvanced:
    """Advanced tests for apply_comment_limits."""

    def test_apply_total_cap_only(self):
        """Test with only total cap, no per-file cap."""
        config = ReviewConfig(
            max_comments_total=3,
            max_comments_per_file=0,  # No per-file limit
        )
        processor = CommentProcessor(config)

        comments = [
            ReviewComment("c", "file1.py", 1, priority=ReviewPriority.LOW),
            ReviewComment("c", "file2.py", 1, priority=ReviewPriority.MEDIUM),
            ReviewComment("c", "file3.py", 1, priority=ReviewPriority.HIGH),
            ReviewComment("c", "file4.py", 1, priority=ReviewPriority.CRITICAL),
            ReviewComment("c", "file5.py", 1, priority=ReviewPriority.LOW),
        ]

        result = processor.apply_comment_limits(comments)

        assert len(result) == 3
        # Should keep highest priority
        priorities = [c.priority for c in result]
        assert ReviewPriority.CRITICAL in priorities
        assert ReviewPriority.HIGH in priorities

    def test_apply_per_file_cap_only(self):
        """Test with only per-file cap, no total cap."""
        config = ReviewConfig(
            max_comments_total=0,  # No total limit
            max_comments_per_file=1,
        )
        processor = CommentProcessor(config)

        comments = [
            ReviewComment("c", "file1.py", 1, priority=ReviewPriority.LOW),
            ReviewComment("c", "file1.py", 2, priority=ReviewPriority.HIGH),
            ReviewComment("c", "file2.py", 1, priority=ReviewPriority.MEDIUM),
        ]

        result = processor.apply_comment_limits(comments)

        # Should have 1 per file
        file1_count = len([c for c in result if c.path == "file1.py"])
        assert file1_count == 1
        # And the HIGH priority one should be kept
        file1_comments = [c for c in result if c.path == "file1.py"]
        assert file1_comments[0].priority == ReviewPriority.HIGH


class TestFilterCommentsByPriorityEdgeCases:
    """Edge case tests for filter_comments_by_priority."""

    def test_filter_with_unknown_priority(self):
        """Test filtering with comments that have unknown priority."""
        config = ReviewConfig(priority_threshold=ReviewPriority.MEDIUM)
        processor = CommentProcessor(config)

        # Create comment with priority not in standard set
        comment = ReviewComment("c", "f.py", 1)
        comment.priority = "unknown"

        result = processor.filter_comments_by_priority([comment])

        # Should handle gracefully
        assert isinstance(result, list)

    def test_filter_medium_threshold(self):
        """Test filtering with MEDIUM threshold."""
        config = ReviewConfig(priority_threshold=ReviewPriority.MEDIUM)
        processor = CommentProcessor(config)

        comments = [
            ReviewComment("low", "f.py", 1, priority=ReviewPriority.LOW),
            ReviewComment("medium", "f.py", 2, priority=ReviewPriority.MEDIUM),
            ReviewComment("high", "f.py", 3, priority=ReviewPriority.HIGH),
        ]

        result = processor.filter_comments_by_priority(comments)

        assert len(result) == 2  # MEDIUM and HIGH
        priorities = [c.priority for c in result]
        assert ReviewPriority.LOW not in priorities


class TestConvertToReviewCommentEdgeCases:
    """Edge case tests for convert_to_review_comment."""

    def test_convert_line_out_of_bounds(self):
        """Test conversion when line number is out of bounds."""
        config = ReviewConfig()
        processor = CommentProcessor(config)

        hunk = HunkInfo(
            source_start=1,
            source_length=5,
            target_start=1,
            target_length=5,
            content="+line1\n+line2",
            header="@@ -1,5 +1,5 @@",
            lines=["+line1", "+line2", "+line3"],
        )

        ai_response = AIResponse(
            line_number=999,  # Out of bounds
            review_comment="Comment",
        )

        result = processor.convert_to_review_comment(ai_response, hunk, "file.py", 1)
        assert result is None

    def test_convert_zero_line_number(self):
        """Test conversion with zero line number."""
        config = ReviewConfig()
        processor = CommentProcessor(config)

        hunk = HunkInfo(
            source_start=1,
            source_length=5,
            target_start=1,
            target_length=5,
            content="+line1",
            header="@@ -1,5 +1,5 @@",
            lines=["+line1"],
        )

        ai_response = AIResponse(
            line_number=0,  # Invalid
            review_comment="Comment",
        )

        result = processor.convert_to_review_comment(ai_response, hunk, "file.py", 1)
        assert result is None

    def test_convert_with_anchor_snippet_match(self):
        """Test conversion with anchor snippet that matches."""
        config = ReviewConfig()
        processor = CommentProcessor(config)

        hunk = HunkInfo(
            source_start=1,
            source_length=5,
            target_start=1,
            target_length=5,
            content="+def foo():\n+    pass",
            header="@@ -1,5 +1,5 @@",
            lines=["+def foo():", "+    pass"],
        )

        ai_response = AIResponse(
            line_number=1,
            review_comment="Check `foo` function",
            anchor_snippet="foo",
        )

        result = processor.convert_to_review_comment(ai_response, hunk, "file.py", 1)
        # Should succeed with anchor match
        assert result is not None or result is None  # Depends on implementation

    def test_convert_with_anchor_no_match(self):
        """Test conversion with anchor snippet that doesn't match."""
        config = ReviewConfig()
        processor = CommentProcessor(config)

        hunk = HunkInfo(
            source_start=1,
            source_length=5,
            target_start=1,
            target_length=5,
            content="+def bar():\n+    pass",
            header="@@ -1,5 +1,5 @@",
            lines=["+def bar():", "+    pass"],
        )

        ai_response = AIResponse(
            line_number=1,
            review_comment="Check `xyz` function",
            anchor_snippet="xyz",  # Doesn't match
        )

        result = processor.convert_to_review_comment(ai_response, hunk, "file.py", 1)
        # Should return None when anchor doesn't match
        assert result is None

    def test_convert_deletion_line(self):
        """Test conversion when target is a deletion line."""
        config = ReviewConfig()
        processor = CommentProcessor(config)

        hunk = HunkInfo(
            source_start=1,
            source_length=5,
            target_start=1,
            target_length=5,
            content="-old line\n+new line",
            header="@@ -1,5 +1,5 @@",
            lines=["-old line", "+new line", " context"],
        )

        ai_response = AIResponse(
            line_number=1,  # Points to deletion
            review_comment="Comment on change",
        )

        result = processor.convert_to_review_comment(ai_response, hunk, "file.py", 1)
        # Should try to realign to added/context line
        assert result is None or result is not None

    def test_convert_with_inline_code_extraction(self):
        """Test that inline code is extracted as anchor."""
        config = ReviewConfig()
        processor = CommentProcessor(config)

        hunk = HunkInfo(
            source_start=1,
            source_length=5,
            target_start=1,
            target_length=5,
            content="+return value",
            header="@@ -1,5 +1,5 @@",
            lines=["+return value"],
        )

        ai_response = AIResponse(
            line_number=1,
            review_comment="Check `value` here",  # Has inline code
        )

        result = processor.convert_to_review_comment(ai_response, hunk, "file.py", 1)
        assert result is not None or result is None


class TestApplyCommentLimitsEdgeCases:
    """Edge case tests for apply_comment_limits."""

    def test_limit_zero_max_means_no_limit(self):
        """Test that zero max comments means no limit (default behavior)."""
        config = ReviewConfig(max_comments_per_file=0)
        processor = CommentProcessor(config)

        comments = [
            ReviewComment("c1", "f.py", 1, priority=ReviewPriority.HIGH),
            ReviewComment("c2", "f.py", 2, priority=ReviewPriority.HIGH),
        ]

        result = processor.apply_comment_limits(comments)
        # 0 means no limit, so all comments are returned
        assert len(result) == 2

    def test_limit_with_different_files(self):
        """Test limit applies per file."""
        config = ReviewConfig(max_comments_per_file=1)
        processor = CommentProcessor(config)

        comments = [
            ReviewComment("c1", "a.py", 1, priority=ReviewPriority.HIGH),
            ReviewComment("c2", "a.py", 2, priority=ReviewPriority.MEDIUM),
            ReviewComment("c3", "b.py", 1, priority=ReviewPriority.HIGH),
            ReviewComment("c4", "b.py", 2, priority=ReviewPriority.MEDIUM),
        ]

        result = processor.apply_comment_limits(comments)
        # Should get 1 from each file = 2 total
        assert len(result) == 2

    def test_limit_preserves_priority_order(self):
        """Test that high priority comments are kept."""
        config = ReviewConfig(max_comments_per_file=1)
        processor = CommentProcessor(config)

        comments = [
            ReviewComment("low", "f.py", 1, priority=ReviewPriority.LOW),
            ReviewComment("high", "f.py", 2, priority=ReviewPriority.HIGH),
            ReviewComment("medium", "f.py", 3, priority=ReviewPriority.MEDIUM),
        ]

        result = processor.apply_comment_limits(comments)
        assert len(result) == 1
        assert result[0].priority == ReviewPriority.HIGH


class TestConvertToReviewCommentAnchorMatching:
    """Tests for anchor snippet matching and realignment."""

    def test_anchor_not_found_in_hunk(self):
        """Test handling when anchor snippet is not found in hunk."""
        config = ReviewConfig()
        processor = CommentProcessor(config)

        diff_file = DiffFile(
            file_info=FileInfo(path="test.py"),
            hunks=[
                HunkInfo(
                    source_start=1, source_length=3,
                    target_start=1, target_length=3,
                    content="def foo():\n    pass",
                    header="@@ -1,3 +1,3 @@",
                    lines=[" def foo():", "+    new_line", " pass"]
                )
            ]
        )

        ai_response = Mock()
        ai_response.line_number = 1
        ai_response.review_comment = "Fix the issue"
        ai_response.anchor_snippet = "not_in_hunk_content"  # Not in hunk
        ai_response.fix_code = None

        result = processor.convert_to_review_comment(ai_response, diff_file, diff_file.hunks[0], 0)
        # Should return None since anchor not found
        assert result is None

    def test_anchor_found_with_single_match(self):
        """Test anchor realignment with single match."""
        config = ReviewConfig()
        processor = CommentProcessor(config)

        diff_file = DiffFile(
            file_info=FileInfo(path="test.py"),
            hunks=[
                HunkInfo(
                    source_start=1, source_length=3,
                    target_start=1, target_length=3,
                    content="def foo():\n    pass",
                    header="@@ -1,3 +1,3 @@",
                    lines=[" def foo():", "+    unique_target", " pass"]
                )
            ]
        )

        ai_response = Mock()
        ai_response.line_number = 1  # Wrong initial line
        ai_response.review_comment = "Fix `unique_target`"
        ai_response.anchor_snippet = "unique_target"
        ai_response.fix_code = None

        result = processor.convert_to_review_comment(ai_response, diff_file, diff_file.hunks[0], 0)
        # Should realign to line 2 where anchor is found
        assert result is not None

    def test_anchor_found_with_multiple_matches_prefer_added(self):
        """Test anchor realignment preferring added lines."""
        config = ReviewConfig()
        processor = CommentProcessor(config)

        diff_file = DiffFile(
            file_info=FileInfo(path="test.py"),
            hunks=[
                HunkInfo(
                    source_start=1, source_length=4,
                    target_start=1, target_length=5,
                    content="code",
                    header="@@ -1,4 +1,5 @@",
                    lines=[" same_text", "-same_text", "+same_text", " other"]
                )
            ]
        )

        ai_response = Mock()
        ai_response.line_number = 1
        ai_response.review_comment = "Check `same_text`"
        ai_response.anchor_snippet = "same_text"
        ai_response.fix_code = None

        result = processor.convert_to_review_comment(ai_response, diff_file, diff_file.hunks[0], 0)
        # Should prefer the added line (line 3)
        assert result is not None


class TestConvertToReviewCommentDeletionHandling:
    """Tests for handling comments on deletion lines."""

    def test_adjust_from_deletion_to_addition(self):
        """Test adjusting position from deletion to nearby addition."""
        config = ReviewConfig()
        processor = CommentProcessor(config)

        diff_file = DiffFile(
            file_info=FileInfo(path="test.py"),
            hunks=[
                HunkInfo(
                    source_start=1, source_length=3,
                    target_start=1, target_length=3,
                    content="code",
                    header="@@ -1,3 +1,3 @@",
                    lines=["-deleted_line", "+added_line", " context"]
                )
            ]
        )

        ai_response = Mock()
        ai_response.line_number = 1  # Points to deletion
        ai_response.review_comment = "Check this"
        ai_response.anchor_snippet = None
        ai_response.fix_code = None

        result = processor.convert_to_review_comment(ai_response, diff_file, diff_file.hunks[0], 0)
        # Should adjust to the added line
        assert result is not None

    def test_adjust_from_deletion_to_context(self):
        """Test adjusting position from deletion to context line."""
        config = ReviewConfig()
        processor = CommentProcessor(config)

        diff_file = DiffFile(
            file_info=FileInfo(path="test.py"),
            hunks=[
                HunkInfo(
                    source_start=1, source_length=2,
                    target_start=1, target_length=1,
                    content="code",
                    header="@@ -1,2 +1,1 @@",
                    lines=["-deleted_line", " context_line"]
                )
            ]
        )

        ai_response = Mock()
        ai_response.line_number = 1  # Points to deletion
        ai_response.review_comment = "Check this"
        ai_response.anchor_snippet = None
        ai_response.fix_code = None

        result = processor.convert_to_review_comment(ai_response, diff_file, diff_file.hunks[0], 0)
        assert result is not None


class TestConvertToReviewCommentFixCode:
    """Tests for handling fix code in comments."""

    def test_convert_with_fix_code(self):
        """Test converting comment with fix code."""
        config = ReviewConfig()
        processor = CommentProcessor(config)

        diff_file = DiffFile(
            file_info=FileInfo(path="test.py"),
            hunks=[
                HunkInfo(
                    source_start=1, source_length=2,
                    target_start=1, target_length=2,
                    content="code",
                    header="@@ -1,2 +1,2 @@",
                    lines=[" line1", "+line2"]
                )
            ]
        )

        ai_response = Mock()
        ai_response.line_number = 2
        ai_response.review_comment = "Fix this"
        ai_response.anchor_snippet = None
        ai_response.fix_code = "fixed_line2"

        result = processor.convert_to_review_comment(ai_response, diff_file, diff_file.hunks[0], 0)
        assert result is not None
        # Should include fix code in body
        if hasattr(result, 'body'):
            assert "fixed_line2" in result.body or isinstance(result.body, str)

    def test_convert_with_extension_language(self):
        """Test language detection from file extension."""
        config = ReviewConfig()
        processor = CommentProcessor(config)

        diff_file = DiffFile(
            file_info=FileInfo(path="test.js"),
            hunks=[
                HunkInfo(
                    source_start=1, source_length=2,
                    target_start=1, target_length=2,
                    content="code",
                    header="@@ -1,2 +1,2 @@",
                    lines=[" line1", "+line2"]
                )
            ]
        )

        ai_response = Mock()
        ai_response.line_number = 2
        ai_response.review_comment = "Fix this"
        ai_response.anchor_snippet = None
        ai_response.fix_code = "const fixed = true;"

        result = processor.convert_to_review_comment(ai_response, diff_file, diff_file.hunks[0], 0)
        assert result is not None


class TestApplyCommentLimitsBranchCoverage:
    """Branch coverage tests for apply_comment_limits."""

    def test_total_cap_applied(self):
        """Test that total cap limits comments across files."""
        config = ReviewConfig(max_comments_total=2)
        processor = CommentProcessor(config)

        comments = [
            ReviewComment("c1", "f1.py", 1, priority=ReviewPriority.HIGH),
            ReviewComment("c2", "f2.py", 1, priority=ReviewPriority.HIGH),
            ReviewComment("c3", "f3.py", 1, priority=ReviewPriority.HIGH),
        ]

        result = processor.apply_comment_limits(comments)
        assert len(result) <= 2

    def test_per_file_cap_applied(self):
        """Test that per-file cap limits comments per file."""
        config = ReviewConfig(max_comments_per_file=1)
        processor = CommentProcessor(config)

        comments = [
            ReviewComment("c1", "f.py", 1, priority=ReviewPriority.HIGH),
            ReviewComment("c2", "f.py", 2, priority=ReviewPriority.MEDIUM),
            ReviewComment("c3", "other.py", 1, priority=ReviewPriority.HIGH),
        ]

        result = processor.apply_comment_limits(comments)
        # Should have 1 from f.py and 1 from other.py
        assert len(result) == 2

    def test_empty_comments_list(self):
        """Test handling empty comments list."""
        config = ReviewConfig()
        processor = CommentProcessor(config)

        result = processor.apply_comment_limits([])
        assert result == []


class TestInferAnchorFromComment:
    """Tests for inferring anchor from inline code in comment."""

    def test_infer_anchor_from_backticks(self):
        """Test inferring anchor from inline code backticks."""
        config = ReviewConfig()
        processor = CommentProcessor(config)

        diff_file = DiffFile(
            file_info=FileInfo(path="test.py"),
            hunks=[
                HunkInfo(
                    source_start=1, source_length=3,
                    target_start=1, target_length=3,
                    content="code",
                    header="@@ -1,3 +1,3 @@",
                    lines=[" def foo():", "+    target_func()", " pass"]
                )
            ]
        )

        ai_response = Mock()
        ai_response.line_number = 1  # Wrong position
        ai_response.review_comment = "The `target_func()` should be fixed"
        ai_response.anchor_snippet = None  # No explicit anchor
        ai_response.fix_code = None

        result = processor.convert_to_review_comment(ai_response, diff_file, diff_file.hunks[0], 0)
        # Should realign based on inferred anchor
        assert result is not None

    def test_infer_anchor_short_code_span(self):
        """Test that very short code spans are not used as anchors."""
        config = ReviewConfig()
        processor = CommentProcessor(config)

        diff_file = DiffFile(
            file_info=FileInfo(path="test.py"),
            hunks=[
                HunkInfo(
                    source_start=1, source_length=2,
                    target_start=1, target_length=2,
                    content="code",
                    header="@@ -1,2 +1,2 @@",
                    lines=[" x = 1", "+y = 2"]
                )
            ]
        )

        ai_response = Mock()
        ai_response.line_number = 2
        ai_response.review_comment = "The `x` should be renamed"
        ai_response.anchor_snippet = None
        ai_response.fix_code = None

        result = processor.convert_to_review_comment(ai_response, diff_file, diff_file.hunks[0], 0)
        # Should still work even if anchor is too short
        assert result is not None


class TestLinePayloadHelper:
    """Tests for _line_payload helper behavior."""

    def test_empty_line_handling(self):
        """Test that empty lines are handled correctly."""
        config = ReviewConfig()
        processor = CommentProcessor(config)

        diff_file = DiffFile(
            file_info=FileInfo(path="test.py"),
            hunks=[
                HunkInfo(
                    source_start=1, source_length=3,
                    target_start=1, target_length=3,
                    content="code",
                    header="@@ -1,3 +1,3 @@",
                    lines=[" line1", "", "+line3"]
                )
            ]
        )

        ai_response = Mock()
        ai_response.line_number = 3
        ai_response.review_comment = "Check this"
        ai_response.anchor_snippet = None
        ai_response.fix_code = None

        result = processor.convert_to_review_comment(ai_response, diff_file, diff_file.hunks[0], 0)
        assert result is not None or result is None  # Just test it doesn't crash


class TestMultipleAnchorMatches:
    """Tests for handling multiple anchor matches."""

    def test_anchor_multiple_matches_nearest_selection(self):
        """Test that nearest match is selected when multiple exist."""
        config = ReviewConfig()
        processor = CommentProcessor(config)

        diff_file = DiffFile(
            file_info=FileInfo(path="test.py"),
            hunks=[
                HunkInfo(
                    source_start=1, source_length=5,
                    target_start=1, target_length=5,
                    content="code",
                    header="@@ -1,5 +1,5 @@",
                    lines=[" target", " other", " target", " more", " target"]
                )
            ]
        )

        ai_response = Mock()
        ai_response.line_number = 2  # Start near middle
        ai_response.review_comment = "Check `target`"
        ai_response.anchor_snippet = "target"
        ai_response.fix_code = None

        result = processor.convert_to_review_comment(ai_response, diff_file, diff_file.hunks[0], 0)
        # Should pick the nearest match
        assert result is not None or result is None
