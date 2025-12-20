"""
Comprehensive tests for gemini_reviewer/models.py
"""

import pytest
from dataclasses import fields

from gemini_reviewer.models import (
    ReviewPriority,
    ReviewFocus,
    PRDetails,
    FileInfo,
    HunkInfo,
    DiffFile,
    ReviewComment,
    AIResponse,
    ReviewResult,
    AnalysisContext,
    ProcessingStats,
)


class TestReviewPriority:
    """Tests for ReviewPriority enum."""

    def test_values(self):
        """Test all priority values."""
        assert ReviewPriority.LOW.value == "low"
        assert ReviewPriority.MEDIUM.value == "medium"
        assert ReviewPriority.HIGH.value == "high"
        assert ReviewPriority.CRITICAL.value == "critical"

    def test_from_value(self):
        """Test creating from string value."""
        assert ReviewPriority("low") == ReviewPriority.LOW
        assert ReviewPriority("high") == ReviewPriority.HIGH


class TestReviewFocus:
    """Tests for ReviewFocus enum."""

    def test_values(self):
        """Test all focus area values."""
        assert ReviewFocus.BUGS.value == "bugs"
        assert ReviewFocus.SECURITY.value == "security"
        assert ReviewFocus.PERFORMANCE.value == "performance"
        assert ReviewFocus.MAINTAINABILITY.value == "maintainability"
        assert ReviewFocus.STYLE.value == "style"
        assert ReviewFocus.ALL.value == "all"


class TestPRDetails:
    """Tests for PRDetails dataclass."""

    def test_basic_creation(self):
        """Test basic PRDetails creation."""
        pr = PRDetails(
            owner="owner",
            repo="repo",
            pull_number=123,
            title="Test PR",
            description="Test description",
        )
        assert pr.owner == "owner"
        assert pr.repo == "repo"
        assert pr.pull_number == 123
        assert pr.title == "Test PR"
        assert pr.description == "Test description"
        assert pr.head_sha is None
        assert pr.base_sha is None

    def test_with_optional_fields(self):
        """Test PRDetails with optional SHA fields."""
        pr = PRDetails(
            owner="owner",
            repo="repo",
            pull_number=1,
            title="Title",
            description="Desc",
            head_sha="abc123",
            base_sha="def456",
        )
        assert pr.head_sha == "abc123"
        assert pr.base_sha == "def456"

    def test_repo_full_name_property(self):
        """Test repo_full_name property."""
        pr = PRDetails(
            owner="myorg",
            repo="myrepo",
            pull_number=1,
            title="Title",
            description="Desc",
        )
        assert pr.repo_full_name == "myorg/myrepo"


class TestFileInfo:
    """Tests for FileInfo dataclass."""

    def test_basic_creation(self):
        """Test basic FileInfo creation."""
        info = FileInfo(path="src/main.py")
        assert info.path == "src/main.py"
        assert info.old_path is None
        assert info.is_new_file is False
        assert info.is_deleted_file is False
        assert info.is_renamed_file is False

    def test_new_file(self):
        """Test FileInfo for a new file."""
        info = FileInfo(path="new_file.py", is_new_file=True)
        assert info.is_new_file is True

    def test_deleted_file(self):
        """Test FileInfo for a deleted file."""
        info = FileInfo(path="old_file.py", is_deleted_file=True)
        assert info.is_deleted_file is True

    def test_renamed_file(self):
        """Test FileInfo for a renamed file."""
        info = FileInfo(
            path="new_name.py",
            old_path="old_name.py",
            is_renamed_file=True,
        )
        assert info.is_renamed_file is True
        assert info.old_path == "old_name.py"

    def test_is_binary_property_true(self):
        """Test is_binary property for binary files."""
        binary_files = [
            "image.png",
            "photo.jpg",
            "pic.jpeg",
            "anim.gif",
            "doc.pdf",
            "archive.zip",
            "archive.tar",
            "compressed.gz",
            "program.exe",
            "library.dll",
            "shared.so",
            "library.dylib",
        ]
        for filename in binary_files:
            info = FileInfo(path=filename)
            assert info.is_binary is True, f"Expected {filename} to be binary"

    def test_is_binary_property_false(self):
        """Test is_binary property for text files."""
        text_files = ["main.py", "script.js", "config.json", "readme.md"]
        for filename in text_files:
            info = FileInfo(path=filename)
            assert info.is_binary is False, f"Expected {filename} to not be binary"

    def test_is_binary_case_insensitive(self):
        """Test that is_binary is case insensitive."""
        info = FileInfo(path="IMAGE.PNG")
        assert info.is_binary is True

    def test_file_extension_property(self):
        """Test file_extension property."""
        assert FileInfo(path="main.py").file_extension == "py"
        assert FileInfo(path="script.js").file_extension == "js"
        assert FileInfo(path="data.json").file_extension == "json"
        assert FileInfo(path="archive.tar.gz").file_extension == "gz"

    def test_file_extension_no_extension(self):
        """Test file_extension for files without extension."""
        info = FileInfo(path="Makefile")
        assert info.file_extension == ""

    def test_file_extension_lowercase(self):
        """Test that file_extension is lowercase."""
        info = FileInfo(path="FILE.PY")
        assert info.file_extension == "py"


class TestHunkInfo:
    """Tests for HunkInfo dataclass."""

    def test_basic_creation(self):
        """Test basic HunkInfo creation."""
        hunk = HunkInfo(
            source_start=1,
            source_length=5,
            target_start=1,
            target_length=7,
            content="content here",
        )
        assert hunk.source_start == 1
        assert hunk.source_length == 5
        assert hunk.target_start == 1
        assert hunk.target_length == 7
        assert hunk.content == "content here"
        assert hunk.header == ""
        assert hunk.lines == []

    def test_with_optional_fields(self):
        """Test HunkInfo with optional fields."""
        hunk = HunkInfo(
            source_start=10,
            source_length=3,
            target_start=10,
            target_length=4,
            content="+new line",
            header="@@ -10,3 +10,4 @@",
            lines=[" context", "+new line", " context"],
        )
        assert hunk.header == "@@ -10,3 +10,4 @@"
        assert len(hunk.lines) == 3


class TestDiffFile:
    """Tests for DiffFile dataclass."""

    def test_basic_creation(self):
        """Test basic DiffFile creation."""
        file_info = FileInfo(path="test.py")
        diff = DiffFile(file_info=file_info)
        assert diff.file_info.path == "test.py"
        assert diff.hunks == []

    def test_with_hunks(self):
        """Test DiffFile with hunks."""
        file_info = FileInfo(path="test.py")
        hunk = HunkInfo(
            source_start=1,
            source_length=1,
            target_start=1,
            target_length=2,
            content="+line",
            lines=["+line1", "+line2"],
        )
        diff = DiffFile(file_info=file_info, hunks=[hunk])
        assert len(diff.hunks) == 1

    def test_total_additions_property(self):
        """Test total_additions property."""
        file_info = FileInfo(path="test.py")
        hunk1 = HunkInfo(
            source_start=1,
            source_length=1,
            target_start=1,
            target_length=3,
            content="",
            lines=["+added1", "+added2", " context"],
        )
        hunk2 = HunkInfo(
            source_start=10,
            source_length=1,
            target_start=12,
            target_length=2,
            content="",
            lines=["+added3", "-removed"],
        )
        diff = DiffFile(file_info=file_info, hunks=[hunk1, hunk2])
        assert diff.total_additions == 3

    def test_total_deletions_property(self):
        """Test total_deletions property."""
        file_info = FileInfo(path="test.py")
        hunk = HunkInfo(
            source_start=1,
            source_length=3,
            target_start=1,
            target_length=1,
            content="",
            lines=["-deleted1", "-deleted2", " context"],
        )
        diff = DiffFile(file_info=file_info, hunks=[hunk])
        assert diff.total_deletions == 2

    def test_empty_hunks(self):
        """Test properties with no hunks."""
        file_info = FileInfo(path="test.py")
        diff = DiffFile(file_info=file_info, hunks=[])
        assert diff.total_additions == 0
        assert diff.total_deletions == 0


class TestReviewComment:
    """Tests for ReviewComment dataclass."""

    def test_basic_creation(self):
        """Test basic ReviewComment creation."""
        comment = ReviewComment(
            body="This is a comment",
            path="file.py",
            position=10,
        )
        assert comment.body == "This is a comment"
        assert comment.path == "file.py"
        assert comment.position == 10
        assert comment.line_number is None
        assert comment.priority == ReviewPriority.MEDIUM
        assert comment.category is None
        assert comment.suggestion is None

    def test_with_all_fields(self):
        """Test ReviewComment with all fields."""
        comment = ReviewComment(
            body="Security issue",
            path="auth.py",
            position=25,
            line_number=50,
            priority=ReviewPriority.CRITICAL,
            category="security",
            suggestion="Use parameterized queries",
        )
        assert comment.line_number == 50
        assert comment.priority == ReviewPriority.CRITICAL
        assert comment.category == "security"
        assert comment.suggestion == "Use parameterized queries"

    def test_to_github_comment(self):
        """Test to_github_comment method."""
        comment = ReviewComment(
            body="Comment body",
            path="src/main.py",
            position=15,
            priority=ReviewPriority.HIGH,
        )
        github_comment = comment.to_github_comment()
        assert github_comment == {
            "body": "Comment body",
            "path": "src/main.py",
            "position": 15,
        }


class TestAIResponse:
    """Tests for AIResponse dataclass."""

    def test_basic_creation(self):
        """Test basic AIResponse creation."""
        response = AIResponse(
            line_number=10,
            review_comment="This needs fixing",
        )
        assert response.line_number == 10
        assert response.review_comment == "This needs fixing"
        assert response.priority == ReviewPriority.MEDIUM
        assert response.category is None
        assert response.confidence is None
        assert response.anchor_snippet is None
        assert response.fix_code is None

    def test_with_all_fields(self):
        """Test AIResponse with all fields."""
        response = AIResponse(
            line_number=25,
            review_comment="Security vulnerability",
            priority=ReviewPriority.CRITICAL,
            category="security",
            confidence=0.95,
            anchor_snippet="password = input()",
            fix_code="password = getpass.getpass()",
        )
        assert response.confidence == 0.95
        assert response.anchor_snippet == "password = input()"
        assert response.fix_code == "password = getpass.getpass()"


class TestReviewResult:
    """Tests for ReviewResult dataclass."""

    def test_basic_creation(self):
        """Test basic ReviewResult creation."""
        pr = PRDetails(
            owner="owner",
            repo="repo",
            pull_number=1,
            title="Title",
            description="Desc",
        )
        result = ReviewResult(pr_details=pr)
        assert result.pr_details == pr
        assert result.comments == []
        assert result.processed_files == 0
        assert result.skipped_files == 0
        assert result.errors == []
        assert result.processing_time is None

    def test_total_comments_property(self):
        """Test total_comments property."""
        pr = PRDetails("o", "r", 1, "t", "d")
        comments = [
            ReviewComment(body="c1", path="f1.py", position=1),
            ReviewComment(body="c2", path="f2.py", position=2),
            ReviewComment(body="c3", path="f3.py", position=3),
        ]
        result = ReviewResult(pr_details=pr, comments=comments)
        assert result.total_comments == 3

    def test_comments_by_priority_property(self):
        """Test comments_by_priority property."""
        pr = PRDetails("o", "r", 1, "t", "d")
        comments = [
            ReviewComment(body="c", path="f.py", position=1, priority=ReviewPriority.LOW),
            ReviewComment(body="c", path="f.py", position=2, priority=ReviewPriority.MEDIUM),
            ReviewComment(body="c", path="f.py", position=3, priority=ReviewPriority.MEDIUM),
            ReviewComment(body="c", path="f.py", position=4, priority=ReviewPriority.HIGH),
            ReviewComment(body="c", path="f.py", position=5, priority=ReviewPriority.CRITICAL),
            ReviewComment(body="c", path="f.py", position=6, priority=ReviewPriority.CRITICAL),
        ]
        result = ReviewResult(pr_details=pr, comments=comments)
        by_priority = result.comments_by_priority
        assert by_priority[ReviewPriority.LOW] == 1
        assert by_priority[ReviewPriority.MEDIUM] == 2
        assert by_priority[ReviewPriority.HIGH] == 1
        assert by_priority[ReviewPriority.CRITICAL] == 2

    def test_comments_by_priority_empty(self):
        """Test comments_by_priority with no comments."""
        pr = PRDetails("o", "r", 1, "t", "d")
        result = ReviewResult(pr_details=pr)
        by_priority = result.comments_by_priority
        assert by_priority[ReviewPriority.LOW] == 0
        assert by_priority[ReviewPriority.MEDIUM] == 0
        assert by_priority[ReviewPriority.HIGH] == 0
        assert by_priority[ReviewPriority.CRITICAL] == 0

    def test_success_property_true(self):
        """Test success property with no errors."""
        pr = PRDetails("o", "r", 1, "t", "d")
        result = ReviewResult(pr_details=pr)
        assert result.success is True

    def test_success_property_false(self):
        """Test success property with errors."""
        pr = PRDetails("o", "r", 1, "t", "d")
        result = ReviewResult(pr_details=pr, errors=["Error 1", "Error 2"])
        assert result.success is False


class TestAnalysisContext:
    """Tests for AnalysisContext dataclass."""

    def test_basic_creation(self):
        """Test basic AnalysisContext creation."""
        pr = PRDetails("o", "r", 1, "t", "d")
        file_info = FileInfo(path="main.py")
        ctx = AnalysisContext(pr_details=pr, file_info=file_info)
        assert ctx.pr_details == pr
        assert ctx.file_info == file_info
        assert ctx.related_files == []
        assert ctx.project_context is None
        assert ctx.language is None
        assert ctx.full_file_content is None
        assert ctx.all_changed_files == []
        assert ctx.change_summary is None

    def test_with_all_fields(self):
        """Test AnalysisContext with all fields."""
        pr = PRDetails("o", "r", 1, "t", "d")
        file_info = FileInfo(path="main.py")
        ctx = AnalysisContext(
            pr_details=pr,
            file_info=file_info,
            related_files=["utils.py", "helpers.py"],
            project_context="Python project",
            language="python",
            full_file_content="# content",
            all_changed_files=["main.py", "test.py"],
            change_summary="Updated main",
        )
        assert len(ctx.related_files) == 2
        assert ctx.language == "python"

    def test_is_test_file_property_true(self):
        """Test is_test_file for test files."""
        pr = PRDetails("o", "r", 1, "t", "d")
        test_paths = [
            "test_main.py",
            "main_test.py",
            "spec_main.py",
            "main_spec.py",
            "tests/test_main.py",
            "src/tests/main.py",  # /tests/ pattern matches
        ]
        for path in test_paths:
            file_info = FileInfo(path=path)
            ctx = AnalysisContext(pr_details=pr, file_info=file_info)
            assert ctx.is_test_file is True, f"Expected {path} to be a test file"

    def test_is_test_file_property_false(self):
        """Test is_test_file for non-test files."""
        pr = PRDetails("o", "r", 1, "t", "d")
        non_test_paths = ["main.py", "utils.py", "src/app.py"]
        for path in non_test_paths:
            file_info = FileInfo(path=path)
            ctx = AnalysisContext(pr_details=pr, file_info=file_info)
            assert ctx.is_test_file is False, f"Expected {path} to not be a test file"


class TestProcessingStats:
    """Tests for ProcessingStats dataclass."""

    def test_basic_creation(self):
        """Test basic ProcessingStats creation."""
        stats = ProcessingStats(start_time=1000.0)
        assert stats.start_time == 1000.0
        assert stats.end_time is None
        assert stats.files_processed == 0
        assert stats.files_skipped == 0
        assert stats.api_calls_made == 0
        assert stats.total_tokens_used == 0
        assert stats.errors_encountered == 0

    def test_duration_property_with_end_time(self):
        """Test duration property when end_time is set."""
        stats = ProcessingStats(start_time=1000.0, end_time=1030.0)
        assert stats.duration == 30.0

    def test_duration_property_without_end_time(self):
        """Test duration property when end_time is not set."""
        stats = ProcessingStats(start_time=1000.0)
        assert stats.duration is None

    def test_processing_rate_property(self):
        """Test processing_rate property."""
        stats = ProcessingStats(
            start_time=1000.0,
            end_time=1010.0,
            files_processed=5,
        )
        assert stats.processing_rate == 0.5  # 5 files / 10 seconds

    def test_processing_rate_zero_duration(self):
        """Test processing_rate with zero duration."""
        stats = ProcessingStats(
            start_time=1000.0,
            end_time=1000.0,
            files_processed=5,
        )
        assert stats.processing_rate is None

    def test_processing_rate_no_end_time(self):
        """Test processing_rate when end_time is not set."""
        stats = ProcessingStats(start_time=1000.0, files_processed=5)
        assert stats.processing_rate is None
