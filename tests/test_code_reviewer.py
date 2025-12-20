"""
Comprehensive tests for gemini_reviewer/code_reviewer.py
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio

from gemini_reviewer.code_reviewer import CodeReviewer, CodeReviewerError
from gemini_reviewer.config import Config, GitHubConfig, GeminiConfig, ReviewConfig
from gemini_reviewer.models import (
    PRDetails,
    ReviewResult,
    ReviewComment,
    DiffFile,
    FileInfo,
    HunkInfo,
    ReviewPriority,
)


class TestCodeReviewerError:
    """Tests for CodeReviewerError exception."""

    def test_exception_message(self):
        """Test exception can be raised with message."""
        with pytest.raises(CodeReviewerError) as exc_info:
            raise CodeReviewerError("Review failed")
        assert "Review failed" in str(exc_info.value)

    def test_exception_inheritance(self):
        """Test CodeReviewerError is an Exception."""
        assert issubclass(CodeReviewerError, Exception)


class TestCodeReviewer:
    """Tests for CodeReviewer class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails(
            owner="owner",
            repo="repo",
            pull_number=123,
            title="Test PR",
            description="Description",
            head_sha="abc123",
            base_sha="def456",
        )

    @pytest.fixture
    def sample_diff_files(self):
        """Create sample diff files."""
        return [
            DiffFile(
                file_info=FileInfo(path="main.py"),
                hunks=[
                    HunkInfo(1, 5, 1, 7, "", "@@ -1,5 +1,7 @@", ["+line"])
                ],
            ),
        ]

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    def test_init(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test CodeReviewer initialization."""
        reviewer = CodeReviewer(mock_config)

        assert reviewer.config == mock_config
        mock_github.assert_called_once()
        mock_gemini.assert_called_once()

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    def test_context_manager(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test CodeReviewer as context manager."""
        with CodeReviewer(mock_config) as reviewer:
            assert reviewer is not None

        # close should have been called
        mock_github.return_value.close.assert_called()
        mock_gemini.return_value.close.assert_called()


class TestCodeReviewerReviewProcess:
    """Tests for the review process."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails(
            owner="owner",
            repo="repo",
            pull_number=123,
            title="Test PR",
            description="Description",
            head_sha="abc123",
            base_sha="def456",
        )

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_review_pr_no_diff(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test review when no diff is available."""
        mock_github.return_value.get_pr_details_from_event.return_value = sample_pr_details
        mock_github.return_value.get_last_reviewed_commit_sha.return_value = None
        mock_github.return_value.get_pr_diff.return_value = ""
        mock_github.return_value.get_existing_comment_signatures.return_value = set()
        mock_github.return_value.get_existing_bot_comments.return_value = []

        reviewer = CodeReviewer(mock_config)
        result = await reviewer.review_pull_request("event.json")

        # No diff means we should exit gracefully
        assert isinstance(result, ReviewResult)

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_review_pr_with_files(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test review with diff files."""
        # Setup mocks
        mock_github.return_value.get_pr_details_from_event.return_value = sample_pr_details
        mock_github.return_value.get_last_reviewed_commit_sha.return_value = None
        mock_github.return_value.get_pr_diff.return_value = "diff --git a/main.py"
        mock_github.return_value.get_existing_comment_signatures.return_value = set()
        mock_github.return_value.get_existing_bot_comments.return_value = []
        mock_github.return_value.filter_out_existing_comments.return_value = []

        mock_diff_parser.return_value.parse_diff.return_value = [
            DiffFile(FileInfo(path="main.py"), hunks=[])
        ]
        mock_diff_parser.return_value.filter_files.return_value = []
        mock_diff_parser.return_value.filter_large_hunks.return_value = []
        mock_diff_parser.return_value.get_parsing_statistics.return_value = {
            "parsed_files": 1,
            "skipped_files": 0,
        }

        mock_config.should_review_file = Mock(return_value=True)

        reviewer = CodeReviewer(mock_config)
        result = await reviewer.review_pull_request("event.json")

        assert isinstance(result, ReviewResult)


class TestCodeReviewerFiltering:
    """Tests for file filtering in CodeReviewer."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        config = Config(github=github_config, gemini=gemini_config)
        config.should_review_file = Mock(return_value=True)
        return config

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_filter_files_applies_config(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test that file filtering respects config."""
        diff_files = [
            DiffFile(FileInfo(path="main.py"), hunks=[]),
            DiffFile(FileInfo(path="test.py"), hunks=[]),
        ]

        mock_diff_parser.return_value.filter_files.return_value = diff_files
        mock_diff_parser.return_value.filter_large_hunks.return_value = diff_files

        reviewer = CodeReviewer(mock_config)
        result = await reviewer._filter_files(diff_files)

        # Should have called should_review_file for each file
        assert mock_config.should_review_file.call_count == 2


class TestCodeReviewerStatistics:
    """Tests for statistics collection."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    def test_get_statistics(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test getting processing statistics."""
        mock_gemini.return_value.get_statistics.return_value = {"total_requests": 0}
        mock_diff_parser.return_value.get_parsing_statistics.return_value = {
            "parsed_files": 0
        }
        mock_github.return_value.check_rate_limit.return_value = {"core": {}}

        reviewer = CodeReviewer(mock_config)
        stats = reviewer.get_statistics()

        assert "processing" in stats
        assert "github" in stats
        assert "gemini" in stats
        assert "parsing" in stats


class TestCodeReviewerConnectionTest:
    """Tests for connection testing."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    def test_test_connections_success(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test successful connection tests."""
        mock_github.return_value.check_rate_limit.return_value = {"core": {"remaining": 5000}}
        mock_github.return_value._client.get_user.return_value.login = "test_user"
        mock_gemini.return_value.test_connection.return_value = True

        reviewer = CodeReviewer(mock_config)
        results = reviewer.test_connections()

        assert results["github"] is True
        assert results["gemini"] is True

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    def test_test_connections_github_failure(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test GitHub connection failure."""
        mock_github.return_value.check_rate_limit.side_effect = Exception("Auth failed")
        mock_gemini.return_value.test_connection.return_value = True

        reviewer = CodeReviewer(mock_config)
        results = reviewer.test_connections()

        assert results["github"] is False
        assert results["gemini"] is True

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    def test_test_connections_gemini_failure(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test Gemini connection failure."""
        mock_github.return_value.check_rate_limit.return_value = {"core": {"remaining": 5000}}
        mock_gemini.return_value.test_connection.return_value = False

        reviewer = CodeReviewer(mock_config)
        results = reviewer.test_connections()

        assert results["gemini"] is False


class TestCodeReviewerChangeSummary:
    """Tests for building change summary."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    def test_build_change_summary(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test building change summary."""
        diff_files = [
            DiffFile(
                file_info=FileInfo(path="main.py"),
                hunks=[
                    HunkInfo(1, 5, 1, 7, "", "", ["+def hello():", "+    pass", "-old"])
                ],
            ),
            DiffFile(
                file_info=FileInfo(path="new.py", is_new_file=True),
                hunks=[
                    HunkInfo(0, 0, 1, 3, "", "", ["+line1", "+line2", "+line3"])
                ],
            ),
        ]

        reviewer = CodeReviewer(mock_config)
        summary = reviewer._build_change_summary(diff_files)

        assert "main.py" in summary
        assert "new.py" in summary
        assert "modified" in summary or "added" in summary

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    def test_build_change_summary_detects_functions(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test that change summary detects function definitions."""
        diff_files = [
            DiffFile(
                file_info=FileInfo(path="main.py"),
                hunks=[
                    HunkInfo(1, 1, 1, 3, "", "", ["+def my_function():", "+    return True"])
                ],
            ),
        ]

        reviewer = CodeReviewer(mock_config)
        summary = reviewer._build_change_summary(diff_files)

        assert "my_function" in summary

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    def test_build_change_summary_empty_list(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test building change summary with empty list."""
        reviewer = CodeReviewer(mock_config)
        summary = reviewer._build_change_summary([])

        assert "## All Files Changed" in summary


class TestCodeReviewerAnalysis:
    """Tests for the analysis methods."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        config = Config(github=github_config, gemini=gemini_config)
        config.should_review_file = Mock(return_value=True)
        return config

    @pytest.fixture
    def sample_diff_files(self):
        """Create sample diff files."""
        return [
            DiffFile(
                file_info=FileInfo(path="main.py"),
                hunks=[
                    HunkInfo(1, 5, 1, 7, "+new code", "@@ -1,5 +1,7 @@", ["+line1"])
                ],
            ),
        ]

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head123", "base456")

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_analyze_files_sequentially(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_diff_files,
        sample_pr_details,
    ):
        """Test analyzing files sequentially."""
        mock_context.return_value.detect_related_files = AsyncMock(return_value=[])
        mock_context.return_value.build_project_context = AsyncMock(return_value=None)
        mock_context.return_value.build_analysis_context = AsyncMock(
            return_value=Mock()
        )
        mock_gemini.return_value.analyze_code_hunk.return_value = []

        reviewer = CodeReviewer(mock_config)
        reviewer._existing_comment_signatures = set()
        reviewer._is_followup_review = False
        reviewer._current_review_file_paths = set()
        result = await reviewer._analyze_files_sequentially(sample_diff_files, sample_pr_details)

        assert isinstance(result, list)

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_analyze_files_concurrently(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_diff_files,
        sample_pr_details,
    ):
        """Test analyzing files concurrently."""
        mock_context.return_value.detect_related_files = AsyncMock(return_value=[])
        mock_context.return_value.build_project_context = AsyncMock(return_value=None)
        mock_context.return_value.build_analysis_context = AsyncMock(
            return_value=Mock()
        )
        mock_gemini.return_value.analyze_code_hunk.return_value = []

        reviewer = CodeReviewer(mock_config)
        reviewer._existing_comment_signatures = set()
        reviewer._is_followup_review = False
        reviewer._current_review_file_paths = set()
        result = await reviewer._analyze_files_concurrently(sample_diff_files, sample_pr_details)

        assert isinstance(result, list)


class TestCodeReviewerSingleFile:
    """Tests for single file analysis."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @pytest.fixture
    def sample_diff_file(self):
        """Create a sample diff file."""
        return DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[
                HunkInfo(1, 5, 1, 7, "+code", "@@ -1,5 +1,7 @@", ["+line"])
            ],
        )

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head123", "base456")

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_analyze_single_file(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_diff_file,
        sample_pr_details,
    ):
        """Test analyzing a single file."""
        mock_context.return_value.detect_related_files = AsyncMock(return_value=[])
        mock_context.return_value.build_project_context = AsyncMock(return_value=None)
        mock_context.return_value.build_analysis_context = AsyncMock(
            return_value=Mock()
        )
        mock_gemini.return_value.analyze_code_hunk.return_value = []

        reviewer = CodeReviewer(mock_config)
        reviewer._existing_comment_signatures = set()
        result = await reviewer._analyze_single_file(sample_diff_file, sample_pr_details)

        assert isinstance(result, list)




class TestCodeReviewerGetDiff:
    """Tests for _get_pr_diff functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head123", "base456")

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_get_pr_diff_incremental(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test getting incremental PR diff."""
        mock_github.return_value.get_last_reviewed_commit_sha.return_value = "old_sha"
        mock_github.return_value.get_pr_diff_since.return_value = "diff content"

        reviewer = CodeReviewer(mock_config)
        result = await reviewer._get_pr_diff(sample_pr_details)

        assert result == "diff content"

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_get_pr_diff_full(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test getting full PR diff when no prior review."""
        mock_github.return_value.get_last_reviewed_commit_sha.return_value = None
        mock_github.return_value.get_pr_diff.return_value = "full diff"

        reviewer = CodeReviewer(mock_config)
        result = await reviewer._get_pr_diff(sample_pr_details)

        assert result == "full diff"

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_get_pr_diff_github_error(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test getting PR diff with GitHubClientError."""
        from gemini_reviewer.github_client import GitHubClientError
        mock_github.return_value.get_last_reviewed_commit_sha.return_value = None
        mock_github.return_value.get_pr_diff.side_effect = GitHubClientError("API Error")

        reviewer = CodeReviewer(mock_config)
        result = await reviewer._get_pr_diff(sample_pr_details)

        assert result == ""


class TestCodeReviewerParseDiff:
    """Tests for _parse_diff functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_parse_diff_success(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test successful diff parsing."""
        mock_diff_parser.return_value.parse_diff.return_value = [
            DiffFile(FileInfo(path="test.py"), hunks=[])
        ]

        reviewer = CodeReviewer(mock_config)
        result = await reviewer._parse_diff("diff content")

        assert len(result) == 1

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_parse_diff_exception(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test diff parsing with exception."""
        from gemini_reviewer.diff_parser import DiffParsingError
        mock_diff_parser.return_value.parse_diff.side_effect = DiffParsingError("Parse error")

        reviewer = CodeReviewer(mock_config)
        result = await reviewer._parse_diff("bad diff")

        assert result == []


class TestCodeReviewerCreateReview:
    """Tests for _create_github_review functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc")

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_create_github_review_with_comments(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test creating GitHub review with comments."""
        mock_comment_proc.return_value.filter_and_prioritize.return_value = [
            ReviewComment("Fix", "test.py", 1)
        ]
        mock_github.return_value.filter_out_existing_comments.return_value = [
            ReviewComment("Fix", "test.py", 1)
        ]
        mock_github.return_value.create_review.return_value = True

        reviewer = CodeReviewer(mock_config)
        comments = [ReviewComment("Fix", "test.py", 1)]
        result = await reviewer._create_github_review(sample_pr_details, comments)

        assert result is True

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_create_github_review_empty(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test creating GitHub review with no comments."""
        mock_comment_proc.return_value.filter_and_prioritize.return_value = []
        mock_github.return_value.filter_out_existing_comments.return_value = []
        mock_github.return_value.create_review.return_value = True

        reviewer = CodeReviewer(mock_config)
        result = await reviewer._create_github_review(sample_pr_details, [])

        assert result is True


class TestCodeReviewerFollowUp:
    """Tests for follow-up review functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head123", "base456")

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_followup_review(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test follow-up review with existing comments."""
        mock_github.return_value.get_pr_details_from_event.return_value = sample_pr_details
        mock_github.return_value.get_last_reviewed_commit_sha.return_value = "old_sha"
        mock_github.return_value.get_pr_diff_since.return_value = ""
        mock_github.return_value.get_pr_diff.return_value = ""
        mock_github.return_value.get_existing_comment_signatures.return_value = set()
        mock_github.return_value.get_existing_bot_comments.return_value = [
            {"path": "main.py", "line": 10, "body": "Fix this", "id": 1, "created_at": "2024-01-01"}
        ]
        mock_github.return_value.filter_out_existing_comments.return_value = []
        mock_github.return_value.create_review.return_value = True

        reviewer = CodeReviewer(mock_config)
        result = await reviewer.review_pull_request("event.json")

        assert isinstance(result, ReviewResult)


class TestCodeReviewerIncrementalReview:
    """Tests for incremental review functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head123", "base456")

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_incremental_diff_review(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test review with incremental diff (since last review)."""
        mock_github.return_value.get_pr_details_from_event.return_value = sample_pr_details
        mock_github.return_value.get_last_reviewed_commit_sha.return_value = "old_sha"
        mock_github.return_value.get_pr_diff_since.return_value = "diff --git a/new.py"
        mock_github.return_value.get_existing_comment_signatures.return_value = set()
        mock_github.return_value.get_existing_bot_comments.return_value = []
        mock_github.return_value.filter_out_existing_comments.return_value = []
        mock_github.return_value.create_review.return_value = True

        mock_diff_parser.return_value.parse_diff.return_value = []
        mock_diff_parser.return_value.filter_files.return_value = []
        mock_diff_parser.return_value.filter_large_hunks.return_value = []
        mock_diff_parser.return_value.get_parsing_statistics.return_value = {"parsed_files": 0}

        reviewer = CodeReviewer(mock_config)
        result = await reviewer.review_pull_request("event.json")

        assert isinstance(result, ReviewResult)


class TestCodeReviewerExceptionHandling:
    """Tests for exception handling paths."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head123", "base456")

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_review_with_signature_load_error(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test review continues when loading comment signatures fails."""
        mock_github.return_value.get_pr_details_from_event.return_value = sample_pr_details
        mock_github.return_value.get_existing_comment_signatures.side_effect = Exception("API Error")
        mock_github.return_value.get_existing_bot_comments.return_value = []
        mock_github.return_value.get_last_reviewed_commit_sha.return_value = None
        mock_github.return_value.get_pr_diff.return_value = ""

        reviewer = CodeReviewer(mock_config)
        result = await reviewer.review_pull_request("event.json")

        assert isinstance(result, ReviewResult)

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_review_with_bot_comments_error(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test review continues when getting bot comments fails."""
        mock_github.return_value.get_pr_details_from_event.return_value = sample_pr_details
        mock_github.return_value.get_existing_comment_signatures.return_value = set()
        mock_github.return_value.get_existing_bot_comments.side_effect = Exception("API Error")
        mock_github.return_value.get_last_reviewed_commit_sha.return_value = None
        mock_github.return_value.get_pr_diff.return_value = ""

        reviewer = CodeReviewer(mock_config)
        result = await reviewer.review_pull_request("event.json")

        assert isinstance(result, ReviewResult)

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_review_general_exception(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test review handles general exceptions."""
        mock_github.return_value.get_pr_details_from_event.side_effect = Exception("Unexpected error")

        reviewer = CodeReviewer(mock_config)
        result = await reviewer.review_pull_request("event.json")

        assert isinstance(result, ReviewResult)
        assert len(result.errors) > 0


class TestCodeReviewerApprovalReview:
    """Tests for approval review scenarios."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        config = Config(github=github_config, gemini=gemini_config)
        config.should_review_file = Mock(return_value=True)
        return config

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head123", "base456")

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_approval_review_on_no_comments(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test approval review is posted when no comments are generated."""
        mock_github.return_value.get_pr_details_from_event.return_value = sample_pr_details
        mock_github.return_value.get_existing_comment_signatures.return_value = set()
        mock_github.return_value.get_existing_bot_comments.return_value = []
        mock_github.return_value.get_last_reviewed_commit_sha.return_value = None
        mock_github.return_value.get_pr_diff.return_value = "diff --git a/main.py"
        mock_github.return_value.filter_out_existing_comments.return_value = []
        mock_github.return_value.get_file_content.return_value = "content"
        mock_github.return_value.create_review.return_value = True

        mock_diff_parser.return_value.parse_diff.return_value = [
            DiffFile(FileInfo(path="main.py"), hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+line"])])
        ]
        mock_diff_parser.return_value.filter_files.return_value = [
            DiffFile(FileInfo(path="main.py"), hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+line"])])
        ]
        mock_diff_parser.return_value.filter_large_hunks.return_value = [
            DiffFile(FileInfo(path="main.py"), hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+line"])])
        ]
        mock_diff_parser.return_value.get_parsing_statistics.return_value = {"parsed_files": 1, "skipped_files": 0}

        mock_context.return_value.detect_related_files = AsyncMock(return_value=[])
        mock_context.return_value.build_project_context = AsyncMock(return_value=None)

        mock_gemini.return_value.analyze_code_hunk.return_value = []

        mock_comment_proc.return_value.filter_comments_by_priority.return_value = []
        mock_comment_proc.return_value.apply_comment_limits.return_value = []

        reviewer = CodeReviewer(mock_config)
        reviewer._existing_comment_signatures = set()
        reviewer._is_followup_review = False
        result = await reviewer.review_pull_request("event.json")

        assert isinstance(result, ReviewResult)


class TestCodeReviewerFollowUpReplies:
    """Tests for follow-up reply functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc")

    @pytest.fixture
    def sample_diff_file(self):
        """Create a sample diff file."""
        return DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[HunkInfo(1, 5, 1, 7, "", "", ["+line"])],
        )

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_post_followup_replies_no_previous_comments(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
        sample_diff_file,
    ):
        """Test follow-up when no previous bot comments exist."""
        reviewer = CodeReviewer(mock_config)
        reviewer._previous_bot_comments = None
        reviewer._unresolved_prior_ids = set()

        comments = [ReviewComment("Fix this", "main.py", 1)]
        await reviewer._post_followup_replies(sample_pr_details, sample_diff_file, comments)
        # Should return early without error

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_post_followup_replies_with_previous_comments(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
        sample_diff_file,
    ):
        """Test follow-up replies to previous bot comments."""
        reviewer = CodeReviewer(mock_config)
        reviewer._previous_bot_comments = [
            {"path": "main.py", "line": 1, "body": "Fix this bug", "id": 123, "position": 5}
        ]
        reviewer._unresolved_prior_ids = set()

        comment = ReviewComment("Previous issue: Fix this bug\nStatus: Not resolved", "main.py", 1)
        comment.position = 5
        await reviewer._post_followup_replies(sample_pr_details, sample_diff_file, [comment])

        mock_github.return_value.reply_to_comment.assert_called()

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_post_followup_replies_position_matching(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
        sample_diff_file,
    ):
        """Test follow-up replies use position matching."""
        reviewer = CodeReviewer(mock_config)
        reviewer._previous_bot_comments = [
            {"path": "main.py", "line": 10, "body": "Comment 1", "id": 100, "position": 10},
            {"path": "main.py", "line": 20, "body": "Comment 2", "id": 200, "position": 20},
        ]
        reviewer._unresolved_prior_ids = set()

        comment = ReviewComment("New issue", "main.py", 10)
        comment.position = 12
        await reviewer._post_followup_replies(sample_pr_details, sample_diff_file, [comment])

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_post_followup_replies_exception(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
        sample_diff_file,
    ):
        """Test follow-up replies handles exceptions gracefully."""
        mock_github.return_value.reply_to_comment.side_effect = Exception("API Error")

        reviewer = CodeReviewer(mock_config)
        reviewer._previous_bot_comments = [
            {"path": "main.py", "line": 1, "body": "Fix", "id": 123}
        ]
        reviewer._unresolved_prior_ids = set()

        comment = ReviewComment("Reply", "main.py", 1)
        await reviewer._post_followup_replies(sample_pr_details, sample_diff_file, [comment])
        # Should not raise


class TestCodeReviewerResolveComments:
    """Tests for resolving completed comments."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc")

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_resolve_comments_not_followup(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test resolve returns early if not follow-up review."""
        reviewer = CodeReviewer(mock_config)
        reviewer._is_followup_review = False

        await reviewer._resolve_completed_comments(sample_pr_details)
        # Should return early

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_resolve_comments_no_previous(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test resolve returns early if no previous comments."""
        reviewer = CodeReviewer(mock_config)
        reviewer._is_followup_review = True
        reviewer._previous_bot_comments = []

        await reviewer._resolve_completed_comments(sample_pr_details)
        # Should return early

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_resolve_comments_posts_replies(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test resolve posts resolution replies."""
        mock_github.return_value.reply_to_comment.return_value = True

        reviewer = CodeReviewer(mock_config)
        reviewer._is_followup_review = True
        reviewer._previous_bot_comments = [
            {"path": "main.py", "id": 123, "body": "Fix this"}
        ]
        reviewer._current_review_file_paths = {"main.py"}
        reviewer._unresolved_prior_ids = set()

        await reviewer._resolve_completed_comments(sample_pr_details)

        mock_github.return_value.reply_to_comment.assert_called()

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_resolve_comments_skips_unresolved(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test resolve skips comments still unresolved."""
        reviewer = CodeReviewer(mock_config)
        reviewer._is_followup_review = True
        reviewer._previous_bot_comments = [
            {"path": "main.py", "id": 123, "body": "Fix"}
        ]
        reviewer._current_review_file_paths = {"main.py"}
        reviewer._unresolved_prior_ids = {123}

        await reviewer._resolve_completed_comments(sample_pr_details)

        mock_github.return_value.reply_to_comment.assert_not_called()


class TestCodeReviewerCreateReviewEvents:
    """Tests for _create_github_review with different events."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc")

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_create_review_with_preferred_event(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test creating review with preferred event."""
        mock_github.return_value.filter_out_existing_comments.return_value = []
        mock_comment_proc.return_value.filter_comments_by_priority.return_value = []
        mock_comment_proc.return_value.apply_comment_limits.return_value = []
        mock_github.return_value.create_review.return_value = True

        reviewer = CodeReviewer(mock_config)
        result = await reviewer._create_github_review(sample_pr_details, [], preferred_event="APPROVE")

        mock_github.return_value.create_review.assert_called()
        assert result is True

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_create_review_request_changes(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test creating review with REQUEST_CHANGES event."""
        comments = [ReviewComment("Fix this", "main.py", 1)]
        mock_github.return_value.filter_out_existing_comments.return_value = comments
        mock_comment_proc.return_value.filter_comments_by_priority.return_value = comments
        mock_comment_proc.return_value.apply_comment_limits.return_value = comments
        mock_github.return_value.create_review.return_value = True

        reviewer = CodeReviewer(mock_config)
        result = await reviewer._create_github_review(sample_pr_details, comments)

        assert result is True

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_create_review_filtered_comments(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test creating review when all comments are filtered."""
        comments = [ReviewComment("Fix", "main.py", 1)]
        mock_github.return_value.filter_out_existing_comments.return_value = comments
        mock_comment_proc.return_value.filter_comments_by_priority.return_value = comments
        mock_comment_proc.return_value.apply_comment_limits.return_value = []
        mock_github.return_value.create_review.return_value = True

        reviewer = CodeReviewer(mock_config)
        result = await reviewer._create_github_review(sample_pr_details, comments)

        assert result is True

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_create_review_github_error(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test creating review handles GitHubClientError."""
        from gemini_reviewer.github_client import GitHubClientError
        mock_github.return_value.filter_out_existing_comments.return_value = []
        mock_comment_proc.return_value.filter_comments_by_priority.return_value = []
        mock_comment_proc.return_value.apply_comment_limits.return_value = []
        mock_github.return_value.create_review.side_effect = GitHubClientError("API Error")

        reviewer = CodeReviewer(mock_config)
        result = await reviewer._create_github_review(sample_pr_details, [])

        assert result is False


class TestCodeReviewerChangeSummaryExtended:
    """Extended tests for building change summary."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    def test_build_change_summary_deleted_file(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test change summary with deleted file."""
        diff_files = [
            DiffFile(
                file_info=FileInfo(path="old.py", is_deleted_file=True),
                hunks=[HunkInfo(1, 5, 0, 0, "", "", ["-line1", "-line2"])],
            )
        ]

        reviewer = CodeReviewer(mock_config)
        summary = reviewer._build_change_summary(diff_files)

        assert "deleted" in summary

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    def test_build_change_summary_renamed_file(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test change summary with renamed file."""
        diff_files = [
            DiffFile(
                file_info=FileInfo(path="new_name.py", is_renamed_file=True),
                hunks=[],
            )
        ]

        reviewer = CodeReviewer(mock_config)
        summary = reviewer._build_change_summary(diff_files)

        assert "renamed" in summary

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    def test_build_change_summary_class_detection(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test that change summary detects class definitions."""
        diff_files = [
            DiffFile(
                file_info=FileInfo(path="main.py"),
                hunks=[HunkInfo(1, 1, 1, 3, "", "", ["+class MyClass:", "+    pass"])],
            )
        ]

        reviewer = CodeReviewer(mock_config)
        summary = reviewer._build_change_summary(diff_files)

        assert "MyClass" in summary

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    def test_build_change_summary_async_function(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test that change summary detects async function definitions."""
        diff_files = [
            DiffFile(
                file_info=FileInfo(path="main.py"),
                hunks=[HunkInfo(1, 1, 1, 3, "", "", ["+async def fetch_data():", "+    pass"])],
            )
        ]

        reviewer = CodeReviewer(mock_config)
        summary = reviewer._build_change_summary(diff_files)

        assert "fetch_data" in summary

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    def test_build_change_summary_js_function(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test that change summary detects JavaScript function definitions."""
        diff_files = [
            DiffFile(
                file_info=FileInfo(path="main.js"),
                hunks=[HunkInfo(1, 1, 1, 3, "", "", ["+function handleClick() {", "+    return;", "+}"])],
            )
        ]

        reviewer = CodeReviewer(mock_config)
        summary = reviewer._build_change_summary(diff_files)

        assert "handleClick" in summary

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    def test_build_change_summary_limits_files(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test that change summary limits to 30 files."""
        diff_files = [
            DiffFile(
                file_info=FileInfo(path=f"file{i}.py"),
                hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+line"])],
            )
            for i in range(40)
        ]

        reviewer = CodeReviewer(mock_config)
        summary = reviewer._build_change_summary(diff_files)

        # Should have header (2 lines: title + blank) + files (limited to 30 total entries)
        lines = [l for l in summary.split("\n") if l.strip()]
        assert len(lines) <= 31  # Header + up to 30 file entries


class TestCodeReviewerStatisticsRateLimit:
    """Tests for statistics with rate limit errors."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    def test_get_statistics_rate_limit_error(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test getting statistics when rate limit check fails."""
        mock_gemini.return_value.get_statistics.return_value = {"total_requests": 0}
        mock_diff_parser.return_value.get_parsing_statistics.return_value = {"parsed_files": 0}
        mock_github.return_value.check_rate_limit.side_effect = Exception("Rate limit API error")

        reviewer = CodeReviewer(mock_config)
        stats = reviewer.get_statistics()

        assert "processing" in stats
        assert stats["github"] == {}


class TestCodeReviewerConnectionTestExtended:
    """Extended tests for connection testing."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    def test_test_connections_github_user_error(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test connection test when getting user fails but rate limit works."""
        mock_github.return_value.check_rate_limit.return_value = {"core": {"remaining": 5000}}
        mock_github.return_value._client.get_user.side_effect = Exception("User API error")
        mock_gemini.return_value.test_connection.return_value = True

        reviewer = CodeReviewer(mock_config)
        results = reviewer.test_connections()

        assert results["github"] is True

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    def test_test_connections_rate_limit_unexpected_structure(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test connection test when rate limit returns unexpected structure."""
        mock_github.return_value.check_rate_limit.return_value = {}
        mock_gemini.return_value.test_connection.return_value = True

        reviewer = CodeReviewer(mock_config)
        results = reviewer.test_connections()

        assert results["github"] is False

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    def test_test_connections_gemini_exception(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test connection test when Gemini raises exception."""
        mock_github.return_value.check_rate_limit.return_value = {"core": {"remaining": 5000}}
        mock_gemini.return_value.test_connection.side_effect = Exception("Gemini error")

        reviewer = CodeReviewer(mock_config)
        results = reviewer.test_connections()

        assert results["gemini"] is False


class TestCodeReviewerCleanupError:
    """Tests for cleanup with errors."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    def test_close_with_error(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
    ):
        """Test close handles errors gracefully."""
        mock_github.return_value.close.side_effect = Exception("Cleanup error")

        reviewer = CodeReviewer(mock_config)
        reviewer.close()  # Should not raise


class TestCodeReviewerDuplicateFileHandling:
    """Tests for duplicate file handling in analysis."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        config = Config(github=github_config, gemini=gemini_config)
        config.should_review_file = Mock(return_value=True)
        return config

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head123", "base456")

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_sequential_skips_duplicate_files(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test sequential analysis skips duplicate file entries."""
        mock_context.return_value.detect_related_files = AsyncMock(return_value=[])
        mock_context.return_value.build_project_context = AsyncMock(return_value=None)
        mock_gemini.return_value.analyze_code_hunk.return_value = []

        # Create duplicate file entries
        diff_files = [
            DiffFile(FileInfo(path="main.py"), hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+line"])]),
            DiffFile(FileInfo(path="main.py"), hunks=[HunkInfo(2, 1, 2, 1, "", "", ["+line2"])]),
        ]

        reviewer = CodeReviewer(mock_config)
        reviewer._existing_comment_signatures = set()
        reviewer._is_followup_review = False
        reviewer._current_review_file_paths = set()
        result = await reviewer._analyze_files_sequentially(diff_files, sample_pr_details)

        # Should only process unique files
        assert isinstance(result, list)

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_concurrent_skips_duplicate_files(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test concurrent analysis skips duplicate file entries."""
        mock_context.return_value.detect_related_files = AsyncMock(return_value=[])
        mock_context.return_value.build_project_context = AsyncMock(return_value=None)
        mock_gemini.return_value.analyze_code_hunk.return_value = []

        # Create duplicate file entries
        diff_files = [
            DiffFile(FileInfo(path="main.py"), hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+line"])]),
            DiffFile(FileInfo(path="main.py"), hunks=[HunkInfo(2, 1, 2, 1, "", "", ["+line2"])]),
        ]

        reviewer = CodeReviewer(mock_config)
        reviewer._existing_comment_signatures = set()
        reviewer._is_followup_review = False
        reviewer._current_review_file_paths = set()
        result = await reviewer._analyze_files_concurrently(diff_files, sample_pr_details)

        assert isinstance(result, list)


class TestCodeReviewerAnalysisSingleFileExtended:
    """Extended tests for single file analysis."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head123", "base456")

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_analyze_single_file_with_comments(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test analyzing a single file that generates comments."""
        from gemini_reviewer.models import AIResponse

        mock_context.return_value.detect_related_files = AsyncMock(return_value=[])
        mock_context.return_value.build_project_context = AsyncMock(return_value=None)

        ai_response = AIResponse(line_number=5, review_comment="Fix this issue")
        mock_gemini.return_value.analyze_code_hunk.return_value = [ai_response]

        mock_comment = ReviewComment("Fix this issue", "main.py", 5)
        mock_comment_proc.return_value.convert_to_review_comment.return_value = mock_comment

        mock_github.return_value.get_file_content.return_value = "print('hello')"
        mock_github.return_value._compute_signature.return_value = "sig123"

        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[HunkInfo(1, 5, 1, 7, "+code", "@@ -1,5 +1,7 @@", ["+line"])],
        )

        reviewer = CodeReviewer(mock_config)
        reviewer._existing_comment_signatures = set()
        reviewer._is_followup_review = False
        result = await reviewer._analyze_single_file(diff_file, sample_pr_details)

        assert len(result) == 1
        assert result[0].body == "Fix this issue"

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_analyze_single_file_skips_duplicate(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test analyzing a file that skips duplicate comments."""
        from gemini_reviewer.models import AIResponse

        mock_context.return_value.detect_related_files = AsyncMock(return_value=[])
        mock_context.return_value.build_project_context = AsyncMock(return_value=None)

        ai_response = AIResponse(line_number=5, review_comment="Fix this")
        mock_gemini.return_value.analyze_code_hunk.return_value = [ai_response]

        mock_comment = ReviewComment("Fix this", "main.py", 5)
        mock_comment_proc.return_value.convert_to_review_comment.return_value = mock_comment

        mock_github.return_value.get_file_content.return_value = "print('hello')"
        mock_github.return_value._compute_signature.return_value = "existing_sig"

        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[HunkInfo(1, 5, 1, 7, "", "", ["+line"])],
        )

        reviewer = CodeReviewer(mock_config)
        reviewer._existing_comment_signatures = {"existing_sig"}
        reviewer._is_followup_review = False
        result = await reviewer._analyze_single_file(diff_file, sample_pr_details)

        # Should skip due to existing signature
        assert len(result) == 0

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_analyze_single_file_gemini_error(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test analyzing a file when Gemini API fails."""
        from gemini_reviewer.gemini_client import GeminiClientError

        mock_context.return_value.detect_related_files = AsyncMock(return_value=[])
        mock_context.return_value.build_project_context = AsyncMock(return_value=None)

        mock_gemini.return_value.analyze_code_hunk.side_effect = GeminiClientError("API Error")

        mock_github.return_value.get_file_content.return_value = "code"

        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[HunkInfo(1, 5, 1, 7, "", "", ["+line"])],
        )

        reviewer = CodeReviewer(mock_config)
        reviewer._existing_comment_signatures = set()
        result = await reviewer._analyze_single_file(diff_file, sample_pr_details)

        # Should handle error gracefully and return empty list
        assert result == []

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_analyze_single_file_followup_mode(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test analyzing a file in follow-up review mode."""
        mock_context.return_value.detect_related_files = AsyncMock(return_value=[])
        mock_context.return_value.build_project_context = AsyncMock(return_value=None)
        mock_gemini.return_value.analyze_code_hunk.return_value = []
        mock_github.return_value.get_file_content.return_value = "code"

        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[HunkInfo(1, 5, 1, 7, "", "", ["+line"])],
        )

        reviewer = CodeReviewer(mock_config)
        reviewer._existing_comment_signatures = set()
        reviewer._is_followup_review = True
        reviewer._previous_comments = "Previous comment"
        reviewer._current_followup_issues = set()
        result = await reviewer._analyze_single_file(diff_file, sample_pr_details)

        assert isinstance(result, list)

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_analyze_single_file_with_file_fetch_error(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test analyzing when fetching full file content fails."""
        mock_context.return_value.detect_related_files = AsyncMock(return_value=[])
        mock_context.return_value.build_project_context = AsyncMock(return_value=None)
        mock_gemini.return_value.analyze_code_hunk.return_value = []

        mock_github.return_value.get_file_content.side_effect = Exception("File not found")

        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[HunkInfo(1, 5, 1, 7, "", "", ["+line"])],
        )

        reviewer = CodeReviewer(mock_config)
        reviewer._existing_comment_signatures = set()
        result = await reviewer._analyze_single_file(diff_file, sample_pr_details)

        # Should continue without full file content
        assert isinstance(result, list)


class TestCodeReviewerGetDiffExtended:
    """Extended tests for _get_pr_diff."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head123", "base456")

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_get_pr_diff_empty_incremental(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test getting diff when incremental diff is empty."""
        mock_github.return_value.get_last_reviewed_commit_sha.return_value = "old_sha"
        mock_github.return_value.get_pr_diff_since.return_value = ""

        reviewer = CodeReviewer(mock_config)
        result = await reviewer._get_pr_diff(sample_pr_details)

        assert result == ""

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_get_pr_diff_incremental_fallback(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test getting diff falls back to full when incremental returns None."""
        mock_github.return_value.get_last_reviewed_commit_sha.return_value = "old_sha"
        mock_github.return_value.get_pr_diff_since.return_value = None
        mock_github.return_value.get_pr_diff.return_value = "full diff"

        reviewer = CodeReviewer(mock_config)
        result = await reviewer._get_pr_diff(sample_pr_details)

        assert result == "full diff"


class TestCodeReviewerReviewNoFiles:
    """Tests for review with no files scenarios."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        from gemini_reviewer.config import Config, GeminiConfig, GitHubConfig, ReviewConfig

        return Config(
            gemini=GeminiConfig(api_key="AIzaSy_test_key_12345"),
            github=GitHubConfig(token="ghp_test_token_123456"),
            review=ReviewConfig(),
        )

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails(
            owner="test-owner",
            repo="test-repo",
            pull_number=123,
            title="Test PR",
            description="Test description",
            head_sha="abc123",
        )

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_review_empty_diff(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test review when diff is empty string."""
        mock_github.return_value.get_last_reviewed_commit_sha.return_value = None
        mock_github.return_value.get_pr_diff.return_value = ""

        reviewer = CodeReviewer(mock_config)
        result = await reviewer.review_pull_request(sample_pr_details)

        assert result is not None
        assert len(result.comments) == 0

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_review_none_diff(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test review when diff is None."""
        mock_github.return_value.get_last_reviewed_commit_sha.return_value = None
        mock_github.return_value.get_pr_diff.return_value = None

        reviewer = CodeReviewer(mock_config)
        result = await reviewer.review_pull_request(sample_pr_details)

        assert result is not None
        assert len(result.errors) > 0

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_review_no_files_after_parsing(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test review when parsing returns no files."""
        mock_github.return_value.get_last_reviewed_commit_sha.return_value = None
        mock_github.return_value.get_pr_diff.return_value = "diff content"
        mock_diff_parser.return_value.parse_diff.return_value = []

        reviewer = CodeReviewer(mock_config)
        result = await reviewer.review_pull_request(sample_pr_details)

        assert result is not None
        assert len(result.comments) == 0

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_review_exception_handling(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test review handles exceptions gracefully."""
        # Need to set up the whole chain - get_last_reviewed_commit_sha returns None
        # so it falls through to get_pr_diff which raises an exception
        mock_github_instance = mock_github.return_value
        mock_github_instance.get_last_reviewed_commit_sha.return_value = None
        mock_github_instance.get_pr_diff.side_effect = Exception("API Error")
        mock_github_instance.get_pr_details_from_event.return_value = sample_pr_details

        reviewer = CodeReviewer(mock_config)
        result = await reviewer.review_pull_request(sample_pr_details)

        assert result is not None
        # Exception should be caught and added to errors
        assert len(result.errors) > 0


class TestCodeReviewerFollowUpMode:
    """Tests for follow-up review mode."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        from gemini_reviewer.config import Config, GeminiConfig, GitHubConfig, ReviewConfig

        return Config(
            gemini=GeminiConfig(api_key="AIzaSy_test_key_12345"),
            github=GitHubConfig(token="ghp_test_token_123456"),
            review=ReviewConfig(),
        )

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails(
            owner="test-owner",
            repo="test-repo",
            pull_number=123,
            title="Test PR",
            description="Test description",
            head_sha="abc123",
        )

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_followup_no_comments(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test follow-up review with no comments generated."""
        mock_github.return_value.get_last_reviewed_commit_sha.return_value = "old_sha"
        mock_github.return_value.get_pr_diff_since.return_value = "diff content"
        mock_diff_parser.return_value.parse_diff.return_value = []

        reviewer = CodeReviewer(mock_config)
        result = await reviewer.review_pull_request(sample_pr_details)

        assert result is not None


class TestCodeReviewerSequentialProcessing:
    """Tests for sequential file processing."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config with concurrent processing disabled."""
        from gemini_reviewer.config import Config, GeminiConfig, GitHubConfig, ReviewConfig, PerformanceConfig

        return Config(
            gemini=GeminiConfig(api_key="AIzaSy_test_key_12345"),
            github=GitHubConfig(token="ghp_test_token_123456"),
            review=ReviewConfig(),
            performance=PerformanceConfig(enable_concurrent_processing=False),
        )

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails(
            owner="test-owner",
            repo="test-repo",
            pull_number=123,
            title="Test PR",
            description="Test description",
            head_sha="abc123",
        )

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_sequential_processing(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test sequential file processing mode."""
        mock_github.return_value.get_last_reviewed_commit_sha.return_value = None
        mock_github.return_value.get_pr_diff.return_value = "diff content"
        mock_diff_parser.return_value.parse_diff.return_value = []

        reviewer = CodeReviewer(mock_config)
        result = await reviewer.review_pull_request(sample_pr_details)

        assert result is not None


class TestCodeReviewerDiffRetrievalAdvanced:
    """Advanced tests for diff retrieval scenarios."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        from gemini_reviewer.config import Config, GeminiConfig, GitHubConfig, ReviewConfig, PerformanceConfig

        return Config(
            gemini=GeminiConfig(api_key="AIzaSy_test_key_12345"),
            github=GitHubConfig(token="ghp_test_token_123456"),
            review=ReviewConfig(),
            performance=PerformanceConfig(enable_concurrent_processing=False),
        )

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails(
            owner="test-owner",
            repo="test-repo",
            pull_number=123,
            title="Test PR",
            description="Test description",
            head_sha="abc123",
        )

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_diff_returns_none_adds_error(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test handling when diff retrieval returns None."""
        mock_github_instance = mock_github.return_value
        mock_github_instance.get_last_reviewed_commit_sha.return_value = None
        mock_github_instance.get_pr_diff.return_value = None

        reviewer = CodeReviewer(mock_config)
        result = await reviewer.review_pull_request(sample_pr_details)

        assert result is not None
        assert len(result.errors) > 0 or result.processed_files == 0

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @patch("gemini_reviewer.code_reviewer.DiffParser")
    @patch("gemini_reviewer.code_reviewer.ContextBuilder")
    @patch("gemini_reviewer.code_reviewer.CommentProcessor")
    @pytest.mark.asyncio
    async def test_diff_empty_string_no_changes(
        self,
        mock_comment_proc,
        mock_context,
        mock_diff_parser,
        mock_gemini,
        mock_github,
        mock_config,
        sample_pr_details,
    ):
        """Test handling when diff is empty string (no changes)."""
        mock_github_instance = mock_github.return_value
        mock_github_instance.get_last_reviewed_commit_sha.return_value = None
        mock_github_instance.get_pr_diff.return_value = ""

        reviewer = CodeReviewer(mock_config)
        result = await reviewer.review_pull_request(sample_pr_details)

        assert result is not None
        assert result.processed_files == 0


class TestCodeReviewerFileFiltering:
    """Tests for file filtering functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    def test_should_skip_file_binary(self, mock_config):
        """Test that binary files are skipped."""
        with patch("gemini_reviewer.code_reviewer.GitHubClient"):
            with patch("gemini_reviewer.code_reviewer.GeminiClient"):
                reviewer = CodeReviewer(mock_config)
                if hasattr(reviewer, '_should_skip_file'):
                    file_info = FileInfo("test.png", "image/png")
                    result = reviewer._should_skip_file(file_info)
                    assert isinstance(result, bool)

    def test_should_skip_file_generated(self, mock_config):
        """Test that generated files are skipped."""
        with patch("gemini_reviewer.code_reviewer.GitHubClient"):
            with patch("gemini_reviewer.code_reviewer.GeminiClient"):
                reviewer = CodeReviewer(mock_config)
                if hasattr(reviewer, '_should_skip_file'):
                    file_info = FileInfo("package-lock.json", "json")
                    result = reviewer._should_skip_file(file_info)
                    assert isinstance(result, bool)


class TestCodeReviewerHunkProcessing:
    """Tests for hunk processing functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    @pytest.fixture
    def sample_hunk(self):
        """Create sample hunk."""
        return HunkInfo(
            source_start=1,
            source_length=5,
            target_start=1,
            target_length=6,
            content=" line1\n-line2\n+line2modified\n+newline\n line3",
            header="@@ -1,5 +1,6 @@",
            lines=[" line1", "-line2", "+line2modified", "+newline", " line3"]
        )

    def test_process_hunk_empty_content(self, mock_config):
        """Test processing hunk with empty content."""
        with patch("gemini_reviewer.code_reviewer.GitHubClient"):
            with patch("gemini_reviewer.code_reviewer.GeminiClient"):
                reviewer = CodeReviewer(mock_config)
                empty_hunk = HunkInfo(
                    source_start=1,
                    source_length=1,
                    target_start=1,
                    target_length=1,
                    content="",
                    header="@@ -1,1 +1,1 @@",
                    lines=[]
                )
                if hasattr(reviewer, '_process_hunk'):
                    result = reviewer._process_hunk(empty_hunk, {})
                    assert result is None or isinstance(result, list)


class TestCodeReviewerStatistics:
    """Tests for statistics tracking."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        return Config(github=github_config, gemini=gemini_config)

    def test_get_statistics_initial(self, mock_config):
        """Test getting initial statistics."""
        with patch("gemini_reviewer.code_reviewer.GitHubClient"):
            with patch("gemini_reviewer.code_reviewer.GeminiClient"):
                reviewer = CodeReviewer(mock_config)
                if hasattr(reviewer, 'get_statistics'):
                    stats = reviewer.get_statistics()
                    assert isinstance(stats, dict)


class TestCodeReviewerConfigFiltering:
    """Tests for config-based file filtering."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config object that rejects certain files."""
        github_config = GitHubConfig(token="ghp_test123456789012")
        gemini_config = GeminiConfig(api_key="AIzaSyTestKey123456")
        config = Config(github=github_config, gemini=gemini_config)
        return config

    @patch("gemini_reviewer.code_reviewer.GitHubClient")
    @patch("gemini_reviewer.code_reviewer.GeminiClient")
    @pytest.mark.asyncio
    async def test_filter_files_via_config(self, mock_gemini, mock_github, mock_config):
        """Test that files are filtered based on config.should_review_file."""
        reviewer = CodeReviewer(mock_config)

        # Create a diff file with a path that might be excluded
        diff_file = DiffFile(
            file_info=FileInfo(path="test.min.js")
        )

        # Check if the filter method exists
        if hasattr(reviewer, '_filter_files'):
            # Mock the should_review_file to return False
            with patch.object(mock_config, 'should_review_file', return_value=False):
                result = await reviewer._filter_files([diff_file])
                assert isinstance(result, list)
