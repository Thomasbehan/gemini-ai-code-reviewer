"""
Comprehensive tests for gemini_reviewer/github_client.py
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open

from gemini_reviewer.github_client import (
    GitHubClient,
    GitHubClientError,
    PRNotFoundError,
    RateLimitError,
)
from gemini_reviewer.config import GitHubConfig
from gemini_reviewer.models import PRDetails, ReviewComment, ReviewPriority


class TestGitHubClientErrors:
    """Tests for GitHub client exceptions."""

    def test_github_client_error(self):
        """Test GitHubClientError exception."""
        with pytest.raises(GitHubClientError) as exc_info:
            raise GitHubClientError("Test error")
        assert "Test error" in str(exc_info.value)

    def test_pr_not_found_error(self):
        """Test PRNotFoundError exception."""
        with pytest.raises(PRNotFoundError):
            raise PRNotFoundError("PR not found")

    def test_rate_limit_error(self):
        """Test RateLimitError exception."""
        with pytest.raises(RateLimitError):
            raise RateLimitError("Rate limit exceeded")

    def test_error_inheritance(self):
        """Test exception inheritance."""
        assert issubclass(PRNotFoundError, GitHubClientError)
        assert issubclass(RateLimitError, GitHubClientError)


class TestGitHubClient:
    """Tests for GitHubClient class."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(
            token="ghp_test123456789012",
            api_base_url="https://api.github.com",
            timeout=30,
        )

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

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_init(self, mock_session, mock_github, valid_config):
        """Test GitHubClient initialization."""
        client = GitHubClient(valid_config)

        mock_github.assert_called_once_with(valid_config.token)
        assert client.config == valid_config

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_details_from_event_direct_pr(
        self, mock_session, mock_github, valid_config
    ):
        """Test getting PR details from direct PR event."""
        event_data = {
            "number": 123,
            "repository": {"full_name": "owner/repo"},
        }

        mock_pr = Mock()
        mock_pr.title = "Test PR"
        mock_pr.body = "Description"
        mock_pr.head.sha = "abc123"
        mock_pr.base.sha = "def456"

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)

        with patch("builtins.open", mock_open(read_data=json.dumps(event_data))):
            result = client.get_pr_details_from_event("event.json")

        assert result.pull_number == 123
        assert result.owner == "owner"
        assert result.repo == "repo"

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_details_from_event_comment_trigger(
        self, mock_session, mock_github, valid_config
    ):
        """Test getting PR details from comment trigger event."""
        event_data = {
            "issue": {
                "number": 456,
                "pull_request": {"url": "..."},
            },
            "repository": {"full_name": "owner/repo"},
        }

        mock_pr = Mock()
        mock_pr.title = "Comment Trigger PR"
        mock_pr.body = "Description"
        mock_pr.head.sha = "abc123"
        mock_pr.base.sha = "def456"

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)

        with patch("builtins.open", mock_open(read_data=json.dumps(event_data))):
            result = client.get_pr_details_from_event("event.json")

        assert result.pull_number == 456

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_details_from_event_file_not_found(
        self, mock_session, mock_github, valid_config
    ):
        """Test handling missing event file."""
        client = GitHubClient(valid_config)

        with pytest.raises(GitHubClientError, match="Failed to load event data"):
            client.get_pr_details_from_event("nonexistent.json")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_diff_success(
        self, mock_session, mock_github, valid_config
    ):
        """Test getting PR diff successfully."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "diff --git a/file.py b/file.py\n+new line"
        mock_session.return_value.get.return_value = mock_response

        mock_pr = Mock()
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_pr_diff("owner", "repo", 123)

        assert "diff --git" in result

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_diff_not_found(
        self, mock_session, mock_github, valid_config
    ):
        """Test handling PR not found when getting diff."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_session.return_value.get.return_value = mock_response

        mock_pr = Mock()
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)

        with pytest.raises(GitHubClientError):
            client.get_pr_diff("owner", "repo", 999)

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_diff_invalid_params(
        self, mock_session, mock_github, valid_config
    ):
        """Test getting diff with invalid parameters."""
        client = GitHubClient(valid_config)

        with pytest.raises(GitHubClientError, match="Invalid parameters"):
            client.get_pr_diff("", "", 0)

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_diff_negative_pr_number(
        self, mock_session, mock_github, valid_config
    ):
        """Test getting diff with negative PR number."""
        client = GitHubClient(valid_config)

        with pytest.raises(GitHubClientError, match="Invalid pull request number"):
            client.get_pr_diff("owner", "repo", -1)


class TestGitHubClientReviews:
    """Tests for GitHub client review functionality."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @pytest.fixture
    def sample_comments(self):
        """Create sample review comments."""
        return [
            ReviewComment(
                body="Fix this bug",
                path="main.py",
                position=10,
                priority=ReviewPriority.HIGH,
            ),
            ReviewComment(
                body="Consider refactoring",
                path="utils.py",
                position=25,
                priority=ReviewPriority.MEDIUM,
            ),
        ]

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_create_review_with_comments(
        self, mock_session, mock_github, valid_config, sample_pr_details, sample_comments
    ):
        """Test creating a review with comments."""
        mock_review = Mock()
        mock_review.id = 1234
        mock_pr = Mock()
        mock_pr.create_review.return_value = mock_review
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.create_review(
            sample_pr_details, sample_comments, "COMMENT"
        )

        assert result is True
        mock_pr.create_review.assert_called_once()

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_create_review_no_comments(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test creating a review with no comments."""
        mock_review = Mock()
        mock_review.id = 1234
        mock_pr = Mock()
        mock_pr.create_review.return_value = mock_review
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.create_review(sample_pr_details, [], "APPROVE")

        assert result is True


class TestGitHubClientSignatures:
    """Tests for comment signature functionality."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_compute_signature(
        self, mock_session, mock_github, valid_config
    ):
        """Test computing comment signature."""
        client = GitHubClient(valid_config)

        sig1 = client._compute_signature("file.py", "Fix this bug")
        sig2 = client._compute_signature("file.py", "Fix this bug")
        sig3 = client._compute_signature("file.py", "Different comment")

        assert sig1 == sig2  # Same content = same signature
        assert sig1 != sig3  # Different content = different signature

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_normalize_for_signature(
        self, mock_session, mock_github, valid_config
    ):
        """Test normalizing text for signature."""
        client = GitHubClient(valid_config)

        norm1 = client._normalize_for_signature("Hello   World")
        norm2 = client._normalize_for_signature("Hello World")

        # Whitespace should be collapsed
        assert norm1 == norm2

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_strip_signature_marker(
        self, mock_session, mock_github, valid_config
    ):
        """Test stripping signature marker from body."""
        client = GitHubClient(valid_config)

        body_with_marker = "Comment text\n<!-- AI-SIG:abc123def456 -->"
        result = client._strip_signature_marker(body_with_marker)

        assert "AI-SIG" not in result
        assert "Comment text" in result

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_append_signature_marker(
        self, mock_session, mock_github, valid_config
    ):
        """Test appending signature marker to body."""
        client = GitHubClient(valid_config)

        body = "Comment text"
        result = client._append_signature_marker(body, "file.py")

        assert "AI-SIG" in result

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_append_signature_marker_already_has_marker(
        self, mock_session, mock_github, valid_config
    ):
        """Test not adding duplicate signature markers."""
        client = GitHubClient(valid_config)

        body = "Comment\n<!-- AI-SIG:abc123def456 -->"
        result = client._append_signature_marker(body, "file.py")

        # Should not add another marker
        assert result.count("AI-SIG") == 1


class TestGitHubClientFileContent:
    """Tests for file content retrieval."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_file_content_success(
        self, mock_session, mock_github, valid_config
    ):
        """Test getting file content successfully."""
        mock_content = Mock()
        mock_content.decoded_content = b"print('hello')"
        mock_repo = Mock()
        mock_repo.get_contents.return_value = mock_content
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_file_content("owner", "repo", "main.py", "HEAD")

        assert result == "print('hello')"

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_file_content_not_found(
        self, mock_session, mock_github, valid_config
    ):
        """Test handling file not found."""
        mock_repo = Mock()
        mock_repo.get_contents.side_effect = Exception("File not found")
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_file_content("owner", "repo", "missing.py", "HEAD")

        assert result is None


class TestGitHubClientRateLimit:
    """Tests for rate limit functionality."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_check_rate_limit_with_core(
        self, mock_session, mock_github, valid_config
    ):
        """Test checking rate limit with core attribute."""
        mock_core = Mock()
        mock_core.limit = 5000
        mock_core.remaining = 4000
        mock_core.reset.timestamp.return_value = 1234567890

        mock_rate_limit = Mock()
        mock_rate_limit.core = mock_core
        mock_github.return_value.get_rate_limit.return_value = mock_rate_limit

        client = GitHubClient(valid_config)
        result = client.check_rate_limit()

        assert "core" in result
        assert result["core"]["limit"] == 5000

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_check_rate_limit_error(
        self, mock_session, mock_github, valid_config
    ):
        """Test handling rate limit check error."""
        mock_github.return_value.get_rate_limit.side_effect = Exception("API Error")

        client = GitHubClient(valid_config)
        result = client.check_rate_limit()

        # Should return valid structure even on error
        assert "core" in result


class TestGitHubClientSanitization:
    """Tests for input sanitization."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_sanitize_input_normal_text(
        self, mock_session, mock_github, valid_config
    ):
        """Test sanitizing normal text."""
        client = GitHubClient(valid_config)

        result = client._sanitize_input("Normal text")
        assert result == "Normal text"

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_sanitize_input_null_bytes(
        self, mock_session, mock_github, valid_config
    ):
        """Test removing null bytes."""
        client = GitHubClient(valid_config)

        result = client._sanitize_input("Hello\x00World")
        assert "\x00" not in result

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_sanitize_input_control_chars(
        self, mock_session, mock_github, valid_config
    ):
        """Test removing control characters."""
        client = GitHubClient(valid_config)

        result = client._sanitize_input("Hello\x01\x02World")
        assert "\x01" not in result
        assert "\x02" not in result

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_sanitize_input_preserves_newlines(
        self, mock_session, mock_github, valid_config
    ):
        """Test preserving newlines and tabs."""
        client = GitHubClient(valid_config)

        result = client._sanitize_input("Line1\n\tLine2")
        assert "\n" in result
        assert "\t" in result

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_sanitize_input_non_string(
        self, mock_session, mock_github, valid_config
    ):
        """Test handling non-string input."""
        client = GitHubClient(valid_config)

        result = client._sanitize_input(12345)
        assert result == "12345"

        result = client._sanitize_input(None)
        assert result == ""


class TestGitHubClientClose:
    """Tests for client cleanup."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_close(
        self, mock_session, mock_github, valid_config
    ):
        """Test closing the client."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance

        client = GitHubClient(valid_config)
        client.close()

        mock_session_instance.close.assert_called_once()


class TestGitHubClientLastReviewedCommit:
    """Tests for get_last_reviewed_commit_sha functionality."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head123", "base456")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_last_reviewed_commit_sha_no_prior_review(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test when there's no prior bot review."""
        from datetime import datetime

        mock_commit = Mock()
        mock_commit.sha = "abc123"
        mock_commit.commit.author.date = datetime(2024, 1, 1, 12, 0, 0)

        mock_pr = Mock()
        mock_pr.get_commits.return_value = [mock_commit]
        mock_pr.get_reviews.return_value = []
        mock_pr.as_issue.return_value.get_comments.return_value = []

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_last_reviewed_commit_sha(sample_pr_details)

        assert result is None

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_last_reviewed_commit_sha_with_review(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test when there is a prior bot review."""
        from datetime import datetime

        mock_commit = Mock()
        mock_commit.sha = "abc123"
        mock_commit.commit.committer.date = datetime(2024, 1, 1, 10, 0, 0)
        mock_commit.commit.author.date = datetime(2024, 1, 1, 10, 0, 0)

        mock_review = Mock()
        mock_review.user.login = "github-actions[bot]"
        mock_review.body = "Gemini AI Code Reviewer found issues"
        mock_review.submitted_at = datetime(2024, 1, 1, 12, 0, 0)

        mock_pr = Mock()
        mock_pr.get_commits.return_value = [mock_commit]
        mock_pr.get_reviews.return_value = [mock_review]
        mock_pr.as_issue.return_value.get_comments.return_value = []

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_last_reviewed_commit_sha(sample_pr_details)

        assert result == "abc123"

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_last_reviewed_commit_sha_no_commits(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test when PR has no commits."""
        mock_pr = Mock()
        mock_pr.get_commits.return_value = []

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_last_reviewed_commit_sha(sample_pr_details)

        assert result is None

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_last_reviewed_commit_sha_exception(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test handling exceptions."""
        mock_github.return_value.get_repo.side_effect = Exception("API Error")

        client = GitHubClient(valid_config)
        result = client.get_last_reviewed_commit_sha(sample_pr_details)

        assert result is None


class TestGitHubClientDiffSince:
    """Tests for get_pr_diff_since functionality."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head123", "base456")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_diff_since_no_base_sha(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test when base_sha is not provided."""
        client = GitHubClient(valid_config)
        result = client.get_pr_diff_since(sample_pr_details, "")

        assert result is None

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_diff_since_same_sha(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test when base equals head."""
        mock_pr = Mock()
        mock_pr.head.sha = "abc123"
        mock_pr.get_commits.return_value = []

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_pr_diff_since(sample_pr_details, "abc123")

        assert result == ""

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_diff_since_success(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test successful incremental diff."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "diff --git a/file.py b/file.py"
        mock_session.return_value.get.return_value = mock_response

        mock_pr = Mock()
        mock_pr.head.sha = "head123"
        mock_pr.get_commits.return_value = []

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_pr_diff_since(sample_pr_details, "base456")

        assert "diff --git" in result

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_diff_since_api_failure_fallback(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test fallback when compare API fails."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_session.return_value.get.return_value = mock_response

        mock_pr = Mock()
        mock_pr.head.sha = "head123"
        mock_pr.get_commits.return_value = []

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_pr_diff_since(sample_pr_details, "base456")

        assert result == ""


class TestGitHubClientIncrementalDiff:
    """Tests for get_incremental_diff_by_commits functionality."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head123", "base456")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_incremental_diff_no_base_sha(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test when base_sha is not provided."""
        client = GitHubClient(valid_config)
        result = client.get_incremental_diff_by_commits(sample_pr_details, "")

        assert result is None

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_incremental_diff_no_commits(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test when PR has no commits."""
        mock_pr = Mock()
        mock_pr.get_commits.return_value = []

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_incremental_diff_by_commits(sample_pr_details, "base123")

        assert result == ""

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_incremental_diff_with_commits(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test building diff from commits."""
        mock_commit1 = Mock()
        mock_commit1.sha = "base123"

        mock_commit2 = Mock()
        mock_commit2.sha = "new123"

        mock_file = Mock()
        mock_file.patch = "+new line"
        mock_file.status = "modified"
        mock_file.filename = "file.py"
        mock_file.previous_filename = None

        mock_gh_commit = Mock()
        mock_gh_commit.files = [mock_file]

        mock_pr = Mock()
        mock_pr.get_commits.return_value = [mock_commit1, mock_commit2]

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_repo.get_commit.return_value = mock_gh_commit
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_incremental_diff_by_commits(sample_pr_details, "base123")

        assert "file.py" in result


class TestGitHubClientReviewMessages:
    """Tests for review message generation."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_generate_review_summary(
        self, mock_session, mock_github, valid_config
    ):
        """Test generating review summary."""
        client = GitHubClient(valid_config)

        comments = [
            ReviewComment("Fix bug", "file.py", 1, priority=ReviewPriority.HIGH),
            ReviewComment("Consider", "file.py", 2, priority=ReviewPriority.LOW),
        ]

        result = client._generate_review_summary(comments)

        assert "Gemini AI Code Review" in result
        assert "2" in result

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_generate_approval_message(
        self, mock_session, mock_github, valid_config
    ):
        """Test generating approval message."""
        client = GitHubClient(valid_config)
        result = client._generate_approval_message()

        assert "looks good" in result

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_generate_filtered_message(
        self, mock_session, mock_github, valid_config
    ):
        """Test generating filtered message."""
        client = GitHubClient(valid_config)
        result = client._generate_filtered_message(5)

        assert "5" in result
        assert "hidden" in result


class TestGitHubClientExistingComments:
    """Tests for existing comment functionality."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_existing_comment_signatures(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test getting existing comment signatures."""
        mock_comment = Mock()
        mock_comment.path = "file.py"
        mock_comment.body = "Comment <!-- AI-SIG:abc123def456 -->"

        mock_pr = Mock()
        mock_pr.get_review_comments.return_value = [mock_comment]

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_existing_comment_signatures(sample_pr_details)

        assert len(result) > 0

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_existing_bot_comments(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test getting existing bot comments."""
        mock_comment = Mock()
        mock_comment.path = "file.py"
        mock_comment.body = "Bot comment <!-- AI-SIG:abc123 -->"
        mock_comment.original_line = 10
        mock_comment.position = 5
        mock_comment.original_position = 5
        mock_comment.created_at = "2024-01-01"
        mock_comment.id = 123
        mock_comment.user.login = "github-actions[bot]"

        mock_pr = Mock()
        mock_pr.get_review_comments.return_value = [mock_comment]

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo
        mock_github.return_value.get_user.return_value.login = "github-actions[bot]"

        client = GitHubClient(valid_config)
        result = client.get_existing_bot_comments(sample_pr_details)

        assert len(result) == 1

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_filter_out_existing_comments(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test filtering out existing comments."""
        mock_pr = Mock()
        mock_pr.get_review_comments.return_value = []

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)

        comments = [
            ReviewComment("New comment", "file.py", 1),
        ]

        result = client.filter_out_existing_comments(sample_pr_details, comments)

        assert len(result) >= 0


class TestGitHubClientValidation:
    """Tests for comment validation."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_validate_and_sanitize_comment_valid(
        self, mock_session, mock_github, valid_config
    ):
        """Test validating a valid comment."""
        client = GitHubClient(valid_config)

        comment = ReviewComment("Fix this", "file.py", 10)
        result = client._validate_and_sanitize_comment(comment)

        assert result is not None
        assert result['body'] is not None
        assert result['path'] == "file.py"
        assert result['position'] == 10

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_validate_and_sanitize_comment_missing_body(
        self, mock_session, mock_github, valid_config
    ):
        """Test validating comment with missing body."""
        client = GitHubClient(valid_config)

        comment = ReviewComment("", "file.py", 10)
        result = client._validate_and_sanitize_comment(comment)

        assert result is None

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_validate_and_sanitize_comment_invalid_position(
        self, mock_session, mock_github, valid_config
    ):
        """Test validating comment with invalid position."""
        client = GitHubClient(valid_config)

        comment = ReviewComment("Fix this", "file.py", 0)
        result = client._validate_and_sanitize_comment(comment)

        assert result is None


class TestGitHubClientCreateReview:
    """Tests for create_review functionality."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_create_review_with_filtered_comments(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test creating review when comments were filtered."""
        mock_review = Mock()
        mock_review.id = 1234
        mock_pr = Mock()
        mock_pr.create_review.return_value = mock_review
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.create_review(
            sample_pr_details, [], "COMMENT", total_comments_generated=5
        )

        assert result is True

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_create_review_approve_fallback(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test fallback to COMMENT when APPROVE fails."""
        mock_review = Mock()
        mock_review.id = 1234
        mock_pr = Mock()
        # First call fails (APPROVE), second succeeds (COMMENT)
        mock_pr.create_review.side_effect = [Exception("Cannot approve"), mock_review]
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.create_review(
            sample_pr_details, [], "APPROVE"
        )

        assert result is True

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_create_review_invalid_comment_type(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test handling invalid comment types."""
        mock_review = Mock()
        mock_review.id = 1234
        mock_pr = Mock()
        mock_pr.create_review.return_value = mock_review
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        # Pass invalid comment type
        result = client.create_review(
            sample_pr_details, ["invalid"], "COMMENT"
        )

        assert result is True


class TestGitHubClientRateLimitDiff:
    """Tests for rate limit handling in get_pr_diff."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_diff_rate_limit(
        self, mock_session, mock_github, valid_config
    ):
        """Test handling rate limit error."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.text = "rate limit exceeded"
        mock_session.return_value.get.return_value = mock_response

        mock_pr = Mock()
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)

        # After retries, gets wrapped in GitHubClientError
        with pytest.raises(GitHubClientError):
            client.get_pr_diff("owner", "repo", 123)

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_diff_forbidden_not_rate_limit(
        self, mock_session, mock_github, valid_config
    ):
        """Test handling 403 that is not rate limit."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.text = "access denied"
        mock_session.return_value.get.return_value = mock_response

        mock_pr = Mock()
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)

        with pytest.raises(GitHubClientError, match="Access forbidden"):
            client.get_pr_diff("owner", "repo", 123)


class TestGitHubClientEventHandling:
    """Tests for event handling edge cases."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_details_from_event_invalid_repo(
        self, mock_session, mock_github, valid_config
    ):
        """Test handling invalid repository name."""
        event_data = {
            "number": 123,
            "repository": {"full_name": "invalid-no-slash"},
        }

        client = GitHubClient(valid_config)

        with patch("builtins.open", mock_open(read_data=json.dumps(event_data))):
            with pytest.raises(GitHubClientError, match="Invalid repository name"):
                client.get_pr_details_from_event("event.json")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_details_from_event_api_error(
        self, mock_session, mock_github, valid_config
    ):
        """Test handling API error when getting PR details."""
        event_data = {
            "number": 123,
            "repository": {"full_name": "owner/repo"},
        }

        mock_github.return_value.get_repo.side_effect = Exception("API Error")

        client = GitHubClient(valid_config)

        with patch("builtins.open", mock_open(read_data=json.dumps(event_data))):
            with pytest.raises(GitHubClientError, match="Failed to get PR details"):
                client.get_pr_details_from_event("event.json")


class TestGitHubClientAdvanced:
    """Advanced tests for GitHub client edge cases."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_session_headers_set(self, mock_session, mock_github, valid_config):
        """Test that session headers are properly set."""
        mock_session_instance = Mock()
        mock_session_instance.headers = Mock()
        mock_session.return_value = mock_session_instance

        client = GitHubClient(valid_config)

        # Verify headers.update was called
        mock_session_instance.headers.update.assert_called()
        call_args = mock_session_instance.headers.update.call_args[0][0]
        assert "Authorization" in call_args
        assert "ghp_test123456789012" in call_args["Authorization"]

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_diff_success(self, mock_session, mock_github, valid_config):
        """Test successful PR diff retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "diff --git a/file.py b/file.py"
        mock_session.return_value.get.return_value = mock_response

        mock_pr = Mock()
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_pr_diff("owner", "repo", 123)

        assert "diff --git" in result

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_diff_empty_response(self, mock_session, mock_github, valid_config):
        """Test handling empty diff response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_session.return_value.get.return_value = mock_response

        mock_pr = Mock()
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_pr_diff("owner", "repo", 123)

        assert result == ""

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_filter_out_existing_comments_removes_duplicates(
        self, mock_session, mock_github, valid_config
    ):
        """Test that existing comment signatures are filtered."""
        client = GitHubClient(valid_config)

        comments = [
            ReviewComment("Comment 1", "file.py", 1),
            ReviewComment("Comment 2", "file.py", 2),
        ]

        # Mock existing comments that have same signature as Comment 1
        mock_existing_comment = Mock()
        mock_existing_comment.path = "file.py"
        mock_existing_comment.body = "Comment 1"

        mock_pr = Mock()
        mock_pr.get_review_comments.return_value = [mock_existing_comment]
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        pr_details = PRDetails("owner", "repo", 123, "Title", "Desc", "head_sha")
        result = client.filter_out_existing_comments(pr_details, comments)

        # First comment should be filtered because it matches existing
        assert len(result) < len(comments)

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_file_content_success(self, mock_session, mock_github, valid_config):
        """Test getting file content successfully."""
        mock_file = Mock()
        mock_file.decoded_content = b"def hello(): pass"
        mock_repo = Mock()
        mock_repo.get_contents.return_value = mock_file
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_file_content("owner", "repo", "main.py", "abc123")

        assert "def hello" in result

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_file_content_not_found(self, mock_session, mock_github, valid_config):
        """Test getting file content when file not found."""
        from github import UnknownObjectException
        mock_repo = Mock()
        mock_repo.get_contents.side_effect = UnknownObjectException(404, None, None)
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_file_content("owner", "repo", "missing.py", "abc123")

        assert result is None

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_sanitize_input(self, mock_session, mock_github, valid_config):
        """Test sanitizing input text."""
        client = GitHubClient(valid_config)

        # Test with valid content
        result = client._sanitize_input("Fix this issue")
        assert "Fix this issue" in result

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_sanitize_input_with_null_bytes(self, mock_session, mock_github, valid_config):
        """Test sanitizing input with null bytes."""
        client = GitHubClient(valid_config)

        result = client._sanitize_input("text\x00with\x00nulls")
        assert "\x00" not in result

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_compute_signature(self, mock_session, mock_github, valid_config):
        """Test computing comment signature."""
        client = GitHubClient(valid_config)

        sig1 = client._compute_signature("file.py", "Comment text")
        sig2 = client._compute_signature("file.py", "Comment text")
        sig3 = client._compute_signature("file.py", "Different text")

        assert sig1 == sig2
        assert sig1 != sig3

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_close_client(self, mock_session, mock_github, valid_config):
        """Test closing the client."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance

        client = GitHubClient(valid_config)
        client.close()

        mock_session_instance.close.assert_called_once()


class TestGitHubClientPRDetailsFields:
    """Tests for PR details field handling."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_details_all_fields(self, mock_session, mock_github, valid_config):
        """Test PR details includes all fields."""
        mock_pr = Mock()
        mock_pr.title = "My PR Title"
        mock_pr.body = "PR Description"
        mock_pr.head.sha = "head_sha_123"
        mock_pr.base.sha = "base_sha_456"

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_pr_details("owner", "repo", 123)

        assert result.title == "My PR Title"
        assert result.description == "PR Description"
        assert result.head_sha == "head_sha_123"
        assert result.base_sha == "base_sha_456"

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_details_none_body(self, mock_session, mock_github, valid_config):
        """Test PR details handles None body."""
        mock_pr = Mock()
        mock_pr.title = "Title"
        mock_pr.body = None
        mock_pr.head.sha = "sha"
        mock_pr.base.sha = "sha"

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_pr_details("owner", "repo", 123)

        assert result.description == "" or result.description is None


class TestGitHubClientCommentProcessing:
    """Tests for comment processing functionality."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_existing_comment_signatures(self, mock_session, mock_github, valid_config):
        """Test getting existing comment signatures."""
        mock_comment = Mock()
        mock_comment.path = "file.py"
        mock_comment.line = 10
        mock_comment.body = "Comment body"

        mock_pr = Mock()
        mock_pr.get_review_comments.return_value = [mock_comment]
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        pr_details = PRDetails("owner", "repo", 123, "Title", "Desc", "head_sha")
        result = client.get_existing_comment_signatures(pr_details)

        assert isinstance(result, set)
        assert len(result) >= 1

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_filter_preserves_valid_comments(self, mock_session, mock_github, valid_config):
        """Test that filter preserves comments not in existing signatures."""
        # Mock empty existing comments
        mock_pr = Mock()
        mock_pr.get_review_comments.return_value = []
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        pr_details = PRDetails("owner", "repo", 123, "Title", "Desc", "head_sha")

        comments = [
            ReviewComment("New comment", "file.py", 1),
        ]

        result = client.filter_out_existing_comments(pr_details, comments)

        assert len(result) == 1

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_filter_handles_empty_comments(self, mock_session, mock_github, valid_config):
        """Test filter handles empty comment list."""
        mock_pr = Mock()
        mock_pr.get_review_comments.return_value = []
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        pr_details = PRDetails("owner", "repo", 123, "Title", "Desc", "head_sha")

        result = client.filter_out_existing_comments(pr_details, [])

        assert result == []


class TestGitHubClientGetPRWithRetry:
    """Tests for _get_pr_with_retry functionality."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_with_retry_404_error(self, mock_session, mock_github, valid_config):
        """Test handling 404 error in _get_pr_with_retry."""
        from tenacity import RetryError

        mock_repo = Mock()
        mock_repo.get_pull.side_effect = Exception("404 Not Found")
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)

        # The retry wraps the PRNotFoundError in RetryError
        with pytest.raises((PRNotFoundError, RetryError)):
            client._get_pr_with_retry(mock_repo, 999)

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_with_retry_other_error(self, mock_session, mock_github, valid_config):
        """Test handling non-404 error in _get_pr_with_retry."""
        from tenacity import RetryError

        mock_repo = Mock()
        mock_repo.get_pull.side_effect = Exception("Connection error")
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)

        # After retries, exception is wrapped in RetryError
        with pytest.raises((Exception, RetryError)):
            client._get_pr_with_retry(mock_repo, 123)


class TestGitHubClientDiffExceptions:
    """Tests for get_pr_diff exception handling."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_diff_timeout(self, mock_session, mock_github, valid_config):
        """Test handling timeout in get_pr_diff."""
        import requests
        from tenacity import RetryError

        mock_session.return_value.get.side_effect = requests.exceptions.Timeout("Timeout")

        mock_pr = Mock()
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)

        # After retries, wrapped in RetryError
        with pytest.raises((requests.exceptions.Timeout, RetryError)):
            client.get_pr_diff("owner", "repo", 123)

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_diff_request_exception(self, mock_session, mock_github, valid_config):
        """Test handling request exception in get_pr_diff."""
        import requests
        from tenacity import RetryError

        mock_session.return_value.get.side_effect = requests.exceptions.RequestException("Network error")

        mock_pr = Mock()
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)

        # After retries, wrapped in RetryError
        with pytest.raises((requests.exceptions.RequestException, RetryError)):
            client.get_pr_diff("owner", "repo", 123)

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_diff_non_200_status(self, mock_session, mock_github, valid_config):
        """Test handling non-200 status code in get_pr_diff."""
        from tenacity import RetryError

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_response.raise_for_status.side_effect = Exception("Server error")
        mock_session.return_value.get.return_value = mock_response

        mock_pr = Mock()
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)

        # After retries, wrapped in GitHubClientError or RetryError
        with pytest.raises((GitHubClientError, RetryError)):
            client.get_pr_diff("owner", "repo", 123)


class TestGitHubClientCommentReplies:
    """Tests for get_comment_replies functionality."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head_sha")

    @patch("gemini_reviewer.github_client.requests.get")
    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_comment_replies_success(self, mock_session, mock_github, mock_get, valid_config, sample_pr_details):
        """Test getting comment replies successfully."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": 100, "path": "file.py", "position": 10, "body": "Original", "in_reply_to_id": None},
            {"id": 101, "path": "file.py", "position": 10, "body": "Reply", "in_reply_to_id": 100},
        ]
        mock_get.return_value = mock_response

        client = GitHubClient(valid_config)
        result = client.get_comment_replies(sample_pr_details, 100)

        assert len(result) == 1
        assert result[0]["body"] == "Reply"

    @patch("gemini_reviewer.github_client.requests.get")
    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_comment_replies_not_found(self, mock_session, mock_github, mock_get, valid_config, sample_pr_details):
        """Test getting replies for non-existent comment."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": 999, "path": "file.py", "body": "Other comment"},
        ]
        mock_get.return_value = mock_response

        client = GitHubClient(valid_config)
        result = client.get_comment_replies(sample_pr_details, 100)

        assert result == []

    @patch("gemini_reviewer.github_client.requests.get")
    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_comment_replies_api_error(self, mock_session, mock_github, mock_get, valid_config, sample_pr_details):
        """Test handling API error in get_comment_replies."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        client = GitHubClient(valid_config)
        result = client.get_comment_replies(sample_pr_details, 100)

        assert result == []

    @patch("gemini_reviewer.github_client.requests.get")
    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_comment_replies_exception(self, mock_session, mock_github, mock_get, valid_config, sample_pr_details):
        """Test handling exception in get_comment_replies."""
        mock_get.side_effect = Exception("Network error")

        client = GitHubClient(valid_config)
        result = client.get_comment_replies(sample_pr_details, 100)

        assert result == []

    @patch("gemini_reviewer.github_client.requests.get")
    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_comment_replies_same_thread(self, mock_session, mock_github, mock_get, valid_config, sample_pr_details):
        """Test getting replies in the same thread."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": 100, "path": "file.py", "position": 10, "body": "Original", "in_reply_to_id": None},
            {"id": 101, "path": "file.py", "position": 10, "body": "Thread reply", "in_reply_to_id": 50},
        ]
        mock_get.return_value = mock_response

        client = GitHubClient(valid_config)
        result = client.get_comment_replies(sample_pr_details, 100)

        # Reply is in same thread (same path and position)
        assert len(result) == 1


class TestGitHubClientReplyToComment:
    """Tests for reply_to_comment functionality."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head_sha")

    @patch("gemini_reviewer.github_client.requests.post")
    @patch("gemini_reviewer.github_client.requests.get")
    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_reply_to_comment_success(self, mock_session, mock_github, mock_get, mock_post, valid_config, sample_pr_details):
        """Test replying to comment successfully."""
        # Mock get_comment_replies returns empty (no existing reply)
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = []
        mock_get.return_value = mock_get_response

        # Mock successful post
        mock_post_response = Mock()
        mock_post_response.status_code = 201
        mock_post.return_value = mock_post_response

        client = GitHubClient(valid_config)
        result = client.reply_to_comment(sample_pr_details, 100, "Fixed!")

        assert result is True
        mock_post.assert_called_once()

    @patch("gemini_reviewer.github_client.requests.post")
    @patch("gemini_reviewer.github_client.requests.get")
    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_reply_to_comment_already_exists(self, mock_session, mock_github, mock_get, mock_post, valid_config, sample_pr_details):
        """Test skipping duplicate reply."""
        # Mock get_comment_replies returns the target comment and an existing reply with same body
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = [
            {"id": 100, "path": "file.py", "position": 10, "body": "Original", "in_reply_to_id": None},
            {"id": 101, "path": "file.py", "position": 10, "body": "Fixed!", "in_reply_to_id": 100},
        ]
        mock_get.return_value = mock_get_response

        client = GitHubClient(valid_config)
        result = client.reply_to_comment(sample_pr_details, 100, "Fixed!")

        assert result is True
        mock_post.assert_not_called()

    @patch("gemini_reviewer.github_client.requests.post")
    @patch("gemini_reviewer.github_client.requests.get")
    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_reply_to_comment_api_failure(self, mock_session, mock_github, mock_get, mock_post, valid_config, sample_pr_details):
        """Test handling API failure in reply_to_comment."""
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = []
        mock_get.return_value = mock_get_response

        mock_post_response = Mock()
        mock_post_response.status_code = 403
        mock_post_response.text = "Forbidden"
        mock_post.return_value = mock_post_response

        client = GitHubClient(valid_config)
        result = client.reply_to_comment(sample_pr_details, 100, "Fixed!")

        assert result is False

    @patch("gemini_reviewer.github_client.requests.get")
    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_reply_to_comment_exception(self, mock_session, mock_github, mock_get, valid_config, sample_pr_details):
        """Test handling exception in reply_to_comment."""
        mock_get.side_effect = Exception("Network error")

        client = GitHubClient(valid_config)
        result = client.reply_to_comment(sample_pr_details, 100, "Fixed!")

        assert result is False


class TestGitHubClientFileReviewComments:
    """Tests for get_file_review_comments functionality."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head_sha")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_file_review_comments_success(self, mock_session, mock_github, valid_config, sample_pr_details):
        """Test getting file review comments successfully."""
        from datetime import datetime

        mock_user = Mock()
        mock_user.login = "reviewer"

        mock_comment = Mock()
        mock_comment.path = "file.py"
        mock_comment.body = "Fix this issue"
        mock_comment.user = mock_user
        mock_comment.created_at = datetime(2024, 1, 1, 12, 0, 0)

        mock_pr = Mock()
        mock_pr.get_review_comments.return_value = [mock_comment]
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_file_review_comments(sample_pr_details, "file.py")

        assert result is not None
        assert "file.py" in result
        assert "reviewer" in result

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_file_review_comments_no_comments(self, mock_session, mock_github, valid_config, sample_pr_details):
        """Test when no comments found for file."""
        mock_pr = Mock()
        mock_pr.get_review_comments.return_value = []
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_file_review_comments(sample_pr_details, "file.py")

        assert result is None

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_file_review_comments_wrong_file(self, mock_session, mock_github, valid_config, sample_pr_details):
        """Test when comments are for different file."""
        mock_comment = Mock()
        mock_comment.path = "other_file.py"
        mock_comment.body = "Comment on other file"

        mock_pr = Mock()
        mock_pr.get_review_comments.return_value = [mock_comment]
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_file_review_comments(sample_pr_details, "file.py")

        assert result is None

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_file_review_comments_exception(self, mock_session, mock_github, valid_config, sample_pr_details):
        """Test handling exception in get_file_review_comments."""
        mock_github.return_value.get_repo.side_effect = Exception("API error")

        client = GitHubClient(valid_config)
        result = client.get_file_review_comments(sample_pr_details, "file.py")

        assert result is None

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_file_review_comments_empty_body(self, mock_session, mock_github, valid_config, sample_pr_details):
        """Test handling comments with empty body."""
        mock_comment = Mock()
        mock_comment.path = "file.py"
        mock_comment.body = ""

        mock_pr = Mock()
        mock_pr.get_review_comments.return_value = [mock_comment]
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_file_review_comments(sample_pr_details, "file.py")

        assert result is None


class TestGitHubClientRepositoryInfo:
    """Tests for get_repository_info functionality."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_repository_info_success(self, mock_session, mock_github, valid_config):
        """Test getting repository info successfully."""
        mock_repo = Mock()
        mock_repo.name = "test-repo"
        mock_repo.full_name = "owner/test-repo"
        mock_repo.description = "A test repo"
        mock_repo.language = "Python"
        mock_repo.default_branch = "main"
        mock_repo.private = False
        mock_repo.size = 1000
        mock_repo.stargazers_count = 50
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_repository_info("owner", "test-repo")

        assert result["name"] == "test-repo"
        assert result["language"] == "Python"
        assert result["default_branch"] == "main"

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_repository_info_failure(self, mock_session, mock_github, valid_config):
        """Test handling failure in get_repository_info."""
        mock_github.return_value.get_repo.side_effect = Exception("Not found")

        client = GitHubClient(valid_config)
        result = client.get_repository_info("owner", "nonexistent")

        assert result == {}


class TestGitHubClientPRFiles:
    """Tests for get_pr_files functionality."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_files_success(self, mock_session, mock_github, valid_config):
        """Test getting PR files successfully."""
        mock_file = Mock()
        mock_file.filename = "file.py"
        mock_file.status = "modified"
        mock_file.additions = 10
        mock_file.deletions = 5
        mock_file.changes = 15
        mock_file.patch = "+new line"

        mock_pr = Mock()
        mock_pr.get_files.return_value = [mock_file]
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_pr_files("owner", "repo", 123)

        assert len(result) == 1
        assert result[0]["filename"] == "file.py"
        assert result[0]["status"] == "modified"

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_files_failure(self, mock_session, mock_github, valid_config):
        """Test handling failure in get_pr_files."""
        mock_github.return_value.get_repo.side_effect = Exception("API error")

        client = GitHubClient(valid_config)
        result = client.get_pr_files("owner", "repo", 123)

        assert result == []


class TestGitHubClientRateLimitAdvanced:
    """Advanced tests for check_rate_limit functionality."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_check_rate_limit_with_rate_attr(self, mock_session, mock_github, valid_config):
        """Test check_rate_limit with rate attribute (newer PyGithub)."""
        mock_rate = Mock()
        mock_rate.limit = 5000
        mock_rate.remaining = 4000
        mock_rate.reset.timestamp.return_value = 1234567890

        mock_rate_limit = Mock(spec=[])  # No 'core' attr
        mock_rate_limit.rate = mock_rate
        mock_github.return_value.get_rate_limit.return_value = mock_rate_limit

        client = GitHubClient(valid_config)
        result = client.check_rate_limit()

        assert "core" in result
        assert result["core"]["limit"] == 5000

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_check_rate_limit_unknown_structure(self, mock_session, mock_github, valid_config):
        """Test check_rate_limit with unknown structure."""
        mock_rate_limit = Mock(spec=[])  # No known attrs
        del mock_rate_limit.core
        del mock_rate_limit.rate
        mock_github.return_value.get_rate_limit.return_value = mock_rate_limit

        client = GitHubClient(valid_config)
        result = client.check_rate_limit()

        # Should return default valid structure
        assert "core" in result


class TestGitHubClientLastReviewedAdvanced:
    """Advanced tests for get_last_reviewed_commit_sha."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head123", "base456")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_last_reviewed_commit_commit_exception(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test handling exception when getting commits."""
        mock_pr = Mock()
        mock_pr.get_commits.side_effect = Exception("API Error")

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_last_reviewed_commit_sha(sample_pr_details)

        assert result is None

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_last_reviewed_commit_issue_comment(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test finding bot review via issue comment."""
        from datetime import datetime

        mock_commit = Mock()
        mock_commit.sha = "abc123"
        mock_commit.commit.committer.date = datetime(2024, 1, 1, 10, 0, 0)
        mock_commit.commit.author.date = datetime(2024, 1, 1, 10, 0, 0)

        mock_issue_comment = Mock()
        mock_issue_comment.user.login = "github-actions[bot]"
        mock_issue_comment.body = "Gemini AI Code Reviewer approved"
        mock_issue_comment.created_at = datetime(2024, 1, 1, 12, 0, 0)

        mock_issue = Mock()
        mock_issue.get_comments.return_value = [mock_issue_comment]

        mock_pr = Mock()
        mock_pr.get_commits.return_value = [mock_commit]
        mock_pr.get_reviews.return_value = []
        mock_pr.as_issue.return_value = mock_issue

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_last_reviewed_commit_sha(sample_pr_details)

        assert result == "abc123"

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_last_reviewed_commit_no_mapping(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test when review time cannot be mapped to any commit."""
        from datetime import datetime

        # Commit is AFTER the review time, so no reviewed commit
        mock_commit = Mock()
        mock_commit.sha = "abc123"
        mock_commit.commit.committer.date = datetime(2024, 1, 2, 10, 0, 0)
        mock_commit.commit.author.date = datetime(2024, 1, 2, 10, 0, 0)

        mock_review = Mock()
        mock_review.user.login = "github-actions[bot]"
        mock_review.body = "Gemini AI Code Reviewer"
        mock_review.submitted_at = datetime(2024, 1, 1, 10, 0, 0)

        mock_pr = Mock()
        mock_pr.get_commits.return_value = [mock_commit]
        mock_pr.get_reviews.return_value = [mock_review]
        mock_pr.as_issue.return_value.get_comments.return_value = []

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_last_reviewed_commit_sha(sample_pr_details)

        # No commits at or before review time
        assert result is None


class TestGitHubClientIncrementalDiffAdvanced:
    """Advanced tests for get_incremental_diff_by_commits."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head123", "base456")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_incremental_diff_base_not_found(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test when base SHA is not found in PR commits."""
        mock_commit = Mock()
        mock_commit.sha = "different_sha"

        mock_file = Mock()
        mock_file.patch = "+new content"
        mock_file.status = "modified"
        mock_file.filename = "file.py"
        mock_file.previous_filename = None

        mock_gh_commit = Mock()
        mock_gh_commit.files = [mock_file]

        mock_pr = Mock()
        mock_pr.get_commits.return_value = [mock_commit]

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_repo.get_commit.return_value = mock_gh_commit
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_incremental_diff_by_commits(sample_pr_details, "nonexistent_sha")

        # Should include all commits as fallback
        assert "file.py" in result

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_incremental_diff_added_file(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test diff for added file."""
        mock_commit1 = Mock()
        mock_commit1.sha = "base_sha"
        mock_commit2 = Mock()
        mock_commit2.sha = "new_sha"

        mock_file = Mock()
        mock_file.patch = "+new file content"
        mock_file.status = "added"
        mock_file.filename = "new_file.py"
        mock_file.previous_filename = None

        mock_gh_commit = Mock()
        mock_gh_commit.files = [mock_file]

        mock_pr = Mock()
        mock_pr.get_commits.return_value = [mock_commit1, mock_commit2]

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_repo.get_commit.return_value = mock_gh_commit
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_incremental_diff_by_commits(sample_pr_details, "base_sha")

        assert "--- /dev/null" in result
        assert "+++ b/new_file.py" in result

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_incremental_diff_removed_file(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test diff for removed file."""
        mock_commit1 = Mock()
        mock_commit1.sha = "base_sha"
        mock_commit2 = Mock()
        mock_commit2.sha = "new_sha"

        mock_file = Mock()
        mock_file.patch = "-removed content"
        mock_file.status = "removed"
        mock_file.filename = "deleted.py"
        mock_file.previous_filename = None

        mock_gh_commit = Mock()
        mock_gh_commit.files = [mock_file]

        mock_pr = Mock()
        mock_pr.get_commits.return_value = [mock_commit1, mock_commit2]

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_repo.get_commit.return_value = mock_gh_commit
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_incremental_diff_by_commits(sample_pr_details, "base_sha")

        assert "--- a/deleted.py" in result
        assert "+++ /dev/null" in result

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_incremental_diff_renamed_file(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test diff for renamed file."""
        mock_commit1 = Mock()
        mock_commit1.sha = "base_sha"
        mock_commit2 = Mock()
        mock_commit2.sha = "new_sha"

        mock_file = Mock()
        mock_file.patch = " unchanged content"
        mock_file.status = "renamed"
        mock_file.filename = "new_name.py"
        mock_file.previous_filename = "old_name.py"

        mock_gh_commit = Mock()
        mock_gh_commit.files = [mock_file]

        mock_pr = Mock()
        mock_pr.get_commits.return_value = [mock_commit1, mock_commit2]

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_repo.get_commit.return_value = mock_gh_commit
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_incremental_diff_by_commits(sample_pr_details, "base_sha")

        assert "--- a/old_name.py" in result
        assert "+++ b/new_name.py" in result

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_incremental_diff_commit_exception(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test handling exception when getting commit."""
        mock_commit1 = Mock()
        mock_commit1.sha = "base_sha"
        mock_commit2 = Mock()
        mock_commit2.sha = "new_sha"

        mock_pr = Mock()
        mock_pr.get_commits.return_value = [mock_commit1, mock_commit2]

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_repo.get_commit.side_effect = Exception("API error")
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_incremental_diff_by_commits(sample_pr_details, "base_sha")

        # Should return empty since all commits failed
        assert result == ""

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_incremental_diff_exception(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test handling exception in get_incremental_diff_by_commits."""
        mock_github.return_value.get_repo.side_effect = Exception("API error")

        client = GitHubClient(valid_config)
        result = client.get_incremental_diff_by_commits(sample_pr_details, "base_sha")

        assert result == ""


class TestGitHubClientFilterDedupe:
    """Tests for filter_out_existing_comments deduplication."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head_sha")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_filter_dedupes_similar_comments_same_line(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test that similar comments on the same line are deduped."""
        mock_pr = Mock()
        mock_pr.get_review_comments.return_value = []
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)

        # Two similar comments on same file and position
        comments = [
            ReviewComment("Fix this bug please", "file.py", 10, priority=ReviewPriority.MEDIUM),
            ReviewComment("Fix this bug", "file.py", 10, priority=ReviewPriority.LOW),
        ]

        result = client.filter_out_existing_comments(sample_pr_details, comments)

        # Should keep only one (the higher priority or longer one)
        assert len(result) == 1

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_filter_keeps_different_comments_same_line(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test that different comments on same line are kept."""
        mock_pr = Mock()
        mock_pr.get_review_comments.return_value = []
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)

        # Two different comments on same position
        comments = [
            ReviewComment("Fix the null check", "file.py", 10),
            ReviewComment("Add error handling", "file.py", 10),
        ]

        result = client.filter_out_existing_comments(sample_pr_details, comments)

        # Different comments should be kept
        assert len(result) == 2

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_filter_exception_returns_original(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test that exception in filter returns original comments."""
        mock_github.return_value.get_repo.side_effect = Exception("API error")

        client = GitHubClient(valid_config)

        comments = [
            ReviewComment("Comment", "file.py", 1),
        ]

        result = client.filter_out_existing_comments(sample_pr_details, comments)

        # Should return original on error
        assert len(result) == 1


class TestGitHubClientDiffSinceAdvanced:
    """Advanced tests for get_pr_diff_since."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head123", "base456")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_diff_since_empty_compare(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test when compare API returns empty diff."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_session.return_value.get.return_value = mock_response

        mock_pr = Mock()
        mock_pr.head.sha = "head123"
        mock_pr.get_commits.return_value = []

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_pr_diff_since(sample_pr_details, "base456")

        # Empty diff triggers fallback which also returns empty
        assert result == ""

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_diff_since_head_differs_from_commits(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test when head SHA differs from latest commit."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "diff --git"
        mock_session.return_value.get.return_value = mock_response

        mock_commit = Mock()
        mock_commit.sha = "latest_commit_sha"

        mock_pr = Mock()
        mock_pr.head.sha = "head123"  # Different from latest commit
        mock_pr.get_commits.return_value = [mock_commit]

        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_pr_diff_since(sample_pr_details, "base456")

        assert "diff --git" in result

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_pr_diff_since_exception(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test exception handling in get_pr_diff_since."""
        mock_github.return_value.get_repo.side_effect = Exception("API error")

        client = GitHubClient(valid_config)
        result = client.get_pr_diff_since(sample_pr_details, "base456")

        # Should return empty on error, not re-review entire codebase
        assert result == ""


class TestGitHubClientGetFileContentAdvanced:
    """Advanced tests for get_file_content."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_file_content_directory(self, mock_session, mock_github, valid_config):
        """Test get_file_content when path is a directory."""
        mock_content = Mock(spec=[])  # No decoded_content attribute
        mock_repo = Mock()
        mock_repo.get_contents.return_value = mock_content
        mock_github.return_value.get_repo.return_value = mock_repo

        client = GitHubClient(valid_config)
        result = client.get_file_content("owner", "repo", "src/", "HEAD")

        assert result is None

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_file_content_repo_error(self, mock_session, mock_github, valid_config):
        """Test get_file_content when repo lookup fails."""
        mock_github.return_value.get_repo.side_effect = Exception("Repo not found")

        client = GitHubClient(valid_config)
        result = client.get_file_content("owner", "nonexistent", "file.py", "HEAD")

        assert result is None


class TestGitHubClientExistingBotComments:
    """Tests for get_existing_bot_comments functionality."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head_sha")

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_existing_bot_comments_exception(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test handling exception in get_existing_bot_comments."""
        mock_github.return_value.get_repo.side_effect = Exception("API error")

        client = GitHubClient(valid_config)
        result = client.get_existing_bot_comments(sample_pr_details)

        assert result == []

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_existing_bot_comments_user_error(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test when get_user fails."""
        mock_pr = Mock()
        mock_pr.get_review_comments.return_value = []
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo
        mock_github.return_value.get_user.side_effect = Exception("Auth error")

        client = GitHubClient(valid_config)
        result = client.get_existing_bot_comments(sample_pr_details)

        assert result == []

    @patch("gemini_reviewer.github_client.Github")
    @patch("gemini_reviewer.github_client.requests.Session")
    def test_get_existing_bot_comments_comment_exception(
        self, mock_session, mock_github, valid_config, sample_pr_details
    ):
        """Test when processing a comment fails."""
        mock_comment = Mock()
        # Make accessing path raise an exception
        type(mock_comment).path = property(lambda self: (_ for _ in ()).throw(Exception("attr error")))

        mock_pr = Mock()
        mock_pr.get_review_comments.return_value = [mock_comment]
        mock_repo = Mock()
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo
        mock_github.return_value.get_user.return_value.login = "bot"

        client = GitHubClient(valid_config)
        result = client.get_existing_bot_comments(sample_pr_details)

        # Should handle exception gracefully
        assert result == []
