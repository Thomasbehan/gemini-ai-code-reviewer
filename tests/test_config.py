"""
Comprehensive tests for gemini_reviewer/config.py
"""

import pytest
from unittest.mock import patch, MagicMock

from gemini_reviewer.config import (
    LogLevel,
    GitHubConfig,
    GeminiConfig,
    ReviewConfig,
    PerformanceConfig,
    LoggingConfig,
    Config,
)
from gemini_reviewer.prompts import ReviewMode
from gemini_reviewer.models import ReviewPriority, ReviewFocus


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_all_levels(self):
        """Test all log level values."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"


class TestGitHubConfig:
    """Tests for GitHubConfig dataclass."""

    def test_valid_classic_token(self):
        """Test with valid classic 40-char token."""
        token = "a" * 40
        config = GitHubConfig(token=token)
        assert config.token == token
        assert config.api_base_url == "https://api.github.com"
        assert config.timeout == 30
        assert config.max_retries == 3

    def test_valid_fine_grained_token(self):
        """Test with valid fine-grained token."""
        config = GitHubConfig(token="ghp_abcdefghijklmnop")
        assert config.token == "ghp_abcdefghijklmnop"

    def test_custom_values(self):
        """Test with custom configuration values."""
        config = GitHubConfig(
            token="ghp_test123456789012",
            api_base_url="https://github.example.com/api",
            timeout=60,
            max_retries=5,
            retry_delay_min=2,
            retry_delay_max=20,
        )
        assert config.api_base_url == "https://github.example.com/api"
        assert config.timeout == 60
        assert config.max_retries == 5

    def test_empty_token_raises(self):
        """Test that empty token raises ValueError."""
        with pytest.raises(ValueError, match="is required"):
            GitHubConfig(token="")

    def test_invalid_token_format_raises(self):
        """Test that invalid token format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid GitHub token format"):
            GitHubConfig(token="abc")  # Too short


class TestGeminiConfig:
    """Tests for GeminiConfig dataclass."""

    def test_valid_api_key(self):
        """Test with valid API key."""
        config = GeminiConfig(api_key="AIzaSyTestKey123456")
        assert config.api_key == "AIzaSyTestKey123456"
        assert config.model_name == "gemini-2.5-flash"
        assert config.temperature == 0.0
        assert config.top_p == 0.9

    def test_custom_values(self):
        """Test with custom configuration values."""
        config = GeminiConfig(
            api_key="AIzaSyTestKey123456",
            model_name="gemini-pro",
            max_output_tokens=4096,
            temperature=0.5,
            top_p=0.8,
            timeout=120,
            max_retries=5,
        )
        assert config.model_name == "gemini-pro"
        assert config.max_output_tokens == 4096
        assert config.temperature == 0.5
        assert config.top_p == 0.8

    def test_empty_api_key_raises(self):
        """Test that empty API key raises ValueError."""
        with pytest.raises(ValueError, match="is required"):
            GeminiConfig(api_key="")

    def test_invalid_api_key_format_raises(self):
        """Test that invalid API key format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid Gemini API key format"):
            GeminiConfig(api_key="short")

    def test_temperature_too_high_raises(self):
        """Test that temperature > 2.0 raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be between"):
            GeminiConfig(api_key="AIzaSyTestKey123456", temperature=2.5)

    def test_temperature_too_low_raises(self):
        """Test that temperature < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be between"):
            GeminiConfig(api_key="AIzaSyTestKey123456", temperature=-0.1)

    def test_top_p_too_high_raises(self):
        """Test that top_p > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="Top_p must be between"):
            GeminiConfig(api_key="AIzaSyTestKey123456", top_p=1.5)

    def test_top_p_too_low_raises(self):
        """Test that top_p < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="Top_p must be between"):
            GeminiConfig(api_key="AIzaSyTestKey123456", top_p=-0.1)


class TestReviewConfig:
    """Tests for ReviewConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ReviewConfig()
        assert config.review_mode == ReviewMode.STANDARD
        assert config.focus_areas == [ReviewFocus.ALL]
        assert config.max_files_per_review == 50
        assert config.max_lines_per_hunk == 500
        assert config.review_test_files is False
        assert config.review_docs is False
        assert config.priority_threshold == ReviewPriority.LOW

    def test_default_exclude_patterns(self):
        """Test default exclude patterns are set."""
        config = ReviewConfig()
        assert "*.md" in config.exclude_patterns
        assert "*.json" in config.exclude_patterns
        assert "package-lock.json" in config.exclude_patterns

    def test_custom_exclude_patterns(self):
        """Test custom exclude patterns."""
        config = ReviewConfig(exclude_patterns=["*.test.js", "*.spec.ts"])
        assert config.exclude_patterns == ["*.test.js", "*.spec.ts"]

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ReviewConfig(
            review_mode=ReviewMode.STRICT,
            focus_areas=[ReviewFocus.SECURITY, ReviewFocus.BUGS],
            include_patterns=["*.py"],
            max_files_per_review=20,
            max_lines_per_hunk=200,
            review_test_files=True,
            review_docs=True,
            priority_threshold=ReviewPriority.HIGH,
            max_comments_total=50,
            max_comments_per_file=10,
        )
        assert config.review_mode == ReviewMode.STRICT
        assert config.max_files_per_review == 20
        assert config.max_comments_total == 50

    def test_invalid_max_files_raises(self):
        """Test that non-positive max_files_per_review raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            ReviewConfig(max_files_per_review=0)

    def test_invalid_max_lines_raises(self):
        """Test that non-positive max_lines_per_hunk raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            ReviewConfig(max_lines_per_hunk=-1)


class TestPerformanceConfig:
    """Tests for PerformanceConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PerformanceConfig()
        assert config.enable_concurrent_processing is True
        assert config.max_concurrent_files == 3
        assert config.max_concurrent_api_calls == 5
        assert config.chunk_size == 10
        assert config.enable_caching is True
        assert config.cache_ttl == 3600

    def test_non_positive_concurrent_files_uses_default(self):
        """Test that non-positive max_concurrent_files uses default of 1."""
        config = PerformanceConfig(max_concurrent_files=0)
        assert config.max_concurrent_files == 1

        config = PerformanceConfig(max_concurrent_files=-5)
        assert config.max_concurrent_files == 1

    def test_non_positive_concurrent_api_calls_uses_default(self):
        """Test that non-positive max_concurrent_api_calls uses default of 1."""
        config = PerformanceConfig(max_concurrent_api_calls=0)
        assert config.max_concurrent_api_calls == 1


class TestLoggingConfig:
    """Tests for LoggingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LoggingConfig()
        assert config.level == LogLevel.INFO
        assert config.enable_file_logging is False
        assert config.log_file_path == "gemini_reviewer.log"
        assert config.max_log_size == 10 * 1024 * 1024
        assert config.backup_count == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LoggingConfig(
            level=LogLevel.DEBUG,
            enable_file_logging=True,
            log_file_path="/var/log/app.log",
        )
        assert config.level == LogLevel.DEBUG
        assert config.enable_file_logging is True


class TestConfig:
    """Tests for main Config dataclass."""

    @pytest.fixture
    def github_config(self):
        """Create a valid GitHubConfig."""
        return GitHubConfig(token="ghp_test123456789012")

    @pytest.fixture
    def gemini_config(self):
        """Create a valid GeminiConfig."""
        return GeminiConfig(api_key="AIzaSyTestKey123456")

    def test_basic_creation(self, github_config, gemini_config):
        """Test basic Config creation."""
        config = Config(github=github_config, gemini=gemini_config)
        assert config.github == github_config
        assert config.gemini == gemini_config
        assert isinstance(config.review, ReviewConfig)
        assert isinstance(config.performance, PerformanceConfig)
        assert isinstance(config.logging, LoggingConfig)

    def test_from_environment_missing_github_token(self, monkeypatch):
        """Test from_environment raises when GITHUB_TOKEN is missing."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestKey123456")
        with pytest.raises(ValueError, match="GITHUB_TOKEN"):
            Config.from_environment()

    def test_from_environment_missing_gemini_key(self, monkeypatch):
        """Test from_environment raises when GEMINI_API_KEY is missing."""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123456789012")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            Config.from_environment()

    def test_from_environment_basic(self, monkeypatch):
        """Test from_environment with minimal required variables."""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123456789012")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestKey123456")
        # Clear optional variables
        for var in ["GITHUB_TIMEOUT", "GEMINI_MODEL", "REVIEW_MODE", "LOG_LEVEL"]:
            monkeypatch.delenv(var, raising=False)

        config = Config.from_environment()
        assert config.github.token == "ghp_test123456789012"
        assert config.gemini.api_key == "AIzaSyTestKey123456"

    def test_from_environment_with_options(self, monkeypatch):
        """Test from_environment with optional variables."""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123456789012")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestKey123456")
        monkeypatch.setenv("GITHUB_TIMEOUT", "60")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-pro")
        monkeypatch.setenv("GEMINI_TEMPERATURE", "0.5")
        monkeypatch.setenv("REVIEW_MODE", "strict")
        monkeypatch.setenv("MAX_FILES_PER_REVIEW", "25")
        monkeypatch.setenv("REVIEW_TEST_FILES", "true")
        monkeypatch.setenv("ENABLE_CONCURRENT", "false")

        config = Config.from_environment()
        assert config.github.timeout == 60
        assert config.gemini.model_name == "gemini-pro"
        assert config.gemini.temperature == 0.5
        assert config.review.review_mode == ReviewMode.STRICT
        assert config.review.max_files_per_review == 25
        assert config.review.review_test_files is True
        assert config.performance.enable_concurrent_processing is False

    def test_from_environment_with_exclude_patterns(self, monkeypatch):
        """Test from_environment with EXCLUDE patterns."""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123456789012")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestKey123456")
        monkeypatch.setenv("EXCLUDE", "*.test.js,*.spec.ts")

        config = Config.from_environment()
        assert "*.test.js" in config.review.exclude_patterns
        assert "*.spec.ts" in config.review.exclude_patterns

    def test_from_environment_with_input_exclude(self, monkeypatch):
        """Test from_environment with INPUT_EXCLUDE (GitHub Actions style)."""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123456789012")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestKey123456")
        monkeypatch.delenv("EXCLUDE", raising=False)
        monkeypatch.setenv("INPUT_EXCLUDE", "*.md,*.txt")

        config = Config.from_environment()
        assert "*.md" in config.review.exclude_patterns

    def test_from_environment_with_priority_threshold(self, monkeypatch):
        """Test from_environment with priority threshold."""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123456789012")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestKey123456")
        monkeypatch.setenv("REVIEW_PRIORITY_THRESHOLD", "high")

        config = Config.from_environment()
        assert config.review.priority_threshold == ReviewPriority.HIGH

    def test_from_environment_with_custom_prompt(self, monkeypatch):
        """Test from_environment with custom prompt."""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123456789012")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestKey123456")
        monkeypatch.setenv("SYSTEM_PROMPT", "Focus on error handling")

        config = Config.from_environment()
        assert config.review.custom_prompt_template == "Focus on error handling"

    def test_from_environment_with_comment_limits(self, monkeypatch):
        """Test from_environment with comment limits."""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123456789012")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestKey123456")
        monkeypatch.setenv("MAX_COMMENTS_TOTAL", "50")
        monkeypatch.setenv("MAX_COMMENTS_PER_FILE", "10")

        config = Config.from_environment()
        assert config.review.max_comments_total == 50
        assert config.review.max_comments_per_file == 10

    def test_get_review_prompt_template(self, github_config, gemini_config):
        """Test get_review_prompt_template method."""
        config = Config(github=github_config, gemini=gemini_config)
        prompt = config.get_review_prompt_template()
        assert "JSON" in prompt
        assert "reviews" in prompt

    def test_get_review_prompt_template_with_custom(self, github_config, gemini_config):
        """Test get_review_prompt_template with custom template."""
        review_config = ReviewConfig(
            custom_prompt_template="Custom instructions here"
        )
        config = Config(
            github=github_config,
            gemini=gemini_config,
            review=review_config,
        )
        prompt = config.get_review_prompt_template()
        assert "Custom instructions here" in prompt

    def test_get_review_prompt_template_with_previous_comments(self, github_config, gemini_config):
        """Test get_review_prompt_template with previous comments for followup."""
        review_config = ReviewConfig(review_mode=ReviewMode.FOLLOWUP)
        config = Config(
            github=github_config,
            gemini=gemini_config,
            review=review_config,
        )
        prompt = config.get_review_prompt_template(previous_comments="Previous comment 1")
        assert "Previous comment 1" in prompt

    def test_should_review_file_basic(self, github_config, gemini_config):
        """Test should_review_file method."""
        config = Config(github=github_config, gemini=gemini_config)
        assert config.should_review_file("main.py") is True
        assert config.should_review_file("README.md") is False  # excluded by default

    def test_should_review_file_with_include_patterns(self, github_config, gemini_config):
        """Test should_review_file with include patterns."""
        review_config = ReviewConfig(include_patterns=["*.py"])
        config = Config(
            github=github_config,
            gemini=gemini_config,
            review=review_config,
        )
        assert config.should_review_file("main.py") is True
        assert config.should_review_file("script.js") is False

    def test_should_review_file_with_exclude_patterns(self, github_config, gemini_config):
        """Test should_review_file with exclude patterns."""
        review_config = ReviewConfig(exclude_patterns=["*.generated.py"])
        config = Config(
            github=github_config,
            gemini=gemini_config,
            review=review_config,
        )
        assert config.should_review_file("main.py") is True
        assert config.should_review_file("code.generated.py") is False

    def test_should_review_file_test_files(self, github_config, gemini_config):
        """Test should_review_file with test files."""
        # Default: don't review test files
        config = Config(github=github_config, gemini=gemini_config)
        assert config.should_review_file("test_main.py") is False

        # With review_test_files=True
        review_config = ReviewConfig(review_test_files=True)
        config = Config(
            github=github_config,
            gemini=gemini_config,
            review=review_config,
        )
        assert config.should_review_file("test_main.py") is True

    def test_should_review_file_doc_files(self, github_config, gemini_config):
        """Test should_review_file with doc files."""
        # Default: don't review doc files
        config = Config(github=github_config, gemini=gemini_config)
        assert config.should_review_file("guide.rst") is False

        # With review_docs=True
        review_config = ReviewConfig(review_docs=True, exclude_patterns=[])
        config = Config(
            github=github_config,
            gemini=gemini_config,
            review=review_config,
        )
        assert config.should_review_file("guide.rst") is True

    def test_to_dict(self, github_config, gemini_config):
        """Test to_dict method."""
        config = Config(github=github_config, gemini=gemini_config)
        result = config.to_dict()

        assert "github" in result
        assert "gemini" in result
        assert "review" in result
        assert "performance" in result
        assert "logging" in result

        # Check github section (token should not be exposed)
        assert "timeout" in result["github"]
        assert "token" not in result["github"]

        # Check gemini section
        assert result["gemini"]["model_name"] == "gemini-2.5-flash"

        # Check review section
        assert result["review"]["review_mode"] == "standard"
        assert result["review"]["priority_threshold"] == "low"
