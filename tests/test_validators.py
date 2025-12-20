"""
Comprehensive tests for gemini_reviewer/validators.py
"""

import pytest

from gemini_reviewer.validators import (
    validate_required_string,
    validate_positive_int,
    validate_range,
    validate_github_token_format,
    validate_gemini_api_key_format,
    ensure_positive_or_default,
)


class TestValidateRequiredString:
    """Tests for validate_required_string function."""

    def test_valid_string(self):
        """Test with a valid non-empty string."""
        # Should not raise any exception
        validate_required_string("valid_value", "test_field")

    def test_empty_string_raises(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="test_field is required"):
            validate_required_string("", "test_field")

    def test_none_raises(self):
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="field_name is required"):
            validate_required_string(None, "field_name")

    def test_whitespace_only_passes(self):
        """Test that whitespace-only string passes (truthy)."""
        # Whitespace is truthy, so it should not raise
        validate_required_string("   ", "field")

    def test_field_name_in_error_message(self):
        """Test that field name appears in error message."""
        with pytest.raises(ValueError) as exc_info:
            validate_required_string("", "my_custom_field")
        assert "my_custom_field" in str(exc_info.value)


class TestValidatePositiveInt:
    """Tests for validate_positive_int function."""

    def test_positive_value(self):
        """Test with a positive integer."""
        validate_positive_int(1, "count")
        validate_positive_int(100, "count")
        validate_positive_int(999999, "count")

    def test_zero_raises(self):
        """Test that zero raises ValueError."""
        with pytest.raises(ValueError, match="count must be positive"):
            validate_positive_int(0, "count")

    def test_negative_raises(self):
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError, match="value must be positive"):
            validate_positive_int(-1, "value")
        with pytest.raises(ValueError, match="value must be positive"):
            validate_positive_int(-100, "value")

    def test_field_name_in_error_message(self):
        """Test that field name appears in error message."""
        with pytest.raises(ValueError) as exc_info:
            validate_positive_int(-5, "max_files")
        assert "max_files" in str(exc_info.value)


class TestValidateRange:
    """Tests for validate_range function."""

    def test_value_in_range(self):
        """Test with values within range."""
        validate_range(0.5, 0.0, 1.0, "temperature")
        validate_range(0.0, 0.0, 1.0, "temperature")  # min boundary
        validate_range(1.0, 0.0, 1.0, "temperature")  # max boundary
        validate_range(1.5, 0.0, 2.0, "temperature")

    def test_value_below_min_raises(self):
        """Test that value below minimum raises ValueError."""
        with pytest.raises(ValueError, match="temp must be between 0.0 and 1.0"):
            validate_range(-0.1, 0.0, 1.0, "temp")

    def test_value_above_max_raises(self):
        """Test that value above maximum raises ValueError."""
        with pytest.raises(ValueError, match="temp must be between 0.0 and 1.0"):
            validate_range(1.1, 0.0, 1.0, "temp")

    def test_integer_range(self):
        """Test with integer values."""
        validate_range(5, 1, 10, "count")
        validate_range(1, 1, 10, "count")
        validate_range(10, 1, 10, "count")

    def test_field_name_in_error_message(self):
        """Test that field name and range appear in error message."""
        with pytest.raises(ValueError) as exc_info:
            validate_range(5.0, 0.0, 2.0, "my_param")
        error_msg = str(exc_info.value)
        assert "my_param" in error_msg
        assert "0.0" in error_msg
        assert "2.0" in error_msg


class TestValidateGithubTokenFormat:
    """Tests for validate_github_token_format function."""

    def test_classic_token_40_chars(self):
        """Test valid classic 40-character token."""
        token = "a" * 40
        assert validate_github_token_format(token) is True

    def test_fine_grained_token_ghp(self):
        """Test valid fine-grained personal access token (ghp_)."""
        assert validate_github_token_format("ghp_abcdefghijklmnop") is True

    def test_fine_grained_token_ghs(self):
        """Test valid GitHub App installation token (ghs_)."""
        assert validate_github_token_format("ghs_abcdefghijklmnop") is True

    def test_fine_grained_token_gho(self):
        """Test valid OAuth access token (gho_)."""
        assert validate_github_token_format("gho_abcdefghijklmnop") is True

    def test_fine_grained_token_ghu(self):
        """Test valid user-to-server token (ghu_)."""
        assert validate_github_token_format("ghu_abcdefghijklmnop") is True

    def test_empty_string_returns_false(self):
        """Test that empty string returns False."""
        assert validate_github_token_format("") is False

    def test_none_returns_false(self):
        """Test that None returns False."""
        assert validate_github_token_format(None) is False

    def test_short_token_returns_false(self):
        """Test that very short tokens return False."""
        assert validate_github_token_format("abc") is False
        assert validate_github_token_format("a") is False

    def test_non_string_returns_false(self):
        """Test that non-string values return False."""
        assert validate_github_token_format(12345) is False
        assert validate_github_token_format(["token"]) is False

    def test_wrong_prefix_short_returns_false(self):
        """Test that wrong prefix with short length returns False."""
        assert validate_github_token_format("xyz_abc") is False

    def test_minimum_4_chars(self):
        """Test minimum length of 4 characters with valid format."""
        # Must be 4+ chars AND (40 chars total OR valid prefix)
        assert validate_github_token_format("ghp_abc") is True  # Valid prefix
        assert validate_github_token_format("a" * 40) is True  # 40 chars
        assert validate_github_token_format("abc") is False  # Too short
        assert validate_github_token_format("abcd") is False  # 4 chars but no valid prefix/length


class TestValidateGeminiApiKeyFormat:
    """Tests for validate_gemini_api_key_format function."""

    def test_valid_key(self):
        """Test valid API key (more than 10 characters)."""
        assert validate_gemini_api_key_format("AIzaSyAbCdEfGhIjKlMnOpQr") is True
        assert validate_gemini_api_key_format("a" * 11) is True

    def test_minimum_length_boundary(self):
        """Test minimum length boundary (more than 10)."""
        assert validate_gemini_api_key_format("a" * 10) is False
        assert validate_gemini_api_key_format("a" * 11) is True

    def test_empty_string_returns_false(self):
        """Test that empty string returns False."""
        assert validate_gemini_api_key_format("") is False

    def test_none_returns_false(self):
        """Test that None returns False."""
        assert validate_gemini_api_key_format(None) is False

    def test_short_key_returns_false(self):
        """Test that short keys return False."""
        assert validate_gemini_api_key_format("abc") is False
        assert validate_gemini_api_key_format("1234567890") is False

    def test_non_string_returns_false(self):
        """Test that non-string values return False."""
        assert validate_gemini_api_key_format(12345678901) is False
        assert validate_gemini_api_key_format(["key"]) is False


class TestEnsurePositiveOrDefault:
    """Tests for ensure_positive_or_default function."""

    def test_positive_value_returned(self):
        """Test that positive values are returned as-is."""
        assert ensure_positive_or_default(5, 10) == 5
        assert ensure_positive_or_default(1, 100) == 1
        assert ensure_positive_or_default(999, 1) == 999

    def test_zero_returns_default(self):
        """Test that zero returns the default value."""
        assert ensure_positive_or_default(0, 10) == 10
        assert ensure_positive_or_default(0, 1) == 1

    def test_negative_returns_default(self):
        """Test that negative values return the default value."""
        assert ensure_positive_or_default(-1, 10) == 10
        assert ensure_positive_or_default(-100, 5) == 5

    def test_default_value_used_correctly(self):
        """Test that the correct default is returned."""
        assert ensure_positive_or_default(-1, 42) == 42
        assert ensure_positive_or_default(0, 99) == 99
