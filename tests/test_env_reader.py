"""
Comprehensive tests for gemini_reviewer/env_reader.py
"""

import os
import pytest
from enum import Enum

from gemini_reviewer.env_reader import (
    get_env_str,
    get_env_int,
    get_env_float,
    get_env_bool,
    get_env_list,
    get_env_enum,
)


class SampleEnum(Enum):
    """Sample enum for testing."""
    VALUE_A = "value_a"
    VALUE_B = "value_b"
    VALUE_C = "value_c"


class TestGetEnvStr:
    """Tests for get_env_str function."""

    def test_returns_env_value(self, monkeypatch):
        """Test returning environment variable value."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        assert get_env_str("TEST_VAR") == "test_value"

    def test_returns_default_when_not_set(self, monkeypatch):
        """Test returning default when variable not set."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        assert get_env_str("MISSING_VAR", "default") == "default"

    def test_returns_empty_default(self, monkeypatch):
        """Test default empty string when not specified."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        assert get_env_str("MISSING_VAR") == ""

    def test_fallback_key_used(self, monkeypatch):
        """Test fallback key when primary is empty."""
        monkeypatch.delenv("PRIMARY", raising=False)
        monkeypatch.setenv("FALLBACK", "fallback_value")
        assert get_env_str("PRIMARY", "default", "FALLBACK") == "fallback_value"

    def test_primary_preferred_over_fallback(self, monkeypatch):
        """Test primary key is preferred when both are set."""
        monkeypatch.setenv("PRIMARY", "primary_value")
        monkeypatch.setenv("FALLBACK", "fallback_value")
        assert get_env_str("PRIMARY", "default", "FALLBACK") == "primary_value"

    def test_multiple_fallbacks(self, monkeypatch):
        """Test multiple fallback keys."""
        monkeypatch.delenv("PRIMARY", raising=False)
        monkeypatch.delenv("FALLBACK1", raising=False)
        monkeypatch.setenv("FALLBACK2", "second_fallback")
        assert get_env_str("PRIMARY", "default", "FALLBACK1", "FALLBACK2") == "second_fallback"

    def test_default_when_all_empty(self, monkeypatch):
        """Test default when primary and all fallbacks are empty."""
        monkeypatch.delenv("PRIMARY", raising=False)
        monkeypatch.delenv("FALLBACK", raising=False)
        assert get_env_str("PRIMARY", "default_value", "FALLBACK") == "default_value"


class TestGetEnvInt:
    """Tests for get_env_int function."""

    def test_returns_int_value(self, monkeypatch):
        """Test returning integer value."""
        monkeypatch.setenv("INT_VAR", "42")
        assert get_env_int("INT_VAR", 0) == 42

    def test_returns_default_when_not_set(self, monkeypatch):
        """Test returning default when variable not set."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        assert get_env_int("MISSING_VAR", 100) == 100

    def test_returns_default_on_invalid_value(self, monkeypatch):
        """Test returning default when value is not a valid integer."""
        monkeypatch.setenv("INVALID_INT", "not_a_number")
        assert get_env_int("INVALID_INT", 50) == 50

    def test_fallback_key_used(self, monkeypatch):
        """Test fallback key when primary is empty."""
        monkeypatch.delenv("PRIMARY", raising=False)
        monkeypatch.setenv("FALLBACK", "99")
        assert get_env_int("PRIMARY", 0, "FALLBACK") == 99

    def test_negative_values(self, monkeypatch):
        """Test negative integer values."""
        monkeypatch.setenv("NEG_VAR", "-10")
        assert get_env_int("NEG_VAR", 0) == -10

    def test_zero_value(self, monkeypatch):
        """Test zero value."""
        monkeypatch.setenv("ZERO_VAR", "0")
        assert get_env_int("ZERO_VAR", 5) == 0


class TestGetEnvFloat:
    """Tests for get_env_float function."""

    def test_returns_float_value(self, monkeypatch):
        """Test returning float value."""
        monkeypatch.setenv("FLOAT_VAR", "3.14")
        assert get_env_float("FLOAT_VAR", 0.0) == 3.14

    def test_returns_default_when_not_set(self, monkeypatch):
        """Test returning default when variable not set."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        assert get_env_float("MISSING_VAR", 1.5) == 1.5

    def test_returns_default_on_invalid_value(self, monkeypatch):
        """Test returning default when value is not a valid float."""
        monkeypatch.setenv("INVALID_FLOAT", "not_a_float")
        assert get_env_float("INVALID_FLOAT", 2.5) == 2.5

    def test_fallback_key_used(self, monkeypatch):
        """Test fallback key when primary is empty."""
        monkeypatch.delenv("PRIMARY", raising=False)
        monkeypatch.setenv("FALLBACK", "0.99")
        assert get_env_float("PRIMARY", 0.0, "FALLBACK") == 0.99

    def test_integer_as_float(self, monkeypatch):
        """Test integer string is parsed as float."""
        monkeypatch.setenv("INT_AS_FLOAT", "42")
        assert get_env_float("INT_AS_FLOAT", 0.0) == 42.0

    def test_negative_float(self, monkeypatch):
        """Test negative float values."""
        monkeypatch.setenv("NEG_FLOAT", "-1.5")
        assert get_env_float("NEG_FLOAT", 0.0) == -1.5


class TestGetEnvBool:
    """Tests for get_env_bool function."""

    def test_true_values(self, monkeypatch):
        """Test recognized true values."""
        for value in ["true", "TRUE", "True", "yes", "YES", "Yes", "1"]:
            monkeypatch.setenv("BOOL_VAR", value)
            assert get_env_bool("BOOL_VAR", False) is True

    def test_false_values(self, monkeypatch):
        """Test values that result in False."""
        for value in ["false", "FALSE", "no", "NO", "0", "anything"]:
            monkeypatch.setenv("BOOL_VAR", value)
            assert get_env_bool("BOOL_VAR", True) is False

    def test_returns_default_when_not_set(self, monkeypatch):
        """Test returning default when variable not set."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        assert get_env_bool("MISSING_VAR", True) is True
        assert get_env_bool("MISSING_VAR", False) is False

    def test_fallback_key_used(self, monkeypatch):
        """Test fallback key when primary is empty."""
        monkeypatch.delenv("PRIMARY", raising=False)
        monkeypatch.setenv("FALLBACK", "true")
        assert get_env_bool("PRIMARY", False, "FALLBACK") is True

    def test_empty_string_returns_default(self, monkeypatch):
        """Test empty string returns default."""
        monkeypatch.setenv("EMPTY_VAR", "")
        assert get_env_bool("EMPTY_VAR", True) is True


class TestGetEnvList:
    """Tests for get_env_list function."""

    def test_comma_separated(self, monkeypatch):
        """Test comma-separated list parsing."""
        monkeypatch.setenv("LIST_VAR", "a,b,c")
        assert get_env_list("LIST_VAR") == ["a", "b", "c"]

    def test_custom_separator(self, monkeypatch):
        """Test custom separator."""
        monkeypatch.setenv("LIST_VAR", "a;b;c")
        assert get_env_list("LIST_VAR", ";") == ["a", "b", "c"]

    def test_strips_whitespace(self, monkeypatch):
        """Test whitespace is stripped from items."""
        monkeypatch.setenv("LIST_VAR", " a , b , c ")
        assert get_env_list("LIST_VAR") == ["a", "b", "c"]

    def test_empty_items_filtered(self, monkeypatch):
        """Test empty items are filtered out."""
        monkeypatch.setenv("LIST_VAR", "a,,b,  ,c")
        assert get_env_list("LIST_VAR") == ["a", "b", "c"]

    def test_returns_empty_when_not_set(self, monkeypatch):
        """Test returning empty list when variable not set."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        assert get_env_list("MISSING_VAR") == []

    def test_fallback_key_used(self, monkeypatch):
        """Test fallback key when primary is empty."""
        monkeypatch.delenv("PRIMARY", raising=False)
        monkeypatch.setenv("FALLBACK", "x,y,z")
        assert get_env_list("PRIMARY", ",", "FALLBACK") == ["x", "y", "z"]

    def test_single_item(self, monkeypatch):
        """Test single item list."""
        monkeypatch.setenv("LIST_VAR", "single")
        assert get_env_list("LIST_VAR") == ["single"]


class TestGetEnvEnum:
    """Tests for get_env_enum function."""

    def test_returns_enum_value(self, monkeypatch):
        """Test returning enum value from environment."""
        monkeypatch.setenv("ENUM_VAR", "value_a")
        result = get_env_enum("ENUM_VAR", SampleEnum, SampleEnum.VALUE_B)
        assert result == SampleEnum.VALUE_A

    def test_returns_default_when_not_set(self, monkeypatch):
        """Test returning default when variable not set."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        result = get_env_enum("MISSING_VAR", SampleEnum, SampleEnum.VALUE_C)
        assert result == SampleEnum.VALUE_C

    def test_returns_default_on_invalid_value(self, monkeypatch):
        """Test returning default when value doesn't match enum."""
        monkeypatch.setenv("ENUM_VAR", "invalid_value")
        result = get_env_enum("ENUM_VAR", SampleEnum, SampleEnum.VALUE_B)
        assert result == SampleEnum.VALUE_B

    def test_case_insensitive(self, monkeypatch):
        """Test case insensitivity in enum matching."""
        monkeypatch.setenv("ENUM_VAR", "VALUE_A")
        result = get_env_enum("ENUM_VAR", SampleEnum, SampleEnum.VALUE_B)
        assert result == SampleEnum.VALUE_A

    def test_strips_whitespace(self, monkeypatch):
        """Test whitespace is stripped."""
        monkeypatch.setenv("ENUM_VAR", "  value_b  ")
        result = get_env_enum("ENUM_VAR", SampleEnum, SampleEnum.VALUE_A)
        assert result == SampleEnum.VALUE_B

    def test_fallback_key_used(self, monkeypatch):
        """Test fallback key when primary is empty."""
        monkeypatch.delenv("PRIMARY", raising=False)
        monkeypatch.setenv("FALLBACK", "value_c")
        result = get_env_enum("PRIMARY", SampleEnum, SampleEnum.VALUE_A, "FALLBACK")
        assert result == SampleEnum.VALUE_C
