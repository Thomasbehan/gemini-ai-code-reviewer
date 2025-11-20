import pytest

from gemini_reviewer import validators as V


def test_validate_required_string():
    # non-empty ok
    V.validate_required_string("abc", "name")
    # empty -> error
    with pytest.raises(ValueError):
        V.validate_required_string("", "name")
    # None treated as empty -> error
    with pytest.raises(ValueError):
        V.validate_required_string(None, "field")


def test_validate_positive_int():
    V.validate_positive_int(1, "count")
    with pytest.raises(ValueError):
        V.validate_positive_int(0, "count")
    with pytest.raises(ValueError):
        V.validate_positive_int(-5, "count")


def test_validate_range():
    # Inclusive bounds
    V.validate_range(5, 5, 10, "value")
    V.validate_range(10, 5, 10, "value")
    V.validate_range(7, 5, 10, "value")
    with pytest.raises(ValueError):
        V.validate_range(4.9, 5, 10, "value")
    with pytest.raises(ValueError):
        V.validate_range(10.1, 5, 10, "value")


def test_validate_github_token_format():
    assert V.validate_github_token_format("a" * 40) is True  # classic length
    assert V.validate_github_token_format("ghp_1234") is True
    assert V.validate_github_token_format("ghs_1234") is True
    assert V.validate_github_token_format("gho_1234") is True
    assert V.validate_github_token_format("ghu_1234") is True
    # too short and no prefix
    assert V.validate_github_token_format("abc") is False
    # wrong type / empty
    assert V.validate_github_token_format(123) is False
    assert V.validate_github_token_format("") is False


def test_validate_gemini_api_key_format():
    assert V.validate_gemini_api_key_format("X" * 11) is True
    assert V.validate_gemini_api_key_format("short") is False
    assert V.validate_gemini_api_key_format(12345) is False
    assert V.validate_gemini_api_key_format("") is False


def test_ensure_positive_or_default():
    assert V.ensure_positive_or_default(5, 2) == 5
    assert V.ensure_positive_or_default(0, 2) == 2
    assert V.ensure_positive_or_default(-1, 2) == 2
