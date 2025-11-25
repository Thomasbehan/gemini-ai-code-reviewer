import os
import pytest

from enum import Enum
import gemini_reviewer.env_reader as E
from gemini_reviewer.prompts import ReviewMode


def test_get_env_str_with_and_without_fallback(monkeypatch):
    monkeypatch.delenv("PRIMARY_KEY", raising=False)
    monkeypatch.delenv("FALLBACK1", raising=False)
    assert E.get_env_str("PRIMARY_KEY", "x", "FALLBACK1") == "x"

    monkeypatch.setenv("FALLBACK1", "fb")
    assert E.get_env_str("PRIMARY_KEY", "x", "FALLBACK1") == "fb"

    monkeypatch.setenv("PRIMARY_KEY", "main")
    assert E.get_env_str("PRIMARY_KEY", "x", "FALLBACK1") == "main"


def test_get_env_int_success_and_failure(monkeypatch):
    monkeypatch.delenv("N_INT", raising=False)
    assert E.get_env_int("N_INT", 7) == 7
    monkeypatch.setenv("N_INT", "42")
    assert E.get_env_int("N_INT", 7) == 42
    # invalid -> default
    monkeypatch.setenv("N_INT", "not-an-int")
    assert E.get_env_int("N_INT", 7) == 7


def test_get_env_float_success_and_failure(monkeypatch):
    monkeypatch.delenv("N_FLOAT", raising=False)
    assert E.get_env_float("N_FLOAT", 1.5) == 1.5
    monkeypatch.setenv("N_FLOAT", "2.25")
    assert E.get_env_float("N_FLOAT", 1.5) == 2.25
    monkeypatch.setenv("N_FLOAT", "oops")
    assert E.get_env_float("N_FLOAT", 1.5) == 1.5


def test_get_env_bool_variants(monkeypatch):
    monkeypatch.delenv("N_BOOL", raising=False)
    assert E.get_env_bool("N_BOOL", True) is True
    for v in ["true", "TRUE", "yes", "1"]:
        monkeypatch.setenv("N_BOOL", v)
        assert E.get_env_bool("N_BOOL", False) is True
    for v in ["false", "no", "0", "random"]:
        monkeypatch.setenv("N_BOOL", v)
        assert E.get_env_bool("N_BOOL", True) is False


def test_get_env_list_with_separator(monkeypatch):
    monkeypatch.delenv("N_LIST", raising=False)
    assert E.get_env_list("N_LIST") == []
    monkeypatch.setenv("N_LIST", "a, b , ,c")
    assert E.get_env_list("N_LIST") == ["a", "b", "c"]
    monkeypatch.setenv("N_LIST", "a|b|c")
    assert E.get_env_list("N_LIST", separator="|") == ["a", "b", "c"]


def test_get_env_enum_with_valid_and_invalid(monkeypatch):
    # Use existing ReviewMode enum to ensure lowercase strings map correctly
    monkeypatch.setenv("MODE", "standard")
    assert E.get_env_enum("MODE", ReviewMode, ReviewMode.LENIENT) is ReviewMode.STANDARD

    monkeypatch.setenv("MODE", "INVALID")
    assert E.get_env_enum("MODE", ReviewMode, ReviewMode.LENIENT) is ReviewMode.LENIENT

    # Also test with a custom Enum
    class Color(Enum):
        RED = "red"
        BLUE = "blue"
    monkeypatch.setenv("COLOR", "Blue")
    assert E.get_env_enum("COLOR", Color, Color.RED) is Color.BLUE
