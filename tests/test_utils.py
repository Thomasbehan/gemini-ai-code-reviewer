import pytest

import gemini_reviewer.utils as U


def test_matches_pattern():
    assert U.matches_pattern("src/app.py", "src/*.py")
    assert not U.matches_pattern("src/app.js", "src/*.py")


def test_is_test_file():
    assert U.is_test_file("tests/test_example.py")  # '/tests/' pattern
    assert U.is_test_file("src/module_test.go")  # '_test.' pattern
    assert U.is_test_file("src/spec_helper.py")  # 'spec_' pattern
    assert U.is_test_file("lib/module_spec.py")  # '_spec.' pattern
    assert U.is_test_file("test_main.py")  # 'test_' prefix
    assert U.is_test_file("C\\proj\\tests\\file.py")  # Windows '\\tests\\' pattern
    assert U.is_test_file("spec/helpers.py") is False  # 'spec_' required, not folder name
    assert not U.is_test_file("src/app.py")


def test_is_doc_file():
    for ext in [".md", ".rst", ".txt", ".doc", ".docx"]:
        assert U.is_doc_file(f"README{ext}")
    assert not U.is_doc_file("main.py")


def test_get_file_language_mapping_and_unknown():
    mapping = {
        "py": "python", "js": "javascript", "ts": "typescript", "jsx": "javascript",
        "tsx": "typescript", "java": "java", "kt": "kotlin", "go": "go",
        "rs": "rust", "cpp": "c++", "cc": "c++", "cxx": "c++", "c": "c",
        "h": "c", "hpp": "c++", "cs": "c#", "rb": "ruby", "php": "php",
        "swift": "swift", "scala": "scala", "r": "r", "sql": "sql",
        "sh": "shell", "bash": "shell", "yaml": "yaml", "yml": "yaml",
        "json": "json", "xml": "xml", "html": "html", "css": "css",
        "scss": "scss", "sass": "sass", "vue": "vue", "dart": "dart",
        "lua": "lua", "pl": "perl", "ex": "elixir", "exs": "elixir",
    }
    for ext, lang in mapping.items():
        assert U.get_file_language(f"file.{ext}") == lang
    assert U.get_file_language("Makefile") == "unknown"


def test_sanitize_text_controls_and_whitespace():
    raw = "Hello\x00\x01World"  # contains control chars
    assert U.sanitize_text(raw) == "HelloWorld"

    # excessive newlines -> capped at 3
    text = "A\n\n\n\n\nB"
    assert U.sanitize_text(text) == "A\n\n\nB"

    # excessive spaces/tabs -> capped at 8 spaces
    text2 = "X" + (" " * 20) + "Y"
    assert U.sanitize_text(text2) == "X" + (" " * 8) + "Y"

    # None or empty returns empty string
    assert U.sanitize_text("") == ""
    assert U.sanitize_text(None) == ""


def test_sanitize_code_content_only_null_removed():
    code = "print(1)\x00\n\t  a =  1"
    assert U.sanitize_code_content(code) == "print(1)\n\t  a =  1"
    assert U.sanitize_code_content("") == ""
    assert U.sanitize_code_content(None) == ""


def test_is_binary_file_by_extension():
    for ext in ['.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.tar', '.gz', '.exe', '.dll', '.so', '.dylib', '.bin', '.dat', '.pyc', '.pyo', '.class']:
        assert U.is_binary_file(f"image{ext}")
    assert not U.is_binary_file("script.py")
