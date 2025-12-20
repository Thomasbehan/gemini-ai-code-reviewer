"""
Comprehensive tests for gemini_reviewer/utils.py
"""

import pytest

from gemini_reviewer.utils import (
    matches_pattern,
    is_test_file,
    is_doc_file,
    get_file_language,
    sanitize_text,
    sanitize_code_content,
    is_binary_file,
)


class TestMatchesPattern:
    """Tests for matches_pattern function."""

    def test_exact_match(self):
        """Test exact filename match."""
        assert matches_pattern("file.py", "file.py") is True
        assert matches_pattern("file.py", "file.js") is False

    def test_wildcard_extension(self):
        """Test wildcard extension matching."""
        assert matches_pattern("main.py", "*.py") is True
        assert matches_pattern("main.js", "*.py") is False

    def test_wildcard_prefix(self):
        """Test wildcard prefix matching."""
        assert matches_pattern("test_main.py", "test_*") is True
        assert matches_pattern("main_test.py", "test_*") is False

    def test_double_star_pattern(self):
        """Test double-star pattern for directory matching."""
        assert matches_pattern("src/main.py", "src/*.py") is True
        # fnmatch's * matches any characters including /
        assert matches_pattern("src/sub/main.py", "src/*.py") is True
        # Use ** for explicit recursive matching
        assert matches_pattern("src/sub/main.py", "src/**/*.py") is True

    def test_question_mark_wildcard(self):
        """Test single character wildcard."""
        assert matches_pattern("file1.py", "file?.py") is True
        assert matches_pattern("file12.py", "file?.py") is False

    def test_no_extension_match(self):
        """Test matching files without extensions."""
        assert matches_pattern("Makefile", "Makefile") is True
        assert matches_pattern("Dockerfile", "Dockerfile") is True


class TestIsTestFile:
    """Tests for is_test_file function."""

    def test_test_prefix(self):
        """Test files with test_ prefix."""
        assert is_test_file("test_main.py") is True
        assert is_test_file("test_utils.py") is True

    def test_test_suffix(self):
        """Test files with _test suffix."""
        assert is_test_file("main_test.py") is True
        assert is_test_file("utils_test.js") is True

    def test_spec_prefix(self):
        """Test files with spec_ prefix."""
        assert is_test_file("spec_main.py") is True

    def test_spec_suffix(self):
        """Test files with _spec suffix."""
        assert is_test_file("main_spec.rb") is True

    def test_tests_directory(self):
        """Test files in tests/ directory."""
        # Pattern requires /tests/ - so needs leading slash in path
        assert is_test_file("src/tests/main.py") is True
        assert is_test_file("/project/tests/utils.py") is True

    def test_test_directory(self):
        """Test files in test/ directory."""
        # Pattern requires /test/ - so needs leading slash in path
        assert is_test_file("src/test/main.py") is True
        assert is_test_file("/project/test/utils.py") is True

    def test_windows_path_separator(self):
        """Test Windows-style path separators."""
        # Pattern requires \test\ or \tests\ - needs surrounding backslashes
        assert is_test_file("src\\test\\main.py") is True
        assert is_test_file("src\\tests\\utils.py") is True

    def test_non_test_files(self):
        """Test non-test files."""
        assert is_test_file("main.py") is False
        assert is_test_file("utils.py") is False
        assert is_test_file("src/app.py") is False

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert is_test_file("TEST_main.py") is True
        assert is_test_file("Main_TEST.py") is True


class TestIsDocFile:
    """Tests for is_doc_file function."""

    def test_markdown_files(self):
        """Test markdown files."""
        assert is_doc_file("README.md") is True
        assert is_doc_file("docs/guide.md") is True

    def test_rst_files(self):
        """Test reStructuredText files."""
        assert is_doc_file("index.rst") is True

    def test_txt_files(self):
        """Test plain text files."""
        assert is_doc_file("notes.txt") is True

    def test_doc_files(self):
        """Test Word document files."""
        assert is_doc_file("document.doc") is True
        assert is_doc_file("document.docx") is True

    def test_non_doc_files(self):
        """Test non-documentation files."""
        assert is_doc_file("main.py") is False
        assert is_doc_file("script.js") is False
        assert is_doc_file("config.json") is False

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert is_doc_file("README.MD") is True
        assert is_doc_file("guide.TXT") is True


class TestGetFileLanguage:
    """Tests for get_file_language function."""

    def test_python(self):
        """Test Python file detection."""
        assert get_file_language("main.py") == "python"

    def test_javascript(self):
        """Test JavaScript file detection."""
        assert get_file_language("app.js") == "javascript"
        assert get_file_language("component.jsx") == "javascript"

    def test_typescript(self):
        """Test TypeScript file detection."""
        assert get_file_language("app.ts") == "typescript"
        assert get_file_language("component.tsx") == "typescript"

    def test_java(self):
        """Test Java file detection."""
        assert get_file_language("Main.java") == "java"

    def test_kotlin(self):
        """Test Kotlin file detection."""
        assert get_file_language("Main.kt") == "kotlin"

    def test_go(self):
        """Test Go file detection."""
        assert get_file_language("main.go") == "go"

    def test_rust(self):
        """Test Rust file detection."""
        assert get_file_language("main.rs") == "rust"

    def test_cpp(self):
        """Test C++ file detection."""
        assert get_file_language("main.cpp") == "c++"
        assert get_file_language("main.cc") == "c++"
        assert get_file_language("main.cxx") == "c++"
        assert get_file_language("header.hpp") == "c++"

    def test_c(self):
        """Test C file detection."""
        assert get_file_language("main.c") == "c"
        assert get_file_language("header.h") == "c"

    def test_csharp(self):
        """Test C# file detection."""
        assert get_file_language("Program.cs") == "c#"

    def test_ruby(self):
        """Test Ruby file detection."""
        assert get_file_language("app.rb") == "ruby"

    def test_php(self):
        """Test PHP file detection."""
        assert get_file_language("index.php") == "php"

    def test_swift(self):
        """Test Swift file detection."""
        assert get_file_language("App.swift") == "swift"

    def test_scala(self):
        """Test Scala file detection."""
        assert get_file_language("Main.scala") == "scala"

    def test_r(self):
        """Test R file detection."""
        assert get_file_language("analysis.r") == "r"

    def test_sql(self):
        """Test SQL file detection."""
        assert get_file_language("query.sql") == "sql"

    def test_shell(self):
        """Test shell script detection."""
        assert get_file_language("script.sh") == "shell"
        assert get_file_language("script.bash") == "shell"

    def test_yaml(self):
        """Test YAML file detection."""
        assert get_file_language("config.yaml") == "yaml"
        assert get_file_language("config.yml") == "yaml"

    def test_json(self):
        """Test JSON file detection."""
        assert get_file_language("config.json") == "json"

    def test_xml(self):
        """Test XML file detection."""
        assert get_file_language("config.xml") == "xml"

    def test_html(self):
        """Test HTML file detection."""
        assert get_file_language("index.html") == "html"

    def test_css(self):
        """Test CSS file detection."""
        assert get_file_language("styles.css") == "css"

    def test_scss_sass(self):
        """Test SCSS/SASS file detection."""
        assert get_file_language("styles.scss") == "scss"
        assert get_file_language("styles.sass") == "sass"

    def test_vue(self):
        """Test Vue file detection."""
        assert get_file_language("App.vue") == "vue"

    def test_dart(self):
        """Test Dart file detection."""
        assert get_file_language("main.dart") == "dart"

    def test_lua(self):
        """Test Lua file detection."""
        assert get_file_language("script.lua") == "lua"

    def test_perl(self):
        """Test Perl file detection."""
        assert get_file_language("script.pl") == "perl"

    def test_elixir(self):
        """Test Elixir file detection."""
        assert get_file_language("app.ex") == "elixir"
        assert get_file_language("app.exs") == "elixir"

    def test_unknown_extension(self):
        """Test unknown extension returns 'unknown'."""
        assert get_file_language("file.xyz") == "unknown"
        assert get_file_language("file.unknown") == "unknown"

    def test_no_extension(self):
        """Test file with no extension returns 'unknown'."""
        assert get_file_language("Makefile") == "unknown"
        assert get_file_language("Dockerfile") == "unknown"

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert get_file_language("MAIN.PY") == "python"
        assert get_file_language("App.JS") == "javascript"


class TestSanitizeText:
    """Tests for sanitize_text function."""

    def test_empty_string(self):
        """Test empty string returns empty."""
        assert sanitize_text("") == ""

    def test_none_returns_empty(self):
        """Test None returns empty string."""
        assert sanitize_text(None) == ""

    def test_normal_text_unchanged(self):
        """Test normal text is preserved."""
        text = "This is normal text."
        assert sanitize_text(text) == text

    def test_preserves_newlines_and_tabs(self):
        """Test that newlines and tabs are preserved."""
        text = "Line 1\n\tIndented line"
        assert sanitize_text(text) == text

    def test_removes_null_bytes(self):
        """Test that null bytes are removed."""
        text = "Hello\x00World"
        assert sanitize_text(text) == "HelloWorld"

    def test_removes_control_characters(self):
        """Test that control characters are removed."""
        text = "Hello\x01\x02\x03World"
        assert sanitize_text(text) == "HelloWorld"

    def test_strips_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        text = "  text with spaces  "
        assert sanitize_text(text) == "text with spaces"

    def test_normalizes_excessive_newlines(self):
        """Test normalization of excessive newlines (max 3)."""
        text = "Line 1\n\n\n\n\n\nLine 2"
        result = sanitize_text(text)
        assert "\n\n\n\n" not in result
        assert "\n\n\n" in result

    def test_normalizes_excessive_spaces(self):
        """Test normalization of excessive spaces (max 8)."""
        text = "Word1" + " " * 15 + "Word2"
        result = sanitize_text(text)
        assert " " * 10 not in result


class TestSanitizeCodeContent:
    """Tests for sanitize_code_content function."""

    def test_empty_string(self):
        """Test empty string returns empty."""
        assert sanitize_code_content("") == ""

    def test_none_returns_empty(self):
        """Test None returns empty string."""
        assert sanitize_code_content(None) == ""

    def test_preserves_formatting(self):
        """Test that code formatting is preserved."""
        code = "def foo():\n    return 42"
        assert sanitize_code_content(code) == code

    def test_preserves_multiple_newlines(self):
        """Test that multiple newlines are preserved in code."""
        code = "class A:\n\n\n\n    pass"
        assert sanitize_code_content(code) == code

    def test_preserves_multiple_spaces(self):
        """Test that multiple spaces are preserved in code."""
        code = "text = 'hello     world'"
        assert sanitize_code_content(code) == code

    def test_removes_null_bytes(self):
        """Test that null bytes are removed from code."""
        code = "print('hello\x00world')"
        assert sanitize_code_content(code) == "print('helloworld')"


class TestIsBinaryFile:
    """Tests for is_binary_file function."""

    def test_image_files(self):
        """Test image file detection."""
        assert is_binary_file("image.png") is True
        assert is_binary_file("photo.jpg") is True
        assert is_binary_file("photo.jpeg") is True
        assert is_binary_file("animation.gif") is True

    def test_pdf_files(self):
        """Test PDF file detection."""
        assert is_binary_file("document.pdf") is True

    def test_archive_files(self):
        """Test archive file detection."""
        assert is_binary_file("archive.zip") is True
        assert is_binary_file("archive.tar") is True
        assert is_binary_file("archive.gz") is True

    def test_executable_files(self):
        """Test executable file detection."""
        assert is_binary_file("program.exe") is True
        assert is_binary_file("library.dll") is True
        assert is_binary_file("library.so") is True
        assert is_binary_file("library.dylib") is True

    def test_compiled_python(self):
        """Test compiled Python file detection."""
        assert is_binary_file("module.pyc") is True
        assert is_binary_file("module.pyo") is True

    def test_java_class(self):
        """Test Java class file detection."""
        assert is_binary_file("Main.class") is True

    def test_other_binary(self):
        """Test other binary file types."""
        assert is_binary_file("data.bin") is True
        assert is_binary_file("data.dat") is True

    def test_text_files(self):
        """Test that text files return False."""
        assert is_binary_file("main.py") is False
        assert is_binary_file("script.js") is False
        assert is_binary_file("config.json") is False
        assert is_binary_file("README.md") is False

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert is_binary_file("IMAGE.PNG") is True
        assert is_binary_file("Archive.ZIP") is True
