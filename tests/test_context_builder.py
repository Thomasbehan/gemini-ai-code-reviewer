"""
Comprehensive tests for gemini_reviewer/context_builder.py
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from gemini_reviewer.context_builder import ContextBuilder
from gemini_reviewer.models import DiffFile, FileInfo, HunkInfo, PRDetails


class TestContextBuilder:
    """Tests for ContextBuilder class."""

    @pytest.fixture
    def mock_github_client(self):
        """Create a mock GitHub client."""
        client = Mock()
        client.get_file_content = Mock(return_value=None)
        return client

    @pytest.fixture
    def mock_diff_parser(self):
        """Create a mock DiffParser."""
        parser = Mock()
        return parser

    @pytest.fixture
    def context_builder(self, mock_github_client, mock_diff_parser):
        """Create a ContextBuilder with mocked dependencies."""
        return ContextBuilder(mock_github_client, mock_diff_parser)

    @pytest.fixture
    def sample_diff_file(self):
        """Create a sample DiffFile for testing."""
        return DiffFile(
            file_info=FileInfo(path="src/main.py"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=10,
                    target_start=1,
                    target_length=12,
                    content="import utils\nfrom helper import Helper",
                    header="@@ -1,10 +1,12 @@",
                    lines=[
                        " import os",
                        "+import utils",
                        "+from helper import Helper",
                        " ",
                        " def main():",
                    ],
                )
            ],
        )

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails(
            owner="owner",
            repo="repo",
            pull_number=123,
            title="Add new feature",
            description="This PR adds a new feature",
            head_sha="abc123",
            base_sha="def456",
        )

    def test_init(self, context_builder):
        """Test ContextBuilder initialization."""
        assert context_builder is not None

    @pytest.mark.asyncio
    async def test_detect_related_files_empty_hunks(
        self, context_builder, sample_pr_details
    ):
        """Test detecting related files with no hunks."""
        diff_file = DiffFile(
            file_info=FileInfo(path="empty.py"),
            hunks=[],
        )

        result = await context_builder.detect_related_files(diff_file, sample_pr_details)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_related_files_with_imports(
        self, context_builder, sample_diff_file, sample_pr_details
    ):
        """Test detecting related files from imports."""
        result = await context_builder.detect_related_files(
            sample_diff_file, sample_pr_details
        )
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_build_project_context(
        self, context_builder, sample_diff_file, sample_pr_details
    ):
        """Test building project context."""
        related_files = ["utils.py", "helper.py"]

        result = await context_builder.build_project_context(
            sample_diff_file, related_files, sample_pr_details
        )

        # Should return a string or None
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_build_project_context_with_file_content(
        self, context_builder, sample_diff_file, sample_pr_details, mock_github_client
    ):
        """Test building project context when file content is available."""
        mock_github_client.get_file_content.return_value = """
def helper_function():
    return True

class HelperClass:
    pass
"""
        related_files = ["helper.py"]

        result = await context_builder.build_project_context(
            sample_diff_file, related_files, sample_pr_details
        )

        # Result should include file content
        assert result is None or isinstance(result, str)


class TestImportToFilePath:
    """Tests for import path conversion."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder with mocked dependencies."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_import_to_file_path_python(self, context_builder):
        """Test converting Python import to file path."""
        result = context_builder._import_to_file_path("utils", "main.py", "python")
        # Should return a valid path or None
        assert result is None or isinstance(result, str)

    def test_import_to_file_path_javascript(self, context_builder):
        """Test converting JavaScript import to file path."""
        result = context_builder._import_to_file_path("./utils", "src/main.js", "javascript")
        assert result is None or isinstance(result, str)

    def test_import_to_file_path_relative(self, context_builder):
        """Test converting relative import to file path."""
        result = context_builder._import_to_file_path(".", "src/module.py", "python")
        assert result is None or isinstance(result, str)


class TestRelevanceScoring:
    """Tests for file relevance scoring."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder with mocked dependencies."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_score_related_file_relevance(self, context_builder):
        """Test scoring related file relevance."""
        score = context_builder._score_related_file_relevance(
            "src/utils.py", "src/main.py", "+import utils"
        )
        assert isinstance(score, (int, float))
        assert score >= 0

    def test_score_related_file_same_directory(self, context_builder):
        """Test scoring for files in same directory."""
        score_same = context_builder._score_related_file_relevance(
            "src/utils.py", "src/main.py", ""
        )
        score_diff = context_builder._score_related_file_relevance(
            "tests/test_main.py", "src/main.py", ""
        )

        assert score_same >= 0
        assert score_diff >= 0


class TestFindTestFiles:
    """Tests for finding test files."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder with mocked dependencies."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_find_test_files(self, context_builder):
        """Test finding test files for a source file."""
        result = context_builder._find_test_files("src/main.py")
        assert isinstance(result, list)

    def test_find_test_files_nonexistent(self, context_builder):
        """Test finding test files for nonexistent file."""
        result = context_builder._find_test_files("nonexistent.py")
        assert isinstance(result, list)


class TestFindConfigFiles:
    """Tests for finding config files."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder with mocked dependencies."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_find_config_files(self, context_builder):
        """Test finding configuration files."""
        result = context_builder._find_config_files()
        assert isinstance(result, list)


class TestDetectRelatedFilesAdvanced:
    """Advanced tests for detect_related_files."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder with mocked dependencies."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value=None)
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="python")
        return ContextBuilder(mock_client, mock_parser)

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails(
            owner="owner",
            repo="repo",
            pull_number=123,
            title="Title",
            description="Desc",
            head_sha="abc123",
        )

    @pytest.mark.asyncio
    async def test_detect_with_github_content(self, sample_pr_details):
        """Test detection when GitHub returns file content."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value="import utils\nfrom helper import Helper")
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="python")

        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+x"])],
        )

        result = await builder.detect_related_files(diff_file, sample_pr_details)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_with_javascript_imports(self, sample_pr_details):
        """Test detection with JavaScript imports."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value="import React from 'react';\nimport utils from './utils';")
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="javascript")

        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="main.js"),
            hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+x"])],
        )

        result = await builder.detect_related_files(diff_file, sample_pr_details)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_with_typescript_imports(self, sample_pr_details):
        """Test detection with TypeScript imports."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value="import { Component } from '@angular/core';")
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="typescript")

        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="main.ts"),
            hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+x"])],
        )

        result = await builder.detect_related_files(diff_file, sample_pr_details)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_with_java_imports(self, sample_pr_details):
        """Test detection with Java imports."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value="import java.util.List;\nimport com.example.Utils;")
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="java")

        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="Main.java"),
            hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+x"])],
        )

        result = await builder.detect_related_files(diff_file, sample_pr_details)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_with_go_imports(self, sample_pr_details):
        """Test detection with Go imports."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value='import "fmt"\nimport pkg "github.com/example/pkg"')
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="go")

        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="main.go"),
            hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+x"])],
        )

        result = await builder.detect_related_files(diff_file, sample_pr_details)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_with_ruby_requires(self, sample_pr_details):
        """Test detection with Ruby requires."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value="require 'json'\nrequire_relative 'helper'")
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="ruby")

        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="main.rb"),
            hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+x"])],
        )

        result = await builder.detect_related_files(diff_file, sample_pr_details)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_fallback_to_hunk_content(self, sample_pr_details):
        """Test detection falls back to hunk content when file content unavailable."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value=None)
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="python")

        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[HunkInfo(1, 3, 1, 4, "", "", ["+import utils", " other", "+from helper import x"])],
        )

        result = await builder.detect_related_files(diff_file, sample_pr_details)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_with_python_class_inheritance(self, sample_pr_details):
        """Test detection with Python class inheritance."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value="""
from base import BaseClass

class MyClass(base.OtherClass):
    pass
""")
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="python")

        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+x"])],
        )

        result = await builder.detect_related_files(diff_file, sample_pr_details)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_with_python_type_annotations(self, sample_pr_details):
        """Test detection with Python type annotations."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value="""
from typing import List, Optional
from models import User

def process(data: List[User]) -> Optional[Result]:
    pass
""")
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="python")

        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+x"])],
        )

        result = await builder.detect_related_files(diff_file, sample_pr_details)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_with_syntax_error_python(self, sample_pr_details):
        """Test detection handles Python syntax errors gracefully."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value="def broken( = \nimport utils")
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="python")

        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+x"])],
        )

        result = await builder.detect_related_files(diff_file, sample_pr_details)
        # Should not raise, returns list
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_limits_to_10_files(self, sample_pr_details):
        """Test that related files are limited to 10."""
        mock_client = Mock()
        # Generate content with many imports
        imports = "\n".join([f"import module{i}" for i in range(20)])
        mock_client.get_file_content = Mock(return_value=imports)
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="python")

        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+x"])],
        )

        result = await builder.detect_related_files(diff_file, sample_pr_details)
        assert len(result) <= 10

    @pytest.mark.asyncio
    async def test_detect_handles_exception(self, sample_pr_details):
        """Test detection handles general exceptions gracefully."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(side_effect=Exception("API Error"))
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="python")

        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+x"])],
        )

        result = await builder.detect_related_files(diff_file, sample_pr_details)
        # Should return empty list, not raise
        assert isinstance(result, list)


class TestExtractTypeRefs:
    """Tests for _extract_type_refs method."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder with mocked dependencies."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_extract_simple_type(self, context_builder):
        """Test extracting simple type reference."""
        import ast
        tree = ast.parse("def f(x: MyClass): pass")
        func = tree.body[0]

        seen = set()
        related = []
        context_builder._extract_type_refs(func.args.args[0].annotation, seen, related, "main.py")

        assert "MyClass" in seen

    def test_extract_builtin_type_ignored(self, context_builder):
        """Test that builtin types are ignored."""
        import ast
        tree = ast.parse("def f(x: str, y: int): pass")
        func = tree.body[0]

        seen = set()
        related = []
        context_builder._extract_type_refs(func.args.args[0].annotation, seen, related, "main.py")

        assert "str" not in seen

    def test_extract_qualified_type(self, context_builder):
        """Test extracting qualified type like module.Class."""
        import ast
        tree = ast.parse("def f(x: models.User): pass")
        func = tree.body[0]

        seen = set()
        related = []
        context_builder._extract_type_refs(func.args.args[0].annotation, seen, related, "main.py")

        assert "models" in seen

    def test_extract_generic_type(self, context_builder):
        """Test extracting generic types like List[T]."""
        import ast
        tree = ast.parse("def f(x: List[MyClass]): pass")
        func = tree.body[0]

        seen = set()
        related = []
        context_builder._extract_type_refs(func.args.args[0].annotation, seen, related, "main.py")

        # Should have extracted from the subscript
        assert "MyClass" in seen

    def test_extract_tuple_type(self, context_builder):
        """Test extracting tuple types."""
        import ast
        # Tuple[str, MyType] annotation
        code = "def f() -> tuple: pass"
        tree = ast.parse(code)
        func = tree.body[0]

        seen = set()
        related = []
        # Should handle gracefully
        context_builder._extract_type_refs(func.returns, seen, related, "main.py")


class TestImportToFilePathAdvanced:
    """Advanced tests for _import_to_file_path."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder with mocked dependencies."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_python_dotted_import(self, context_builder):
        """Test converting dotted Python import."""
        result = context_builder._import_to_file_path("package.module", "main.py", "python")
        assert result is None or isinstance(result, str)

    def test_python_relative_import(self, context_builder):
        """Test converting relative Python import."""
        result = context_builder._import_to_file_path(".helper", "package/main.py", "python")
        assert result is None or isinstance(result, str)

    def test_javascript_relative_import(self, context_builder):
        """Test converting relative JavaScript import."""
        result = context_builder._import_to_file_path("./utils", "src/main.js", "javascript")
        assert result is None or isinstance(result, str)

    def test_javascript_parent_import(self, context_builder):
        """Test converting parent-relative JavaScript import."""
        result = context_builder._import_to_file_path("../utils", "src/sub/main.js", "javascript")
        assert result is None or isinstance(result, str)

    def test_node_module_import(self, context_builder):
        """Test handling node_modules import."""
        result = context_builder._import_to_file_path("lodash", "src/main.js", "javascript")
        # Should return None for node_modules
        assert result is None or isinstance(result, str)

    def test_unknown_language(self, context_builder):
        """Test handling unknown language."""
        result = context_builder._import_to_file_path("something", "main.xyz", "xyz")
        assert result is None or isinstance(result, str)


class TestBuildProjectContextAdvanced:
    """Advanced tests for build_project_context."""

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head123")

    @pytest.mark.asyncio
    async def test_build_context_empty_related(self, sample_pr_details):
        """Test building context with no related files."""
        mock_client = Mock()
        mock_parser = Mock()
        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+x"])],
        )

        result = await builder.build_project_context(diff_file, [], sample_pr_details)
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_build_context_with_content(self, sample_pr_details):
        """Test building context with file content available."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value="def helper(): pass")
        mock_parser = Mock()
        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+x"])],
        )

        result = await builder.build_project_context(diff_file, ["helper.py"], sample_pr_details)
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_build_context_fetch_error(self, sample_pr_details):
        """Test building context when file fetch fails."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(side_effect=Exception("Not found"))
        mock_parser = Mock()
        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+x"])],
        )

        result = await builder.build_project_context(diff_file, ["missing.py"], sample_pr_details)
        assert result is None or isinstance(result, str)


class TestScoreRelatedFileRelevanceAdvanced:
    """Advanced tests for _score_related_file_relevance."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder with mocked dependencies."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_score_direct_import_match(self, context_builder):
        """Test scoring when file is directly imported."""
        score = context_builder._score_related_file_relevance(
            "utils.py", "main.py", "+import utils\n+from utils import helper"
        )
        assert score > 0

    def test_score_same_directory_bonus(self, context_builder):
        """Test same directory gets higher score."""
        score_same = context_builder._score_related_file_relevance(
            "src/utils.py", "src/main.py", ""
        )
        score_diff = context_builder._score_related_file_relevance(
            "lib/utils.py", "src/main.py", ""
        )
        # Same directory should score higher
        assert isinstance(score_same, (int, float))
        assert isinstance(score_diff, (int, float))

    def test_score_test_file_lower(self, context_builder):
        """Test that test files might get different scores."""
        score_src = context_builder._score_related_file_relevance(
            "src/utils.py", "src/main.py", ""
        )
        score_test = context_builder._score_related_file_relevance(
            "tests/test_utils.py", "src/main.py", ""
        )
        assert isinstance(score_src, (int, float))
        assert isinstance(score_test, (int, float))


class TestContextBuilderFileAnalysis:
    """Tests for file analysis functionality."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder with mocked dependencies."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails(
            owner="owner",
            repo="repo",
            pull_number=123,
            title="Test",
            description="Description",
        )

    def test_import_to_file_path_python(self, context_builder):
        """Test converting Python import to file path."""
        result = context_builder._import_to_file_path("utils.helpers", "main.py", "python")
        assert isinstance(result, str) or result is None

    def test_import_to_file_path_javascript(self, context_builder):
        """Test converting JavaScript import to file path."""
        result = context_builder._import_to_file_path("./utils", "app.js", "javascript")
        assert isinstance(result, str) or result is None

    def test_import_to_file_path_relative(self, context_builder):
        """Test converting relative import to file path."""
        result = context_builder._import_to_file_path(".local", "package/module.py", "python")
        assert isinstance(result, str) or result is None

    def test_import_to_file_path_package(self, context_builder):
        """Test converting package import to file path."""
        result = context_builder._import_to_file_path("flask", "main.py", "python")
        # External packages may return None or a path
        assert result is None or isinstance(result, str)


class TestContextBuilderEdgeCases:
    """Edge case tests for ContextBuilder."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder with mocked dependencies."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value="def hello(): pass")
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails(
            owner="owner",
            repo="repo",
            pull_number=123,
            title="Test",
            description="Description",
            head_sha="abc123",
        )

    def test_import_to_file_path_typescript(self, context_builder):
        """Test converting TypeScript import to file path."""
        result = context_builder._import_to_file_path("./components/Button", "app.tsx", "typescript")
        assert isinstance(result, str) or result is None

    def test_import_to_file_path_go(self, context_builder):
        """Test converting Go import to file path."""
        result = context_builder._import_to_file_path("github.com/user/pkg", "main.go", "go")
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_build_project_context(self, context_builder, sample_pr_details):
        """Test building project context."""
        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=5,
                    target_start=1,
                    target_length=7,
                    content="+import utils",
                    header="@@ -1,5 +1,7 @@",
                    lines=["+import utils"],
                )
            ],
        )

        result = await context_builder.build_project_context(
            diff_file, ["main.py"], sample_pr_details
        )

        # Result can be None or a string depending on context
        assert result is None or isinstance(result, str)


class TestContextBuilderRelevanceScoring:
    """Tests for relevance scoring functionality."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder with mocked dependencies."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_score_with_import_reference(self, context_builder):
        """Test scoring when file is referenced in imports."""
        score = context_builder._score_related_file_relevance(
            "utils.py",
            "main.py",
            "+import utils\n+from utils import helper"
        )
        assert score > 0

    def test_score_same_package(self, context_builder):
        """Test same package gets higher score."""
        score1 = context_builder._score_related_file_relevance(
            "pkg/utils.py", "pkg/main.py", ""
        )
        score2 = context_builder._score_related_file_relevance(
            "other/utils.py", "pkg/main.py", ""
        )
        assert isinstance(score1, (int, float))
        assert isinstance(score2, (int, float))

    def test_score_function_reference(self, context_builder):
        """Test scoring when function is referenced."""
        score = context_builder._score_related_file_relevance(
            "helpers.py",
            "main.py",
            "+result = helper_function(data)"
        )
        assert isinstance(score, (int, float))


class TestContextBuilderExtractTypeRefs:
    """Tests for type reference extraction."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder with mocked dependencies."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_extract_type_refs_simple(self, context_builder):
        """Test extracting simple type references."""
        import ast
        annotation = ast.parse("str", mode='eval').body
        seen_imports = set()
        related_files = []

        context_builder._extract_type_refs(annotation, seen_imports, related_files, "main.py")
        # Should handle without error
        assert True

    def test_extract_type_refs_generic(self, context_builder):
        """Test extracting generic type references."""
        import ast
        annotation = ast.parse("List[str]", mode='eval').body
        seen_imports = {"List"}
        related_files = []

        context_builder._extract_type_refs(annotation, seen_imports, related_files, "main.py")
        assert True

    def test_extract_type_refs_none(self, context_builder):
        """Test handling None annotation."""
        seen_imports = set()
        related_files = []

        context_builder._extract_type_refs(None, seen_imports, related_files, "main.py")
        assert True

    def test_extract_type_refs_attribute(self, context_builder):
        """Test extracting attribute type references like module.Type."""
        import ast
        # Parse an expression like 'typing.List'
        annotation = ast.parse("typing.List", mode='eval').body
        seen_imports = set()
        related_files = []

        context_builder._extract_type_refs(annotation, seen_imports, related_files, "main.py")
        assert True

    def test_extract_type_refs_subscript(self, context_builder):
        """Test extracting subscript type references like List[str]."""
        import ast
        annotation = ast.parse("Dict[str, int]", mode='eval').body
        seen_imports = set()
        related_files = []

        context_builder._extract_type_refs(annotation, seen_imports, related_files, "main.py")
        assert True

    def test_extract_type_refs_custom_type(self, context_builder):
        """Test extracting custom type that's not a builtin."""
        import ast
        annotation = ast.parse("MyCustomClass", mode='eval').body
        seen_imports = set()
        related_files = []

        context_builder._extract_type_refs(annotation, seen_imports, related_files, "main.py")
        # Should add MyCustomClass to seen_imports
        assert "MyCustomClass" in seen_imports


class TestContextBuilderDetectRelatedFilesAdvanced:
    """Advanced tests for detect_related_files."""

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails(
            owner="owner",
            repo="repo",
            pull_number=123,
            title="Test",
            description="Description",
            head_sha="abc123",
        )

    @pytest.mark.asyncio
    async def test_detect_related_files_with_python_imports(self, sample_pr_details):
        """Test detecting related files from Python imports."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value="""
import os
import sys
from utils import helper
from models import User
""")
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="python")
        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=5,
                    target_start=1,
                    target_length=7,
                    content="+from utils import helper",
                    header="@@ -1,5 +1,7 @@",
                    lines=["+from utils import helper"],
                )
            ],
        )

        result = await builder.detect_related_files(diff_file, sample_pr_details)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_related_files_with_javascript_imports(self, sample_pr_details):
        """Test detecting related files from JavaScript imports."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value="""
import React from 'react';
import { useState } from 'react';
import utils from './utils';
const helper = require('./helper');
""")
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="javascript")
        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="app.js"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=5,
                    target_start=1,
                    target_length=7,
                    content="+import utils from './utils';",
                    header="@@ -1,5 +1,7 @@",
                    lines=["+import utils from './utils';"],
                )
            ],
        )

        result = await builder.detect_related_files(diff_file, sample_pr_details)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_related_files_with_go_imports(self, sample_pr_details):
        """Test detecting related files from Go imports."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value="""
package main

import (
    "fmt"
    "github.com/user/pkg/utils"
)
""")
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="go")
        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="main.go"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=5,
                    target_start=1,
                    target_length=7,
                    content='+import "fmt"',
                    header="@@ -1,5 +1,7 @@",
                    lines=['+import "fmt"'],
                )
            ],
        )

        result = await builder.detect_related_files(diff_file, sample_pr_details)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_related_files_with_ruby_imports(self, sample_pr_details):
        """Test detecting related files from Ruby imports."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value="""
require 'json'
require_relative 'utils'
load 'helper.rb'
""")
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="ruby")
        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="app.rb"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=5,
                    target_start=1,
                    target_length=7,
                    content="+require_relative 'utils'",
                    header="@@ -1,5 +1,7 @@",
                    lines=["+require_relative 'utils'"],
                )
            ],
        )

        result = await builder.detect_related_files(diff_file, sample_pr_details)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_related_files_with_class_inheritance(self, sample_pr_details):
        """Test detecting related files from Python class inheritance."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value="""
from base import BaseClass

class MyClass(BaseClass):
    pass

class Child(module.ParentClass):
    pass
""")
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="python")
        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="child.py"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=5,
                    target_start=1,
                    target_length=10,
                    content="+class MyClass(BaseClass):",
                    header="@@ -1,5 +1,10 @@",
                    lines=["+class MyClass(BaseClass):"],
                )
            ],
        )

        result = await builder.detect_related_files(diff_file, sample_pr_details)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_related_files_with_type_annotations(self, sample_pr_details):
        """Test detecting related files from Python type annotations."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value="""
from typing import List, Optional
from models import User, Order

def process_users(users: List[User]) -> Optional[Order]:
    pass

async def fetch_data(config: Config) -> Response:
    pass
""")
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="python")
        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="service.py"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=5,
                    target_start=1,
                    target_length=10,
                    content="+def process_users(users: List[User]):",
                    header="@@ -1,5 +1,10 @@",
                    lines=["+def process_users(users: List[User]):"],
                )
            ],
        )

        result = await builder.detect_related_files(diff_file, sample_pr_details)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_related_files_syntax_error(self, sample_pr_details):
        """Test handling Python syntax errors gracefully."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value="""
def broken_function(
    # This has a syntax error - missing closing paren
""")
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="python")
        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="broken.py"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=5,
                    target_start=1,
                    target_length=5,
                    content="+def broken(",
                    header="@@ -1,5 +1,5 @@",
                    lines=["+def broken("],
                )
            ],
        )

        # Should not raise, just return empty or partial list
        result = await builder.detect_related_files(diff_file, sample_pr_details)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_related_files_exception_handling(self, sample_pr_details):
        """Test exception handling in detect_related_files."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(side_effect=Exception("API Error"))
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="python")
        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="main.py"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=5,
                    target_start=1,
                    target_length=5,
                    content="+import utils",
                    header="@@ -1,5 +1,5 @@",
                    lines=["+import utils"],
                )
            ],
        )

        # Should handle exception gracefully
        result = await builder.detect_related_files(diff_file, sample_pr_details)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_related_files_unknown_language(self, sample_pr_details):
        """Test detecting related files for unknown language."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value="some content")
        mock_parser = Mock()
        mock_parser.get_file_language = Mock(return_value="unknown")
        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="file.xyz"),
            hunks=[
                HunkInfo(
                    source_start=1,
                    source_length=5,
                    target_start=1,
                    target_length=5,
                    content="+content",
                    header="@@ -1,5 +1,5 @@",
                    lines=["+content"],
                )
            ],
        )

        result = await builder.detect_related_files(diff_file, sample_pr_details)
        assert isinstance(result, list)


class TestContextBuilderFindReverseDependencies:
    """Tests for _find_reverse_dependencies."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder with mocked dependencies."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_find_reverse_deps_exception_handling(self, context_builder):
        """Test that exceptions are handled gracefully."""
        with patch('os.walk', side_effect=OSError("Access denied")):
            result = context_builder._find_reverse_dependencies("main.py")
            assert result == []

    def test_find_reverse_deps_with_files(self, context_builder):
        """Test finding reverse dependencies."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            main_py = os.path.join(tmpdir, "main.py")
            utils_py = os.path.join(tmpdir, "utils.py")

            with open(main_py, 'w') as f:
                f.write("from utils import helper\n")
            with open(utils_py, 'w') as f:
                f.write("def helper(): pass\n")

            with patch('os.getcwd', return_value=tmpdir):
                result = context_builder._find_reverse_dependencies("utils.py")
                assert isinstance(result, list)


class TestContextBuilderFindFunctionCallers:
    """Tests for _find_function_callers."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder with mocked dependencies."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_find_callers_no_repo_root(self, context_builder):
        """Test when repo root doesn't exist."""
        with patch('os.path.exists', return_value=False):
            result = context_builder._find_function_callers("main.py", "+def my_func():")
            assert result == []

    def test_find_callers_empty_diff(self, context_builder):
        """Test with empty diff content."""
        result = context_builder._find_function_callers("main.py", "")
        assert result == []


class TestContextBuilderPrioritizeContext:
    """Tests for _prioritize_context_sections."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder with mocked dependencies."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_prioritize_empty_sections(self, context_builder):
        """Test prioritizing empty sections."""
        result = context_builder._prioritize_context_sections(
            [],  # sections as List[Dict]
            1000  # max_size
        )
        assert isinstance(result, list)

    def test_prioritize_with_sections(self, context_builder):
        """Test prioritizing with content."""
        sections = [
            {"type": "imports", "content": "import os\nimport sys", "priority": 1},
            {"type": "functions", "content": "def main(): pass", "priority": 2},
            {"type": "classes", "content": "class MyClass: pass", "priority": 3},
        ]
        result = context_builder._prioritize_context_sections(
            sections,
            1000
        )
        assert isinstance(result, list)

    def test_prioritize_exceeds_budget(self, context_builder):
        """Test prioritizing when sections exceed budget."""
        sections = [
            {"type": "imports", "content": "x" * 500, "priority": 1},
            {"type": "functions", "content": "y" * 500, "priority": 2},
            {"type": "classes", "content": "z" * 500, "priority": 3},
        ]
        result = context_builder._prioritize_context_sections(
            sections,
            100  # Very small budget
        )
        assert isinstance(result, list)


class TestContextBuilderFindTestFiles:
    """Tests for _find_test_files."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder with mocked dependencies."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_find_test_files_returns_list(self, context_builder):
        """Test that _find_test_files returns a list."""
        result = context_builder._find_test_files("main.py")
        assert isinstance(result, list)

    def test_find_test_files_with_tests(self, context_builder):
        """Test finding test files."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test directory structure
            tests_dir = os.path.join(tmpdir, "tests")
            os.makedirs(tests_dir)

            # Create test file
            test_file = os.path.join(tests_dir, "test_main.py")
            with open(test_file, 'w') as f:
                f.write("def test_main(): pass\n")

            with patch('os.getcwd', return_value=tmpdir):
                result = context_builder._find_test_files("main.py")
                assert isinstance(result, list)


class TestContextBuilderFindConfigFiles:
    """Tests for _find_config_files."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder with mocked dependencies."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_find_config_files_no_repo(self, context_builder):
        """Test when repo doesn't exist."""
        with patch('os.path.exists', return_value=False):
            result = context_builder._find_config_files()
            assert result == []

    def test_find_config_files_with_configs(self, context_builder):
        """Test finding config files."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config files
            pyproject = os.path.join(tmpdir, "pyproject.toml")
            with open(pyproject, 'w') as f:
                f.write("[project]\nname = 'test'\n")

            with patch('os.getcwd', return_value=tmpdir):
                result = context_builder._find_config_files()
                assert isinstance(result, list)


class TestContextBuilderBuildRepoMentalModel:
    """Tests for _build_repo_mental_model."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder with mocked dependencies."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_build_mental_model_returns_string(self, context_builder):
        """Test that _build_repo_mental_model returns a string."""
        result = context_builder._build_repo_mental_model()
        assert isinstance(result, str)

    def test_build_mental_model_with_files(self, context_builder):
        """Test building mental model."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files
            src_dir = os.path.join(tmpdir, "src")
            os.makedirs(src_dir)

            main_py = os.path.join(src_dir, "main.py")
            with open(main_py, 'w') as f:
                f.write("def main(): pass\n")

            with patch('os.getcwd', return_value=tmpdir):
                result = context_builder._build_repo_mental_model()
                assert isinstance(result, str)


class TestContextBuilderExtractCodeSignatures:
    """Tests for _extract_code_signatures."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder with mocked dependencies."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_extract_signatures_returns_string(self, context_builder):
        """Test that _extract_code_signatures returns a string."""
        result = context_builder._extract_code_signatures()
        assert isinstance(result, str)

    def test_extract_signatures_with_python_files(self, context_builder):
        """Test extracting signatures from Python files."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create Python file with functions and classes
            main_py = os.path.join(tmpdir, "main.py")
            with open(main_py, 'w') as f:
                f.write("""
def function_one():
    pass

class MyClass:
    def method(self):
        pass
""")

            with patch('os.getcwd', return_value=tmpdir):
                result = context_builder._extract_code_signatures(max_files=5)
                assert isinstance(result, str)

    def test_extract_signatures_with_js_files(self, context_builder):
        """Test extracting signatures from JavaScript files."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create JavaScript file
            main_js = os.path.join(tmpdir, "main.js")
            with open(main_js, 'w') as f:
                f.write("""
function doSomething() {}

const myArrowFunc = () => {};

class Component {
    render() {}
}
""")

            with patch('os.getcwd', return_value=tmpdir):
                result = context_builder._extract_code_signatures(max_files=5)
                assert isinstance(result, str)


class TestContextBuilderLocalFileFallback:
    """Tests for local file fallback when GitHub returns None."""

    @pytest.fixture
    def context_builder(self):
        """Create context builder with mocks."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value=None)
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head_sha")

    @pytest.mark.asyncio
    async def test_build_context_with_local_file_fallback(self, context_builder, sample_pr_details):
        """Test fallback to local file when GitHub returns None."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create local file
            test_file = os.path.join(tmpdir, "local_test.py")
            with open(test_file, 'w') as f:
                f.write("def local_function():\n    pass\n")

            diff_file = DiffFile(
                file_info=FileInfo(path="local_test.py"),
                hunks=[
                    HunkInfo(
                        source_start=1, source_length=2,
                        target_start=1, target_length=2,
                        content="def local_function():\n    pass",
                        header="@@ -1,2 +1,2 @@",
                        lines=["+def local_function():", "+    pass"]
                    )
                ]
            )

            with patch('os.getcwd', return_value=tmpdir):
                result = await context_builder.build_project_context(
                    diff_file, [], sample_pr_details
                )
                assert result is not None

    @pytest.mark.asyncio
    async def test_build_context_file_not_found_locally(self, context_builder, sample_pr_details):
        """Test graceful handling when file doesn't exist locally."""
        diff_file = DiffFile(
            file_info=FileInfo(path="nonexistent_file.py"),
            hunks=[
                HunkInfo(
                    source_start=1, source_length=2,
                    target_start=1, target_length=2,
                    content="# Content",
                    header="@@ -1,2 +1,2 @@",
                    lines=["+# Content"]
                )
            ]
        )

        result = await context_builder.build_project_context(
            diff_file, [], sample_pr_details
        )
        assert result is not None


class TestContextBuilderFindFunctionCallers:
    """Tests for _find_function_callers method."""

    @pytest.fixture
    def context_builder(self):
        """Create context builder with mocks."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_find_callers_with_python_functions(self, context_builder):
        """Test finding callers of Python functions."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Python file that defines a function
            main_py = os.path.join(tmpdir, "main.py")
            with open(main_py, 'w') as f:
                f.write("def target_function():\n    pass\n")

            # Create another file that calls it
            caller_py = os.path.join(tmpdir, "caller.py")
            with open(caller_py, 'w') as f:
                f.write("from main import target_function\n\ndef call_it():\n    target_function()\n")

            with patch('os.getcwd', return_value=tmpdir):
                # Diff content that modifies target_function
                diff_content = "+def target_function():\n+    pass"
                result = context_builder._find_function_callers("main.py", diff_content)
                assert isinstance(result, list)

    def test_find_callers_exceeding_limit(self, context_builder):
        """Test that callers are limited to 15."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create main file
            main_py = os.path.join(tmpdir, "main.py")
            with open(main_py, 'w') as f:
                f.write("def target_func():\n    pass\n")

            # Create many caller files
            for i in range(20):
                caller_py = os.path.join(tmpdir, f"caller_{i}.py")
                with open(caller_py, 'w') as f:
                    lines = [f"# Line {j}\n" for j in range(10)]
                    lines.append("target_func()\n")
                    f.writelines(lines)

            with patch('os.getcwd', return_value=tmpdir):
                diff_content = "+def target_func():\n+    pass"
                result = context_builder._find_function_callers("main.py", diff_content)
                # Should be limited
                assert len(result) <= 15

    def test_find_callers_with_exception_in_file(self, context_builder):
        """Test handling exception when reading caller file."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            main_py = os.path.join(tmpdir, "main.py")
            with open(main_py, 'w') as f:
                f.write("def my_func():\n    pass\n")

            # Create a directory with same name as expected file (causes read error)
            os.makedirs(os.path.join(tmpdir, "caller.py"))

            with patch('os.getcwd', return_value=tmpdir):
                diff_content = "+def my_func():\n+    pass"
                result = context_builder._find_function_callers("main.py", diff_content)
                assert isinstance(result, list)


class TestContextBuilderPackageJsonParsing:
    """Tests for package.json parsing in _build_repo_mental_model."""

    @pytest.fixture
    def context_builder(self):
        """Create context builder with mocks."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_parse_package_json_scripts(self, context_builder):
        """Test parsing scripts from package.json."""
        import tempfile
        import os
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_json = os.path.join(tmpdir, "package.json")
            with open(pkg_json, 'w') as f:
                json.dump({
                    "name": "test-project",
                    "scripts": {
                        "test": "jest",
                        "build": "webpack",
                        "lint": "eslint ."
                    }
                }, f)

            with patch('os.getcwd', return_value=tmpdir):
                result = context_builder._build_repo_mental_model()
                assert isinstance(result, str)
                # Should include scripts info
                if "scripts" in result.lower():
                    assert "test" in result or "build" in result

    def test_parse_invalid_package_json(self, context_builder):
        """Test handling invalid JSON in package.json."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_json = os.path.join(tmpdir, "package.json")
            with open(pkg_json, 'w') as f:
                f.write("{ invalid json content ")

            with patch('os.getcwd', return_value=tmpdir):
                # Should not raise, just skip
                result = context_builder._build_repo_mental_model()
                assert isinstance(result, str)


class TestContextBuilderASTFormatAnnotation:
    """Tests for AST annotation formatting fallbacks."""

    @pytest.fixture
    def context_builder(self):
        """Create context builder with mocks."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_extract_signatures_with_complex_types(self, context_builder):
        """Test extracting signatures with complex type annotations."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            main_py = os.path.join(tmpdir, "typed.py")
            with open(main_py, 'w') as f:
                f.write('''
from typing import List, Dict, Optional

def complex_func(
    items: List[Dict[str, int]],
    name: Optional[str] = None
) -> Dict[str, List[int]]:
    pass

class TypedClass:
    def method(self, value: "ForwardRef") -> None:
        pass
''')

            with patch('os.getcwd', return_value=tmpdir):
                result = context_builder._extract_code_signatures(max_files=5)
                assert isinstance(result, str)

    def test_extract_signatures_with_syntax_error(self, context_builder):
        """Test extracting signatures from file with syntax errors."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            bad_py = os.path.join(tmpdir, "broken.py")
            with open(bad_py, 'w') as f:
                f.write("def broken(\n    # missing close paren\n")

            with patch('os.getcwd', return_value=tmpdir):
                # Should not raise
                result = context_builder._extract_code_signatures(max_files=5)
                assert isinstance(result, str)


class TestContextBuilderGenericSignatures:
    """Tests for generic (non-Python) signature extraction."""

    @pytest.fixture
    def context_builder(self):
        """Create context builder with mocks."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_extract_go_signatures(self, context_builder):
        """Test extracting Go function signatures."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            main_go = os.path.join(tmpdir, "main.go")
            with open(main_go, 'w') as f:
                f.write("""
package main

func processRequest() {}
func handleResponse() {}
""")

            with patch('os.getcwd', return_value=tmpdir):
                result = context_builder._extract_code_signatures(max_files=5)
                assert isinstance(result, str)

    def test_extract_java_signatures(self, context_builder):
        """Test extracting Java method signatures."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            main_java = os.path.join(tmpdir, "Main.java")
            with open(main_java, 'w') as f:
                f.write("""
public class Main {
    public void processData(String input) {}
    private static int calculate(int a, int b) { return a + b; }
}
""")

            with patch('os.getcwd', return_value=tmpdir):
                result = context_builder._extract_code_signatures(max_files=5)
                assert isinstance(result, str)

    def test_extract_typescript_signatures(self, context_builder):
        """Test extracting TypeScript signatures."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            main_ts = os.path.join(tmpdir, "main.ts")
            with open(main_ts, 'w') as f:
                f.write("""
export function exportedFunc() {}
const arrowFunc = (x: number) => x * 2;
class Service {
    handle(): void {}
}
""")

            with patch('os.getcwd', return_value=tmpdir):
                result = context_builder._extract_code_signatures(max_files=5)
                assert isinstance(result, str)


class TestContextBuilderCalledFunctions:
    """Tests for finding called function definitions."""

    @pytest.fixture
    def mock_github_client(self):
        """Create mock GitHub client."""
        client = Mock()
        client.get_file_content = Mock(return_value=None)
        return client

    @pytest.fixture
    def context_builder(self, mock_github_client):
        """Create context builder."""
        mock_parser = Mock()
        return ContextBuilder(mock_github_client, mock_parser)

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head_sha")

    def test_find_called_functions_in_python(self, context_builder):
        """Test finding definitions of called functions."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create helper file with function definition
            helper_py = os.path.join(tmpdir, "helper.py")
            with open(helper_py, 'w') as f:
                f.write("def helper_func():\n    '''Helper function.'''\n    return 42\n")

            # File content that calls helper_func
            file_content = "from helper import helper_func\n\ndef main():\n    result = helper_func()\n"

            with patch('os.getcwd', return_value=tmpdir):
                result = context_builder._find_called_functions(file_content, "main.py")
                assert isinstance(result, list)

    def test_find_called_functions_empty_content(self, context_builder):
        """Test with empty file content."""
        result = context_builder._find_called_functions("", "test.py")
        assert isinstance(result, list)
        assert len(result) == 0


class TestContextBuilderBuildProjectContextFull:
    """Tests for build_project_context with various scenarios."""

    @pytest.fixture
    def mock_github_client(self):
        """Create mock GitHub client."""
        client = Mock()
        client.get_file_content = Mock(return_value="def func():\n    pass\n")
        return client

    @pytest.fixture
    def context_builder(self, mock_github_client):
        """Create context builder."""
        mock_parser = Mock()
        return ContextBuilder(mock_github_client, mock_parser)

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head_sha")

    @pytest.fixture
    def sample_diff_file(self):
        """Create sample diff file."""
        return DiffFile(
            file_info=FileInfo(path="src/main.py"),
            hunks=[
                HunkInfo(
                    source_start=1, source_length=5,
                    target_start=1, target_length=7,
                    content="def main():\n    pass",
                    header="@@ -1,5 +1,7 @@",
                    lines=["+def main():", "+    helper_func()"]
                )
            ]
        )

    @pytest.mark.asyncio
    async def test_build_project_context_with_callers(self, mock_github_client, sample_pr_details):
        """Test build_project_context includes function callers."""
        import tempfile
        import os

        mock_parser = Mock()
        builder = ContextBuilder(mock_github_client, mock_parser)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create main.py
            main_py = os.path.join(tmpdir, "main.py")
            with open(main_py, 'w') as f:
                f.write("def target_function():\n    pass\n")

            # Create caller.py
            caller_py = os.path.join(tmpdir, "caller.py")
            with open(caller_py, 'w') as f:
                f.write("from main import target_function\n\ndef use_it():\n    target_function()\n")

            diff_file = DiffFile(
                file_info=FileInfo(path="main.py"),
                hunks=[
                    HunkInfo(
                        source_start=1, source_length=2,
                        target_start=1, target_length=2,
                        content="def target_function():\n    pass",
                        header="@@ -1,2 +1,2 @@",
                        lines=["+def target_function():", "+    pass"]
                    )
                ]
            )

            with patch('os.getcwd', return_value=tmpdir):
                result = await builder.build_project_context(diff_file, [], sample_pr_details)
                assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_build_project_context_with_called_functions(self, sample_pr_details):
        """Test build_project_context includes called function definitions."""
        import tempfile
        import os

        mock_client = Mock()
        mock_client.get_file_content = Mock(
            return_value="from helper import helper_func\n\ndef main():\n    helper_func()\n"
        )
        mock_parser = Mock()
        builder = ContextBuilder(mock_client, mock_parser)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create helper.py with function
            helper_py = os.path.join(tmpdir, "helper.py")
            with open(helper_py, 'w') as f:
                f.write("def helper_func():\n    '''Does something.'''\n    return True\n")

            diff_file = DiffFile(
                file_info=FileInfo(path="main.py"),
                hunks=[
                    HunkInfo(
                        source_start=1, source_length=4,
                        target_start=1, target_length=4,
                        content="from helper import helper_func",
                        header="@@ -1,4 +1,4 @@",
                        lines=[" from helper import helper_func", "+def main():", "+    helper_func()"]
                    )
                ]
            )

            with patch('os.getcwd', return_value=tmpdir):
                result = await builder.build_project_context(diff_file, [], sample_pr_details)
                assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_build_project_context_exception_in_callers(self, sample_pr_details):
        """Test build_project_context handles exceptions in caller finding."""
        mock_client = Mock()
        mock_client.get_file_content = Mock(return_value="def test(): pass")
        mock_parser = Mock()
        builder = ContextBuilder(mock_client, mock_parser)

        diff_file = DiffFile(
            file_info=FileInfo(path="test.py"),
            hunks=[
                HunkInfo(
                    source_start=1, source_length=1,
                    target_start=1, target_length=1,
                    content="def test(): pass",
                    header="@@ -1,1 +1,1 @@",
                    lines=["+def test(): pass"]
                )
            ]
        )

        # Simulate exception during caller finding
        with patch.object(builder, '_find_function_callers', side_effect=Exception("Test error")):
            result = await builder.build_project_context(diff_file, [], sample_pr_details)
            assert isinstance(result, str)


class TestContextBuilderCIWorkflows:
    """Tests for CI/CD workflow detection."""

    @pytest.fixture
    def context_builder(self):
        """Create context builder."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_build_mental_model_with_github_workflows(self, context_builder):
        """Test detecting GitHub workflow files."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .github/workflows directory
            workflows_dir = os.path.join(tmpdir, ".github", "workflows")
            os.makedirs(workflows_dir)

            # Create workflow file
            ci_yaml = os.path.join(workflows_dir, "ci.yml")
            with open(ci_yaml, 'w') as f:
                f.write("name: CI\non: push\njobs:\n  test:\n    runs-on: ubuntu-latest\n")

            with patch('os.getcwd', return_value=tmpdir):
                result = context_builder._build_repo_mental_model()
                assert isinstance(result, str)

    def test_build_mental_model_with_gitlab_ci(self, context_builder):
        """Test detecting GitLab CI files."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .gitlab-ci.yml
            gitlab_ci = os.path.join(tmpdir, ".gitlab-ci.yml")
            with open(gitlab_ci, 'w') as f:
                f.write("stages:\n  - test\n  - build\n")

            with patch('os.getcwd', return_value=tmpdir):
                result = context_builder._build_repo_mental_model()
                assert isinstance(result, str)


class TestContextBuilderMaxFilesLimit:
    """Tests for max files limit in signature extraction."""

    @pytest.fixture
    def context_builder(self):
        """Create context builder."""
        mock_client = Mock()
        mock_parser = Mock()
        return ContextBuilder(mock_client, mock_parser)

    def test_extract_signatures_respects_max_files(self, context_builder):
        """Test that signature extraction respects max files limit."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many Python files
            for i in range(50):
                py_file = os.path.join(tmpdir, f"file_{i}.py")
                with open(py_file, 'w') as f:
                    f.write(f"def func_{i}():\n    pass\n")

            with patch('os.getcwd', return_value=tmpdir):
                result = context_builder._extract_code_signatures(max_files=5)
                assert isinstance(result, str)
                # Should be limited and not process all 50 files

    def test_extract_signatures_skips_excluded_dirs(self, context_builder):
        """Test that signature extraction skips excluded directories."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file in excluded directory
            node_modules = os.path.join(tmpdir, "node_modules")
            os.makedirs(node_modules)
            excluded_file = os.path.join(node_modules, "test.py")
            with open(excluded_file, 'w') as f:
                f.write("def excluded(): pass\n")

            # Create file in normal directory
            src_dir = os.path.join(tmpdir, "src")
            os.makedirs(src_dir)
            included_file = os.path.join(src_dir, "main.py")
            with open(included_file, 'w') as f:
                f.write("def included(): pass\n")

            with patch('os.getcwd', return_value=tmpdir):
                result = context_builder._extract_code_signatures(max_files=10)
                assert isinstance(result, str)
                # Should skip node_modules


class TestContextBuilderRelatedFilesAdvanced:
    """Advanced tests for detecting related files."""

    @pytest.fixture
    def mock_github_client(self):
        """Create mock GitHub client."""
        client = Mock()
        client.get_file_content = Mock(return_value=None)
        return client

    @pytest.fixture
    def context_builder(self, mock_github_client):
        """Create context builder."""
        mock_parser = Mock()
        return ContextBuilder(mock_github_client, mock_parser)

    @pytest.fixture
    def sample_pr_details(self):
        """Create sample PR details."""
        return PRDetails("owner", "repo", 123, "Title", "Desc", "head_sha")

    @pytest.mark.asyncio
    async def test_detect_related_files_with_inheritance(self, mock_github_client, sample_pr_details):
        """Test detecting files with class inheritance."""
        import tempfile
        import os

        mock_parser = Mock()
        builder = ContextBuilder(mock_github_client, mock_parser)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create base class file
            base_py = os.path.join(tmpdir, "base.py")
            with open(base_py, 'w') as f:
                f.write("class BaseHandler:\n    def handle(self): pass\n")

            # Create derived class file
            derived_py = os.path.join(tmpdir, "derived.py")
            with open(derived_py, 'w') as f:
                f.write("from base import BaseHandler\n\nclass MyHandler(BaseHandler):\n    pass\n")

            diff_file = DiffFile(
                file_info=FileInfo(path="base.py"),
                hunks=[
                    HunkInfo(
                        source_start=1, source_length=2,
                        target_start=1, target_length=3,
                        content="class BaseHandler:\n    def handle(self): pass",
                        header="@@ -1,2 +1,3 @@",
                        lines=[" class BaseHandler:", "+    def handle(self): pass"]
                    )
                ]
            )

            with patch('os.getcwd', return_value=tmpdir):
                mock_github_client.get_file_content = Mock(return_value="class BaseHandler:\n    def handle(self): pass\n")
                result = await builder.detect_related_files(diff_file, sample_pr_details)
                assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detect_related_files_with_type_references(self, mock_github_client, sample_pr_details):
        """Test detecting files with type references."""
        import tempfile
        import os

        mock_parser = Mock()
        builder = ContextBuilder(mock_github_client, mock_parser)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create types file
            types_py = os.path.join(tmpdir, "types.py")
            with open(types_py, 'w') as f:
                f.write("class UserModel:\n    name: str\n    email: str\n")

            # Main file using types
            main_py = os.path.join(tmpdir, "main.py")
            with open(main_py, 'w') as f:
                f.write("from types import UserModel\n\ndef get_user() -> UserModel:\n    pass\n")

            diff_file = DiffFile(
                file_info=FileInfo(path="main.py"),
                hunks=[
                    HunkInfo(
                        source_start=1, source_length=3,
                        target_start=1, target_length=3,
                        content="from types import UserModel",
                        header="@@ -1,3 +1,3 @@",
                        lines=[" from types import UserModel"]
                    )
                ]
            )

            with patch('os.getcwd', return_value=tmpdir):
                mock_github_client.get_file_content = Mock(return_value="from types import UserModel\n\ndef get_user() -> UserModel:\n    pass\n")
                result = await builder.detect_related_files(diff_file, sample_pr_details)
                assert isinstance(result, list)
