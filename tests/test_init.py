"""
Comprehensive tests for gemini_reviewer/__init__.py
"""

import pytest


class TestPackageMetadata:
    """Tests for package metadata."""

    def test_version_exists(self):
        """Test __version__ is defined."""
        from gemini_reviewer import __version__
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_version_format(self):
        """Test version follows semver format."""
        from gemini_reviewer import __version__
        parts = __version__.split(".")
        assert len(parts) >= 2  # At least major.minor

    def test_author_exists(self):
        """Test __author__ is defined."""
        from gemini_reviewer import __author__
        assert __author__ is not None

    def test_description_exists(self):
        """Test __description__ is defined."""
        from gemini_reviewer import __description__
        assert __description__ is not None


class TestLazyImports:
    """Tests for lazy import functionality."""

    def test_import_config(self):
        """Test lazy import of Config."""
        from gemini_reviewer import Config
        assert Config is not None

    def test_import_code_reviewer(self):
        """Test lazy import of CodeReviewer."""
        from gemini_reviewer import CodeReviewer
        assert CodeReviewer is not None

    def test_import_code_reviewer_error(self):
        """Test lazy import of CodeReviewerError."""
        from gemini_reviewer import CodeReviewerError
        assert CodeReviewerError is not None

    def test_import_pr_details(self):
        """Test lazy import of PRDetails."""
        from gemini_reviewer import PRDetails
        assert PRDetails is not None

    def test_import_review_result(self):
        """Test lazy import of ReviewResult."""
        from gemini_reviewer import ReviewResult
        assert ReviewResult is not None

    def test_import_review_comment(self):
        """Test lazy import of ReviewComment."""
        from gemini_reviewer import ReviewComment
        assert ReviewComment is not None

    def test_import_diff_file(self):
        """Test lazy import of DiffFile."""
        from gemini_reviewer import DiffFile
        assert DiffFile is not None

    def test_import_file_info(self):
        """Test lazy import of FileInfo."""
        from gemini_reviewer import FileInfo
        assert FileInfo is not None

    def test_import_hunk_info(self):
        """Test lazy import of HunkInfo."""
        from gemini_reviewer import HunkInfo
        assert HunkInfo is not None

    def test_import_analysis_context(self):
        """Test lazy import of AnalysisContext."""
        from gemini_reviewer import AnalysisContext
        assert AnalysisContext is not None

    def test_import_processing_stats(self):
        """Test lazy import of ProcessingStats."""
        from gemini_reviewer import ProcessingStats
        assert ProcessingStats is not None

    def test_import_review_priority(self):
        """Test lazy import of ReviewPriority."""
        from gemini_reviewer import ReviewPriority
        assert ReviewPriority is not None

    def test_import_review_focus(self):
        """Test lazy import of ReviewFocus."""
        from gemini_reviewer import ReviewFocus
        assert ReviewFocus is not None

    def test_import_github_client(self):
        """Test lazy import of GitHubClient."""
        from gemini_reviewer import GitHubClient
        assert GitHubClient is not None

    def test_import_github_client_error(self):
        """Test lazy import of GitHubClientError."""
        from gemini_reviewer import GitHubClientError
        assert GitHubClientError is not None

    def test_import_gemini_client(self):
        """Test lazy import of GeminiClient."""
        from gemini_reviewer import GeminiClient
        assert GeminiClient is not None

    def test_import_gemini_client_error(self):
        """Test lazy import of GeminiClientError."""
        from gemini_reviewer import GeminiClientError
        assert GeminiClientError is not None

    def test_import_diff_parser(self):
        """Test lazy import of DiffParser."""
        from gemini_reviewer import DiffParser
        assert DiffParser is not None

    def test_import_diff_parsing_error(self):
        """Test lazy import of DiffParsingError."""
        from gemini_reviewer import DiffParsingError
        assert DiffParsingError is not None

    def test_import_context_builder(self):
        """Test lazy import of ContextBuilder."""
        from gemini_reviewer import ContextBuilder
        assert ContextBuilder is not None

    def test_import_comment_processor(self):
        """Test lazy import of CommentProcessor."""
        from gemini_reviewer import CommentProcessor
        assert CommentProcessor is not None


class TestLazyImportCaching:
    """Tests for lazy import caching behavior."""

    def test_repeated_import_returns_same_object(self):
        """Test that repeated imports return the same object."""
        from gemini_reviewer import Config as Config1
        from gemini_reviewer import Config as Config2
        assert Config1 is Config2

    def test_import_is_cached_in_globals(self):
        """Test that imports are cached in module globals."""
        import gemini_reviewer

        # Access to trigger lazy load
        _ = gemini_reviewer.Config

        # Should now be in module globals
        assert "Config" in dir(gemini_reviewer)


class TestInvalidImport:
    """Tests for invalid import handling."""

    def test_invalid_attribute_raises(self):
        """Test that invalid attribute raises AttributeError."""
        import gemini_reviewer

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = gemini_reviewer.NonExistentClass


class TestImportErrors:
    """Tests for import error handling."""

    def test_import_error_provides_clear_message(self):
        """Test that import errors provide clear message."""
        import gemini_reviewer
        from unittest.mock import patch

        # Temporarily modify the lazy exports to point to a non-existent module
        original_exports = gemini_reviewer._lazy_exports.copy()
        gemini_reviewer._lazy_exports["FakeClass"] = (
            "nonexistent.module",
            "FakeClass",
        )

        try:
            with pytest.raises(ImportError, match="Failed to import"):
                _ = gemini_reviewer.FakeClass
        finally:
            # Restore original exports
            gemini_reviewer._lazy_exports = original_exports
            # Clean up any cached value
            if "FakeClass" in gemini_reviewer.__dict__:
                del gemini_reviewer.__dict__["FakeClass"]


class TestAllExports:
    """Tests for __all__ exports."""

    def test_all_defined(self):
        """Test __all__ is defined."""
        from gemini_reviewer import __all__
        assert __all__ is not None
        assert isinstance(__all__, list)

    def test_all_contains_main_classes(self):
        """Test __all__ contains main classes."""
        from gemini_reviewer import __all__

        expected = [
            "Config",
            "CodeReviewer",
            "CodeReviewerError",
            "PRDetails",
            "ReviewResult",
            "ReviewComment",
            "DiffFile",
            "GitHubClient",
            "GeminiClient",
        ]

        for item in expected:
            assert item in __all__, f"Missing {item} in __all__"

    def test_all_items_importable(self):
        """Test all items in __all__ can be imported."""
        from gemini_reviewer import __all__
        import gemini_reviewer

        for name in __all__:
            obj = getattr(gemini_reviewer, name)
            assert obj is not None, f"Failed to import {name}"
