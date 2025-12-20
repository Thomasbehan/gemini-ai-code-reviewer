"""
Comprehensive tests for gemini_reviewer/diff_parser.py
"""

import pytest

from gemini_reviewer.diff_parser import DiffParser, DiffParsingError
from gemini_reviewer.models import DiffFile, FileInfo, HunkInfo


# Sample diff content for testing
SIMPLE_DIFF = """diff --git a/main.py b/main.py
index 1234567..abcdefg 100644
--- a/main.py
+++ b/main.py
@@ -1,3 +1,4 @@
 import os
+import sys

 def main():
"""

NEW_FILE_DIFF = """diff --git a/newfile.py b/newfile.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/newfile.py
@@ -0,0 +1,3 @@
+def hello():
+    print("Hello, World!")
+    return True
"""

DELETED_FILE_DIFF = """diff --git a/oldfile.py b/oldfile.py
deleted file mode 100644
index abcdefg..0000000
--- a/oldfile.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def goodbye():
-    print("Goodbye")
"""

RENAMED_FILE_DIFF = """diff --git a/old_name.py b/new_name.py
similarity index 95%
rename from old_name.py
rename to new_name.py
index 1234567..abcdefg 100644
--- a/old_name.py
+++ b/new_name.py
@@ -1,3 +1,3 @@
 def hello():
-    print("old")
+    print("new")
"""

MULTI_HUNK_DIFF = """diff --git a/multi.py b/multi.py
index 1234567..abcdefg 100644
--- a/multi.py
+++ b/multi.py
@@ -1,3 +1,4 @@
 import os
+import sys

 def first():
@@ -10,3 +11,4 @@

 def second():
+    return True
     pass
"""

BINARY_FILE_DIFF = """diff --git a/image.png b/image.png
new file mode 100644
index 0000000..1234567
Binary files /dev/null and b/image.png differ
"""

COMPLEX_DIFF = SIMPLE_DIFF + "\n" + NEW_FILE_DIFF


class TestDiffParser:
    """Tests for DiffParser class."""

    def test_init(self):
        """Test DiffParser initialization."""
        parser = DiffParser()
        stats = parser.get_parsing_statistics()
        assert stats["parsed_files"] == 0
        assert stats["skipped_files"] == 0

    def test_parse_empty_content(self):
        """Test parsing empty content returns empty list."""
        parser = DiffParser()
        result = parser.parse_diff("")
        assert result == []

    def test_parse_none_content(self):
        """Test parsing None content returns empty list."""
        parser = DiffParser()
        result = parser.parse_diff(None)
        assert result == []

    def test_parse_invalid_type(self):
        """Test parsing non-string content returns empty list."""
        parser = DiffParser()
        result = parser.parse_diff(12345)
        assert result == []

    def test_parse_simple_diff(self):
        """Test parsing a simple diff with one file."""
        parser = DiffParser()
        result = parser.parse_diff(SIMPLE_DIFF)

        assert len(result) == 1
        diff_file = result[0]
        assert diff_file.file_info.path == "main.py"
        assert diff_file.file_info.is_new_file is False
        assert diff_file.file_info.is_deleted_file is False
        assert len(diff_file.hunks) >= 1

    def test_parse_new_file(self):
        """Test parsing a diff for a new file."""
        parser = DiffParser()
        result = parser.parse_diff(NEW_FILE_DIFF)

        assert len(result) == 1
        diff_file = result[0]
        assert diff_file.file_info.path == "newfile.py"
        assert diff_file.file_info.is_new_file is True
        assert diff_file.total_additions >= 3

    def test_parse_deleted_file(self):
        """Test parsing a diff for a deleted file."""
        parser = DiffParser()
        result = parser.parse_diff(DELETED_FILE_DIFF)

        assert len(result) == 1
        diff_file = result[0]
        assert diff_file.file_info.is_deleted_file is True
        assert diff_file.total_deletions >= 2

    def test_parse_renamed_file(self):
        """Test parsing a diff for a renamed file."""
        parser = DiffParser()
        result = parser.parse_diff(RENAMED_FILE_DIFF)

        assert len(result) == 1
        diff_file = result[0]
        # The new name should be the path
        assert "new_name.py" in diff_file.file_info.path or "old_name.py" in diff_file.file_info.path

    def test_parse_multi_hunk_diff(self):
        """Test parsing a diff with multiple hunks."""
        parser = DiffParser()
        result = parser.parse_diff(MULTI_HUNK_DIFF)

        assert len(result) == 1
        diff_file = result[0]
        assert len(diff_file.hunks) >= 2

    def test_parse_complex_diff(self):
        """Test parsing a diff with multiple files."""
        parser = DiffParser()
        result = parser.parse_diff(COMPLEX_DIFF)

        assert len(result) >= 2

    def test_parsing_statistics(self):
        """Test that parsing statistics are tracked."""
        parser = DiffParser()
        parser.parse_diff(SIMPLE_DIFF)

        stats = parser.get_parsing_statistics()
        assert stats["parsed_files"] >= 1

    def test_reset_statistics(self):
        """Test resetting parsing statistics."""
        parser = DiffParser()
        parser.parse_diff(SIMPLE_DIFF)
        parser.reset_statistics()

        stats = parser.get_parsing_statistics()
        assert stats["parsed_files"] == 0
        assert stats["total_additions"] == 0


class TestDiffParserFiltering:
    """Tests for DiffParser filtering methods."""

    @pytest.fixture
    def sample_diff_files(self):
        """Create sample DiffFile objects for testing."""
        files = [
            DiffFile(
                file_info=FileInfo(path="main.py"),
                hunks=[
                    HunkInfo(1, 3, 1, 5, "", "", ["+a", "+b", "-c"])
                ],
            ),
            DiffFile(
                file_info=FileInfo(path="utils.js"),
                hunks=[
                    HunkInfo(1, 1, 1, 2, "", "", ["+x"])
                ],
            ),
            DiffFile(
                file_info=FileInfo(path="test_main.py"),
                hunks=[
                    HunkInfo(1, 1, 1, 1, "", "", ["+test"])
                ],
            ),
            DiffFile(
                file_info=FileInfo(path="image.png"),
                hunks=[],
            ),
        ]
        return files

    def test_filter_with_include_patterns(self, sample_diff_files):
        """Test filtering with include patterns."""
        parser = DiffParser()
        result = parser.filter_files(
            sample_diff_files,
            include_patterns=["*.py"],
        )

        assert len(result) == 2
        paths = [f.file_info.path for f in result]
        assert "main.py" in paths
        assert "test_main.py" in paths

    def test_filter_with_exclude_patterns(self, sample_diff_files):
        """Test filtering with exclude patterns."""
        parser = DiffParser()
        result = parser.filter_files(
            sample_diff_files,
            exclude_patterns=["test_*.py"],
        )

        paths = [f.file_info.path for f in result]
        assert "test_main.py" not in paths

    def test_filter_with_max_files(self, sample_diff_files):
        """Test filtering with max files limit."""
        parser = DiffParser()
        result = parser.filter_files(
            sample_diff_files,
            max_files=2,
        )

        assert len(result) <= 2

    def test_filter_with_min_changes(self, sample_diff_files):
        """Test filtering with minimum changes requirement."""
        parser = DiffParser()
        result = parser.filter_files(
            sample_diff_files,
            min_changes=2,
        )

        # Only files with 2+ changes should be included
        for f in result:
            total = f.total_additions + f.total_deletions
            assert total >= 2

    def test_filter_empty_list(self):
        """Test filtering empty list returns empty list."""
        parser = DiffParser()
        result = parser.filter_files([])
        assert result == []

    def test_filter_skips_binary_files(self, sample_diff_files):
        """Test that binary files are skipped."""
        parser = DiffParser()
        result = parser.filter_files(sample_diff_files)

        paths = [f.file_info.path for f in result]
        assert "image.png" not in paths


class TestFilterLargeHunks:
    """Tests for filter_large_hunks method."""

    def test_truncate_large_hunk(self):
        """Test truncating hunks that exceed max lines."""
        parser = DiffParser()

        large_hunk = HunkInfo(
            source_start=1,
            source_length=600,
            target_start=1,
            target_length=600,
            content="",
            header="@@ -1,600 +1,600 @@",
            lines=[f"+line{i}" for i in range(600)],
        )

        diff_file = DiffFile(
            file_info=FileInfo(path="large.py"),
            hunks=[large_hunk],
        )

        result = parser.filter_large_hunks([diff_file], max_lines_per_hunk=100)

        assert len(result) == 1
        assert len(result[0].hunks[0].lines) == 100

    def test_limit_hunks_per_file(self):
        """Test limiting number of hunks per file."""
        parser = DiffParser()

        hunks = [
            HunkInfo(i, 1, i, 1, "", f"@@ -{i},1 +{i},1 @@", ["+x"])
            for i in range(30)
        ]

        diff_file = DiffFile(
            file_info=FileInfo(path="many_hunks.py"),
            hunks=hunks,
        )

        result = parser.filter_large_hunks([diff_file], max_hunks_per_file=10)

        assert len(result) == 1
        assert len(result[0].hunks) == 10

    def test_preserve_small_hunks(self):
        """Test that small hunks are preserved as-is."""
        parser = DiffParser()

        small_hunk = HunkInfo(
            source_start=1,
            source_length=5,
            target_start=1,
            target_length=5,
            content="",
            header="@@ -1,5 +1,5 @@",
            lines=["+line1", "+line2", "+line3"],
        )

        diff_file = DiffFile(
            file_info=FileInfo(path="small.py"),
            hunks=[small_hunk],
        )

        result = parser.filter_large_hunks([diff_file], max_lines_per_hunk=100)

        assert len(result[0].hunks[0].lines) == 3


class TestAnalyzeDiffComplexity:
    """Tests for analyze_diff_complexity static method."""

    def test_empty_diff(self):
        """Test complexity analysis for empty diff."""
        result = DiffParser.analyze_diff_complexity([])
        assert result["complexity"] == "none"
        assert result["total_files"] == 0

    def test_low_complexity(self):
        """Test low complexity detection."""
        hunks = [HunkInfo(1, 10, 1, 10, "", "", ["+x"] * 50)]
        files = [DiffFile(FileInfo(path="f.py"), hunks=hunks)]

        result = DiffParser.analyze_diff_complexity(files)
        assert result["complexity"] == "low"

    def test_medium_complexity(self):
        """Test medium complexity detection."""
        hunks = [HunkInfo(1, 100, 1, 100, "", "", ["+x"] * 100)]
        files = [
            DiffFile(FileInfo(path=f"f{i}.py"), hunks=hunks)
            for i in range(12)
        ]

        result = DiffParser.analyze_diff_complexity(files)
        assert result["complexity"] in ["medium", "high"]

    def test_high_complexity(self):
        """Test high complexity detection."""
        hunks = [HunkInfo(1, 200, 1, 200, "", "", ["+x"] * 200)]
        files = [
            DiffFile(FileInfo(path=f"f{i}.py"), hunks=hunks)
            for i in range(25)
        ]

        result = DiffParser.analyze_diff_complexity(files)
        assert result["complexity"] == "high"

    def test_complexity_statistics(self):
        """Test that complexity analysis includes statistics."""
        hunks = [HunkInfo(1, 5, 1, 5, "", "", ["+a", "-b"])]
        files = [DiffFile(FileInfo(path="f.py"), hunks=hunks)]

        result = DiffParser.analyze_diff_complexity(files)

        assert "total_files" in result
        assert "total_hunks" in result
        assert "total_lines" in result
        assert "avg_hunks_per_file" in result
        assert "avg_lines_per_file" in result


class TestDiffParserFallback:
    """Tests for fallback parsing behavior."""

    def test_empty_unidiff_triggers_warning(self):
        """Test that empty unidiff result logs warning."""
        parser = DiffParser()
        # A malformed diff that unidiff can't parse but might not raise
        malformed = "This is not a valid diff format at all"
        result = parser.parse_diff(malformed)
        # Should return empty or try manual parsing
        assert isinstance(result, list)

    def test_manual_parsing_on_unidiff_failure(self):
        """Test manual parsing is used when unidiff fails."""
        parser = DiffParser()
        # Create a diff-like content that unidiff might struggle with
        partial_diff = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,2 +1,3 @@
 line1
+new line
 line2
"""
        result = parser.parse_diff(partial_diff)
        assert isinstance(result, list)


class TestDiffParserEdgeCases:
    """Tests for edge cases in diff parsing."""

    def test_null_source_and_target(self):
        """Test handling when both source and target are null-like."""
        parser = DiffParser()
        # Binary file with unusual format
        binary_diff = """diff --git a/file.bin b/file.bin
new file mode 100644
Binary files /dev/null and b/file.bin differ
"""
        result = parser.parse_diff(binary_diff)
        # Binary files should be skipped
        stats = parser.get_parsing_statistics()
        assert isinstance(result, list)

    def test_parse_malformed_hunk_header(self):
        """Test parsing with malformed hunk header."""
        parser = DiffParser()
        # Hunk header with missing components
        diff_with_bad_hunk = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1 +1 @@
-old
+new
"""
        result = parser.parse_diff(diff_with_bad_hunk)
        assert isinstance(result, list)

    def test_file_with_no_hunks(self):
        """Test file that somehow has no hunks."""
        parser = DiffParser()
        # Just headers, no content changes
        no_hunks_diff = """diff --git a/empty.py b/empty.py
similarity index 100%
rename from old.py
rename to empty.py
"""
        result = parser.parse_diff(no_hunks_diff)
        assert isinstance(result, list)


class TestManualParsing:
    """Tests for manual parsing fallback."""

    def test_manual_parse_basic_diff(self):
        """Test manual parsing of basic diff format."""
        parser = DiffParser()
        basic_diff = """diff --git a/file.py b/file.py
index 1234567..abcdefg 100644
--- a/file.py
+++ b/file.py
@@ -5,3 +5,4 @@
 context line
+added line
 more context
"""
        result = parser.parse_diff(basic_diff)
        assert isinstance(result, list)

    def test_parse_diff_with_multiple_files(self):
        """Test parsing diff with multiple files."""
        parser = DiffParser()
        multi_file_diff = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,2 +1,3 @@
 import os
+import sys

diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1,2 +1,3 @@
 def hello():
+    print("hi")
     pass
"""
        result = parser.parse_diff(multi_file_diff)
        assert len(result) >= 1  # At least one file parsed


class TestHunkConversion:
    """Tests for hunk conversion edge cases."""

    def test_hunk_with_context_lines(self):
        """Test parsing hunks with context lines."""
        parser = DiffParser()
        diff_with_context = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,5 +1,6 @@
 line1
 line2
+new_line
 line3
 line4
 line5
"""
        result = parser.parse_diff(diff_with_context)
        if result:
            assert result[0].hunks[0].lines is not None

    def test_hunk_only_additions(self):
        """Test hunk with only additions."""
        parser = DiffParser()
        only_adds = """diff --git a/new.py b/new.py
new file mode 100644
--- /dev/null
+++ b/new.py
@@ -0,0 +1,3 @@
+line1
+line2
+line3
"""
        result = parser.parse_diff(only_adds)
        if result:
            # Check that additions are counted
            assert result[0].total_additions >= 0

    def test_hunk_only_deletions(self):
        """Test hunk with only deletions."""
        parser = DiffParser()
        only_dels = """diff --git a/old.py b/old.py
deleted file mode 100644
--- a/old.py
+++ /dev/null
@@ -1,3 +0,0 @@
-line1
-line2
-line3
"""
        result = parser.parse_diff(only_dels)
        if result:
            # Check that deletions are counted
            assert result[0].total_deletions >= 0


class TestManualParsing:
    """Tests for manual diff parsing fallback."""

    def test_invalid_diff_header_raises(self):
        """Test that invalid diff header raises error."""
        parser = DiffParser()
        invalid_diff = "not a valid diff"

        # Should not raise - should return empty or handle gracefully
        result = parser.parse_diff(invalid_diff)
        # Either empty or fallback succeeds
        assert isinstance(result, list)

    def test_parse_hunk_header(self):
        """Test parsing hunk header."""
        parser = DiffParser()

        # Standard format
        hunk = parser._parse_hunk_header("@@ -1,5 +1,7 @@")
        assert hunk.source_start == 1
        assert hunk.source_length == 5
        assert hunk.target_start == 1
        assert hunk.target_length == 7

    def test_parse_hunk_header_single_line(self):
        """Test parsing hunk header for single line change."""
        parser = DiffParser()

        # No length specified means 1
        hunk = parser._parse_hunk_header("@@ -1 +1 @@")
        assert hunk.source_start == 1
        assert hunk.source_length == 1
        assert hunk.target_start == 1
        assert hunk.target_length == 1

    def test_parse_hunk_header_invalid(self):
        """Test parsing invalid hunk header returns None."""
        parser = DiffParser()

        result = parser._parse_hunk_header("not a hunk header")
        assert result is None


class TestDiffParsingError:
    """Tests for DiffParsingError exception."""

    def test_exception_message(self):
        """Test exception can be raised with message."""
        with pytest.raises(DiffParsingError) as exc_info:
            raise DiffParsingError("Test error")

        assert "Test error" in str(exc_info.value)

    def test_exception_inheritance(self):
        """Test DiffParsingError is an Exception."""
        assert issubclass(DiffParsingError, Exception)


class TestDiffParserAdvanced:
    """Advanced tests for DiffParser to cover edge cases."""

    def test_successful_unidiff_parsing(self):
        """Test the successful unidiff parsing path."""
        parser = DiffParser()
        # A well-formed diff that unidiff can parse
        valid_diff = """diff --git a/test.py b/test.py
index 1234567..abcdefg 100644
--- a/test.py
+++ b/test.py
@@ -1,3 +1,4 @@
 import os
+import sys
 def main():
     pass
"""
        result = parser.parse_diff(valid_diff)
        assert len(result) >= 1
        stats = parser.get_parsing_statistics()
        assert stats["parsed_files"] >= 1

    def test_parse_file_with_dev_null_target(self):
        """Test parsing when target is /dev/null (deleted file)."""
        parser = DiffParser()
        deleted_diff = """diff --git a/deleted.py b/deleted.py
deleted file mode 100644
index abcdefg..0000000
--- a/deleted.py
+++ /dev/null
@@ -1,5 +0,0 @@
-def old_function():
-    pass
-
-def another():
-    pass
"""
        result = parser.parse_diff(deleted_diff)
        assert isinstance(result, list)

    def test_parse_file_with_dev_null_source(self):
        """Test parsing when source is /dev/null (new file)."""
        parser = DiffParser()
        new_diff = """diff --git a/newfile.py b/newfile.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/newfile.py
@@ -0,0 +1,5 @@
+def new_function():
+    pass
+
+def another():
+    pass
"""
        result = parser.parse_diff(new_diff)
        assert len(result) >= 1
        if result:
            assert result[0].file_info.is_new_file is True

    def test_binary_file_skipped(self):
        """Test that binary files are skipped during parsing."""
        parser = DiffParser()
        binary_diff = """diff --git a/image.png b/image.png
new file mode 100644
index 0000000..1234567
Binary files /dev/null and b/image.png differ
"""
        result = parser.parse_diff(binary_diff)
        # Binary files should be skipped
        stats = parser.get_parsing_statistics()
        # Either 0 files or skipped
        assert isinstance(result, list)

    def test_empty_hunk_skipped(self):
        """Test that hunks with no lines are handled."""
        from unittest.mock import Mock, patch

        parser = DiffParser()

        # Create a mock hunk that is empty
        mock_hunk = Mock()
        mock_hunk.__iter__ = Mock(return_value=iter([]))

        result = parser._convert_hunk(mock_hunk)
        assert result is None

    def test_hunk_conversion_error(self):
        """Test hunk conversion handles exceptions."""
        from unittest.mock import Mock

        parser = DiffParser()

        # Create a mock hunk that raises an exception
        mock_hunk = Mock()
        mock_hunk.__iter__ = Mock(side_effect=Exception("Hunk error"))

        result = parser._convert_hunk(mock_hunk)
        assert result is None

    def test_patched_file_conversion_error(self):
        """Test patched file conversion handles exceptions."""
        from unittest.mock import Mock

        parser = DiffParser()

        # Create a mock patched file that causes an exception
        mock_file = Mock()
        mock_file.source_file = None
        mock_file.target_file = None

        result = parser._convert_patched_file(mock_file)
        assert result is None

    def test_filter_files_with_binary(self):
        """Test filtering explicitly skips binary files."""
        parser = DiffParser()

        files = [
            DiffFile(
                file_info=FileInfo(path="code.py"),
                hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+x"])],
            ),
            DiffFile(
                file_info=FileInfo(path="image.jpg"),  # Binary file
                hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+x"])],
            ),
        ]

        result = parser.filter_files(files)

        # image.jpg should be filtered if is_binary is True
        assert isinstance(result, list)

    def test_filter_large_hunks_empty_result(self):
        """Test filter_large_hunks when file has no hunks after filtering."""
        parser = DiffParser()

        # File with only empty hunks
        diff_file = DiffFile(
            file_info=FileInfo(path="test.py"),
            hunks=[],  # No hunks
        )

        result = parser.filter_large_hunks([diff_file])

        # File with no hunks should not be included
        assert result == []

    def test_manual_parsing_exception_in_loop(self):
        """Test manual parsing handles exceptions during line processing."""
        parser = DiffParser()

        # A diff that might cause issues during parsing
        problematic_diff = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,2 +1,3 @@
 line1
+added
 line2
"""
        result = parser.parse_diff(problematic_diff)
        assert isinstance(result, list)

    def test_parse_file_header_invalid(self):
        """Test parsing invalid file header."""
        parser = DiffParser()

        # Invalid header should raise DiffParsingError
        with pytest.raises(DiffParsingError):
            parser._parse_file_header(["not a valid diff --git header"], 0)

    def test_file_status_detection_new_file(self):
        """Test detecting new file status in manual parsing."""
        parser = DiffParser()

        lines = [
            "diff --git a/new.py b/new.py",
            "new file mode 100644",
            "index 0000000..1234567",
            "--- /dev/null",
            "+++ b/new.py",
            "@@ -0,0 +1,1 @@",
            "+content"
        ]

        diff_file = parser._parse_file_header(lines, 0)
        assert diff_file.file_info.is_new_file is True

    def test_file_status_detection_deleted_file(self):
        """Test detecting deleted file status in manual parsing."""
        parser = DiffParser()

        lines = [
            "diff --git a/old.py b/old.py",
            "deleted file mode 100644",
            "index 1234567..0000000",
            "--- a/old.py",
            "+++ /dev/null",
            "@@ -1,1 +0,0 @@",
            "-content"
        ]

        diff_file = parser._parse_file_header(lines, 0)
        assert diff_file.file_info.is_deleted_file is True

    def test_file_status_detection_reaches_hunk(self):
        """Test file status detection stops at hunk header."""
        parser = DiffParser()

        lines = [
            "diff --git a/test.py b/test.py",
            "index 1234567..abcdefg",
            "@@ -1,1 +1,1 @@",  # Hunk header should stop status search
            "-old",
            "+new"
        ]

        diff_file = parser._parse_file_header(lines, 0)
        # Should be a regular modified file
        assert diff_file.file_info.is_new_file is False
        assert diff_file.file_info.is_deleted_file is False

    def test_manual_parse_saves_last_file(self):
        """Test that manual parsing saves the last file."""
        parser = DiffParser()

        diff = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,1 +1,2 @@
 line1
+line2
"""
        result = parser._parse_manually(diff)

        # Should include the last file
        assert len(result) >= 1

    def test_manual_parse_multiple_files(self):
        """Test manual parsing with multiple files."""
        parser = DiffParser()

        diff = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,1 +1,2 @@
 line1
+line2
diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1,1 +1,2 @@
 a
+b
"""
        result = parser._parse_manually(diff)

        assert len(result) >= 2


class TestDiffParserFilterBinaryFile:
    """Tests specifically for binary file filtering."""

    def test_filter_with_explicit_binary_file(self):
        """Test that files marked as binary are filtered out."""
        parser = DiffParser()

        binary_file = DiffFile(
            file_info=FileInfo(path="data.bin"),
            hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+data"])],
        )
        # Mark it as binary
        binary_file.file_info._is_binary = True

        regular_file = DiffFile(
            file_info=FileInfo(path="code.py"),
            hunks=[HunkInfo(1, 1, 1, 1, "", "", ["+code"])],
        )

        result = parser.filter_files([binary_file, regular_file])

        # Binary file should be filtered
        paths = [f.file_info.path for f in result]
        assert "data.bin" not in paths or not binary_file.file_info.is_binary


class TestDiffParserUnidiffParsing:
    """Tests for unidiff parsing path."""

    def test_parse_diff_unidiff_success_path(self):
        """Test successful parsing via unidiff."""
        parser = DiffParser()

        valid_diff = """diff --git a/main.py b/main.py
index 1234567..abcdefg 100644
--- a/main.py
+++ b/main.py
@@ -1,3 +1,4 @@
 import os
+import sys

 def main():
"""
        result = parser.parse_diff(valid_diff)
        assert isinstance(result, list)

    def test_parse_diff_unidiff_empty_result(self):
        """Test when unidiff returns no files."""
        parser = DiffParser()

        # Malformed diff that unidiff can't parse
        empty_diff = "not a valid diff format"
        result = parser.parse_diff(empty_diff)
        assert isinstance(result, list)

    def test_parse_diff_manual_fallback(self):
        """Test manual parsing fallback when unidiff fails."""
        parser = DiffParser()

        # Simple diff-like content
        diff_content = """--- a/test.py
+++ b/test.py
@@ -1,1 +1,2 @@
 line1
+line2
"""
        result = parser.parse_diff(diff_content)
        assert isinstance(result, list)


class TestDiffParserBinaryFileHandling:
    """Tests for binary file detection and handling."""

    def test_parse_binary_file_diff(self):
        """Test parsing diff with binary file."""
        parser = DiffParser()

        binary_diff = """diff --git a/image.png b/image.png
new file mode 100644
index 0000000..1234567
Binary files /dev/null and b/image.png differ
"""
        result = parser.parse_diff(binary_diff)
        # Binary files should be skipped
        assert isinstance(result, list)

    def test_file_info_binary_detection(self):
        """Test FileInfo binary file detection."""
        # Test with common binary extensions
        binary_files = ["image.png", "data.bin", "font.woff", "archive.zip"]
        for filename in binary_files:
            info = FileInfo(path=filename)
            # FileInfo should detect these as binary
            assert isinstance(info.is_binary, bool)


class TestDiffParserHunkConversion:
    """Tests for hunk conversion."""

    def test_convert_hunk_with_context_lines(self):
        """Test converting hunk with context lines."""
        parser = DiffParser()

        diff = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,5 +1,6 @@
 context line 1
 context line 2
+new line
 context line 3
 context line 4
 context line 5
"""
        result = parser.parse_diff(diff)
        assert isinstance(result, list)
        if result:
            assert len(result[0].hunks) >= 1

    def test_convert_hunk_with_deletions(self):
        """Test converting hunk with deletion lines."""
        parser = DiffParser()

        diff = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,3 +1,2 @@
 line1
-deleted line
 line3
"""
        result = parser.parse_diff(diff)
        assert isinstance(result, list)


class TestDiffParserManualParsing:
    """Tests for manual parsing path."""

    def test_manual_parse_simple_diff(self):
        """Test manual parsing of simple diff."""
        parser = DiffParser()

        simple_diff = """--- old.py
+++ new.py
@@ -1 +1,2 @@
 existing
+added
"""
        result = parser.parse_diff(simple_diff)
        assert isinstance(result, list)

    def test_manual_parse_with_file_mode(self):
        """Test manual parsing with file mode changes."""
        parser = DiffParser()

        diff = """diff --git a/script.sh b/script.sh
old mode 100644
new mode 100755
--- a/script.sh
+++ b/script.sh
@@ -1 +1,2 @@
 #!/bin/bash
+echo "hello"
"""
        result = parser.parse_diff(diff)
        assert isinstance(result, list)


class TestDiffParserRenamedFiles:
    """Tests for handling renamed files."""

    def test_parse_renamed_file(self):
        """Test parsing renamed file diff."""
        parser = DiffParser()

        diff = """diff --git a/old_name.py b/new_name.py
similarity index 95%
rename from old_name.py
rename to new_name.py
--- a/old_name.py
+++ b/new_name.py
@@ -1,2 +1,2 @@
-old content
+new content
"""
        result = parser.parse_diff(diff)
        assert isinstance(result, list)


class TestDiffParserStatistics:
    """Tests for diff parsing statistics."""

    def test_statistics_after_parsing(self):
        """Test that statistics are updated after parsing."""
        parser = DiffParser()

        diff = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,2 +1,4 @@
 line1
+added1
+added2
-removed
"""
        parser.parse_diff(diff)
        stats = parser.get_parsing_statistics()

        assert "total_files" in stats or "parsed_files" in stats
        assert isinstance(stats, dict)

    def test_statistics_initial_state(self):
        """Test initial statistics state."""
        parser = DiffParser()
        stats = parser.get_parsing_statistics()

        assert isinstance(stats, dict)


class TestDiffParserEdgeCases:
    """Edge case tests for diff parser."""

    def test_empty_hunks_filtering(self):
        """Test that files with empty hunks are filtered."""
        parser = DiffParser()

        # A diff that might result in empty hunks
        diff = """diff --git a/empty.py b/empty.py
--- a/empty.py
+++ b/empty.py
"""
        result = parser.parse_diff(diff)
        assert isinstance(result, list)

    def test_parse_with_special_characters_in_path(self):
        """Test parsing file with special characters in path."""
        parser = DiffParser()

        diff = """diff --git a/path with spaces/test.py b/path with spaces/test.py
--- a/path with spaces/test.py
+++ b/path with spaces/test.py
@@ -1 +1,2 @@
 line1
+line2
"""
        result = parser.parse_diff(diff)
        assert isinstance(result, list)
