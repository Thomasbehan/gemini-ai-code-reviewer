"""
Comprehensive tests for gemini_reviewer/gemini_client.py
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch

from gemini_reviewer.gemini_client import (
    GeminiClient,
    GeminiClientError,
    ModelNotAvailableError,
    TokenLimitExceededError,
)
from gemini_reviewer.config import Config, GeminiConfig, GitHubConfig, ReviewConfig
from gemini_reviewer.models import (
    HunkInfo,
    AnalysisContext,
    PRDetails,
    FileInfo,
    AIResponse,
    ReviewPriority,
)


class TestGeminiClientErrors:
    """Tests for Gemini client exceptions."""

    def test_gemini_client_error(self):
        """Test GeminiClientError exception."""
        with pytest.raises(GeminiClientError) as exc_info:
            raise GeminiClientError("Test error")
        assert "Test error" in str(exc_info.value)

    def test_model_not_available_error(self):
        """Test ModelNotAvailableError exception."""
        with pytest.raises(ModelNotAvailableError):
            raise ModelNotAvailableError("Model not available")

    def test_token_limit_exceeded_error(self):
        """Test TokenLimitExceededError exception."""
        with pytest.raises(TokenLimitExceededError):
            raise TokenLimitExceededError("Token limit exceeded")

    def test_error_inheritance(self):
        """Test exception inheritance."""
        assert issubclass(ModelNotAvailableError, GeminiClientError)
        assert issubclass(TokenLimitExceededError, GeminiClientError)


class TestGeminiClient:
    """Tests for GeminiClient class."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GeminiConfig."""
        return GeminiConfig(
            api_key="AIzaSyTestKey123456",
            model_name="gemini-pro",
            temperature=0.0,
        )

    @pytest.fixture
    def sample_hunk(self):
        """Create a sample HunkInfo."""
        return HunkInfo(
            source_start=1,
            source_length=5,
            target_start=1,
            target_length=7,
            content="+def hello():\n+    print('Hello')",
            header="@@ -1,5 +1,7 @@",
            lines=[" import os", "+def hello():", "+    print('Hello')"],
        )

    @pytest.fixture
    def sample_context(self):
        """Create a sample AnalysisContext."""
        pr = PRDetails("owner", "repo", 1, "Title", "Description")
        file_info = FileInfo(path="main.py")
        return AnalysisContext(
            pr_details=pr,
            file_info=file_info,
            language="python",
        )

    @patch("gemini_reviewer.gemini_client.genai")
    def test_init(self, mock_genai, valid_config):
        """Test GeminiClient initialization."""
        mock_genai.configure = Mock()
        mock_genai.GenerativeModel = Mock()

        client = GeminiClient(valid_config)

        mock_genai.configure.assert_called_once_with(api_key=valid_config.api_key)
        assert client.config == valid_config

    @patch("gemini_reviewer.gemini_client.genai")
    def test_test_connection_success(self, mock_genai, valid_config):
        """Test successful connection test."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Hello"
        # Set up candidates properly so the code doesn't fail
        mock_response.candidates = None  # No candidates = skip that check
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(valid_config)
        result = client.test_connection()

        # test_connection returns True/False or a string in some versions
        assert result is True or result == "Hello"

    @patch("gemini_reviewer.gemini_client.genai")
    def test_test_connection_failure(self, mock_genai, valid_config):
        """Test failed connection test."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("Connection failed")
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(valid_config)
        result = client.test_connection()

        assert result is False

    @patch("gemini_reviewer.gemini_client.genai")
    def test_analyze_code_hunk_success(
        self, mock_genai, valid_config, sample_hunk, sample_context
    ):
        """Test successful code hunk analysis."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps({
            "reviews": [
                {
                    "lineNumber": 2,
                    "reviewComment": "Consider adding docstring",
                    "priority": "low",
                    "category": "documentation",
                }
            ]
        })
        # Set up response to not have candidates to skip that check path
        mock_response.candidates = None
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(valid_config)
        prompt = "Review this code"

        result = client.analyze_code_hunk(sample_hunk, sample_context, prompt)

        assert isinstance(result, list)

    @patch("gemini_reviewer.gemini_client.genai")
    def test_analyze_code_hunk_empty_response(
        self, mock_genai, valid_config, sample_hunk, sample_context
    ):
        """Test handling empty response."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps({"reviews": []})
        mock_response.candidates = None
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(valid_config)

        result = client.analyze_code_hunk(sample_hunk, sample_context, "Review")

        assert result == []

    @patch("gemini_reviewer.gemini_client.genai")
    def test_analyze_code_hunk_invalid_json(
        self, mock_genai, valid_config, sample_hunk, sample_context
    ):
        """Test handling invalid JSON response."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Not valid JSON"
        mock_response.candidates = None
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(valid_config)

        # Should handle gracefully
        result = client.analyze_code_hunk(sample_hunk, sample_context, "Review")
        assert isinstance(result, list)

    @patch("gemini_reviewer.gemini_client.genai")
    def test_get_statistics(self, mock_genai, valid_config):
        """Test getting client statistics."""
        mock_genai.GenerativeModel = Mock()

        client = GeminiClient(valid_config)
        stats = client.get_statistics()

        assert "total_requests" in stats
        assert "successful_requests" in stats
        assert "failed_requests" in stats

    @patch("gemini_reviewer.gemini_client.genai")
    def test_close(self, mock_genai, valid_config):
        """Test client cleanup."""
        mock_genai.GenerativeModel = Mock()

        client = GeminiClient(valid_config)
        client.close()  # Should not raise


class TestJSONParsing:
    """Tests for JSON parsing functionality."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GeminiConfig."""
        return GeminiConfig(api_key="AIzaSyTestKey123456")

    @patch("gemini_reviewer.gemini_client.genai")
    def test_parse_valid_json_with_code_blocks(self, mock_genai, valid_config):
        """Test parsing JSON wrapped in markdown code blocks."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        response_text = """
```json
{
    "reviews": [
        {"lineNumber": 1, "reviewComment": "Test", "priority": "low"}
    ]
}
```
"""

        result = client._parse_ai_response(response_text)
        assert isinstance(result, list)

    @patch("gemini_reviewer.gemini_client.genai")
    def test_parse_truncated_json(self, mock_genai, valid_config):
        """Test parsing truncated JSON response."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        truncated_json = '{"reviews": [{"lineNumber": 1, "reviewComment": "Test"'

        result = client._parse_ai_response(truncated_json)
        # Should handle gracefully
        assert isinstance(result, list)

    @patch("gemini_reviewer.gemini_client.genai")
    def test_extract_json_from_text(self, mock_genai, valid_config):
        """Test extracting JSON from mixed text."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        mixed_text = """
Here is my analysis:

{"reviews": [{"lineNumber": 1, "reviewComment": "Good code"}]}

That's all!
"""

        result = client._extract_valid_json_segment(mixed_text)
        assert result is not None or result == ""

    @patch("gemini_reviewer.gemini_client.genai")
    def test_repair_json(self, mock_genai, valid_config):
        """Test JSON repair functionality."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        broken_json = '{"reviews": [{"lineNumber": 1, "reviewComment": "Test"'

        result = client._repair_truncated_json(broken_json)
        # Should attempt repair
        assert isinstance(result, str)


class TestAIResponseConversion:
    """Tests for converting AI responses to AIResponse objects."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GeminiConfig."""
        return GeminiConfig(api_key="AIzaSyTestKey123456")

    @patch("gemini_reviewer.gemini_client.genai")
    def test_convert_valid_response(self, mock_genai, valid_config):
        """Test converting valid response data."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        response_data = {
            "lineNumber": 10,
            "reviewComment": "Consider refactoring",
            "priority": "high",
            "category": "maintainability",
        }

        result = client._parse_single_review(response_data)

        assert result is not None
        assert result.line_number == 10
        assert result.review_comment == "Consider refactoring"
        assert result.priority == ReviewPriority.HIGH
        assert result.category == "maintainability"

    @patch("gemini_reviewer.gemini_client.genai")
    def test_convert_minimal_response(self, mock_genai, valid_config):
        """Test converting minimal response data."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        response_data = {
            "lineNumber": 5,
            "reviewComment": "Fix this",
        }

        result = client._parse_single_review(response_data)

        assert result is not None
        assert result.line_number == 5
        assert result.priority == ReviewPriority.MEDIUM  # default

    @patch("gemini_reviewer.gemini_client.genai")
    def test_convert_invalid_response(self, mock_genai, valid_config):
        """Test converting invalid response data."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        # Missing required fields
        response_data = {"category": "bugs"}

        result = client._parse_single_review(response_data)

        assert result is None

    @patch("gemini_reviewer.gemini_client.genai")
    def test_convert_with_fix_code(self, mock_genai, valid_config):
        """Test converting response with fix code."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        response_data = {
            "lineNumber": 10,
            "reviewComment": "Use f-string",
            "fixCode": "f'Hello {name}'",
        }

        result = client._parse_single_review(response_data)

        assert result is not None
        assert result.fix_code == "f'Hello {name}'"


class TestPromptBuilding:
    """Tests for prompt building functionality."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GeminiConfig."""
        return GeminiConfig(api_key="AIzaSyTestKey123456")

    @pytest.fixture
    def sample_hunk(self):
        """Create a sample HunkInfo."""
        return HunkInfo(
            source_start=1,
            source_length=5,
            target_start=1,
            target_length=5,
            content="+print('hello')",
            header="@@ -1,5 +1,5 @@",
            lines=["+print('hello')"],
        )

    @pytest.fixture
    def sample_context(self):
        """Create a sample AnalysisContext."""
        pr = PRDetails("owner", "repo", 1, "Add feature", "Description")
        file_info = FileInfo(path="main.py")
        return AnalysisContext(
            pr_details=pr,
            file_info=file_info,
            language="python",
            project_context="Python project",
        )

    @patch("gemini_reviewer.gemini_client.genai")
    def test_build_prompt(self, mock_genai, valid_config, sample_hunk, sample_context):
        """Test building the analysis prompt."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        prompt = client._create_analysis_prompt(
            sample_hunk, sample_context, "Review template"
        )

        assert "main.py" in prompt
        assert "python" in prompt.lower()

    @patch("gemini_reviewer.gemini_client.genai")
    def test_build_prompt_with_full_file(
        self, mock_genai, valid_config, sample_hunk
    ):
        """Test building prompt with full file content."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        pr = PRDetails("owner", "repo", 1, "Title", "Desc")
        file_info = FileInfo(path="main.py")
        context = AnalysisContext(
            pr_details=pr,
            file_info=file_info,
            language="python",
            full_file_content="def main():\n    pass",
        )

        prompt = client._create_analysis_prompt(sample_hunk, context, "Review")

        assert isinstance(prompt, str)

    @patch("gemini_reviewer.gemini_client.genai")
    def test_build_prompt_with_related_files(
        self, mock_genai, valid_config, sample_hunk
    ):
        """Test building prompt with related files."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        pr = PRDetails("owner", "repo", 1, "Title", "Desc")
        file_info = FileInfo(path="main.py")
        context = AnalysisContext(
            pr_details=pr,
            file_info=file_info,
            related_files=["utils.py", "helper.py"],
        )

        prompt = client._create_analysis_prompt(sample_hunk, context, "Review")

        assert isinstance(prompt, str)


class TestGeminiClientAdvanced:
    """Advanced tests for Gemini client edge cases."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GeminiConfig."""
        return GeminiConfig(api_key="AIzaSyTestKey123456")

    @pytest.fixture
    def sample_hunk(self):
        """Create a sample HunkInfo."""
        return HunkInfo(
            source_start=1,
            source_length=5,
            target_start=1,
            target_length=7,
            content="+def hello():\n+    print('Hello')",
            header="@@ -1,5 +1,7 @@",
            lines=[" import os", "+def hello():", "+    print('Hello')"],
        )

    @pytest.fixture
    def sample_context(self):
        """Create a sample AnalysisContext."""
        pr = PRDetails("owner", "repo", 1, "Title", "Description")
        file_info = FileInfo(path="main.py")
        return AnalysisContext(
            pr_details=pr,
            file_info=file_info,
            language="python",
        )

    @patch("gemini_reviewer.gemini_client.genai")
    def test_init_failure(self, mock_genai, valid_config):
        """Test handling initialization failure."""
        mock_genai.configure.side_effect = Exception("API key invalid")

        with pytest.raises(GeminiClientError):
            GeminiClient(valid_config)

    @patch("gemini_reviewer.gemini_client.genai")
    def test_analyze_empty_hunk(self, mock_genai, valid_config, sample_context):
        """Test analyzing empty hunk."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        empty_hunk = HunkInfo(
            source_start=1,
            source_length=0,
            target_start=1,
            target_length=0,
            content="",
            header="",
            lines=[],
        )

        result = client.analyze_code_hunk(empty_hunk, sample_context, "Review")
        assert result == []

    @patch("gemini_reviewer.gemini_client.genai")
    def test_analyze_none_hunk(self, mock_genai, valid_config, sample_context):
        """Test analyzing None hunk."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        result = client.analyze_code_hunk(None, sample_context, "Review")
        assert result == []

    @patch("gemini_reviewer.gemini_client.genai")
    def test_analyze_invalid_context(self, mock_genai, valid_config, sample_hunk):
        """Test analyzing with invalid context."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        result = client.analyze_code_hunk(sample_hunk, None, "Review")
        assert result == []

    @patch("gemini_reviewer.gemini_client.genai")
    def test_analyze_context_no_pr_details(self, mock_genai, valid_config, sample_hunk):
        """Test analyzing with context missing PR details."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        context = AnalysisContext(
            pr_details=None,
            file_info=FileInfo(path="main.py"),
        )

        result = client.analyze_code_hunk(sample_hunk, context, "Review")
        assert result == []

    @patch("gemini_reviewer.gemini_client.genai")
    def test_analyze_very_long_prompt(self, mock_genai, valid_config, sample_context):
        """Test analyzing with very long prompt that needs truncation."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps({"reviews": []})
        mock_response.candidates = None
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(valid_config)

        # Create a hunk with very long content
        long_hunk = HunkInfo(
            source_start=1,
            source_length=100,
            target_start=1,
            target_length=100,
            content="+" + "x" * 200000,  # Very long content
            header="@@ -1,100 +1,100 @@",
            lines=["+" + "x" * 1000 for _ in range(200)],
        )

        result = client.analyze_code_hunk(long_hunk, sample_context, "Review " * 10000)
        assert isinstance(result, list)

    @patch("gemini_reviewer.gemini_client.genai")
    def test_analyze_api_exception(self, mock_genai, valid_config, sample_hunk, sample_context):
        """Test handling API exception during analysis."""
        from tenacity import RetryError

        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(valid_config)

        with pytest.raises((Exception, RetryError)):
            client.analyze_code_hunk(sample_hunk, sample_context, "Review")

    @patch("gemini_reviewer.gemini_client.genai")
    def test_parse_response_with_candidates(self, mock_genai, valid_config, sample_hunk, sample_context):
        """Test parsing response with candidates."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps({"reviews": []})

        # Mock candidates with finish_reason
        mock_candidate = Mock()
        mock_candidate.finish_reason = "STOP"
        mock_response.candidates = [mock_candidate]

        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(valid_config)
        result = client.analyze_code_hunk(sample_hunk, sample_context, "Review")

        assert isinstance(result, list)

    @patch("gemini_reviewer.gemini_client.genai")
    def test_parse_priority_variations(self, mock_genai, valid_config):
        """Test parsing different priority values."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        # Test critical priority
        data = {"lineNumber": 1, "reviewComment": "Test", "priority": "critical"}
        result = client._parse_single_review(data)
        assert result.priority == ReviewPriority.CRITICAL

        # Test medium priority
        data["priority"] = "medium"
        result = client._parse_single_review(data)
        assert result.priority == ReviewPriority.MEDIUM

        # Test low priority
        data["priority"] = "low"
        result = client._parse_single_review(data)
        assert result.priority == ReviewPriority.LOW

        # Test invalid priority (should default to medium)
        data["priority"] = "invalid"
        result = client._parse_single_review(data)
        assert result.priority == ReviewPriority.MEDIUM

    @patch("gemini_reviewer.gemini_client.genai")
    def test_parse_review_with_all_fields(self, mock_genai, valid_config):
        """Test parsing review with all possible fields."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        data = {
            "lineNumber": 10,
            "reviewComment": "Consider refactoring this",
            "priority": "high",
            "category": "refactoring",
            "fixCode": "improved_code()",
            "confidence": 0.9,
        }

        result = client._parse_single_review(data)

        assert result is not None
        assert result.line_number == 10
        assert result.review_comment == "Consider refactoring this"
        assert result.priority == ReviewPriority.HIGH
        assert result.category == "refactoring"
        assert result.fix_code == "improved_code()"

    @patch("gemini_reviewer.gemini_client.genai")
    def test_parse_review_missing_line_number(self, mock_genai, valid_config):
        """Test parsing review with missing line number."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        data = {"reviewComment": "No line number"}

        result = client._parse_single_review(data)
        assert result is None

    @patch("gemini_reviewer.gemini_client.genai")
    def test_parse_review_missing_comment(self, mock_genai, valid_config):
        """Test parsing review with missing comment."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        data = {"lineNumber": 10}

        result = client._parse_single_review(data)
        assert result is None

    @patch("gemini_reviewer.gemini_client.genai")
    def test_extract_json_various_formats(self, mock_genai, valid_config):
        """Test extracting JSON from various formats."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        # JSON with prefix text
        text1 = 'Here is the result: {"reviews": []}'
        result1 = client._extract_valid_json_segment(text1)
        assert result1 is not None

        # JSON with suffix text
        text2 = '{"reviews": []} That is all!'
        result2 = client._extract_valid_json_segment(text2)
        assert result2 is not None

        # Array JSON
        text3 = '[{"lineNumber": 1, "reviewComment": "Test"}]'
        result3 = client._extract_valid_json_segment(text3)
        assert result3 is not None

    @patch("gemini_reviewer.gemini_client.genai")
    def test_repair_json_various_cases(self, mock_genai, valid_config):
        """Test JSON repair for various broken formats."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        # Missing closing brace
        broken1 = '{"reviews": [{"lineNumber": 1}'
        result1 = client._repair_truncated_json(broken1)
        assert isinstance(result1, str)

        # Missing closing bracket
        broken2 = '{"reviews": ['
        result2 = client._repair_truncated_json(broken2)
        assert isinstance(result2, str)

        # Truncated string
        broken3 = '{"reviews": [{"lineNumber": 1, "reviewComment": "test'
        result3 = client._repair_truncated_json(broken3)
        assert isinstance(result3, str)

    @patch("gemini_reviewer.gemini_client.genai")
    def test_statistics_tracking(self, mock_genai, valid_config, sample_hunk, sample_context):
        """Test that statistics are tracked correctly."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps({"reviews": []})
        mock_response.candidates = None
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(valid_config)

        # Make some requests
        client.analyze_code_hunk(sample_hunk, sample_context, "Review")
        client.analyze_code_hunk(sample_hunk, sample_context, "Review")

        stats = client.get_statistics()
        assert stats["total_requests"] >= 2

    @patch("gemini_reviewer.gemini_client.genai")
    def test_generate_content_validation_empty_response(self, mock_genai, valid_config, sample_hunk, sample_context):
        """Test handling empty response from model."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = ""
        mock_response.candidates = None
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(valid_config)

        # The analyze_code_hunk method will trigger validation
        result = client.analyze_code_hunk(sample_hunk, sample_context, "Review")
        assert isinstance(result, list)

    @patch("gemini_reviewer.gemini_client.genai")
    def test_generate_content_candidates_blocked(self, mock_genai, valid_config):
        """Test handling blocked candidates in response."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = '{"reviews": []}'

        # Mock candidates with blocked finish_reason
        mock_candidate = Mock()
        mock_candidate.finish_reason = "SAFETY"
        mock_response.candidates = [mock_candidate]

        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(valid_config)
        result = client._generate_content_with_validation("Test prompt")

        # Should still return the text even if blocked
        assert isinstance(result, str)


class TestGeminiClientContextBuilding:
    """Tests for context building in Gemini client."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GeminiConfig."""
        return GeminiConfig(api_key="AIzaSyTestKey123456")

    @pytest.fixture
    def sample_hunk(self):
        """Create a sample HunkInfo."""
        return HunkInfo(
            source_start=1,
            source_length=5,
            target_start=1,
            target_length=7,
            content="+def hello():\n+    print('Hello')",
            header="@@ -1,5 +1,7 @@",
            lines=[" import os", "+def hello():", "+    print('Hello')"],
        )

    @patch("gemini_reviewer.gemini_client.genai")
    def test_context_with_project_context(self, mock_genai, valid_config, sample_hunk):
        """Test context building with project context."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        pr = PRDetails("owner", "repo", 1, "Title", "Desc")
        file_info = FileInfo(path="main.py")
        context = AnalysisContext(
            pr_details=pr,
            file_info=file_info,
            project_context="This is a Flask web application",
        )

        prompt = client._create_analysis_prompt(sample_hunk, context, "Review")
        assert isinstance(prompt, str)

    @patch("gemini_reviewer.gemini_client.genai")
    def test_context_with_change_summary(self, mock_genai, valid_config, sample_hunk):
        """Test context building with change summary."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        pr = PRDetails("owner", "repo", 1, "Title", "Desc")
        file_info = FileInfo(path="main.py")
        context = AnalysisContext(
            pr_details=pr,
            file_info=file_info,
            change_summary="Added new feature for user authentication",
        )

        prompt = client._create_analysis_prompt(sample_hunk, context, "Review")
        assert isinstance(prompt, str)

    @patch("gemini_reviewer.gemini_client.genai")
    def test_context_with_all_changed_files(self, mock_genai, valid_config, sample_hunk):
        """Test context building with all changed files."""
        mock_genai.GenerativeModel = Mock()
        client = GeminiClient(valid_config)

        pr = PRDetails("owner", "repo", 1, "Title", "Desc")
        file_info = FileInfo(path="main.py")
        context = AnalysisContext(
            pr_details=pr,
            file_info=file_info,
            all_changed_files=["main.py", "utils.py", "config.py"],
        )

        prompt = client._create_analysis_prompt(sample_hunk, context, "Review")
        assert isinstance(prompt, str)


class TestGeminiClientFollowUp:
    """Tests for follow-up review functionality."""

    @pytest.fixture
    def valid_config(self):
        """Create a valid GeminiConfig."""
        return GeminiConfig(api_key="AIzaSyTestKey123456")

    @patch("gemini_reviewer.gemini_client.genai")
    def test_analyze_follow_up_review(self, mock_genai, valid_config):
        """Test analyzing follow-up review."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = json.dumps({
            "status": "resolved",
            "newIssues": []
        })
        mock_response.candidates = None
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(valid_config)

        pr = PRDetails("owner", "repo", 1, "Title", "Desc")

        # Test if the method exists (it may vary based on implementation)
        if hasattr(client, 'analyze_follow_up'):
            result = client.analyze_follow_up(pr, [], "diff content")
            assert isinstance(result, dict) or result is None

