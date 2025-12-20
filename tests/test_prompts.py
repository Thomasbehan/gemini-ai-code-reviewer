"""
Comprehensive tests for gemini_reviewer/prompts.py
"""

import pytest

from gemini_reviewer.prompts import (
    ReviewMode,
    BASE_PROMPT_TEMPLATE,
    NOISE_CONTROL,
    MODE_INSTRUCTIONS,
    FOLLOWUP_PROMPT_TEMPLATE,
    get_review_prompt_template,
)


class TestReviewMode:
    """Tests for ReviewMode enum."""

    def test_all_modes_exist(self):
        """Test all review modes exist."""
        assert ReviewMode.STRICT.value == "strict"
        assert ReviewMode.STANDARD.value == "standard"
        assert ReviewMode.LENIENT.value == "lenient"
        assert ReviewMode.SECURITY_FOCUSED.value == "security_focused"
        assert ReviewMode.PERFORMANCE_FOCUSED.value == "performance_focused"
        assert ReviewMode.FOLLOWUP.value == "followup"

    def test_from_value(self):
        """Test creating enum from string value."""
        assert ReviewMode("strict") == ReviewMode.STRICT
        assert ReviewMode("standard") == ReviewMode.STANDARD
        assert ReviewMode("lenient") == ReviewMode.LENIENT


class TestPromptTemplates:
    """Tests for prompt template constants."""

    def test_base_prompt_contains_required_elements(self):
        """Test BASE_PROMPT_TEMPLATE contains essential elements."""
        assert "JSON" in BASE_PROMPT_TEMPLATE
        assert "lineNumber" in BASE_PROMPT_TEMPLATE
        assert "reviews" in BASE_PROMPT_TEMPLATE
        assert "fixCode" in BASE_PROMPT_TEMPLATE
        assert "explanation" in BASE_PROMPT_TEMPLATE
        assert "priority" in BASE_PROMPT_TEMPLATE

    def test_noise_control_contains_guidance(self):
        """Test NOISE_CONTROL contains noise reduction guidance."""
        assert "false positives" in NOISE_CONTROL.lower() or "speculation" in NOISE_CONTROL.lower()

    def test_mode_instructions_for_all_modes(self):
        """Test MODE_INSTRUCTIONS has entries for all modes."""
        for mode in ReviewMode:
            if mode != ReviewMode.FOLLOWUP:
                assert mode in MODE_INSTRUCTIONS, f"Missing instructions for {mode}"

    def test_strict_mode_instructions(self):
        """Test STRICT mode instructions."""
        instructions = MODE_INSTRUCTIONS[ReviewMode.STRICT]
        assert "critical" in instructions.lower() or "thorough" in instructions.lower()

    def test_standard_mode_instructions(self):
        """Test STANDARD mode instructions."""
        instructions = MODE_INSTRUCTIONS[ReviewMode.STANDARD]
        assert "critical" in instructions.lower() or "bug" in instructions.lower()

    def test_lenient_mode_instructions(self):
        """Test LENIENT mode instructions."""
        instructions = MODE_INSTRUCTIONS[ReviewMode.LENIENT]
        assert "conservative" in instructions.lower() or "critical" in instructions.lower()

    def test_security_focused_instructions(self):
        """Test SECURITY_FOCUSED mode instructions."""
        instructions = MODE_INSTRUCTIONS[ReviewMode.SECURITY_FOCUSED]
        assert "security" in instructions.lower()

    def test_performance_focused_instructions(self):
        """Test PERFORMANCE_FOCUSED mode instructions."""
        instructions = MODE_INSTRUCTIONS[ReviewMode.PERFORMANCE_FOCUSED]
        assert "performance" in instructions.lower()

    def test_followup_template_structure(self):
        """Test FOLLOWUP_PROMPT_TEMPLATE structure."""
        assert "FOLLOW-UP REVIEW" in FOLLOWUP_PROMPT_TEMPLATE
        assert "previous_comments" in FOLLOWUP_PROMPT_TEMPLATE
        assert "resolved" in FOLLOWUP_PROMPT_TEMPLATE.lower()


class TestGetReviewPromptTemplate:
    """Tests for get_review_prompt_template function."""

    def test_standard_mode(self):
        """Test standard mode prompt generation."""
        prompt = get_review_prompt_template(ReviewMode.STANDARD)
        assert BASE_PROMPT_TEMPLATE in prompt
        assert NOISE_CONTROL in prompt
        assert MODE_INSTRUCTIONS[ReviewMode.STANDARD] in prompt

    def test_strict_mode(self):
        """Test strict mode prompt generation."""
        prompt = get_review_prompt_template(ReviewMode.STRICT)
        assert MODE_INSTRUCTIONS[ReviewMode.STRICT] in prompt

    def test_lenient_mode(self):
        """Test lenient mode prompt generation."""
        prompt = get_review_prompt_template(ReviewMode.LENIENT)
        assert MODE_INSTRUCTIONS[ReviewMode.LENIENT] in prompt

    def test_security_focused_mode(self):
        """Test security focused mode prompt generation."""
        prompt = get_review_prompt_template(ReviewMode.SECURITY_FOCUSED)
        assert MODE_INSTRUCTIONS[ReviewMode.SECURITY_FOCUSED] in prompt

    def test_performance_focused_mode(self):
        """Test performance focused mode prompt generation."""
        prompt = get_review_prompt_template(ReviewMode.PERFORMANCE_FOCUSED)
        assert MODE_INSTRUCTIONS[ReviewMode.PERFORMANCE_FOCUSED] in prompt

    def test_followup_mode_uses_special_template(self):
        """Test followup mode uses special template."""
        prompt = get_review_prompt_template(ReviewMode.FOLLOWUP)
        assert "FOLLOW-UP REVIEW" in prompt
        # Should NOT contain base prompt elements
        assert BASE_PROMPT_TEMPLATE not in prompt

    def test_followup_with_previous_comments(self):
        """Test followup mode includes previous comments."""
        previous = "1. Line 10: Fix the bug"
        prompt = get_review_prompt_template(ReviewMode.FOLLOWUP, previous_comments=previous)
        assert "Fix the bug" in prompt

    def test_followup_without_previous_comments(self):
        """Test followup mode with no previous comments."""
        prompt = get_review_prompt_template(ReviewMode.FOLLOWUP, previous_comments="")
        assert "No previous comments found" in prompt

    def test_custom_instructions_included(self):
        """Test custom instructions are included."""
        custom = "Pay special attention to error handling."
        prompt = get_review_prompt_template(ReviewMode.STANDARD, custom_instructions=custom)
        assert custom in prompt
        assert "ADDITIONAL INSTRUCTIONS" in prompt

    def test_custom_instructions_empty(self):
        """Test empty custom instructions are not included."""
        prompt = get_review_prompt_template(ReviewMode.STANDARD, custom_instructions="")
        assert "ADDITIONAL INSTRUCTIONS" not in prompt

    def test_custom_instructions_not_in_followup(self):
        """Test custom instructions behavior in followup mode."""
        # Followup mode uses a completely different template
        custom = "Custom instruction"
        prompt = get_review_prompt_template(ReviewMode.FOLLOWUP, custom_instructions=custom)
        # Custom instructions are not added to followup template
        assert custom not in prompt

    def test_all_regular_modes_include_base_and_noise(self):
        """Test all non-followup modes include base template and noise control."""
        regular_modes = [
            ReviewMode.STRICT,
            ReviewMode.STANDARD,
            ReviewMode.LENIENT,
            ReviewMode.SECURITY_FOCUSED,
            ReviewMode.PERFORMANCE_FOCUSED,
        ]
        for mode in regular_modes:
            prompt = get_review_prompt_template(mode)
            assert BASE_PROMPT_TEMPLATE in prompt, f"Base template missing for {mode}"
            assert NOISE_CONTROL in prompt, f"Noise control missing for {mode}"
