import re
import pytest

from gemini_reviewer.prompts import ReviewMode, get_review_prompt_template, BASE_PROMPT_TEMPLATE, FOLLOWUP_PROMPT_TEMPLATE


def test_standard_prompt_contains_base_and_not_followup():
    prompt = get_review_prompt_template(ReviewMode.STANDARD)
    assert prompt.startswith(BASE_PROMPT_TEMPLATE.splitlines()[0])
    assert "FOLLOW-UP REVIEW" not in prompt
    # Strict output rules present
    assert "REQUIRED OUTPUT FORMAT" in prompt


def test_followup_prompt_with_comments_uses_template_and_injects_comments():
    prev = "1. File A: something\n2. File B: else"
    prompt = get_review_prompt_template(ReviewMode.FOLLOWUP, previous_comments=prev)
    assert "THIS IS A FOLLOW-UP REVIEW" in prompt
    assert "DO NOT raise any new issues" in prompt
    assert prev in prompt
    # Ensure it matches the followup template structure braces formatting
    assert "REQUIRED OUTPUT FORMAT" in prompt


def test_followup_prompt_without_comments_inserts_default_message():
    prompt = get_review_prompt_template(ReviewMode.FOLLOWUP, previous_comments="")
    assert "No previous comments found." in prompt


def test_custom_instructions_appended_for_regular_reviews():
    custom = "Only consider files in src/."
    prompt = get_review_prompt_template(ReviewMode.STANDARD, custom_instructions=custom)
    assert custom in prompt


@pytest.mark.parametrize("mode", [
    ReviewMode.STRICT,
    ReviewMode.STANDARD,
    ReviewMode.LENIENT,
    ReviewMode.SECURITY_FOCUSED,
    ReviewMode.PERFORMANCE_FOCUSED,
])
def test_all_regular_modes_return_non_empty_prompts(mode):
    prompt = get_review_prompt_template(mode)
    assert isinstance(prompt, str)
    assert len(prompt) > 50
