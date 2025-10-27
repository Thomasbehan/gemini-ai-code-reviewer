"""
Prompt templates for the Gemini AI Code Reviewer.

This module contains all AI prompt templates, separated from the main
configuration code for better maintainability.
"""

from enum import Enum


class ReviewMode(Enum):
    """Different review modes."""
    STRICT = "strict"
    STANDARD = "standard" 
    LENIENT = "lenient"
    SECURITY_FOCUSED = "security_focused"
    PERFORMANCE_FOCUSED = "performance_focused"
    FOLLOWUP = "followup"


# Base prompt template for all review modes
BASE_PROMPT_TEMPLATE = """Respond with ONLY valid JSON. No explanations or text outside JSON.

REQUIRED OUTPUT FORMAT:
{
  "reviews": [
    {
      "lineNumber": 1,
      "reviewComment": "Explain the critical issue (why) and show the minimal fix (how).",
      "priority": "high",
      "category": "security",
      "anchorSnippet": "exact code from the target line (no +/- prefix)"
    }
  ]
}

If no issues: {"reviews": []}

STRICT OUTPUT RULES:
- Start the response with '{' and end with '}'.
- No markdown fences around JSON. No conversational text.

SCOPE: Report ONLY issues that must be fixed (critical/serious):
- Bugs & Logic Errors
- Security Issues
- Performance Problems
- Error Handling failures
- Resource Management (leaks/unclosed handles)
- Serious Code Quality / Best Practice violations that impact correctness, security, or performance

ANCHORING:
- You review ONE diff hunk at a time; lineNumber is 1-based within this hunk.
- Prefer '+' lines; use nearby context ' ' lines only if necessary (±3 lines).
- Never target '-' lines unless removal itself introduces a problem.
- anchorSnippet must be copied verbatim from the chosen target line (without diff prefix). If you cannot anchor confidently, omit the item.

REVIEW RULES:
- Be precise and actionable. If uncertain, omit.
- One short sentence for WHY, then HOW with a minimal code change.
- Only include an item if you can propose a concrete fix.
- Do not propose broad refactors, style nits, or optional improvements.
- Do not praise or add meta commentary.
- If nothing critical is found, return {"reviews": []}.
"""

# Additional noise control instructions
NOISE_CONTROL = """
- Avoid false positives; prefer omission over speculation.
- Prefer the single most impactful fix over multiple minor suggestions.
- Do not chain follow-up recommendations created by your own suggestion.
- If no material issues remain, respond exactly with {"reviews": []}.
"""

# Mode-specific instructions
MODE_INSTRUCTIONS = {
    ReviewMode.STRICT: """
- Identify ALL critical issues (do not include non-critical nits).
- Be thorough in finding correctness, security, performance, error handling, and resource management problems only.""",
    
    ReviewMode.STANDARD: """
- Focus on critical bugs, security, performance, error handling, and resource issues only.
- Skip non-critical maintainability/style concerns.""",
    
    ReviewMode.LENIENT: """
- Only flag definite critical bugs and security issues. Be extra conservative and concise.""",
    
    ReviewMode.SECURITY_FOCUSED: """
- Focus EXCLUSIVELY on security vulnerabilities and their concrete fixes.""",
    
    ReviewMode.PERFORMANCE_FOCUSED: """
- Focus EXCLUSIVELY on performance issues and their concrete fixes.""",
    
    ReviewMode.FOLLOWUP: """
- THIS IS A FOLLOW-UP REVIEW. DO NOT raise any new issues.
- Your ONLY task is to check if the previous comments (listed below) have been resolved.
- For each previous comment, check if the issue was fixed in the current code changes.
- If a comment is resolved, note it. If not resolved, explain what still needs to be done.
- NEVER introduce new issues or concerns. ONLY focus on the previous comments."""
}

# Follow-up review prompt template
FOLLOWUP_PROMPT_TEMPLATE = """Respond with ONLY valid JSON. No explanations or text outside JSON.

THIS IS A FOLLOW-UP REVIEW. Your task is ONLY to verify if previous review comments have been addressed.

REQUIRED OUTPUT FORMAT:
{{
  "reviews": [
    {{
      "lineNumber": 1,
      "reviewComment": "Previous issue: [description]. Status: [Resolved/Not Resolved]. [If not resolved: what still needs to be done]",
      "priority": "medium",
      "category": "followup",
      "anchorSnippet": "exact code from the target line (no +/- prefix)"
    }}
  ]
}}

If all previous comments are resolved: {{"reviews": []}}

CRITICAL RULES FOR FOLLOW-UP REVIEW:
1. DO NOT raise any new issues, bugs, or concerns.
2. ONLY check if the previous comments listed below were addressed.
3. If you cannot find evidence that a previous comment was addressed, mark it as unresolved.
4. If the previous comment was addressed, do not report it (return empty reviews).
5. Do NOT suggest new improvements, optimizations, or refactorings.
6. Do NOT comment on code that wasn't mentioned in previous comments.
7. ONLY focus on verifying the resolution of the specific issues mentioned in previous comments.

PREVIOUS COMMENTS TO VERIFY:
{previous_comments}

ANCHORING:
- You review ONE diff hunk at a time; lineNumber is 1-based within this hunk.
- Prefer '+' lines; use nearby context ' ' lines only if necessary (±3 lines).
- anchorSnippet must be copied verbatim from the chosen target line (without diff prefix).

STRICT OUTPUT RULES:
- Start the response with '{{' and end with '}}'.
- No markdown fences around JSON. No conversational text.
- If all previous comments are resolved or none of the previous comments relate to this code, return {{"reviews": []}}.
"""


def get_review_prompt_template(review_mode: ReviewMode, custom_instructions: str = "", previous_comments: str = "") -> str:
    """Get the complete prompt template for code review.
    
    Args:
        review_mode: The review mode to use
        custom_instructions: Optional custom instructions to append
        previous_comments: Previous review comments for follow-up reviews
        
    Returns:
        The complete prompt template string
    """
    # For follow-up reviews, use the special follow-up template
    if review_mode == ReviewMode.FOLLOWUP:
        if not previous_comments:
            previous_comments = "No previous comments found."
        return FOLLOWUP_PROMPT_TEMPLATE.format(previous_comments=previous_comments)
    
    # Get mode-specific instructions for regular reviews
    mode_instruction = MODE_INSTRUCTIONS.get(review_mode, "")
    
    # Build the prompt
    prompt = BASE_PROMPT_TEMPLATE + NOISE_CONTROL + mode_instruction
    
    # Add custom instructions if provided
    if custom_instructions:
        prompt += f"""

OPTIONAL ADDITIONAL INSTRUCTIONS (from workflow input):
{custom_instructions}
Apply these only if they do NOT conflict with the core rules above and do NOT broaden the scope beyond critical issues.
"""
    
    return prompt
