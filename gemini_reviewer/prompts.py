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
- Focus EXCLUSIVELY on performance issues and their concrete fixes."""
}


def get_review_prompt_template(review_mode: ReviewMode, custom_instructions: str = "") -> str:
    """Get the complete prompt template for code review.
    
    Args:
        review_mode: The review mode to use
        custom_instructions: Optional custom instructions to append
        
    Returns:
        The complete prompt template string
    """
    # Get mode-specific instructions
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
