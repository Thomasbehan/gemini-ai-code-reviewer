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
      "explanation": "XYZ is wrong and heres why... (Explain the issue clearly)",
      "fixCode": "code block content (just the code, no markdown backticks)",
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
- 'fixCode' must contain valid code replacement.
- 'explanation' must be concise and actionable.
  State briefly WHY this is a problem and HOW the fix resolves it.

SCOPE: Report ONLY issues that must be fixed (critical/serious):
- Bugs & Logic Errors
- Security Issues
- Performance Problems
- Error Handling failures
- Resource Management (leaks/unclosed handles)
- Poor Variable/Function/Class Naming that severely impacts readability:
  Zero‑shorthand naming policy is enforced across this repository.
  Treat violations as critical readability issues when they harm clarity.
  Disallow non-descriptive or abbreviated identifiers. Examples:
  - Not allowed: s, svc, srv, cfg, conf, req, resp, usr, repo, mgr, util, lst, dt
  - Prefer full, descriptive words: service, service_client, config, request, response,
    user, repository, manager, utilities, items, date_time
  - Single-letter loop indices (i, j, k) are acceptable ONLY for tight loops; for anything
    outside simple indices, prefer descriptive names.
  - Coordinate/math conventions (x, y) are acceptable when contextually appropriate.
  - Names must match purpose and be consistent within the same scope.
- Serious Code Quality / Best Practice violations that impact correctness, security, or performance

ANCHORING:
- You review ONE diff hunk at a time; lineNumber is 1-based within this hunk.
- Prefer '+' (added) lines; use nearby context ' ' lines only if necessary (±3 lines).
- Never target '-' (removed) lines unless the act of removal introduces a problem.
- Do NOT suggest re-adding code that was intentionally removed unless you can show
  a concrete breakage (e.g., API contract violation, missing required behavior, clear bug).
- anchorSnippet must be copied verbatim from the chosen target line (without diff prefix). If you cannot anchor confidently, omit the item.

CONTEXT AWARENESS:
- Use any provided repository/project context to understand how this hunk integrates
  with the rest of the codebase. Prefer fixes that align with existing patterns and APIs.
- If functionality was moved or simplified by deletions, treat that as a potential improvement,
  not a regression, unless you can demonstrate a specific problem introduced by the change.

REVIEW RULES:
- Be precise and actionable. If uncertain, omit.
- In 'explanation', follow the format: "XYZ is wrong and heres why..."
- In 'fixCode', provide the corrected code snippet.
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
- Do not recommend reintroducing code that the diff removes unless removal breaks
  existing behavior or violates a public contract validated by surrounding context.
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
4. If a previous comment is addressed, do NOT include it in the output (omit it). Only include unresolved items.
5. Do NOT suggest new improvements, optimizations, or refactorings.
6. Do NOT comment on code that wasn't mentioned in previous comments.
7. ONLY focus on verifying the resolution of the specific issues mentioned in previous comments.
8. IMPORTANT: Fixes that remove problematic code count as valid resolutions. If the fix consists of deleting the previously problematic code (e.g., removing a try/except around lock.acquire()), treat the '-' deletion lines in the diff as evidence of resolution.
9. Do NOT recommend re-adding removed code in follow-up mode. If the prior issue is resolved by deletion and no new breakage is introduced, consider it resolved.

PREVIOUS COMMENTS TO VERIFY:
{previous_comments}

ANCHORING:
- You review ONE diff hunk at a time; lineNumber is 1-based within this hunk.
- Prefer '+' lines; use nearby context ' ' lines only if necessary (±3 lines).
- For deletion-only fixes, it's acceptable to reference nearby context lines if no added lines exist; you may also cite the deleted code in your explanation, but do not output a review item if the issue is resolved.
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
