"""
Gemini AI client for the Gemini AI Code Reviewer.

This module handles all interactions with Google's Gemini AI including
prompt engineering, response validation, and error handling.
"""

import json
import logging
import time
import re
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import GeminiConfig, ReviewMode
from .models import AIResponse, ReviewPriority, AnalysisContext, HunkInfo, PRDetails
from .utils import get_file_language, sanitize_text, sanitize_code_content


logger = logging.getLogger(__name__)


class GeminiClientError(Exception):
    """Base exception for Gemini client errors."""
    pass


class ModelNotAvailableError(GeminiClientError):
    """Exception raised when the specified model is not available."""
    pass


class TokenLimitExceededError(GeminiClientError):
    """Exception raised when token limit is exceeded."""
    pass


class GeminiClient:
    """Gemini AI client with retry logic and comprehensive error handling."""
    
    def __init__(self, config: GeminiConfig):
        """Initialize Gemini client with configuration."""
        self.config = config
        
        try:
            genai.configure(api_key=config.api_key)
            self._model = genai.GenerativeModel(config.model_name)
            logger.info(f"Initialized Gemini client with model: {config.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            raise GeminiClientError(f"Failed to initialize Gemini client: {str(e)}")
        
        self._generation_config = {
            "max_output_tokens": config.max_output_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            # Force the model to emit JSON only, reducing chances of conversational wrappers
            "response_mime_type": "application/json",
        }
        
        # Statistics tracking
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_tokens_used = 0
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((Exception,))
    )
    def analyze_code_hunk(
        self,
        hunk: HunkInfo,
        context: AnalysisContext,
        prompt_template: str
    ) -> List[AIResponse]:
        """Analyze a code hunk and return AI responses with retry logic."""
        self._total_requests += 1
        
        if not hunk or not hunk.content:
            logger.warning("Empty hunk provided for analysis")
            return []
        
        if not context or not context.pr_details:
            logger.warning("Invalid analysis context provided")
            return []
        
        try:
            prompt = self._create_analysis_prompt(hunk, context, prompt_template)
            
            if len(prompt) > self.config.max_prompt_length:
                logger.warning(f"Prompt too long ({len(prompt)} chars), truncating...")
                prompt = prompt[:self.config.max_prompt_length] + "...[truncated]"
            
            logger.debug(f"Analyzing hunk with {len(hunk.content)} characters of content")
            logger.debug(f"Prompt preview: {prompt[:200]}...")
            
            response = self._generate_content_with_validation(prompt)
            ai_responses = self._parse_ai_response(response)
            
            self._successful_requests += 1
            logger.info(f"Generated {len(ai_responses)} AI responses for hunk")
            
            return ai_responses
            
        except Exception as e:
            self._failed_requests += 1
            logger.error(f"Error analyzing code hunk: {str(e)}")
            raise
    
    def _generate_content_with_validation(self, prompt: str) -> str:
        """Generate content with validation and error handling."""
        logger.info("Sending request to Gemini API...")
        
        try:
            response = self._model.generate_content(prompt, generation_config=self._generation_config)
            
            if not response:
                raise GeminiClientError("Empty response from Gemini API")
            
            # Check if response has valid parts before accessing text
            # finish_reason values: 1=STOP (normal), 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, 'finish_reason', None)
                
                # If content was filtered (safety/recitation), return empty result with warning
                if finish_reason in [3, 4]:  # SAFETY or RECITATION
                    reason_name = "SAFETY" if finish_reason == 3 else "RECITATION"
                    logger.warning(f"Gemini API response filtered due to {reason_name} settings (finish_reason={finish_reason}). "
                                 "This is expected for some code patterns and does not indicate an error. "
                                 "Returning empty review for this hunk.")
                    # Return empty JSON array to indicate no reviews for this hunk
                    return "[]"
                
                # Check if there are valid parts in the response
                if not hasattr(candidate, 'content') or not candidate.content or not candidate.content.parts:
                    logger.warning(f"Response has no valid parts (finish_reason={finish_reason}). Returning empty review.")
                    return "[]"
            
            # Try to access the text - this may still fail for other reasons
            if not hasattr(response, 'text'):
                logger.warning("Response object has no text attribute. Returning empty review.")
                return "[]"
            
            response_text = response.text.strip()
            if not response_text:
                logger.warning("Empty response text from Gemini API. Returning empty review.")
                return "[]"
            
            logger.debug(f"Received response (length: {len(response_text)})")
            
            # Track token usage if available
            if hasattr(response, 'usage_metadata'):
                try:
                    tokens_used = response.usage_metadata.total_token_count
                    self._total_tokens_used += tokens_used
                    logger.debug(f"Tokens used: {tokens_used}")
                except Exception:
                    pass  # Token counting not critical
            
            return response_text
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if this is the specific "response.text requires valid Part" error
            if "response.text" in error_msg and "valid" in error_msg and "part" in error_msg:
                logger.warning(f"Response filtered by Gemini API (likely safety/content policy). "
                             "This is expected for some code patterns. Returning empty review for this hunk.")
                return "[]"
            
            if "quota" in error_msg or "rate limit" in error_msg:
                logger.warning("Gemini API rate limit or quota exceeded")
                raise GeminiClientError("API rate limit exceeded")
            elif "not found" in error_msg or "model" in error_msg:
                raise ModelNotAvailableError(f"Model {self.config.model_name} not available")
            elif "token" in error_msg and "limit" in error_msg:
                raise TokenLimitExceededError("Token limit exceeded")
            else:
                logger.error(f"Gemini API error: {str(e)}")
                raise GeminiClientError(f"Gemini API error: {str(e)}")
    
    def _create_analysis_prompt(
        self,
        hunk: HunkInfo,
        context: AnalysisContext,
        prompt_template: str
    ) -> str:
        """Create a comprehensive analysis prompt with project context."""
        # Sanitize inputs
        sanitized_content = sanitize_code_content(hunk.content)
        sanitized_title = sanitize_text(context.pr_details.title)
        sanitized_description = sanitize_text(context.pr_details.description or "No description provided")
        
        # Add context information
        context_info = []
        if context.file_info:
            context_info.append(f"File: {context.file_info.path}")
            if context.file_info.file_extension:
                language = get_file_language(context.file_info.path)
                if language and language != 'unknown':
                    context_info.append(f"Language: {language}")
        
        if context.is_test_file:
            context_info.append("Note: This is a test file")
        
        if context.related_files:
            context_info.append(f"Related files detected: {', '.join(context.related_files)}")
        
        context_string = "\n".join(context_info) if context_info else ""
        
        # Build the complete prompt with enhanced instructions
        prompt_parts = [
            prompt_template,
            "",
            "IMPORTANT: You have FULL REPOSITORY CONTEXT including the complete file structure and all functions/classes.",
            "Use this context to:",
            "- Verify that functions/classes mentioned or called in the changes actually exist in the codebase",
            "- Check if the code correctly uses existing APIs, functions, and patterns from other files",
            "- Identify when the changes duplicate functionality that already exists elsewhere",
            "- Ensure proper integration with the broader codebase architecture",
            "- Consider how changes affect other parts of the system based on the repository structure",
            "- Catch bugs that only become apparent when you know what exists in other files",
            "",
            "DO NOT suggest adding functions/classes that already exist in the repository.",
            "DO NOT flag missing implementations if they exist elsewhere in the codebase.",
            "",
            f"Pull request title: {sanitized_title}",
            "Pull request description:",
            "---",
            sanitized_description,
            "---",
            ""
        ]
        
        if context_string:
            prompt_parts.extend([
                "File Context:",
                context_string,
                ""
            ])
        
        # Add project context if available (related file contents)
        if context.project_context:
            prompt_parts.extend([
                "Project Context (Related Files):",
                "---",
                context.project_context,
                "---",
                "",
                "Use the above related files to understand how this code fits into the larger project.",
                "Look for:",
                "- Incorrect usage of APIs or functions defined in related files",
                "- Violations of patterns established in the codebase",
                "- Missing error handling based on how related code behaves",
                "- Type mismatches or contract violations",
                ""
            ])
        
        prompt_parts.extend([
            "Git diff to review:",
            "```diff",
            sanitized_content,
            "```"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_ai_response(self, response_text: str) -> List[AIResponse]:
        """Parse AI response and validate the structure.
        Accepts both object-with-'reviews' and top-level list schemas.
        """
        try:
            # Log raw response details for debugging
            logger.debug(f"Raw response length: {len(response_text)} characters")
            logger.debug(f"Raw response preview: {response_text[:200]}...")
            
            # Check if response is empty or whitespace-only
            if not response_text or not response_text.strip():
                logger.error("Received empty or whitespace-only response from Gemini API")
                logger.debug(f"Raw response repr: {repr(response_text[:100])}")
                return []
            
            # Clean the response text
            cleaned_response = self._clean_response_text(response_text)
            logger.debug(f"Cleaned response length: {len(cleaned_response)} characters")
            logger.debug(f"Cleaned response preview: {cleaned_response[:200]}...")
            
            # Validate cleaned response is not empty
            if not cleaned_response or not cleaned_response.strip():
                logger.error("Cleaned response is empty after removing markdown formatting")
                logger.error(f"Raw response was: {response_text[:500]}...")
                return []
            
            # Parse JSON (can be dict or list)
            data = json.loads(cleaned_response)
            logger.debug("Successfully parsed JSON response from Gemini")
            
            reviews_list: Optional[List[Dict[str, Any]]] = None
            
            if isinstance(data, dict):
                # Standard path
                if "reviews" in data and isinstance(data["reviews"], list):
                    reviews_list = data["reviews"]
                else:
                    # Accept alternate top-level keys commonly used by LLMs
                    alt_keys = ["comments", "findings", "issues", "items", "results", "reviewComments"]
                    for k in alt_keys:
                        if k in data and isinstance(data[k], list):
                            logger.info(f"Parsed reviews from alternate top-level key: '{k}'")
                            reviews_list = data[k]
                            break
                    # If still none, maybe the object itself represents a single review item
                    if reviews_list is None:
                        logger.warning("Response JSON object lacks 'reviews' (or alternates); attempting single-item parse")
                        reviews_list = [data]
            elif isinstance(data, list):
                logger.info("Parsed reviews from top-level JSON array")
                reviews_list = data
            else:
                logger.warning(f"Unexpected JSON root type: {type(data)}")
                return []
            
            if not reviews_list:
                logger.warning("No review items found after schema normalization")
                return []
            
            # Convert to AIResponse objects
            ai_responses: List[AIResponse] = []
            for review in reviews_list:
                ai_response = self._parse_single_review(review)
                if ai_response:
                    ai_responses.append(ai_response)
            
            logger.info(f"Successfully parsed {len(ai_responses)} AI responses")
            return ai_responses
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error(f"Raw response length: {len(response_text)}")
            logger.error(f"Raw response preview: {response_text[:500]}...")
            logger.error(f"Cleaned response preview: {cleaned_response[:500] if 'cleaned_response' in locals() else 'N/A'}...")

            # Fallback: try to extract a valid JSON object or array from the raw response
            fallback_json_text = self._extract_valid_json_segment(response_text)
            if fallback_json_text:
                try:
                    data = json.loads(fallback_json_text)
                    logger.info("Recovered by extracting a valid JSON segment from the response")

                    reviews_list: Optional[List[Dict[str, Any]]] = None
                    if isinstance(data, dict):
                        if "reviews" in data and isinstance(data["reviews"], list):
                            reviews_list = data["reviews"]
                        else:
                            for k in ["comments", "findings", "issues", "items", "results", "reviewComments"]:
                                if k in data and isinstance(data[k], list):
                                    logger.info(f"Recovered reviews from alternate key '{k}' in fallback JSON")
                                    reviews_list = data[k]
                                    break
                            if reviews_list is None:
                                reviews_list = [data]
                    elif isinstance(data, list):
                        reviews_list = data
                    else:
                        return []

                    ai_responses: List[AIResponse] = []
                    for review in reviews_list:
                        ai_response = self._parse_single_review(review)
                        if ai_response:
                            ai_responses.append(ai_response)

                    logger.info(f"Successfully parsed {len(ai_responses)} AI responses (fallback path)")
                    return ai_responses
                except Exception as inner_e:
                    logger.error(f"Fallback JSON extraction also failed: {str(inner_e)}")

            return []
        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            logger.debug(f"Raw response: {response_text[:1000]}...")
            return []
    
    def _parse_single_review(self, review: Dict[str, Any]) -> Optional[AIResponse]:
        """Parse a single review from the AI response.
        Accepts multiple field name variants from different response schemas.
        """
        try:
            if not isinstance(review, dict):
                logger.warning(f"Invalid review format: {type(review)}")
                return None
            
            # Normalize line number field
            line_keys = [
                "lineNumber", "line", "ln", "line_no", "lineIndex", "position", "pos"
            ]
            line_number: Optional[int] = None
            for k in line_keys:
                if k in review:
                    try:
                        ln_val = review[k]
                        # Sometimes it can be 1-based index in string form
                        line_number = int(str(ln_val).strip())
                        break
                    except Exception:
                        continue
            if not line_number or line_number <= 0:
                logger.warning(f"Review missing or invalid line number field: keys tried {line_keys}; review keys: {list(review.keys())}")
                return None
            
            # Normalize comment/message field
            comment_keys = [
                "explanation", "reviewComment", "comment", "message", "finding", "text", "body", "description"
            ]
            comment: Optional[str] = None
            for k in comment_keys:
                if k in review and review[k] is not None:
                    comment = str(review[k])
                    break
            if not comment or len(comment.strip()) == 0:
                logger.warning("Empty review comment after normalization")
                return None
            comment = sanitize_text(comment)
            
            # Extract fix code
            fix_code = None
            for k in ["fixCode", "fix", "suggestion", "codeBlock"]:
                if k in review and review[k]:
                    fix_code = str(review[k]).strip()
                    break

            # Optional: anchor snippet tying the comment to a concrete line
            anchor_snippet = None
            for k in ["anchorSnippet", "anchor", "snippet", "code", "anchorText"]:
                if k in review and isinstance(review[k], str) and review[k].strip():
                    anchor_snippet = review[k].strip()
                    break
            
            # If not provided explicitly, try to infer from first inline code span using backticks
            if not anchor_snippet and '`' in comment:
                try:
                    # Prefer shortest inline code span (single backticks) to reduce false matches
                    import re as _re
                    inline_matches = list(_re.finditer(r"`([^`\n]+)`", comment))
                    if inline_matches:
                        # pick the first non-trivial snippet
                        for m in inline_matches:
                            candidate = m.group(1).strip()
                            if len(candidate) >= 2:
                                anchor_snippet = candidate
                                break
                        if not anchor_snippet:
                            anchor_snippet = inline_matches[0].group(1).strip()
                except Exception:
                    pass
            
            # Require actionable fix: keep only comments that include some code indication
            # if not anchor_snippet and '`' not in comment:
            #    logger.info("Discarding non-actionable review (no code snippet/anchor provided)")
            #    return None
            
            # Optional: priority/severity/level
            priority_val = None
            for k in ["priority", "severity", "level", "rating"]:
                if k in review:
                    priority_val = review[k]
                    break
            priority = self._parse_priority(priority_val)
            
            # Optional: category/tag/label/type
            category_val = None
            for k in ["category", "tag", "label", "type", "area"]:
                if k in review:
                    category_val = review[k]
                    break
            category = category_val
            
            # Optional: confidence/score/probability
            confidence_val = None
            for k in ["confidence", "score", "probability", "likelihood"]:
                if k in review:
                    confidence_val = review[k]
                    break
            confidence = self._parse_confidence(confidence_val)
            
            # Handle percentage-like confidences (>1.0 up to 100)
            if confidence is None and confidence_val is not None:
                try:
                    c = float(confidence_val)
                    if c > 1.0 and c <= 100.0:
                        confidence = max(0.0, min(1.0, c / 100.0))
                except Exception:
                    pass
            
            return AIResponse(
                line_number=line_number,
                review_comment=comment,
                priority=priority,
                category=category,
                confidence=confidence,
                anchor_snippet=anchor_snippet,
                fix_code=fix_code
            )
            
        except Exception as e:
            logger.warning(f"Error parsing single review: {str(e)}")
            return None
    
    def _clean_response_text(self, response_text: str) -> str:
        """Clean the response text from common formatting issues and extract JSON or JSON array.
        
        This method handles cases where the AI adds conversational text before/after the JSON,
        despite instructions to output only JSON. It searches for and extracts the JSON object or array.
        """
        cleaned = response_text.strip()
        
        # First, try to remove markdown code block markers if present
        if '```' in cleaned:
            # Remove opening markdown block
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:].lstrip()
            elif cleaned.startswith('```JSON'):
                cleaned = cleaned[7:].lstrip()
            elif cleaned.startswith('```\n'):
                cleaned = cleaned[4:]
            elif cleaned.startswith('```'):
                first_newline = cleaned.find('\n')
                if first_newline != -1:
                    cleaned = cleaned[first_newline + 1:]
                else:
                    cleaned = cleaned[3:].lstrip()
            
            # Remove closing ``` markers
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3].rstrip()
        
        # Find the earliest JSON start either object '{' or array '['
        obj_start = cleaned.find('{')
        arr_start = cleaned.find('[')
        
        # Determine which JSON construct appears first
        start_idx = -1
        if arr_start != -1 and (obj_start == -1 or arr_start < obj_start):
            start_idx = arr_start
        elif obj_start != -1:
            start_idx = obj_start
        else:
            logger.warning("No JSON object or array start found in response")
            return cleaned
        
        # Use JSONDecoder to robustly find the end of the JSON object
        try:
            # raw_decode parses from the start and returns (obj, end_index)
            _, end_offset = json.JSONDecoder().raw_decode(cleaned[start_idx:])
            end_idx = start_idx + end_offset
            
            # Log if we had to strip conversational text
            if start_idx > 0:
                stripped_prefix = cleaned[:start_idx].strip()
                if stripped_prefix:
                    logger.info(f"Stripped conversational prefix from response: {stripped_prefix[:100]}...")
            
            if end_idx < len(cleaned):
                stripped_suffix = cleaned[end_idx:].strip()
                if stripped_suffix:
                    logger.info(f"Stripped conversational suffix from response: {stripped_suffix[:100]}...")
            
            return cleaned[start_idx:end_idx]
            
        except json.JSONDecodeError:
            logger.warning("Failed to decode JSON segment using raw_decode, falling back to heuristic extraction")
            # Return the substring from start_idx so fallback methods can try to handle it
            return cleaned[start_idx:]

    def _extract_valid_json_segment(self, text: str) -> Optional[str]:
        """Attempt to extract a valid JSON object or array that contains reviews from free-form text.
        Strategy:
        1) Prefer fenced ```json code blocks (object or array)
        2) Otherwise, collect all balanced array '[' ']' and object '{' '}' segments and
           try to parse them, prioritizing arrays and dicts that look like reviews.
        3) If all else fails (e.g. truncated JSON), try to recover by extracting valid objects.
        """
        if not text:
            return None
        raw = text.strip()

        # 1) Look for fenced json blocks first (object or array)
        try:
            code_fence_pattern = re.compile(r"```(?:json|JSON)\s*([\s\S]*?)\s*```", re.MULTILINE)
            matches = list(code_fence_pattern.finditer(raw))
            for m in matches:
                candidate = m.group(1).strip()
                try:
                    data = json.loads(candidate)
                    if isinstance(data, list) and data:
                        if isinstance(data[0], dict):
                            logger.info("Using JSON array extracted from ```json fenced block")
                            return candidate
                    if isinstance(data, dict):
                        logger.info("Using JSON object extracted from ```json fenced block")
                        return candidate
                except Exception:
                    continue
        except Exception:
            # Regex failure shouldn't break parsing
            pass

        # 2) Collect balanced segments for objects and arrays
        obj_segments = self._collect_balanced_brace_segments(raw)
        arr_segments = self._collect_balanced_bracket_segments(raw)
        candidates = []
        # Prefer arrays first, then objects
        candidates.extend(arr_segments or [])
        candidates.extend(obj_segments or [])
        
        if candidates:
            # Sort candidates: prefer those containing 'reviews' or review-like keys, then longer
            def looks_reviewish(seg: str) -> bool:
                keywords = ["\"reviews\"", "\"line\"", "\"lineNumber\"", "\"comment\"", "\"message\""]
                return any(k in seg for k in keywords)
            
            candidates.sort(key=lambda s: (not looks_reviewish(s), -len(s)))

            for seg in candidates:
                try:
                    data = json.loads(seg)
                    if isinstance(data, list):
                        if data and isinstance(data[0], dict):
                            return seg
                    elif isinstance(data, dict):
                        # Accept any dict; downstream will normalize single-item dicts too
                        return seg
                except Exception:
                    continue

        # 3) Fallback: Try to recover from truncated response
        # If we reached here, it means we couldn't parse a complete valid JSON structure.
        # This often happens if the response was truncated (max tokens).
        logger.info("No valid JSON found by standard methods, attempting to recover from truncated response...")
        
        # First try to recover complete objects from the list
        recovered = self._recover_json_list_from_objects(raw)
        if recovered:
             logger.info("Successfully recovered JSON list from truncated response")
             return recovered

        # 4) Fallback: Try to repair the truncated JSON string directly
        # This handles the case where the last object itself is truncated
        logger.info("Attempting to repair truncated JSON string...")
        repaired = self._repair_truncated_json(raw)
        try:
            data = json.loads(repaired)
            # Basic validation
            if isinstance(data, (dict, list)):
                logger.info("Successfully repaired truncated JSON")
                return repaired
        except Exception:
            pass

        return None

    def _extract_json_objects(self, text: str) -> List[str]:
        """Find valid JSON objects in text, handling nested braces and strings correctly."""
        objects = []
        stack = []
        in_string = False
        escape = False
        
        for i, char in enumerate(text):
            if escape:
                escape = False
                continue
            if char == '\\':
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
                
            if char == '{':
                stack.append(i)
            elif char == '}':
                if stack:
                    start = stack.pop()
                    if not stack: # Top level object closed
                        objects.append(text[start:i+1])
        return objects

    def _recover_json_list_from_objects(self, text: str) -> Optional[str]:
        """Attempt to recover a list of review objects from text that might be a truncated JSON array."""
        objects = self._extract_json_objects(text)
        if not objects:
            return None
            
        # Filter objects that look like reviews
        valid_reviews = []
        # Keywords common in Review object
        keywords = ['"reviewComment"', '"lineNumber"', '"priority"', '"category"']
        
        for obj_str in objects:
            # Check if it parses as valid JSON
            try:
                # We also verify it has at least one review keyword to avoid picking up garbage
                if any(k in obj_str for k in keywords):
                    # Parse to double check validity
                    json.loads(obj_str)
                    valid_reviews.append(obj_str)
            except Exception:
                continue
        
        if not valid_reviews:
            return None
            
        # Construct a new JSON array
        return "[" + ",".join(valid_reviews) + "]"

    def _repair_truncated_json(self, text: str) -> str:
        """Attempt to repair a truncated JSON string by closing open strings and brackets/braces."""
        stack = []
        in_string = False
        escape = False
        
        for char in text:
            if in_string:
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            
            # Not in string
            if char == '"':
                in_string = True
            elif char == '{':
                stack.append('}')
            elif char == '[':
                stack.append(']')
            elif char == '}' or char == ']':
                if stack:
                    # Ideally the character should match stack[-1]
                    # If not, the JSON is likely malformed, but we proceed
                    if stack[-1] == char:
                        stack.pop()

        # Construct repaired string
        repaired = text
        if in_string:
            repaired += '"'
            
        # Close everything remaining on stack in reverse order
        while stack:
            repaired += stack.pop()
            
        return repaired

    def _collect_balanced_brace_segments(self, text: str) -> List[str]:
        """Collect all top-level balanced brace substrings from text.
        Returns a list of substrings that start with '{' and end with the matching '}'.
        """
        segments: List[str] = []
        brace_count = 0
        start_idx: Optional[int] = None
        for i, ch in enumerate(text):
            if ch == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif ch == '}':
                if brace_count > 0:
                    brace_count -= 1
                    if brace_count == 0 and start_idx is not None:
                        segment = text[start_idx:i+1]
                        segments.append(segment)
                        start_idx = None
        return segments

    def _collect_balanced_bracket_segments(self, text: str) -> List[str]:
        """Collect all top-level balanced bracket substrings from text.
        Returns a list of substrings that start with '[' and end with the matching ']'.
        """
        segments: List[str] = []
        bracket_count = 0
        start_idx: Optional[int] = None
        for i, ch in enumerate(text):
            if ch == '[':
                if bracket_count == 0:
                    start_idx = i
                bracket_count += 1
            elif ch == ']':
                if bracket_count > 0:
                    bracket_count -= 1
                    if bracket_count == 0 and start_idx is not None:
                        segment = text[start_idx:i+1]
                        segments.append(segment)
                        start_idx = None
        return segments
    
    def _parse_priority(self, priority_value: Any) -> ReviewPriority:
        """Parse priority from AI response."""
        if not priority_value:
            return ReviewPriority.MEDIUM
        
        try:
            priority_str = str(priority_value).lower()
            priority_mapping = {
                'critical': ReviewPriority.CRITICAL,
                'high': ReviewPriority.HIGH,
                'medium': ReviewPriority.MEDIUM,
                'low': ReviewPriority.LOW
            }
            return priority_mapping.get(priority_str, ReviewPriority.MEDIUM)
        except Exception:
            return ReviewPriority.MEDIUM
    
    def _parse_confidence(self, confidence_value: Any) -> Optional[float]:
        """Parse confidence score from AI response."""
        if confidence_value is None:
            return None
        
        try:
            confidence = float(confidence_value)
            return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        except (ValueError, TypeError):
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get client usage statistics."""
        success_rate = 0.0
        if self._total_requests > 0:
            success_rate = self._successful_requests / self._total_requests
        
        return {
            'total_requests': self._total_requests,
            'successful_requests': self._successful_requests,
            'failed_requests': self._failed_requests,
            'success_rate': success_rate,
            'total_tokens_used': self._total_tokens_used,
            'model_name': self.config.model_name
        }
    
    def test_connection(self) -> bool:
        """Test connection to Gemini API."""
        try:
            test_prompt = "Respond with 'OK' if you can read this message."
            response = self._model.generate_content(test_prompt)
            return response and hasattr(response, 'text') and response.text.strip()
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text (rough approximation)."""
        # Rough approximation: 1 token ≈ 4 characters for English text
        # This is just an estimate since we don't have direct access to the tokenizer
        return len(text) // 4
    
    def close(self):
        """Clean up resources."""
        logger.debug("Gemini client closed")
