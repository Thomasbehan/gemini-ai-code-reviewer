"""
Configuration management for the Gemini AI Code Reviewer.

This module handles all configuration aspects including environment variables,
validation, and default settings.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum

from .models import ReviewFocus, ReviewPriority


class ReviewMode(Enum):
    """Different review modes."""
    STRICT = "strict"
    STANDARD = "standard" 
    LENIENT = "lenient"
    SECURITY_FOCUSED = "security_focused"
    PERFORMANCE_FOCUSED = "performance_focused"


class LogLevel(Enum):
    """Available log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class GitHubConfig:
    """Configuration for GitHub integration."""
    token: str
    api_base_url: str = "https://api.github.com"
    timeout: int = 30
    max_retries: int = 3
    retry_delay_min: int = 4
    retry_delay_max: int = 10
    
    def __post_init__(self):
        """Validate GitHub configuration."""
        if not self.token:
            raise ValueError("GitHub token is required")
        
        if not self._validate_token_format(self.token):
            raise ValueError("Invalid GitHub token format")
    
    @staticmethod
    def _validate_token_format(token: str) -> bool:
        """Validate GitHub token format."""
        if not token or not isinstance(token, str):
            return False
        # GitHub tokens are typically 40 characters (classic) or start with specific prefixes
        return len(token) >= 4 and (
            len(token) == 40 or 
            token.startswith(('ghp_', 'ghs_', 'gho_', 'ghu_'))
        )


@dataclass  
class GeminiConfig:
    """Configuration for Gemini AI integration."""
    api_key: str
    model_name: str = "gemini-2.5-flash"
    max_output_tokens: int = 8192
    temperature: float = 0.0  # Lower temperature for more precise, deterministic code reviews
    top_p: float = 0.9  # Slightly lower for more focused output
    timeout: int = 60
    max_retries: int = 3
    retry_delay_min: int = 4
    retry_delay_max: int = 60
    max_prompt_length: int = 100000
    
    def __post_init__(self):
        """Validate Gemini configuration."""
        if not self.api_key:
            raise ValueError("Gemini API key is required")
            
        if not self._validate_api_key_format(self.api_key):
            raise ValueError("Invalid Gemini API key format")
            
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
            
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("Top_p must be between 0.0 and 1.0")
    
    @staticmethod
    def _validate_api_key_format(api_key: str) -> bool:
        """Validate Gemini API key format."""
        if not api_key or not isinstance(api_key, str):
            return False
        # Gemini API keys are typically alphanumeric with some special characters
        return len(api_key) > 10


@dataclass
class ReviewConfig:
    """Configuration for code review behavior."""
    review_mode: ReviewMode = ReviewMode.STANDARD
    focus_areas: List[ReviewFocus] = field(default_factory=lambda: [ReviewFocus.ALL])
    exclude_patterns: List[str] = field(default_factory=list)
    include_patterns: List[str] = field(default_factory=list)
    max_files_per_review: int = 50
    max_lines_per_hunk: int = 500
    max_hunks_per_file: int = 20
    min_line_changes: int = 1
    review_test_files: bool = False
    review_docs: bool = False
    custom_prompt_template: Optional[str] = None
    priority_threshold: ReviewPriority = ReviewPriority.LOW
    
    def __post_init__(self):
        """Validate review configuration."""
        if self.max_files_per_review <= 0:
            raise ValueError("max_files_per_review must be positive")
            
        if self.max_lines_per_hunk <= 0:
            raise ValueError("max_lines_per_hunk must be positive")
            
        # Set default exclude patterns if none specified
        if not self.exclude_patterns:
            self.exclude_patterns = [
                "*.md", "*.txt", "*.yml", "*.yaml", "*.json",
                "package-lock.json", "yarn.lock", "*.log"
            ]


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    enable_concurrent_processing: bool = True
    max_concurrent_files: int = 3
    max_concurrent_api_calls: int = 5
    chunk_size: int = 10
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    
    def __post_init__(self):
        """Validate performance configuration."""
        if self.max_concurrent_files <= 0:
            self.max_concurrent_files = 1
            
        if self.max_concurrent_api_calls <= 0:
            self.max_concurrent_api_calls = 1


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    enable_file_logging: bool = False
    log_file_path: str = "gemini_reviewer.log"
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 3


@dataclass
class Config:
    """Main configuration class that combines all configuration sections."""
    github: GitHubConfig
    gemini: GeminiConfig
    review: ReviewConfig = field(default_factory=ReviewConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_environment(cls) -> 'Config':
        """Create configuration from environment variables."""
        # Required environment variables
        github_token = os.environ.get("GITHUB_TOKEN", "")
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        
        if not github_token:
            raise ValueError("GITHUB_TOKEN environment variable is required")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # GitHub configuration
        github_config = GitHubConfig(
            token=github_token,
            timeout=int(os.environ.get("GITHUB_TIMEOUT", "30")),
            max_retries=int(os.environ.get("GITHUB_MAX_RETRIES", "3"))
        )
        
        # Gemini configuration  
        gemini_config = GeminiConfig(
            api_key=gemini_api_key,
            model_name=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            temperature=float(os.environ.get("GEMINI_TEMPERATURE", "0.0")),  # Lower for precise code reviews
            top_p=float(os.environ.get("GEMINI_TOP_P", "0.9")),  # Lower for more focused output
            max_output_tokens=int(os.environ.get("GEMINI_MAX_TOKENS", "8192"))
        )
        
        # Review configuration
        exclude_patterns_raw = os.environ.get("EXCLUDE", "") or os.environ.get("INPUT_EXCLUDE", "")
        include_patterns_raw = os.environ.get("INCLUDE", "") or os.environ.get("INPUT_INCLUDE", "")
        exclude_patterns = [p.strip() for p in exclude_patterns_raw.split(",") if p.strip()] if exclude_patterns_raw else []
        include_patterns = [p.strip() for p in include_patterns_raw.split(",") if p.strip()] if include_patterns_raw else []
        
        review_mode_str = os.environ.get("REVIEW_MODE", "standard").lower()
        review_mode = ReviewMode.STANDARD
        try:
            review_mode = ReviewMode(review_mode_str)
        except ValueError:
            logging.warning(f"Invalid review mode '{review_mode_str}', using 'standard'")
        
        # Priority threshold
        priority_str = (
            os.environ.get("REVIEW_PRIORITY_THRESHOLD")
            or os.environ.get("INPUT_REVIEW_PRIORITY_THRESHOLD")
            or os.environ.get("PRIORITY_THRESHOLD")
        )
        priority_threshold = ReviewPriority.LOW
        if priority_str:
            try:
                priority_threshold = ReviewPriority(priority_str.strip().lower())
            except ValueError:
                logging.warning(f"Invalid priority threshold '{priority_str}', using 'low'")
        
        # Get custom system prompt if provided
        system_prompt = os.environ.get("SYSTEM_PROMPT", "") or os.environ.get("INPUT_SYSTEM_PROMPT", "")
        custom_prompt_template = system_prompt.strip() if system_prompt else None
        
        review_config = ReviewConfig(
            review_mode=review_mode,
            exclude_patterns=exclude_patterns,
            include_patterns=include_patterns,
            max_files_per_review=int(os.environ.get("MAX_FILES_PER_REVIEW", "50")),
            max_lines_per_hunk=int(os.environ.get("MAX_LINES_PER_HUNK", "500")),
            review_test_files=os.environ.get("REVIEW_TEST_FILES", "false").lower() == "true",
            review_docs=os.environ.get("REVIEW_DOCS", "false").lower() == "true",
            custom_prompt_template=custom_prompt_template,
            priority_threshold=priority_threshold
        )
        
        # Performance configuration
        performance_config = PerformanceConfig(
            enable_concurrent_processing=os.environ.get("ENABLE_CONCURRENT", "true").lower() == "true",
            max_concurrent_files=int(os.environ.get("MAX_CONCURRENT_FILES", "3")),
            max_concurrent_api_calls=int(os.environ.get("MAX_CONCURRENT_API_CALLS", "5")),
            enable_caching=os.environ.get("ENABLE_CACHING", "true").lower() == "true"
        )
        
        # Logging configuration
        log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
        log_level = LogLevel.INFO
        try:
            log_level = LogLevel(log_level_str)
        except ValueError:
            logging.warning(f"Invalid log level '{log_level_str}', using 'INFO'")
        
        logging_config = LoggingConfig(
            level=log_level,
            enable_file_logging=os.environ.get("ENABLE_FILE_LOGGING", "false").lower() == "true"
        )
        
        return cls(
            github=github_config,
            gemini=gemini_config,
            review=review_config,
            performance=performance_config,
            logging=logging_config
        )
    
    def get_review_prompt_template(self) -> str:
        """Get the prompt template based on review mode."""
        if self.review.custom_prompt_template:
            return self.review.custom_prompt_template
        
        base_prompt = """CRITICAL: You MUST respond with ONLY valid JSON. NO conversational text, NO explanations outside the JSON, NO introductions, NO conclusions.

REQUIRED OUTPUT FORMAT - RESPOND WITH THIS EXACT STRUCTURE:
{
  "reviews": [
    {
      "lineNumber": 1,
      "reviewComment": "Your detailed review comment here with specific issue and how to fix it",
      "priority": "high",
      "category": "security",
      "anchorSnippet": "exact code from the target line (without +/ - prefix)"
    }
  ]
}

If no issues found, respond with:
{
  "reviews": []
}

DO NOT:
- DO NOT write "Here is my review" or any conversational text
- DO NOT explain what you're doing
- DO NOT add markdown code blocks around the JSON
- DO NOT include any text before or after the JSON
- DO NOT be polite or conversational
- DO start your response directly with the { character

YOU ARE A CODE ANALYSIS API - OUTPUT ONLY JSON.

WHAT TO REVIEW:
You are an expert code reviewer. Look for and comment on:
- **Bugs & Logic Errors**: Incorrect logic, edge cases not handled, potential runtime errors
- **Security Issues**: Vulnerabilities, injection risks, authentication/authorization problems, data exposure
- **Performance Problems**: Inefficient algorithms, unnecessary loops, memory leaks, N+1 queries
- **Code Quality**: Duplicated code, poor naming, complex functions that should be split
- **Error Handling**: Missing try-catch, unhandled edge cases, silent failures
- **Best Practices**: Violations of language-specific conventions, anti-patterns
- **Potential Null/Undefined**: Missing null checks, unsafe optional chaining
- **Resource Management**: Unclosed connections, file handles, memory management issues

LINE ANCHORING RULES (to avoid unrelated comments):
- You are reviewing ONE diff hunk at a time. The lineNumber is 1-based and MUST refer to the line index within the shown hunk (not the original file).
- Prefer lines that start with '+' (added lines). If the issue involves surrounding context, you may target a nearby context line ' ' within ±3 lines.
- NEVER target lines that start with '-' unless the issue is specifically about removed code causing a problem.
- The "anchorSnippet" MUST be copied verbatim from the chosen target line (remove the leading diff marker). If you cannot confidently pick a target line, OMIT the review entirely.

REVIEW GUIDELINES:
- Be specific and actionable; focus on concrete code changes
- Reference the exact line number where the issue occurs (relative to this hunk)
- Briefly state WHY (one short sentence max), then show HOW to fix it with a code change or patch
- Only include a review item if you can propose a specific code change; otherwise OMIT it
- Always provide a solution: include a minimal code snippet illustrating the fix when possible
- Do NOT ask for more information or additional context; propose the best fix with the given code
- Do NOT praise, congratulate, or acknowledge good changes; only report problems
- Avoid nitpicks and subjective style comments unless they impact correctness, security, or performance
- Prioritize critical bugs and security issues as "high" or "critical"
- Mark minor improvements as "low"
- NEVER suggest adding code comments or documentation (focus on code issues only)
- Use GitHub Markdown for formatting within the reviewComment field
- If the code is genuinely good with no issues, return {"reviews": []}

VALIDATION:
- If you cannot include an accurate anchorSnippet from the targeted line, return {"reviews": []} instead of guessing.

REMEMBER: Your entire response must be valid, parseable JSON starting with { and ending with }"""
        
        mode_specific_instructions = {
            ReviewMode.STRICT: """
- Focus on ALL potential issues including minor style problems
- Be very thorough and pedantic
- Flag any deviation from best practices""",
            
            ReviewMode.STANDARD: """
- Focus on bugs, security issues, and performance problems
- Include maintainability concerns
- Skip minor style issues unless they impact readability""",
            
            ReviewMode.LENIENT: """
- Focus ONLY on critical bugs and security vulnerabilities
- Skip style and minor maintainability issues
- Be concise in feedback""",
            
            ReviewMode.SECURITY_FOCUSED: """
- Focus EXCLUSIVELY on security vulnerabilities
- Look for injection attacks, authentication issues, data exposure
- Flag any security anti-patterns""",
            
            ReviewMode.PERFORMANCE_FOCUSED: """
- Focus EXCLUSIVELY on performance issues
- Look for inefficient algorithms, memory leaks, unnecessary operations
- Flag performance anti-patterns"""
        }
        
        focus_instruction = mode_specific_instructions.get(self.review.review_mode, "")
        return base_prompt + focus_instruction
    
    def should_review_file(self, file_path: str) -> bool:
        """Determine if a file should be reviewed based on configuration."""
        # Check include patterns first
        if self.review.include_patterns:
            if not any(self._matches_pattern(file_path, pattern) 
                      for pattern in self.review.include_patterns):
                return False
        
        # Check exclude patterns
        if any(self._matches_pattern(file_path, pattern) 
               for pattern in self.review.exclude_patterns):
            return False
        
        # Check test files
        if not self.review.review_test_files and self._is_test_file(file_path):
            return False
        
        # Check documentation files
        if not self.review.review_docs and self._is_doc_file(file_path):
            return False
        
        return True
    
    @staticmethod
    def _matches_pattern(file_path: str, pattern: str) -> bool:
        """Check if file path matches a pattern."""
        import fnmatch
        return fnmatch.fnmatch(file_path, pattern)
    
    @staticmethod
    def _is_test_file(file_path: str) -> bool:
        """Check if file is a test file (cross-platform)."""
        lowered = file_path.lower()
        test_patterns = ['test_', '_test.', 'spec_', '_spec.', '/test/', '/tests/', '\\test\\', '\\tests\\']
        return any(pattern in lowered for pattern in test_patterns)
    
    @staticmethod
    def _is_doc_file(file_path: str) -> bool:
        """Check if file is a documentation file."""
        doc_extensions = {'.md', '.rst', '.txt', '.doc', '.docx'}
        return any(file_path.lower().endswith(ext) for ext in doc_extensions)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "github": {
                "timeout": self.github.timeout,
                "max_retries": self.github.max_retries,
            },
            "gemini": {
                "model_name": self.gemini.model_name,
                "temperature": self.gemini.temperature,
                "top_p": self.gemini.top_p,
                "max_output_tokens": self.gemini.max_output_tokens,
            },
            "review": {
                "review_mode": self.review.review_mode.value,
                "focus_areas": [area.value for area in self.review.focus_areas],
                "exclude_patterns": self.review.exclude_patterns,
                "include_patterns": self.review.include_patterns,
                "priority_threshold": self.review.priority_threshold.value,
                "max_files_per_review": self.review.max_files_per_review,
                "max_lines_per_hunk": self.review.max_lines_per_hunk,
            },
            "performance": {
                "enable_concurrent_processing": self.performance.enable_concurrent_processing,
                "max_concurrent_files": self.performance.max_concurrent_files,
                "enable_caching": self.performance.enable_caching,
            },
            "logging": {
                "level": self.logging.level.value,
                "enable_file_logging": self.logging.enable_file_logging,
            }
        }
