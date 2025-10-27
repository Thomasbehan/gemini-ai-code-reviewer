"""
Configuration management for the Gemini AI Code Reviewer.

This module handles all configuration aspects including environment variables,
validation, and default settings.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from .models import ReviewFocus, ReviewPriority
from .validators import (
    validate_required_string, validate_positive_int, validate_range,
    validate_github_token_format, validate_gemini_api_key_format,
    ensure_positive_or_default
)
from .env_reader import get_env_str, get_env_int, get_env_float, get_env_bool, get_env_list, get_env_enum
from .utils import matches_pattern, is_test_file, is_doc_file
from .prompts import ReviewMode, get_review_prompt_template as get_prompt_template


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
        validate_required_string(self.token, "GitHub token")
        if not validate_github_token_format(self.token):
            raise ValueError("Invalid GitHub token format")


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
        validate_required_string(self.api_key, "Gemini API key")
        if not validate_gemini_api_key_format(self.api_key):
            raise ValueError("Invalid Gemini API key format")
        validate_range(self.temperature, 0.0, 2.0, "Temperature")
        validate_range(self.top_p, 0.0, 1.0, "Top_p")


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
    # Comment caps (optional). 0 disables limits (default behavior).
    max_comments_total: int = 0
    max_comments_per_file: int = 0
    
    def __post_init__(self):
        """Validate review configuration."""
        validate_positive_int(self.max_files_per_review, "max_files_per_review")
        validate_positive_int(self.max_lines_per_hunk, "max_lines_per_hunk")
        
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
        self.max_concurrent_files = ensure_positive_or_default(self.max_concurrent_files, 1)
        self.max_concurrent_api_calls = ensure_positive_or_default(self.max_concurrent_api_calls, 1)


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
        github_token = get_env_str("GITHUB_TOKEN")
        gemini_api_key = get_env_str("GEMINI_API_KEY")
        
        if not github_token:
            raise ValueError("GITHUB_TOKEN environment variable is required")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # GitHub configuration
        github_config = GitHubConfig(
            token=github_token,
            timeout=get_env_int("GITHUB_TIMEOUT", 30),
            max_retries=get_env_int("GITHUB_MAX_RETRIES", 3)
        )
        
        # Gemini configuration  
        gemini_config = GeminiConfig(
            api_key=gemini_api_key,
            model_name=get_env_str("GEMINI_MODEL", "gemini-2.5-flash"),
            temperature=get_env_float("GEMINI_TEMPERATURE", 0.0),
            top_p=get_env_float("GEMINI_TOP_P", 0.9),
            max_output_tokens=get_env_int("GEMINI_MAX_TOKENS", 8192)
        )
        
        # Review configuration
        exclude_patterns = get_env_list("EXCLUDE", ",", "INPUT_EXCLUDE")
        include_patterns = get_env_list("INCLUDE", ",", "INPUT_INCLUDE")
        review_mode = get_env_enum("REVIEW_MODE", ReviewMode, ReviewMode.STANDARD)
        priority_threshold = get_env_enum(
            "REVIEW_PRIORITY_THRESHOLD", ReviewPriority, ReviewPriority.LOW,
            "INPUT_REVIEW_PRIORITY_THRESHOLD", "PRIORITY_THRESHOLD"
        )
        custom_prompt = get_env_str("SYSTEM_PROMPT", "", "INPUT_SYSTEM_PROMPT")
        
        review_config = ReviewConfig(
            review_mode=review_mode,
            exclude_patterns=exclude_patterns,
            include_patterns=include_patterns,
            max_files_per_review=get_env_int("MAX_FILES_PER_REVIEW", 50),
            max_lines_per_hunk=get_env_int("MAX_LINES_PER_HUNK", 500),
            review_test_files=get_env_bool("REVIEW_TEST_FILES", False),
            review_docs=get_env_bool("REVIEW_DOCS", False),
            custom_prompt_template=custom_prompt if custom_prompt else None,
            priority_threshold=priority_threshold,
            max_comments_total=get_env_int("MAX_COMMENTS_TOTAL", 0, "INPUT_MAX_COMMENTS_TOTAL"),
            max_comments_per_file=get_env_int("MAX_COMMENTS_PER_FILE", 0, "INPUT_MAX_COMMENTS_PER_FILE")
        )
        
        # Performance configuration
        performance_config = PerformanceConfig(
            enable_concurrent_processing=get_env_bool("ENABLE_CONCURRENT", True),
            max_concurrent_files=get_env_int("MAX_CONCURRENT_FILES", 3),
            max_concurrent_api_calls=get_env_int("MAX_CONCURRENT_API_CALLS", 5),
            enable_caching=get_env_bool("ENABLE_CACHING", True)
        )
        
        # Logging configuration
        log_level = get_env_enum("LOG_LEVEL", LogLevel, LogLevel.INFO)
        logging_config = LoggingConfig(
            level=log_level,
            enable_file_logging=get_env_bool("ENABLE_FILE_LOGGING", False)
        )
        
        return cls(
            github=github_config,
            gemini=gemini_config,
            review=review_config,
            performance=performance_config,
            logging=logging_config
        )
    
    def get_review_prompt_template(self) -> str:
        """Get the prompt template based on review mode and custom instructions."""
        return get_prompt_template(
            self.review.review_mode,
            self.review.custom_prompt_template or ""
        )
    
    def should_review_file(self, file_path: str) -> bool:
        """Determine if a file should be reviewed based on configuration."""
        # Check include patterns first
        if self.review.include_patterns:
            if not any(matches_pattern(file_path, pattern) 
                      for pattern in self.review.include_patterns):
                return False
        
        # Check exclude patterns
        if any(matches_pattern(file_path, pattern) 
               for pattern in self.review.exclude_patterns):
            return False
        
        # Check test files
        if not self.review.review_test_files and is_test_file(file_path):
            return False
        
        # Check documentation files
        if not self.review.review_docs and is_doc_file(file_path):
            return False
        
        return True
    
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
