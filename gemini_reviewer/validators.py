"""
Validation utilities for the Gemini AI Code Reviewer.

This module provides reusable validation functions to eliminate duplication
and follow DRY principles.
"""

from typing import Any


def validate_required_string(value: str, field_name: str) -> None:
    """Validate that a required string field is not empty.
    
    Args:
        value: The string value to validate
        field_name: Name of the field for error messages
        
    Raises:
        ValueError: If the value is empty or not a string
    """
    if not value:
        raise ValueError(f"{field_name} is required")


def validate_positive_int(value: int, field_name: str) -> None:
    """Validate that an integer value is positive.
    
    Args:
        value: The integer value to validate
        field_name: Name of the field for error messages
        
    Raises:
        ValueError: If the value is not positive
    """
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


def validate_range(value: float, min_val: float, max_val: float, field_name: str) -> None:
    """Validate that a value is within a specified range.
    
    Args:
        value: The value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        field_name: Name of the field for error messages
        
    Raises:
        ValueError: If the value is outside the range
    """
    if not min_val <= value <= max_val:
        raise ValueError(f"{field_name} must be between {min_val} and {max_val}")


def validate_github_token_format(token: str) -> bool:
    """Validate GitHub token format.
    
    Args:
        token: The GitHub token to validate
        
    Returns:
        True if the token format is valid, False otherwise
    """
    if not token or not isinstance(token, str):
        return False
    # GitHub tokens are typically 40 characters (classic) or start with specific prefixes
    return len(token) >= 4 and (
        len(token) == 40 or 
        token.startswith(('ghp_', 'ghs_', 'gho_', 'ghu_'))
    )


def validate_gemini_api_key_format(api_key: str) -> bool:
    """Validate Gemini API key format.
    
    Args:
        api_key: The Gemini API key to validate
        
    Returns:
        True if the API key format is valid, False otherwise
    """
    if not api_key or not isinstance(api_key, str):
        return False
    # Gemini API keys are typically alphanumeric with some special characters
    return len(api_key) > 10


def ensure_positive_or_default(value: int, default: int) -> int:
    """Ensure a value is positive, otherwise return default.
    
    Args:
        value: The value to check
        default: The default value to return if value is not positive
        
    Returns:
        The value if positive, otherwise the default
    """
    return value if value > 0 else default
