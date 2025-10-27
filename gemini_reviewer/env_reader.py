"""
Environment variable reading utilities for the Gemini AI Code Reviewer.

This module provides reusable functions for reading environment variables
with type conversion and fallback values, following DRY principles.
"""

import os
from typing import TypeVar, Callable, Optional, List
from enum import Enum


T = TypeVar('T')


def get_env_str(key: str, default: str = "", *fallback_keys: str) -> str:
    """Get string value from environment with fallback keys.
    
    Args:
        key: Primary environment variable key
        default: Default value if not found
        *fallback_keys: Additional keys to try if primary is not found
        
    Returns:
        The environment variable value or default
    """
    value = os.environ.get(key, "")
    if value:
        return value
    
    for fallback_key in fallback_keys:
        value = os.environ.get(fallback_key, "")
        if value:
            return value
    
    return default


def get_env_int(key: str, default: int, *fallback_keys: str) -> int:
    """Get integer value from environment with fallback keys.
    
    Args:
        key: Primary environment variable key
        default: Default value if not found or conversion fails
        *fallback_keys: Additional keys to try if primary is not found
        
    Returns:
        The environment variable value as integer or default
    """
    value = get_env_str(key, "", *fallback_keys)
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return default


def get_env_float(key: str, default: float, *fallback_keys: str) -> float:
    """Get float value from environment with fallback keys.
    
    Args:
        key: Primary environment variable key
        default: Default value if not found or conversion fails
        *fallback_keys: Additional keys to try if primary is not found
        
    Returns:
        The environment variable value as float or default
    """
    value = get_env_str(key, "", *fallback_keys)
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    return default


def get_env_bool(key: str, default: bool, *fallback_keys: str) -> bool:
    """Get boolean value from environment with fallback keys.
    
    Recognizes 'true', 'yes', '1' as True (case-insensitive).
    
    Args:
        key: Primary environment variable key
        default: Default value if not found
        *fallback_keys: Additional keys to try if primary is not found
        
    Returns:
        The environment variable value as boolean or default
    """
    value = get_env_str(key, "", *fallback_keys)
    if value:
        return value.lower() in ('true', 'yes', '1')
    return default


def get_env_list(key: str, separator: str = ",", *fallback_keys: str) -> List[str]:
    """Get list of strings from environment with fallback keys.
    
    Args:
        key: Primary environment variable key
        separator: String separator for splitting
        *fallback_keys: Additional keys to try if primary is not found
        
    Returns:
        List of strings split from the environment variable
    """
    value = get_env_str(key, "", *fallback_keys)
    if value:
        return [item.strip() for item in value.split(separator) if item.strip()]
    return []


def get_env_enum(key: str, enum_class: type[Enum], default: Enum, *fallback_keys: str) -> Enum:
    """Get enum value from environment with fallback keys.
    
    Args:
        key: Primary environment variable key
        enum_class: The enum class to convert to
        default: Default enum value if not found or conversion fails
        *fallback_keys: Additional keys to try if primary is not found
        
    Returns:
        The environment variable value as enum or default
    """
    value = get_env_str(key, "", *fallback_keys)
    if value:
        try:
            return enum_class(value.strip().lower())
        except (ValueError, AttributeError):
            pass
    return default
