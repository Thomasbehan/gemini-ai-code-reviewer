"""
Shared utility functions for the Gemini AI Code Reviewer.

This module provides common utility functions used across multiple modules
to follow DRY principles.
"""

import fnmatch
import re
from typing import Set


def matches_pattern(file_path: str, pattern: str) -> bool:
    """Check if file path matches a pattern.
    
    Args:
        file_path: The file path to check
        pattern: The pattern to match against (supports wildcards)
        
    Returns:
        True if the file path matches the pattern, False otherwise
    """
    return fnmatch.fnmatch(file_path, pattern)


def is_test_file(file_path: str) -> bool:
    """Check if file is a test file (cross-platform).
    
    Args:
        file_path: The file path to check
        
    Returns:
        True if the file is a test file, False otherwise
    """
    lowered = file_path.lower()
    test_patterns = ['test_', '_test.', 'spec_', '_spec.', '/test/', '/tests/', '\\test\\', '\\tests\\']
    return any(pattern in lowered for pattern in test_patterns)


def is_doc_file(file_path: str) -> bool:
    """Check if file is a documentation file.
    
    Args:
        file_path: The file path to check
        
    Returns:
        True if the file is a documentation file, False otherwise
    """
    doc_extensions = {'.md', '.rst', '.txt', '.doc', '.docx'}
    return any(file_path.lower().endswith(ext) for ext in doc_extensions)


def get_file_language(file_path: str) -> str:
    """Detect programming language from file extension.
    
    Args:
        file_path: The file path to analyze
        
    Returns:
        The detected language name or 'unknown'
    """
    extension = file_path.split('.')[-1].lower() if '.' in file_path else ''
    
    language_map = {
        'py': 'python',
        'js': 'javascript',
        'ts': 'typescript',
        'jsx': 'javascript',
        'tsx': 'typescript',
        'java': 'java',
        'kt': 'kotlin',
        'go': 'go',
        'rs': 'rust',
        'cpp': 'c++',
        'cc': 'c++',
        'cxx': 'c++',
        'c': 'c',
        'h': 'c',
        'hpp': 'c++',
        'cs': 'c#',
        'rb': 'ruby',
        'php': 'php',
        'swift': 'swift',
        'scala': 'scala',
        'r': 'r',
        'sql': 'sql',
        'sh': 'shell',
        'bash': 'shell',
        'yaml': 'yaml',
        'yml': 'yaml',
        'json': 'json',
        'xml': 'xml',
        'html': 'html',
        'css': 'css',
        'scss': 'scss',
        'sass': 'sass',
        'vue': 'vue',
        'dart': 'dart',
        'lua': 'lua',
        'pl': 'perl',
        'ex': 'elixir',
        'exs': 'elixir',
    }
    
    return language_map.get(extension, 'unknown')


def sanitize_text(text: str) -> str:
    """Sanitize text by removing control characters and normalizing whitespace.
    
    Args:
        text: The text to sanitize
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Remove null bytes and other control characters except newlines and tabs
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    # Normalize excessive whitespace but preserve intentional formatting
    text = re.sub(r'\n{4,}', '\n\n\n', text)  # Max 3 consecutive newlines
    text = re.sub(r'[ \t]{10,}', ' ' * 8, text)  # Max 8 consecutive spaces
    
    return text.strip()


def sanitize_code_content(content: str) -> str:
    """Sanitize code content more conservatively.
    
    Args:
        content: The code content to sanitize
        
    Returns:
        Sanitized code content
    """
    if not content:
        return ""
    
    # Only remove null bytes for code - preserve formatting
    content = content.replace('\x00', '')
    
    return content


def is_binary_file(file_path: str) -> bool:
    """Check if the file is likely binary based on extension.
    
    Args:
        file_path: The file path to check
        
    Returns:
        True if the file is likely binary, False otherwise
    """
    binary_extensions: Set[str] = {
        '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', 
        '.tar', '.gz', '.exe', '.dll', '.so', '.dylib',
        '.bin', '.dat', '.pyc', '.pyo', '.class'
    }
    return any(file_path.lower().endswith(ext) for ext in binary_extensions)
