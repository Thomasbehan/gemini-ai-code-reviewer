"""
Gemini AI Code Reviewer Package

A comprehensive code review system powered by Google's Gemini AI.
This package provides modular components for automated code review in GitHub pull requests.
"""

__version__ = "2.0.0"
__author__ = "truongnh1992"
__description__ = "AI-powered code review system using Google's Gemini AI"

# To keep package import lightweight (so modules like `prompts` can be imported
# without optional heavy dependencies like `requests` or `google.generativeai`),
# we avoid importing submodules at package import time. Instead, we expose a
# lazy attribute loader that imports on first access.

__all__ = [
    # Main classes
    'Config', 'CodeReviewer', 'CodeReviewerError',
    # Data models
    'PRDetails', 'ReviewResult', 'ReviewComment', 'DiffFile', 'FileInfo',
    'HunkInfo', 'AnalysisContext', 'ProcessingStats', 'ReviewPriority', 'ReviewFocus',
    # Client classes
    'GitHubClient', 'GitHubClientError', 'GeminiClient', 'GeminiClientError',
    'DiffParser', 'DiffParsingError', 'ContextBuilder', 'CommentProcessor',
]

# Lazy import map: attribute -> (module_path, attr_name)
_lazy_exports = {
    # Main classes
    'Config': ('gemini_reviewer.config', 'Config'),
    'CodeReviewer': ('gemini_reviewer.code_reviewer', 'CodeReviewer'),
    'CodeReviewerError': ('gemini_reviewer.code_reviewer', 'CodeReviewerError'),
    # Models
    'PRDetails': ('gemini_reviewer.models', 'PRDetails'),
    'ReviewResult': ('gemini_reviewer.models', 'ReviewResult'),
    'ReviewComment': ('gemini_reviewer.models', 'ReviewComment'),
    'DiffFile': ('gemini_reviewer.models', 'DiffFile'),
    'FileInfo': ('gemini_reviewer.models', 'FileInfo'),
    'HunkInfo': ('gemini_reviewer.models', 'HunkInfo'),
    'AnalysisContext': ('gemini_reviewer.models', 'AnalysisContext'),
    'ProcessingStats': ('gemini_reviewer.models', 'ProcessingStats'),
    'ReviewPriority': ('gemini_reviewer.models', 'ReviewPriority'),
    'ReviewFocus': ('gemini_reviewer.models', 'ReviewFocus'),
    # Clients and utilities
    'GitHubClient': ('gemini_reviewer.github_client', 'GitHubClient'),
    'GitHubClientError': ('gemini_reviewer.github_client', 'GitHubClientError'),
    'GeminiClient': ('gemini_reviewer.gemini_client', 'GeminiClient'),
    'GeminiClientError': ('gemini_reviewer.gemini_client', 'GeminiClientError'),
    'DiffParser': ('gemini_reviewer.diff_parser', 'DiffParser'),
    'DiffParsingError': ('gemini_reviewer.diff_parser', 'DiffParsingError'),
    'ContextBuilder': ('gemini_reviewer.context_builder', 'ContextBuilder'),
    'CommentProcessor': ('gemini_reviewer.comment_processor', 'CommentProcessor'),
}


def __getattr__(name):
    target = _lazy_exports.get(name)
    if not target:
        raise AttributeError(f"module 'gemini_reviewer' has no attribute '{name}'")
    module_path, attr_name = target
    try:
        module = __import__(module_path, fromlist=[attr_name])
        value = getattr(module, attr_name)
        globals()[name] = value  # cache for future access
        return value
    except Exception as e:
        # Provide a clear error if a heavy dependency is missing at import time
        raise ImportError(f"Failed to import '{name}' from '{module_path}': {e}")
