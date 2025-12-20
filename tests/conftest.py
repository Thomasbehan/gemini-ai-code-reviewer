"""
Pytest configuration and fixtures for gemini_reviewer tests.
"""

import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_diff_content():
    """Provide sample diff content for tests."""
    return """diff --git a/main.py b/main.py
index 1234567..abcdefg 100644
--- a/main.py
+++ b/main.py
@@ -1,3 +1,4 @@
 import os
+import sys

 def main():
"""


@pytest.fixture
def sample_event_data():
    """Provide sample GitHub event data."""
    return {
        "number": 123,
        "repository": {"full_name": "owner/repo"},
        "pull_request": {
            "title": "Test PR",
            "body": "Description",
            "head": {"sha": "abc123"},
            "base": {"sha": "def456"},
        },
    }


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch):
    """Reset environment variables for each test."""
    # Clear potentially interfering environment variables
    for key in list(os.environ.keys()):
        if key.startswith(("GITHUB_", "GEMINI_", "INPUT_", "REVIEW_")):
            monkeypatch.delenv(key, raising=False)
