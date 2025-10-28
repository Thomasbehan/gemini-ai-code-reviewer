#!/usr/bin/env python3
"""
Test script to verify repository context building functionality.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemini_reviewer.context_builder import ContextBuilder


class MockGitHubClient:
    """Mock GitHub client for testing."""
    def get_file_content(self, owner, repo, path, ref):
        return None
    
    def get_file_review_comments(self, pr_details, path, limit=30):
        return None


class MockDiffParser:
    """Mock diff parser for testing."""
    def get_file_language(self, path):
        if path.endswith('.py'):
            return 'python'
        elif path.endswith('.js'):
            return 'javascript'
        return 'unknown'


def test_repository_scanning():
    """Test repository structure scanning."""
    print("Testing repository structure scanning...")
    
    github_client = MockGitHubClient()
    diff_parser = MockDiffParser()
    context_builder = ContextBuilder(github_client, diff_parser)
    
    # Test repository structure scanning
    repo_structure = context_builder._scan_repository_structure(max_depth=2)
    
    if repo_structure:
        print("✓ Repository structure scanning works!")
        print(f"  Generated {len(repo_structure)} characters of structure")
        print("\nSample output (first 500 chars):")
        print(repo_structure[:500])
    else:
        print("✗ Repository structure scanning returned empty")
        return False
    
    return True


def test_code_signature_extraction():
    """Test code signature extraction."""
    print("\n" + "="*70)
    print("Testing code signature extraction...")
    
    github_client = MockGitHubClient()
    diff_parser = MockDiffParser()
    context_builder = ContextBuilder(github_client, diff_parser)
    
    # Test code signature extraction
    code_signatures = context_builder._extract_code_signatures(max_files=10)
    
    if code_signatures:
        print("✓ Code signature extraction works!")
        print(f"  Generated {len(code_signatures)} characters of signatures")
        print("\nSample output (first 800 chars):")
        print(code_signatures[:800])
    else:
        print("✗ Code signature extraction returned empty")
        return False
    
    return True


def main():
    """Run all tests."""
    print("="*70)
    print("Repository Context Builder Tests")
    print("="*70)
    
    results = []
    
    # Test repository scanning
    results.append(test_repository_scanning())
    
    # Test code signature extraction
    results.append(test_code_signature_extraction())
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! Repository context functionality is working.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
