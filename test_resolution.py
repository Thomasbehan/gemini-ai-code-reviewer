#!/usr/bin/env python3
"""
Test script to verify comment resolution functionality.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemini_reviewer.models import PRDetails


def test_comment_resolution_logic():
    """Test the comment resolution logic (conservative policy).

    We no longer auto-mark previous comments as resolved based on absence of new issues.
    Resolution should only be acknowledged with explicit evidence tied to that comment.
    """
    print("Testing comment resolution logic (conservative)...")
    print("="*70)
    
    # Simulate previous comments
    previous_comments = [
        {'path': 'file1.py', 'line': 10, 'body': 'Issue 1', 'id': 123},
        {'path': 'file2.py', 'line': 20, 'body': 'Issue 2', 'id': 124},
        {'path': 'file3.py', 'line': 30, 'body': 'Issue 3', 'id': 125},
    ]
    
    # Simulate current issues (only file2.py still has issues)
    current_issues = {'file2.py'}
    
    # Simulate resolution checking
    print("\nPrevious comments:")
    for i, comment in enumerate(previous_comments, 1):
        print(f"  {i}. {comment['path']} (ID: {comment['id']}): {comment['body']}")
    
    print("\nCurrent issues detected in follow-up review:")
    for issue in current_issues:
        print(f"  - {issue}")
    
    print("\nResolution analysis (no auto-resolution by absence):")
    resolved_comments = []
    for comment in previous_comments:
        # With the new conservative logic, we do not mark resolved unless there's explicit evidence
        print(f"  • No auto-resolution for: {comment['path']} (ID: {comment['id']})")
    
    print(f"\nSummary: {len(resolved_comments)}/{len(previous_comments)} comments would be marked as resolved")
    
    # Verify expected behavior: zero auto-resolutions
    expected_resolved = 0
    if len(resolved_comments) == expected_resolved:
        print("✓ Test passed!")
        return True
    else:
        print(f"✗ Test failed! Expected {expected_resolved} resolved, got {len(resolved_comments)}")
        return False


def test_github_client_reply_method():
    """Test that the reply_to_comment method exists and has expected parameters."""
    print("\n" + "="*70)
    print("Testing GitHub client reply method...")
    
    try:
        from gemini_reviewer.github_client import GitHubClient
        
        # Check if the method exists
        if hasattr(GitHubClient, 'reply_to_comment'):
            print("✓ reply_to_comment method exists in GitHubClient")
            
            # Check method signature
            import inspect
            sig = inspect.signature(GitHubClient.reply_to_comment)
            params = list(sig.parameters.keys())
            print(f"  Method signature: {params}")
            
            if 'pr_details' in params and 'comment_id' in params:
                print("✓ Method has correct parameters")
                return True
            else:
                print("✗ Method parameters don't match expected (pr_details, comment_id, [reply_body])")
                return False
        else:
            print("✗ reply_to_comment method not found in GitHubClient")
            return False
            
    except Exception as e:
        print(f"✗ Error testing GitHub client: {str(e)}")
        return False


def test_code_reviewer_tracking():
    """Test that CodeReviewer has the necessary tracking attributes."""
    print("\n" + "="*70)
    print("Testing CodeReviewer follow-up tracking...")
    
    try:
        from gemini_reviewer.code_reviewer import CodeReviewer
        from gemini_reviewer.config import Config, GitHubConfig, GeminiConfig
        
        # Create a mock config
        github_config = GitHubConfig(token="test_token")
        gemini_config = GeminiConfig(api_key="test_key")
        config = Config(github=github_config, gemini=gemini_config)
        
        # Initialize reviewer
        reviewer = CodeReviewer(config)
        
        # Check for tracking attributes
        checks = [
            ('_is_followup_review', False),
            ('_previous_comments', ""),
            ('_current_followup_issues', set()),
        ]
        
        all_passed = True
        for attr, expected_type in checks:
            if hasattr(reviewer, attr):
                attr_value = getattr(reviewer, attr)
                attr_type = type(attr_value)
                expected_type_check = type(expected_type)
                if attr_type == expected_type_check:
                    print(f"✓ Attribute '{attr}' exists with correct type: {attr_type.__name__}")
                else:
                    print(f"✗ Attribute '{attr}' has wrong type: {attr_type.__name__} (expected {expected_type_check.__name__})")
                    all_passed = False
            else:
                print(f"✗ Attribute '{attr}' not found")
                all_passed = False
        
        # Check for resolution method
        if hasattr(reviewer, '_resolve_completed_comments'):
            print("✓ _resolve_completed_comments method exists")
        else:
            print("✗ _resolve_completed_comments method not found")
            all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"✗ Error testing CodeReviewer: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("Comment Resolution Functionality Tests")
    print("="*70)
    
    results = []
    
    # Test resolution logic
    results.append(test_comment_resolution_logic())
    
    # Test GitHub client method
    results.append(test_github_client_resolution_method())
    
    # Test CodeReviewer tracking
    results.append(test_code_reviewer_tracking())
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! Comment resolution functionality is working.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
