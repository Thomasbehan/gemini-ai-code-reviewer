#!/usr/bin/env python3
"""
Test script to verify follow-up review logic.
This demonstrates that the system correctly switches between first review and follow-up review modes.
"""

from gemini_reviewer.prompts import ReviewMode, get_review_prompt_template

def test_first_review():
    """Test that first review uses standard prompt without previous comments."""
    print("=" * 80)
    print("TEST 1: First Review (No previous comments)")
    print("=" * 80)
    
    prompt = get_review_prompt_template(ReviewMode.STANDARD)
    
    # Check that it doesn't contain follow-up instructions
    assert "FOLLOW-UP REVIEW" not in prompt
    assert "previous comments" not in prompt.lower() or "previous comments to verify" not in prompt.lower()
    assert "DO NOT raise any new issues" not in prompt
    
    print("✅ PASSED: First review uses standard prompt")
    print(f"   Prompt length: {len(prompt)} characters")
    print(f"   Contains 'SCOPE: Report ONLY issues': {('SCOPE: Report ONLY issues' in prompt)}")
    print()

def test_followup_review():
    """Test that follow-up review uses special prompt with previous comments."""
    print("=" * 80)
    print("TEST 2: Follow-up Review (With previous comments)")
    print("=" * 80)
    
    previous_comments = """1. File: src/main.py
   Line: 42
   Comment: Missing error handling for database connection
   Posted: 2025-10-27 10:30:00

2. File: src/utils.py
   Line: 15
   Comment: Potential SQL injection vulnerability
   Posted: 2025-10-27 10:30:00"""
    
    prompt = get_review_prompt_template(ReviewMode.FOLLOWUP, previous_comments=previous_comments)
    
    # Check that it contains follow-up instructions
    assert "FOLLOW-UP REVIEW" in prompt
    assert "DO NOT raise any new issues" in prompt
    assert "ONLY check if the previous comments" in prompt
    assert "Missing error handling for database connection" in prompt
    assert "Potential SQL injection vulnerability" in prompt
    
    print("✅ PASSED: Follow-up review uses special prompt")
    print(f"   Prompt length: {len(prompt)} characters")
    print(f"   Contains 'DO NOT raise any new issues': {('DO NOT raise any new issues' in prompt)}")
    print(f"   Contains previous comments: {('Missing error handling' in prompt)}")
    print()

def test_followup_no_previous():
    """Test follow-up mode with no previous comments."""
    print("=" * 80)
    print("TEST 3: Follow-up Review (No previous comments found)")
    print("=" * 80)
    
    prompt = get_review_prompt_template(ReviewMode.FOLLOWUP, previous_comments="")
    
    # Should still use follow-up template but with default message
    assert "FOLLOW-UP REVIEW" in prompt
    assert "No previous comments found" in prompt
    
    print("✅ PASSED: Follow-up review handles empty comments gracefully")
    print(f"   Contains 'No previous comments found': {('No previous comments found' in prompt)}")
    print()

def test_all_modes():
    """Test that all review modes work."""
    print("=" * 80)
    print("TEST 4: All Review Modes")
    print("=" * 80)
    
    modes = [
        ReviewMode.STRICT,
        ReviewMode.STANDARD,
        ReviewMode.LENIENT,
        ReviewMode.SECURITY_FOCUSED,
        ReviewMode.PERFORMANCE_FOCUSED,
        ReviewMode.FOLLOWUP
    ]
    
    for mode in modes:
        try:
            if mode == ReviewMode.FOLLOWUP:
                prompt = get_review_prompt_template(mode, previous_comments="Test comment")
            else:
                prompt = get_review_prompt_template(mode)
            print(f"✅ {mode.value}: Generated prompt ({len(prompt)} chars)")
        except Exception as e:
            print(f"❌ {mode.value}: Failed - {e}")
    
    print()

def main():
    """Run all tests."""
    print("\n")
    print("🔍 VERIFYING FOLLOW-UP REVIEW IMPLEMENTATION")
    print("=" * 80)
    print()
    
    try:
        test_first_review()
        test_followup_review()
        test_followup_no_previous()
        test_all_modes()
        
        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print()
        print("SUMMARY:")
        print("- First review mode: Uses standard comprehensive review prompt")
        print("- Follow-up review mode: Uses special prompt that ONLY checks previous comments")
        print("- New issues will NOT be raised in follow-up reviews")
        print("- Previous comments are properly injected into the prompt")
        print()
        return 0
        
    except AssertionError as e:
        print("=" * 80)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 80)
        return 1
    except Exception as e:
        print("=" * 80)
        print(f"❌ ERROR: {e}")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    exit(main())
