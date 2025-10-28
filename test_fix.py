#!/usr/bin/env python3
"""Test script to verify the get_file_language fix."""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Test that the import works
    from gemini_reviewer.utils import get_file_language
    print("✓ Successfully imported get_file_language from utils")
    
    # Test that it works with various file paths
    test_files = [
        "test.py",
        "test.go",
        "test.js",
        "test.java",
        "test.sh",
        "unknown.xyz"
    ]
    
    for file_path in test_files:
        language = get_file_language(file_path)
        print(f"✓ get_file_language('{file_path}') = '{language}'")
    
    # Test that DiffParser doesn't have the method
    from gemini_reviewer.diff_parser import DiffParser
    parser = DiffParser()
    
    if hasattr(parser, 'get_file_language'):
        print("✗ ERROR: DiffParser still has get_file_language method")
        sys.exit(1)
    else:
        print("✓ DiffParser correctly does not have get_file_language method")
    
    print("\n✅ All tests passed! The fix is working correctly.")
    sys.exit(0)
    
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
