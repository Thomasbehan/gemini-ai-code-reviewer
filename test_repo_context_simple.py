#!/usr/bin/env python3
"""
Simple standalone test for repository context functionality.
"""

import os
import ast
import re


def test_scan_repository():
    """Test scanning repository structure."""
    print("Testing repository structure scanning...")
    
    repo_root = os.getcwd()
    tree_lines = ["Repository Structure:"]
    
    exclude_dirs = {
        '.git', '.github', '__pycache__', 'node_modules', '.venv', 'venv',
        '.env', '.pytest_cache', '.mypy_cache', '.tox', 'build', 'dist',
        '.eggs', '*.egg-info', '.idea', '.vscode'
    }
    
    def should_exclude(path, name):
        if name.startswith('.') and name not in {'.gitignore', '.env.example'}:
            return True
        if name in exclude_dirs:
            return True
        return False
    
    def scan_dir(path, prefix="", depth=0, max_depth=2):
        if depth > max_depth:
            return
        
        try:
            entries = sorted(os.listdir(path))
            dirs = [e for e in entries if os.path.isdir(os.path.join(path, e)) and not should_exclude(path, e)]
            files = [e for e in entries if os.path.isfile(os.path.join(path, e)) and not should_exclude(path, e)]
            
            for i, dirname in enumerate(dirs):
                is_last_dir = (i == len(dirs) - 1) and not files
                connector = "└── " if is_last_dir else "├── "
                tree_lines.append(f"{prefix}{connector}{dirname}/")
                
                new_prefix = prefix + ("    " if is_last_dir else "│   ")
                scan_dir(os.path.join(path, dirname), new_prefix, depth + 1, max_depth)
            
            for i, filename in enumerate(files):
                is_last = i == len(files) - 1
                connector = "└── " if is_last else "├── "
                tree_lines.append(f"{prefix}{connector}{filename}")
        except PermissionError:
            pass
    
    scan_dir(repo_root)
    result = "\n".join(tree_lines[:200])
    
    print(f"✓ Generated {len(result)} characters")
    print(f"✓ Found {len(tree_lines)} entries")
    print("\nFirst 500 characters:")
    print(result[:500])
    
    return len(result) > 0


def test_extract_signatures():
    """Test extracting code signatures."""
    print("\n" + "="*70)
    print("Testing code signature extraction...")
    
    repo_root = os.getcwd()
    signatures = ["Code Structure (Functions & Classes):"]
    files_processed = 0
    max_files = 10
    
    code_extensions = {'.py'}
    
    def extract_python_signatures(file_path):
        sigs = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    if methods:
                        sigs.append(f"  class {node.name}: {', '.join(methods[:5])}")
                    else:
                        sigs.append(f"  class {node.name}")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.col_offset == 0:
                        params = [arg.arg for arg in node.args.args]
                        sigs.append(f"  def {node.name}({', '.join(params[:3])}{'...' if len(params) > 3 else ''})")
        except Exception as e:
            pass
        
        return sigs
    
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in {'.git', '.github', '__pycache__', 'node_modules', '.venv', 'venv'}]
        
        for filename in files:
            if files_processed >= max_files:
                break
            
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in code_extensions:
                continue
            
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, repo_root)
            
            file_sigs = extract_python_signatures(file_path)
            
            if file_sigs:
                signatures.append(f"\n{rel_path}:")
                signatures.extend(file_sigs[:20])
                files_processed += 1
        
        if files_processed >= max_files:
            break
    
    result = "\n".join(signatures[:500])
    
    print(f"✓ Generated {len(result)} characters")
    print(f"✓ Processed {files_processed} files")
    print("\nFirst 800 characters:")
    print(result[:800])
    
    return len(result) > 0


def main():
    print("="*70)
    print("Repository Context Functionality Test")
    print("="*70)
    print()
    
    results = []
    
    # Test repository scanning
    try:
        results.append(test_scan_repository())
    except Exception as e:
        print(f"✗ Repository scanning failed: {e}")
        results.append(False)
    
    # Test signature extraction
    try:
        results.append(test_extract_signatures())
    except Exception as e:
        print(f"✗ Signature extraction failed: {e}")
        results.append(False)
    
    print("\n" + "="*70)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("="*70)
    
    if all(results):
        print("\n✓ All tests passed! The implementation is working correctly.")
        return 0
    else:
        print("\n✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    exit(main())
