"""
Context builder for the Gemini AI Code Reviewer.

This module is responsible for building analysis context by detecting
related files and gathering project context.
"""

import logging
import os
import re
import ast
from typing import List, Optional, Dict, Set
from pathlib import Path

from .models import DiffFile, PRDetails

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds analysis context for code review."""
    
    def __init__(self, github_client, diff_parser):
        """Initialize context builder with required dependencies.
        
        Args:
            github_client: GitHub client for fetching file contents
            diff_parser: Diff parser for language detection
        """
        self.github_client = github_client
        self.diff_parser = diff_parser
    
    async def detect_related_files(self, diff_file: DiffFile, pr_details: PRDetails) -> List[str]:
        """Detect related files based on imports and dependencies.
        
        Args:
            diff_file: The diff file to analyze
            pr_details: Pull request details
            
        Returns:
            List of related file paths (limited to 5 most relevant)
        """
        file_path = diff_file.file_info.path
        related_files = []
        
        try:
            # Extract imports/requires/includes from the file content
            import_patterns = {
                'python': [r'from\s+(\S+)\s+import', r'import\s+(\S+)'],
                'javascript': [r'import\s+.*\s+from\s+["\']([^"\']+)["\']', r'require\(["\']([^"\']+)["\']\)'],
                'typescript': [r'import\s+.*\s+from\s+["\']([^"\']+)["\']', r'require\(["\']([^"\']+)["\']\)'],
                'java': [r'import\s+([^;]+);'],
                'go': [r'import\s+["\']([^"\']+)["\']', r'import\s+\(\s*["\']([^"\']+)["\']'],
                'ruby': [r'require\s+["\']([^"\']+)["\']'],
            }
            
            language = self.diff_parser.get_file_language(file_path)
            
            if language and language.lower() in import_patterns:
                patterns = import_patterns[language.lower()]
                
                # Analyze the hunk content for imports
                for hunk in diff_file.hunks:
                    for line in hunk.lines:
                        # Only look at added or context lines, not deleted lines
                        if line.startswith('-'):
                            continue
                        
                        clean_line = line[1:] if line else line  # Remove +/- prefix
                        
                        for pattern in patterns:
                            matches = re.findall(pattern, clean_line)
                            for match in matches:
                                # Convert import path to file path
                                related_file = self._import_to_file_path(match, file_path, language)
                                if related_file and related_file not in related_files:
                                    related_files.append(related_file)
            
            # Limit to most relevant files to avoid token bloat
            related_files = related_files[:5]
            
            if related_files:
                logger.info(f"Detected {len(related_files)} related files for {file_path}")
            
        except Exception as e:
            logger.debug(f"Error detecting related files for {file_path}: {str(e)}")
        
        return related_files
    
    def _import_to_file_path(self, import_path: str, current_file: str, language: str) -> Optional[str]:
        """Convert an import path to a file path.
        
        Args:
            import_path: The import statement path
            current_file: Current file path for relative imports
            language: Programming language
            
        Returns:
            Converted file path or None if conversion fails
        """
        try:
            # Handle relative imports
            if import_path.startswith('.'):
                current_dir = os.path.dirname(current_file)
                # Count leading dots
                dots = len(import_path) - len(import_path.lstrip('.'))
                # Go up directories
                for _ in range(dots - 1):
                    current_dir = os.path.dirname(current_dir)
                import_path = import_path.lstrip('.')
                base_path = os.path.join(current_dir, import_path.replace('.', '/'))
            else:
                # For absolute imports, try common patterns
                base_path = import_path.replace('.', '/')
            
            # Add appropriate extensions based on language
            extensions = {
                'python': ['.py', '/__init__.py'],
                'javascript': ['.js', '/index.js', '.jsx'],
                'typescript': ['.ts', '/index.ts', '.tsx'],
                'java': ['.java'],
                'go': ['.go'],
                'ruby': ['.rb']
            }
            
            possible_paths = []
            if language.lower() in extensions:
                for ext in extensions[language.lower()]:
                    possible_paths.append(base_path + ext)
            
            # Return the first plausible path
            return possible_paths[0] if possible_paths else None
            
        except Exception:
            return None
    
    def _scan_repository_structure(self, max_depth: int = 3) -> str:
        """Scan the repository and create a file tree structure.
        
        Args:
            max_depth: Maximum directory depth to scan
            
        Returns:
            Formatted string showing the repository structure
        """
        try:
            repo_root = os.getcwd()
            tree_lines = ["Repository Structure:"]
            
            # Directories and files to exclude
            exclude_dirs = {
                '.git', '.github', '__pycache__', 'node_modules', '.venv', 'venv',
                '.env', '.pytest_cache', '.mypy_cache', '.tox', 'build', 'dist',
                '.eggs', '*.egg-info', '.idea', '.vscode'
            }
            exclude_patterns = {'.pyc', '.pyo', '.pyd', '.so', '.dylib', '.dll', '.egg'}
            
            def should_exclude(path: str, name: str) -> bool:
                """Check if a path should be excluded."""
                if name.startswith('.') and name not in {'.gitignore', '.env.example'}:
                    return True
                if name in exclude_dirs:
                    return True
                if any(name.endswith(pattern) for pattern in exclude_patterns):
                    return True
                return False
            
            def scan_dir(path: str, prefix: str = "", depth: int = 0) -> None:
                """Recursively scan directory."""
                if depth > max_depth:
                    return
                
                try:
                    entries = sorted(os.listdir(path))
                    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e)) and not should_exclude(path, e)]
                    files = [e for e in entries if os.path.isfile(os.path.join(path, e)) and not should_exclude(path, e)]
                    
                    # Show directories first
                    for i, dirname in enumerate(dirs):
                        is_last_dir = (i == len(dirs) - 1) and not files
                        connector = "└── " if is_last_dir else "├── "
                        tree_lines.append(f"{prefix}{connector}{dirname}/")
                        
                        new_prefix = prefix + ("    " if is_last_dir else "│   ")
                        scan_dir(os.path.join(path, dirname), new_prefix, depth + 1)
                    
                    # Show files
                    for i, filename in enumerate(files):
                        is_last = i == len(files) - 1
                        connector = "└── " if is_last else "├── "
                        tree_lines.append(f"{prefix}{connector}{filename}")
                
                except PermissionError:
                    pass
            
            scan_dir(repo_root)
            return "\n".join(tree_lines[:200])  # Limit to 200 lines
            
        except Exception as e:
            logger.debug(f"Error scanning repository structure: {str(e)}")
            return ""
    
    def _extract_code_signatures(self, max_files: int = 50) -> str:
        """Extract function and class signatures from code files in the repository.
        
        Args:
            max_files: Maximum number of files to process
            
        Returns:
            Formatted string with code signatures
        """
        try:
            repo_root = os.getcwd()
            signatures = ["Code Structure (Functions & Classes):"]
            files_processed = 0
            
            # File extensions to process
            code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.java', '.rb'}
            
            def extract_python_signatures(file_path: str, rel_path: str) -> List[str]:
                """Extract signatures from Python files."""
                sigs = []
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            # Get class with methods
                            methods = [m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                            if methods:
                                sigs.append(f"  class {node.name}: {', '.join(methods[:5])}")
                            else:
                                sigs.append(f"  class {node.name}")
                        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            # Only top-level functions (not methods)
                            if node.col_offset == 0:
                                params = [arg.arg for arg in node.args.args]
                                sigs.append(f"  def {node.name}({', '.join(params[:3])}{'...' if len(params) > 3 else ''})")
                    
                except Exception:
                    pass
                
                return sigs
            
            def extract_generic_signatures(file_path: str, rel_path: str) -> List[str]:
                """Extract signatures from other code files using regex."""
                sigs = []
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Match function/method declarations (simple patterns)
                    # JavaScript/TypeScript: function name(...) or const name = (...) =>
                    patterns = [
                        r'function\s+(\w+)\s*\(',
                        r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>',
                        r'export\s+function\s+(\w+)\s*\(',
                        r'class\s+(\w+)',
                        # Go: func Name(
                        r'func\s+(\w+)\s*\(',
                        # Java: public/private void/type Name(
                        r'(?:public|private|protected)\s+(?:static\s+)?(?:\w+)\s+(\w+)\s*\(',
                    ]
                    
                    found_names = set()
                    for pattern in patterns:
                        matches = re.findall(pattern, content)
                        for match in matches[:10]:  # Limit per pattern
                            if match not in found_names:
                                found_names.add(match)
                                sigs.append(f"  {match}")
                
                except Exception:
                    pass
                
                return sigs
            
            # Walk through repository
            for root, dirs, files in os.walk(repo_root):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if d not in {'.git', '.github', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist'}]
                
                for filename in files:
                    if files_processed >= max_files:
                        break
                    
                    file_ext = os.path.splitext(filename)[1]
                    if file_ext not in code_extensions:
                        continue
                    
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, repo_root)
                    
                    # Extract signatures based on file type
                    file_sigs = []
                    if file_ext == '.py':
                        file_sigs = extract_python_signatures(file_path, rel_path)
                    else:
                        file_sigs = extract_generic_signatures(file_path, rel_path)
                    
                    if file_sigs:
                        signatures.append(f"\n{rel_path}:")
                        signatures.extend(file_sigs[:20])  # Limit signatures per file
                        files_processed += 1
                
                if files_processed >= max_files:
                    break
            
            result = "\n".join(signatures[:500])  # Limit total lines
            return result if len(signatures) > 1 else ""
            
        except Exception as e:
            logger.debug(f"Error extracting code signatures: {str(e)}")
            return ""
    
    async def build_project_context(
        self, 
        diff_file: DiffFile, 
        related_files: List[str], 
        pr_details: PRDetails
    ) -> Optional[str]:
        """Build project context including full repository structure, code signatures, 
        previous comments, and related file contents.
        
        Args:
            diff_file: The diff file being analyzed
            related_files: List of related file paths
            pr_details: Pull request details
            
        Returns:
            Formatted project context string or None if no context available
        """
        context_parts = []
        max_context_size = 20000  # Increased limit to accommodate full repo context
        current_size = 0
        
        try:
            # STEP 1: Add full repository structure
            logger.info("Building full repository context for comprehensive code review")
            repo_structure = self._scan_repository_structure()
            if repo_structure:
                context_parts.append(f"### Full Repository Structure\n{repo_structure}\n")
                current_size += len(repo_structure)
                logger.info(f"Added repository structure ({len(repo_structure)} chars)")
            
            # STEP 2: Add code signatures from all files
            if current_size < max_context_size:
                code_signatures = self._extract_code_signatures()
                if code_signatures:
                    context_parts.append(f"### {code_signatures}\n")
                    current_size += len(code_signatures)
                    logger.info(f"Added code signatures ({len(code_signatures)} chars)")
            
            # STEP 3: Always include previous inline comments on this file (if any)
            if current_size < max_context_size:
                try:
                    prev = self.github_client.get_file_review_comments(
                        pr_details, 
                        diff_file.file_info.path, 
                        limit=30
                    )
                    if prev:
                        # Keep this small; it helps verify resolutions
                        snippet = prev
                        if len(snippet) > 2000:
                            snippet = snippet[:2000] + "\n... (truncated)"
                        context_parts.append(f"### Previous review history for {diff_file.file_info.path}\n{snippet}\n")
                        current_size += len(snippet)
                except Exception:
                    pass
            
            # Related files content
            for related_file in related_files or []:
                if current_size >= max_context_size:
                    break
                
                # Try to fetch the file content from the base branch
                content = self.github_client.get_file_content(
                    pr_details.owner,
                    pr_details.repo,
                    related_file,
                    pr_details.base_sha or 'main'
                )
                
                if content:
                    # Limit individual file size
                    if len(content) > 2000:
                        content = content[:2000] + "\n... (truncated)"
                    
                    context_parts.append(f"### Related file: {related_file}\n```\n{content}\n```\n")
                    current_size += len(content)
            
            if context_parts:
                logger.info(f"Built project context with {len(context_parts)} block(s) (~{current_size} chars)")
                return "\n".join(context_parts)
                
        except Exception as e:
            logger.debug(f"Error building project context: {str(e)}")
        
        return None
