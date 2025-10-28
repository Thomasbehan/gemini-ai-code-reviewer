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
    
    def _find_reverse_dependencies(self, changed_file: str) -> List[str]:
        """Find files that import or depend on the changed file.
        
        Args:
            changed_file: Path to the changed file
            
        Returns:
            List of file paths that depend on the changed file
        """
        reverse_deps = []
        try:
            repo_root = os.getcwd()
            changed_module = changed_file.replace('/', '.').replace('\\', '.')
            # Remove extension for module name
            for ext in ['.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.java', '.rb']:
                if changed_module.endswith(ext):
                    changed_module = changed_module[:-len(ext)]
                    break
            
            # Also check for the filename without path for relative imports
            changed_filename = os.path.basename(changed_file)
            changed_name = os.path.splitext(changed_filename)[0]
            
            # Patterns to search for imports
            import_patterns = [
                rf'from\s+{re.escape(changed_module)}\s+import',
                rf'import\s+{re.escape(changed_module)}',
                rf'from\s+.*{re.escape(changed_name)}\s+import',
                rf'import\s+.*{re.escape(changed_name)}',
                rf'require\(["\'].*{re.escape(changed_filename)}["\']\)',
                rf'import\s+.*from\s+["\'].*{re.escape(changed_filename)}["\']',
            ]
            
            # Search through code files
            code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.java', '.rb'}
            exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist'}
            
            for root, dirs, files in os.walk(repo_root):
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                
                for filename in files:
                    file_ext = os.path.splitext(filename)[1]
                    if file_ext not in code_extensions:
                        continue
                    
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, repo_root)
                    
                    # Skip the changed file itself
                    if rel_path == changed_file:
                        continue
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Check if any pattern matches
                        for pattern in import_patterns:
                            if re.search(pattern, content):
                                reverse_deps.append(rel_path)
                                break  # Found a match, no need to check other patterns
                        
                        if len(reverse_deps) >= 10:  # Limit to 10 reverse dependencies
                            break
                    except Exception:
                        continue
                
                if len(reverse_deps) >= 10:
                    break
            
            logger.info(f"Found {len(reverse_deps)} reverse dependencies for {changed_file}")
        except Exception as e:
            logger.debug(f"Error finding reverse dependencies: {str(e)}")
        
        return reverse_deps
    
    def _find_test_files(self, changed_file: str) -> List[str]:
        """Find test files related to the changed file.
        
        Args:
            changed_file: Path to the changed file
            
        Returns:
            List of related test file paths
        """
        test_files = []
        try:
            repo_root = os.getcwd()
            changed_name = os.path.splitext(os.path.basename(changed_file))[0]
            changed_dir = os.path.dirname(changed_file)
            
            # Common test patterns
            test_patterns = [
                f'test_{changed_name}',
                f'{changed_name}_test',
                f'test{changed_name}',
                changed_name,  # Test file might have same name in test directory
            ]
            
            test_dirs = {'test', 'tests', '__tests__', 'spec', 'specs'}
            test_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.java', '.rb'}
            
            # Search in test directories
            for root, dirs, files in os.walk(repo_root):
                # Prioritize test directories
                dir_name = os.path.basename(root)
                in_test_dir = any(test_dir in root for test_dir in test_dirs)
                
                for filename in files:
                    file_ext = os.path.splitext(filename)[1]
                    if file_ext not in test_extensions:
                        continue
                    
                    file_lower = filename.lower()
                    # Check if filename matches test patterns
                    if in_test_dir or file_lower.startswith('test') or file_lower.endswith('test' + file_ext):
                        # Check if it's related to our changed file
                        for pattern in test_patterns:
                            if pattern in file_lower:
                                file_path = os.path.join(root, filename)
                                rel_path = os.path.relpath(file_path, repo_root)
                                test_files.append(rel_path)
                                break
                    
                    if len(test_files) >= 5:  # Limit to 5 test files
                        break
                
                if len(test_files) >= 5:
                    break
            
            logger.info(f"Found {len(test_files)} test files for {changed_file}")
        except Exception as e:
            logger.debug(f"Error finding test files: {str(e)}")
        
        return test_files
    
    def _find_config_files(self) -> List[str]:
        """Find configuration files (linters, build scripts, Dockerfiles, etc.).
        
        Returns:
            List of configuration file paths
        """
        config_files = []
        try:
            repo_root = os.getcwd()
            
            # Important config file patterns
            config_patterns = [
                # Linters and formatters
                '.eslintrc', '.eslintrc.js', '.eslintrc.json', '.eslintrc.yml',
                '.prettierrc', '.prettierrc.js', '.prettierrc.json',
                '.pylintrc', 'pylint.cfg', '.flake8', 'tox.ini',
                '.mypy.ini', 'mypy.ini', 'pyproject.toml',
                # Build and dependency management
                'package.json', 'package-lock.json', 'yarn.lock',
                'requirements.txt', 'Pipfile', 'Pipfile.lock', 'poetry.lock',
                'go.mod', 'go.sum', 'Gemfile', 'Gemfile.lock',
                'build.gradle', 'pom.xml', 'Makefile',
                # Docker and containerization
                'Dockerfile', 'docker-compose.yml', 'docker-compose.yaml',
                '.dockerignore',
                # CI/CD
                '.travis.yml', 'circle.yml', '.gitlab-ci.yml',
                'azure-pipelines.yml', 'Jenkinsfile',
                # GitHub specific
                'action.yml', 'action.yaml',
                # Editor and IDE
                '.editorconfig',
                # Git
                '.gitignore', '.gitattributes',
            ]
            
            # Search for config files in root and common directories
            search_dirs = [repo_root, os.path.join(repo_root, '.github')]
            
            for search_dir in search_dirs:
                if not os.path.exists(search_dir):
                    continue
                
                try:
                    for item in os.listdir(search_dir):
                        if item in config_patterns or any(item.startswith(p) for p in ['Dockerfile', '.env']):
                            file_path = os.path.join(search_dir, item)
                            if os.path.isfile(file_path):
                                rel_path = os.path.relpath(file_path, repo_root)
                                config_files.append(rel_path)
                except Exception:
                    continue
            
            logger.info(f"Found {len(config_files)} config files")
        except Exception as e:
            logger.debug(f"Error finding config files: {str(e)}")
        
        return config_files
    
    def _build_repo_mental_model(self) -> str:
        """Build a compact repo-level mental model with key information.
        
        Returns:
            Formatted string with repo overview
        """
        model_parts = []
        
        try:
            repo_root = os.getcwd()
            
            # 1. Documentation summaries (README, CONTRIBUTING, etc.)
            doc_files = ['README.md', 'README.rst', 'CONTRIBUTING.md', 'ARCHITECTURE.md', 
                        'ADR.md', 'DESIGN.md', 'docs/README.md']
            
            for doc_file in doc_files:
                doc_path = os.path.join(repo_root, doc_file)
                if os.path.exists(doc_path):
                    try:
                        with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Extract first 1000 chars or first few sections
                        lines = content.split('\n')
                        summary_lines = []
                        char_count = 0
                        
                        for line in lines[:50]:  # First 50 lines max
                            summary_lines.append(line)
                            char_count += len(line)
                            if char_count > 1000:
                                break
                        
                        summary = '\n'.join(summary_lines)
                        if len(content) > char_count:
                            summary += "\n... (truncated)"
                        
                        model_parts.append(f"## {doc_file} Summary:\n{summary}\n")
                    except Exception:
                        pass
            
            # 2. Entrypoints identification
            entrypoint_files = [
                'main.py', 'app.py', '__main__.py', 'run.py', 'server.py',
                'index.js', 'index.ts', 'main.js', 'main.ts', 'app.js', 'app.ts',
                'main.go', 'cmd/main.go',
                'setup.py', 'setup.cfg', 'pyproject.toml',
                'action.yml', 'action.yaml',
                'Dockerfile', 'docker-compose.yml',
            ]
            
            found_entrypoints = []
            for entry in entrypoint_files:
                entry_path = os.path.join(repo_root, entry)
                if os.path.exists(entry_path) and os.path.isfile(entry_path):
                    found_entrypoints.append(entry)
            
            if found_entrypoints:
                model_parts.append(f"## Entrypoints:\n" + "\n".join(f"- {e}" for e in found_entrypoints) + "\n")
            
            # 3. Package/dependency management and scripts
            pkg_files = {
                'package.json': ['scripts', 'dependencies', 'devDependencies'],
                'pyproject.toml': ['tool.poetry.scripts', 'project.scripts'],
                'requirements.txt': None,
                'Pipfile': ['scripts'],
                'go.mod': ['module', 'require'],
                'Gemfile': None,
            }
            
            for pkg_file, keys_to_extract in pkg_files.items():
                pkg_path = os.path.join(repo_root, pkg_file)
                if os.path.exists(pkg_path):
                    try:
                        with open(pkg_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # For JSON files, try to extract scripts
                        if pkg_file.endswith('.json'):
                            import json
                            try:
                                data = json.loads(content)
                                scripts_info = []
                                for key in keys_to_extract or []:
                                    if key in data and isinstance(data[key], dict):
                                        scripts_info.append(f"  {key}:")
                                        for name, cmd in list(data[key].items())[:10]:
                                            scripts_info.append(f"    - {name}: {cmd}")
                                
                                if scripts_info:
                                    model_parts.append(f"## {pkg_file}:\n" + "\n".join(scripts_info) + "\n")
                            except json.JSONDecodeError:
                                pass
                        else:
                            # For other files, include a snippet
                            lines = content.split('\n')[:20]
                            model_parts.append(f"## {pkg_file} (first 20 lines):\n" + "\n".join(lines) + "\n")
                    except Exception:
                        pass
            
            # 4. CI/CD workflows
            ci_dirs = [
                os.path.join(repo_root, '.github', 'workflows'),
                os.path.join(repo_root, '.gitlab'),
                os.path.join(repo_root, '.circleci'),
            ]
            
            workflow_files = []
            for ci_dir in ci_dirs:
                if os.path.exists(ci_dir):
                    try:
                        for item in os.listdir(ci_dir):
                            if item.endswith(('.yml', '.yaml')):
                                workflow_files.append(os.path.relpath(os.path.join(ci_dir, item), repo_root))
                    except Exception:
                        pass
            
            if workflow_files:
                model_parts.append(f"## CI/CD Workflows:\n" + "\n".join(f"- {w}" for w in workflow_files) + "\n")
            
            # 5. Environment samples and lockfiles
            env_files = []
            for item in ['.env.example', '.env.sample', '.env.template', 
                        'package-lock.json', 'yarn.lock', 'poetry.lock', 
                        'Pipfile.lock', 'go.sum', 'Gemfile.lock']:
                item_path = os.path.join(repo_root, item)
                if os.path.exists(item_path):
                    env_files.append(item)
            
            if env_files:
                model_parts.append(f"## Environment & Lockfiles:\n" + "\n".join(f"- {e}" for e in env_files) + "\n")
            
            # 6. Service boundaries / Architecture (detect major directories)
            major_dirs = []
            try:
                for item in os.listdir(repo_root):
                    item_path = os.path.join(repo_root, item)
                    if os.path.isdir(item_path) and not item.startswith('.'):
                        if item not in {'node_modules', '__pycache__', 'venv', '.venv', 'build', 'dist'}:
                            major_dirs.append(item)
            except Exception:
                pass
            
            if major_dirs:
                model_parts.append(f"## Major Packages/Modules:\n" + "\n".join(f"- {d}/" for d in major_dirs[:15]) + "\n")
            
        except Exception as e:
            logger.debug(f"Error building repo mental model: {str(e)}")
        
        if model_parts:
            return "# Repository Mental Model\n\n" + "\n".join(model_parts)
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
        """Build project context including repo mental model, dependency-adjacent code,
        full repository structure, code signatures, previous comments, and related file contents.
        
        Args:
            diff_file: The diff file being analyzed
            related_files: List of related file paths
            pr_details: Pull request details
            
        Returns:
            Formatted project context string or None if no context available
        """
        context_parts = []
        max_context_size = 30000  # Increased limit to accommodate enhanced context layers
        current_size = 0
        
        try:
            logger.info("Building comprehensive repository context for code review")
            
            # LAYER 1: Repo-level mental model (HIGHEST PRIORITY)
            # Provides understanding of the entire codebase structure, entrypoints, docs, etc.
            repo_mental_model = self._build_repo_mental_model()
            if repo_mental_model:
                context_parts.append(repo_mental_model)
                current_size += len(repo_mental_model)
                logger.info(f"Added repo mental model ({len(repo_mental_model)} chars)")
            
            # LAYER 2: Dependency-adjacent code
            # Files that import/are imported by changed files, tests, and configs
            if current_size < max_context_size:
                dep_adjacent_parts = []
                
                # 2a. Reverse dependencies (files that import this file)
                reverse_deps = self._find_reverse_dependencies(diff_file.file_info.path)
                if reverse_deps:
                    dep_adjacent_parts.append(f"#### Reverse Dependencies (files that import {diff_file.file_info.path}):\n" + 
                                             "\n".join(f"- {rd}" for rd in reverse_deps))
                    logger.info(f"Found {len(reverse_deps)} reverse dependencies")
                
                # 2b. Related test files
                test_files = self._find_test_files(diff_file.file_info.path)
                if test_files:
                    dep_adjacent_parts.append(f"#### Related Test Files:\n" + 
                                             "\n".join(f"- {tf}" for tf in test_files))
                    logger.info(f"Found {len(test_files)} test files")
                
                # 2c. Configuration files
                config_files = self._find_config_files()
                if config_files:
                    dep_adjacent_parts.append(f"#### Configuration Files:\n" + 
                                             "\n".join(f"- {cf}" for cf in config_files[:15]))  # Limit to 15
                    logger.info(f"Found {len(config_files)} config files")
                
                if dep_adjacent_parts:
                    dep_section = "### Dependency-Adjacent Code\n\n" + "\n\n".join(dep_adjacent_parts) + "\n"
                    context_parts.append(dep_section)
                    current_size += len(dep_section)
                    logger.info(f"Added dependency-adjacent code section ({len(dep_section)} chars)")
            
            # STEP 3: Add full repository structure
            if current_size < max_context_size:
                repo_structure = self._scan_repository_structure()
                if repo_structure:
                    context_parts.append(f"### Full Repository Structure\n{repo_structure}\n")
                    current_size += len(repo_structure)
                    logger.info(f"Added repository structure ({len(repo_structure)} chars)")
            
            # STEP 4: Add code signatures from all files
            if current_size < max_context_size:
                code_signatures = self._extract_code_signatures()
                if code_signatures:
                    context_parts.append(f"### {code_signatures}\n")
                    current_size += len(code_signatures)
                    logger.info(f"Added code signatures ({len(code_signatures)} chars)")
            
            # STEP 5: Always include previous inline comments on this file (if any)
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
            
            # STEP 6: Forward dependencies (related files content)
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
                    
                    context_parts.append(f"### Related file (forward dependency): {related_file}\n```\n{content}\n```\n")
                    current_size += len(content)
            
            if context_parts:
                logger.info(f"Built comprehensive project context with {len(context_parts)} sections (~{current_size} chars)")
                return "\n".join(context_parts)
                
        except Exception as e:
            logger.debug(f"Error building project context: {str(e)}")
        
        return None
