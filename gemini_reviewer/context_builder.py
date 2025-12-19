"""
Context builder for the Gemini AI Code Reviewer.

This module is responsible for building analysis context by detecting
related files and gathering project context.
"""

import logging
import os
import re
import ast
from typing import List, Optional, Dict, Set, Any
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
        """Detect related files based on imports, dependencies, and inheritance.

        Analyzes the FULL file content (not just hunks) to find all dependencies.

        Args:
            diff_file: The diff file to analyze
            pr_details: Pull request details

        Returns:
            List of related file paths (limited to 10 most relevant)
        """
        file_path = diff_file.file_info.path
        related_files = []
        seen_imports = set()

        try:
            # Get the FULL file content, not just the diff hunks
            full_content = None
            try:
                full_content = self.github_client.get_file_content(
                    pr_details.owner, pr_details.repo, file_path, pr_details.head_sha or 'HEAD'
                )
            except Exception as e:
                logger.debug(f"Could not fetch full file content for {file_path}: {e}")

            # Fall back to reading from local filesystem if GitHub fetch fails
            if not full_content:
                try:
                    local_path = os.path.join(os.getcwd(), file_path)
                    if os.path.exists(local_path):
                        with open(local_path, 'r', encoding='utf-8', errors='ignore') as f:
                            full_content = f.read()
                except Exception:
                    pass

            # If still no content, fall back to hunk content
            if not full_content:
                full_content = "\n".join(
                    line[1:] if line and line[0] in '+ ' else line
                    for hunk in diff_file.hunks
                    for line in hunk.lines
                    if not line.startswith('-')
                )

            # Import patterns for various languages
            import_patterns = {
                'python': [
                    r'from\s+([\w.]+)\s+import',
                    r'import\s+([\w.]+)',
                    # Dynamic imports
                    r'importlib\.import_module\(["\']([^"\']+)["\']\)',
                    r'__import__\(["\']([^"\']+)["\']\)',
                ],
                'javascript': [
                    r'import\s+.*\s+from\s+["\']([^"\']+)["\']',
                    r'import\s+["\']([^"\']+)["\']',
                    r'require\(["\']([^"\']+)["\']\)',
                    r'import\(["\']([^"\']+)["\']\)',  # Dynamic imports
                ],
                'typescript': [
                    r'import\s+.*\s+from\s+["\']([^"\']+)["\']',
                    r'import\s+["\']([^"\']+)["\']',
                    r'require\(["\']([^"\']+)["\']\)',
                    r'import\(["\']([^"\']+)["\']\)',
                ],
                'java': [r'import\s+([^;]+);'],
                'go': [
                    r'import\s+["\']([^"\']+)["\']',
                    r'import\s+\w+\s+["\']([^"\']+)["\']',  # Named imports
                    r'import\s+\(\s*(?:[^)]*\s)?["\']([^"\']+)["\']',
                ],
                'ruby': [
                    r'require\s+["\']([^"\']+)["\']',
                    r'require_relative\s+["\']([^"\']+)["\']',
                    r'load\s+["\']([^"\']+)["\']',
                ],
            }

            language = self.diff_parser.get_file_language(file_path)

            if language and language.lower() in import_patterns:
                patterns = import_patterns[language.lower()]

                for pattern in patterns:
                    matches = re.findall(pattern, full_content)
                    for match in matches:
                        if match in seen_imports:
                            continue
                        seen_imports.add(match)

                        # Convert import path to file path
                        related_file = self._import_to_file_path(match, file_path, language)
                        if related_file and related_file not in related_files:
                            related_files.append(related_file)

            # For Python, also detect class inheritance and type annotations
            if language and language.lower() == 'python':
                try:
                    tree = ast.parse(full_content)
                    for node in ast.walk(tree):
                        # Class inheritance
                        if isinstance(node, ast.ClassDef):
                            for base in node.bases:
                                if isinstance(base, ast.Attribute):
                                    # module.ClassName
                                    if isinstance(base.value, ast.Name):
                                        module_name = base.value.id
                                        if module_name not in seen_imports:
                                            seen_imports.add(module_name)
                                            rel = self._import_to_file_path(module_name, file_path, 'python')
                                            if rel and rel not in related_files:
                                                related_files.append(rel)

                        # Type annotations in function signatures
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            # Check return annotation
                            if node.returns:
                                self._extract_type_refs(node.returns, seen_imports, related_files, file_path)
                            # Check parameter annotations
                            for arg in node.args.args + node.args.kwonlyargs:
                                if arg.annotation:
                                    self._extract_type_refs(arg.annotation, seen_imports, related_files, file_path)
                except SyntaxError:
                    pass  # File might have syntax errors
                except Exception as e:
                    logger.debug(f"Error parsing Python AST for {file_path}: {e}")

            # Limit to most relevant files
            related_files = related_files[:10]

            if related_files:
                logger.info(f"Detected {len(related_files)} related files for {file_path}")

        except Exception as e:
            logger.debug(f"Error detecting related files for {file_path}: {str(e)}")

        return related_files

    def _extract_type_refs(self, annotation, seen_imports: Set[str], related_files: List[str], current_file: str):
        """Extract type references from annotations for dependency detection."""
        try:
            if isinstance(annotation, ast.Name):
                # Simple type like 'MyClass'
                type_name = annotation.id
                if type_name not in seen_imports and type_name not in {'str', 'int', 'float', 'bool', 'None', 'Any', 'List', 'Dict', 'Optional', 'Tuple', 'Set'}:
                    seen_imports.add(type_name)
            elif isinstance(annotation, ast.Attribute):
                # Qualified type like 'module.MyClass'
                if isinstance(annotation.value, ast.Name):
                    module_name = annotation.value.id
                    if module_name not in seen_imports:
                        seen_imports.add(module_name)
                        rel = self._import_to_file_path(module_name, current_file, 'python')
                        if rel and rel not in related_files:
                            related_files.append(rel)
            elif isinstance(annotation, ast.Subscript):
                # Generic types like List[MyClass] or Dict[str, MyClass]
                self._extract_type_refs(annotation.slice, seen_imports, related_files, current_file)
            elif isinstance(annotation, ast.Tuple):
                # Multiple types in a tuple
                for elt in annotation.elts:
                    self._extract_type_refs(elt, seen_imports, related_files, current_file)
        except Exception:
            pass
    
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

    def _find_function_callers(self, changed_file: str, diff_content: str) -> List[Dict[str, Any]]:
        """Find code that calls functions being modified in this diff.

        Args:
            changed_file: Path to the changed file
            diff_content: The diff content to analyze for changed functions

        Returns:
            List of dicts with caller info: {file, function_name, calling_code}
        """
        callers = []
        try:
            repo_root = os.getcwd()

            # Extract function names that are being added/modified in the diff
            changed_functions = set()
            for line in diff_content.split('\n'):
                if line.startswith('+') and not line.startswith('+++'):
                    clean_line = line[1:].strip()
                    # Python function definitions
                    if clean_line.startswith('def ') or clean_line.startswith('async def '):
                        func_name = clean_line.split('(')[0].replace('def ', '').replace('async ', '').strip()
                        if func_name:
                            changed_functions.add(func_name)
                    # JavaScript/TypeScript functions
                    match = re.search(r'function\s+(\w+)', clean_line)
                    if match:
                        changed_functions.add(match.group(1))
                    # Arrow functions assigned to const
                    match = re.search(r'const\s+(\w+)\s*=', clean_line)
                    if match and '=>' in clean_line:
                        changed_functions.add(match.group(1))

            if not changed_functions:
                return []

            logger.debug(f"Looking for callers of functions: {changed_functions}")

            # Search for callers across the codebase
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
                            lines = content.split('\n')

                        for func_name in changed_functions:
                            # Pattern to find function calls (not definitions)
                            call_pattern = rf'(?<!def\s)(?<!function\s)\b{re.escape(func_name)}\s*\('

                            for i, line in enumerate(lines):
                                if re.search(call_pattern, line):
                                    # Get context around the call (3 lines before and after)
                                    start = max(0, i - 3)
                                    end = min(len(lines), i + 4)
                                    context_lines = lines[start:end]

                                    callers.append({
                                        'file': rel_path,
                                        'function_name': func_name,
                                        'line_number': i + 1,
                                        'calling_code': '\n'.join(context_lines)
                                    })

                                    if len(callers) >= 15:  # Limit total callers
                                        break

                        if len(callers) >= 15:
                            break

                    except Exception:
                        continue

                if len(callers) >= 15:
                    break

            if callers:
                logger.info(f"Found {len(callers)} callers for changed functions in {changed_file}")

        except Exception as e:
            logger.debug(f"Error finding function callers: {str(e)}")

        return callers

    def _find_called_functions(self, file_content: str, file_path: str) -> List[Dict[str, Any]]:
        """Find definitions of functions that are called in the given file.

        Args:
            file_content: Content of the file being reviewed
            file_path: Path to the file

        Returns:
            List of dicts with called function info
        """
        called_functions = []
        try:
            repo_root = os.getcwd()

            # For Python files, use AST to find function calls
            if file_path.endswith('.py'):
                try:
                    tree = ast.parse(file_content)
                    function_calls = set()

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Name):
                                function_calls.add(node.func.id)
                            elif isinstance(node.func, ast.Attribute):
                                function_calls.add(node.func.attr)

                    # Now search for definitions of these functions
                    code_extensions = {'.py'}
                    exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist'}

                    for root, dirs, files in os.walk(repo_root):
                        dirs[:] = [d for d in dirs if d not in exclude_dirs]

                        for filename in files:
                            if not filename.endswith('.py'):
                                continue

                            search_path = os.path.join(root, filename)
                            rel_path = os.path.relpath(search_path, repo_root)

                            if rel_path == file_path:
                                continue

                            try:
                                with open(search_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    search_content = f.read()

                                search_tree = ast.parse(search_content)

                                for node in ast.walk(search_tree):
                                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                        if node.name in function_calls:
                                            # Get the function definition with a few lines of body
                                            lines = search_content.split('\n')
                                            start_line = node.lineno - 1
                                            end_line = min(start_line + 15, len(lines))  # Get up to 15 lines
                                            func_code = '\n'.join(lines[start_line:end_line])

                                            called_functions.append({
                                                'file': rel_path,
                                                'function_name': node.name,
                                                'definition': func_code
                                            })

                                            if len(called_functions) >= 10:
                                                break

                                if len(called_functions) >= 10:
                                    break

                            except Exception:
                                continue

                        if len(called_functions) >= 10:
                            break

                except SyntaxError:
                    pass

            if called_functions:
                logger.info(f"Found {len(called_functions)} called function definitions for {file_path}")

        except Exception as e:
            logger.debug(f"Error finding called functions: {str(e)}")

        return called_functions

    def _score_related_file_relevance(self, related_file: str, changed_file: str, diff_content: str) -> float:
        """Score how relevant a related file is to the changed file.

        Higher scores mean more relevant. Used for prioritizing which context to include.

        Args:
            related_file: Path to a potentially related file
            changed_file: Path to the file being changed
            diff_content: The diff content being reviewed

        Returns:
            Relevance score (0.0 to 1.0)
        """
        score = 0.0

        try:
            # Same directory = higher relevance
            if os.path.dirname(related_file) == os.path.dirname(changed_file):
                score += 0.3

            # Same package/module prefix = moderate relevance
            changed_parts = changed_file.replace('\\', '/').split('/')
            related_parts = related_file.replace('\\', '/').split('/')
            common_prefix = 0
            for c, r in zip(changed_parts, related_parts):
                if c == r:
                    common_prefix += 1
                else:
                    break
            if common_prefix > 0:
                score += min(0.2, common_prefix * 0.05)

            # File is mentioned in the diff = very high relevance
            related_name = os.path.basename(related_file).rsplit('.', 1)[0]
            if related_name in diff_content:
                score += 0.4

            # Related file is a utility/helper (lower priority)
            if any(kw in related_file.lower() for kw in ['util', 'helper', 'common', 'base']):
                score -= 0.1

            # Test files have lower priority for production code context
            if any(kw in related_file.lower() for kw in ['test', 'spec', 'mock']):
                score -= 0.15

            # Config files are useful
            if any(kw in related_file.lower() for kw in ['config', 'settings', 'constants']):
                score += 0.1

            # Same file extension = slightly more relevant
            if os.path.splitext(related_file)[1] == os.path.splitext(changed_file)[1]:
                score += 0.05

        except Exception:
            pass

        return max(0.0, min(1.0, score))

    def _prioritize_context_sections(
        self,
        sections: List[Dict[str, Any]],
        max_size: int
    ) -> List[str]:
        """Prioritize and select context sections to fit within budget.

        Args:
            sections: List of dicts with 'content', 'priority', 'name'
            max_size: Maximum total character budget

        Returns:
            List of selected content strings in priority order
        """
        # Sort by priority (higher first)
        sorted_sections = sorted(sections, key=lambda x: x.get('priority', 0), reverse=True)

        selected = []
        current_size = 0

        for section in sorted_sections:
            content = section.get('content', '')
            if not content:
                continue

            content_size = len(content)

            # If this section fits, include it
            if current_size + content_size <= max_size:
                selected.append(content)
                current_size += content_size
            else:
                # Try to include a truncated version if it's high priority
                if section.get('priority', 0) >= 0.8:
                    remaining = max_size - current_size
                    if remaining > 500:  # Only include if we have meaningful space
                        truncated = content[:remaining - 50] + "\n... (truncated due to space)"
                        selected.append(truncated)
                        current_size += len(truncated)
                        break  # No more space after this

        return selected

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
                """Extract signatures from Python files with full type annotations."""
                sigs = []
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    tree = ast.parse(content)

                    def format_annotation(ann) -> str:
                        """Format an AST annotation node to string."""
                        if ann is None:
                            return ""
                        try:
                            return ast.unparse(ann)
                        except Exception:
                            # Fallback for older Python versions
                            if isinstance(ann, ast.Name):
                                return ann.id
                            elif isinstance(ann, ast.Constant):
                                return repr(ann.value)
                            elif isinstance(ann, ast.Subscript):
                                return f"{format_annotation(ann.value)}[...]"
                            return "?"

                    def format_function_signature(node) -> str:
                        """Format a function/method with full signature."""
                        params = []
                        all_args = node.args.args + node.args.kwonlyargs

                        for arg in all_args[:6]:  # Limit params shown
                            param_str = arg.arg
                            if arg.annotation:
                                type_str = format_annotation(arg.annotation)
                                param_str = f"{arg.arg}: {type_str}"
                            params.append(param_str)

                        if len(all_args) > 6:
                            params.append("...")

                        params_str = ", ".join(params)
                        return_type = ""
                        if node.returns:
                            return_type = f" -> {format_annotation(node.returns)}"

                        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                        return f"  {prefix} {node.name}({params_str}){return_type}"

                    # Process top-level nodes
                    for node in ast.iter_child_nodes(tree):
                        if isinstance(node, ast.ClassDef):
                            # Get base classes
                            bases = []
                            for base in node.bases[:3]:
                                bases.append(format_annotation(base))
                            base_str = f"({', '.join(bases)})" if bases else ""

                            # Get methods with their signatures
                            methods = []
                            for item in node.body:
                                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                    # Format method signature compactly
                                    method_params = []
                                    for arg in item.args.args[1:4]:  # Skip 'self', limit to 3
                                        if arg.annotation:
                                            method_params.append(f"{arg.arg}: {format_annotation(arg.annotation)}")
                                        else:
                                            method_params.append(arg.arg)
                                    ret = f" -> {format_annotation(item.returns)}" if item.returns else ""
                                    methods.append(f"{item.name}({', '.join(method_params)}){ret}")

                            sigs.append(f"  class {node.name}{base_str}:")
                            for method in methods[:8]:  # Limit methods per class
                                sigs.append(f"    {method}")

                        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            sigs.append(format_function_signature(node))

                except SyntaxError:
                    pass  # File has syntax errors
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
        max_context_size = 60000  # Increased limit for comprehensive context
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
                        # Previous comments help verify resolutions
                        snippet = prev
                        if len(snippet) > 4000:
                            snippet = snippet[:4000] + "\n... (truncated)"
                        context_parts.append(f"### Previous review history for {diff_file.file_info.path}\n{snippet}\n")
                        current_size += len(snippet)
                except Exception:
                    pass
            
            # STEP 6: Forward dependencies (related files content) - prioritized by relevance
            if related_files:
                # Build diff content for scoring
                diff_content = "\n".join(
                    line for hunk in diff_file.hunks for line in hunk.lines
                )

                # Score and sort related files by relevance
                scored_files = []
                for related_file in related_files:
                    score = self._score_related_file_relevance(
                        related_file, diff_file.file_info.path, diff_content
                    )
                    scored_files.append((related_file, score))

                # Sort by score descending (most relevant first)
                scored_files.sort(key=lambda x: x[1], reverse=True)
                logger.debug(f"Prioritized related files: {[(f, f'{s:.2f}') for f, s in scored_files[:5]]}")

                for related_file, score in scored_files:
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
                        # Allocate more space to higher relevance files
                        if score >= 0.5:
                            max_file_size = 10000
                        elif score >= 0.3:
                            max_file_size = 6000
                        else:
                            max_file_size = 3000

                        if len(content) > max_file_size:
                            content = content[:max_file_size] + "\n... (truncated)"

                        relevance_note = f" [relevance: {score:.2f}]" if score > 0 else ""
                        context_parts.append(
                            f"### Related file (forward dependency): {related_file}{relevance_note}\n```\n{content}\n```\n"
                        )
                        current_size += len(content)

            # STEP 7: Find callers of functions being changed
            if current_size < max_context_size:
                try:
                    # Build diff content from hunks
                    diff_content = "\n".join(
                        line for hunk in diff_file.hunks for line in hunk.lines
                    )
                    callers = self._find_function_callers(diff_file.file_info.path, diff_content)
                    if callers:
                        caller_parts = ["### Code That Calls Changed Functions\n"]
                        caller_parts.append("These code snippets call functions that are being modified in this PR.\n")
                        caller_parts.append("Changes to function signatures or behavior may affect these callers:\n")

                        for caller in callers[:10]:  # Limit displayed
                            caller_parts.append(f"\n#### {caller['file']} calls `{caller['function_name']}` (line {caller['line_number']}):")
                            caller_parts.append(f"```\n{caller['calling_code']}\n```")

                        caller_section = "\n".join(caller_parts)
                        context_parts.append(caller_section)
                        current_size += len(caller_section)
                        logger.info(f"Added {len(callers)} function caller contexts ({len(caller_section)} chars)")
                except Exception as e:
                    logger.debug(f"Error finding function callers: {e}")

            # STEP 8: Find definitions of functions being called
            if current_size < max_context_size:
                try:
                    # Get full file content to analyze calls
                    file_content = self.github_client.get_file_content(
                        pr_details.owner, pr_details.repo,
                        diff_file.file_info.path,
                        pr_details.head_sha or 'HEAD'
                    )
                    if file_content:
                        called_funcs = self._find_called_functions(file_content, diff_file.file_info.path)
                        if called_funcs:
                            called_parts = ["### Definitions of Functions Being Called\n"]
                            called_parts.append("These are definitions of functions called in the changed file:\n")

                            for func in called_funcs[:8]:  # Limit displayed
                                called_parts.append(f"\n#### `{func['function_name']}` from {func['file']}:")
                                called_parts.append(f"```\n{func['definition']}\n```")

                            called_section = "\n".join(called_parts)
                            context_parts.append(called_section)
                            current_size += len(called_section)
                            logger.info(f"Added {len(called_funcs)} called function definitions ({len(called_section)} chars)")
                except Exception as e:
                    logger.debug(f"Error finding called functions: {e}")

            if context_parts:
                logger.info(f"Built comprehensive project context with {len(context_parts)} sections (~{current_size} chars)")
                return "\n".join(context_parts)
                
        except Exception as e:
            logger.debug(f"Error building project context: {str(e)}")
        
        return None
