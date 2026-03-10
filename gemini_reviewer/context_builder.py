"""
Context builder for the Gemini AI Code Reviewer.

This module is responsible for building analysis context by detecting
related files and gathering project context.
"""

import logging
import os
import re
import ast
from typing import List, Optional, Dict, Set, Any, Tuple
from pathlib import Path

from .models import DiffFile, PRDetails
from .utils import get_file_language

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds analysis context for code review."""

    def __init__(self, github_client, diff_parser, project_context_budget: int = 60000):
        """Initialize context builder with required dependencies.

        Args:
            github_client: GitHub client for fetching file contents
            diff_parser: Diff parser for language detection
            project_context_budget: Maximum character budget for project context
        """
        self.github_client = github_client
        self.diff_parser = diff_parser
        self.project_context_budget = project_context_budget

        # Session-level file content cache (keyed by "path@ref")
        self._file_cache: Dict[str, Optional[str]] = {}

        # Single-scan repo data (populated lazily by _scan_repo_once)
        self._repo_scanned: bool = False
        self._repo_structure: str = ""
        self._code_files: List[Tuple[str, str]] = []  # (rel_path, abs_path)
        self._test_file_map: Dict[str, List[str]] = {}  # source base name -> test file rel paths
        self._config_files: List[str] = []

        # PR-stable context caches (reused across files in same review)
        self._cached_mental_model: Optional[str] = None
        self._cached_repo_structure: Optional[str] = None
        self._cached_code_signatures: Optional[str] = None
        self._cached_config_files: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # File content cache
    # ------------------------------------------------------------------

    def _get_file_content_cached(self, owner: str, repo: str, path: str, ref: str) -> Optional[str]:
        """Fetch file content via GitHub API with session-level caching."""
        cache_key = f"{path}@{ref}"
        if cache_key not in self._file_cache:
            try:
                self._file_cache[cache_key] = self.github_client.get_file_content(owner, repo, path, ref)
            except Exception:
                self._file_cache[cache_key] = None
        return self._file_cache[cache_key]

    def get_cached_file_content(self, owner: str, repo: str, path: str, ref: str) -> Optional[str]:
        """Public accessor for cached file content (used by code_reviewer)."""
        return self._get_file_content_cached(owner, repo, path, ref)

    def clear_cache(self) -> None:
        """Clear all caches. Call between PR reviews."""
        self._file_cache.clear()
        self._repo_scanned = False
        self._repo_structure = ""
        self._code_files = []
        self._test_file_map = {}
        self._config_files = []
        self._cached_mental_model = None
        self._cached_repo_structure = None
        self._cached_code_signatures = None
        self._cached_config_files = None

    # ------------------------------------------------------------------
    # Consolidated repo scan
    # ------------------------------------------------------------------

    def _scan_repo_once(self) -> None:
        """Walk the repository once and populate all scan-derived caches."""
        if self._repo_scanned:
            return

        repo_root = os.getcwd()
        tree_lines = ["Repository Structure:"]
        code_files: List[Tuple[str, str]] = []
        test_file_map: Dict[str, List[str]] = {}
        config_files: List[str] = []

        # Directories to exclude
        exclude_dirs = {
            '.git', '.github', '__pycache__', 'node_modules', '.venv', 'venv',
            '.env', '.pytest_cache', '.mypy_cache', '.tox', 'build', 'dist',
            '.eggs', '*.egg-info', '.idea', '.vscode'
        }
        exclude_file_patterns = {'.pyc', '.pyo', '.pyd', '.so', '.dylib', '.dll', '.egg'}
        code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.java', '.rb'}
        test_dirs = {'test', 'tests', '__tests__', 'spec', 'specs'}
        test_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.java', '.rb'}

        # Important config file names
        config_names = {
            '.eslintrc', '.eslintrc.js', '.eslintrc.json', '.eslintrc.yml',
            '.prettierrc', '.prettierrc.js', '.prettierrc.json',
            '.pylintrc', 'pylint.cfg', '.flake8', 'tox.ini',
            '.mypy.ini', 'mypy.ini', 'pyproject.toml',
            'package.json', 'package-lock.json', 'yarn.lock',
            'requirements.txt', 'Pipfile', 'Pipfile.lock', 'poetry.lock',
            'go.mod', 'go.sum', 'Gemfile', 'Gemfile.lock',
            'build.gradle', 'pom.xml', 'Makefile',
            'Dockerfile', 'docker-compose.yml', 'docker-compose.yaml',
            '.dockerignore',
            '.travis.yml', 'circle.yml', '.gitlab-ci.yml',
            'azure-pipelines.yml', 'Jenkinsfile',
            'action.yml', 'action.yaml',
            '.editorconfig', '.gitignore', '.gitattributes',
        }

        def should_exclude_entry(name: str) -> bool:
            if name in exclude_dirs:
                return True
            if name.startswith('.') and name not in {'.gitignore', '.env.example'}:
                return True
            return False

        def should_exclude_file(name: str) -> bool:
            return any(name.endswith(p) for p in exclude_file_patterns)

        try:
            # Build tree structure + collect code/test/config files in one walk
            def scan_dir(path: str, prefix: str = "", depth: int = 0, max_depth: int = 3) -> None:
                if depth > max_depth:
                    return
                try:
                    entries = sorted(os.listdir(path))
                except PermissionError:
                    return

                dirs = [e for e in entries if os.path.isdir(os.path.join(path, e)) and not should_exclude_entry(e)]
                files = [e for e in entries if os.path.isfile(os.path.join(path, e)) and not should_exclude_entry(e) and not should_exclude_file(e)]

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

                    abs_path = os.path.join(path, filename)
                    rel_path = os.path.relpath(abs_path, repo_root)
                    file_ext = os.path.splitext(filename)[1]

                    # Collect code files
                    if file_ext in code_extensions:
                        code_files.append((rel_path, abs_path))

                    # Collect test files
                    if file_ext in test_extensions:
                        in_test_dir = any(td in rel_path.split(os.sep) for td in test_dirs)
                        file_lower = filename.lower()
                        if in_test_dir or file_lower.startswith('test') or file_lower.endswith('test' + file_ext):
                            # Map base name (without test prefix/suffix) to this test file
                            base = file_lower
                            for pref in ('test_', 'test'):
                                if base.startswith(pref):
                                    base = base[len(pref):]
                                    break
                            for suf in ('_test' + file_ext, 'test' + file_ext):
                                if base.endswith(suf):
                                    base = base[:-len(suf)]
                                    break
                            # Also store the original name without extension as a key
                            source_name = os.path.splitext(base)[0] if '.' in base else base
                            if source_name:
                                test_file_map.setdefault(source_name, []).append(rel_path)

                    # Collect config files (only from root and .github)
                    depth_from_root = rel_path.count(os.sep)
                    in_github = rel_path.startswith('.github' + os.sep)
                    if depth_from_root <= 1 or in_github:
                        if filename in config_names or filename.startswith('Dockerfile') or filename.startswith('.env'):
                            config_files.append(rel_path)

            scan_dir(repo_root)
        except Exception as e:
            logger.debug(f"Error during repo scan: {e}")

        self._repo_structure = "\n".join(tree_lines[:200])
        self._code_files = code_files
        self._test_file_map = test_file_map
        self._config_files = config_files
        self._repo_scanned = True

    # ------------------------------------------------------------------
    # Related files detection
    # ------------------------------------------------------------------

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
            full_content = self._get_file_content_cached(
                pr_details.owner, pr_details.repo, file_path, pr_details.head_sha or 'HEAD'
            )

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

            language = get_file_language(file_path)

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
            Converted file path or None if conversion fails or file doesn't exist
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

            if language.lower() in extensions:
                repo_root = os.getcwd()
                for ext in extensions[language.lower()]:
                    candidate = base_path + ext
                    # Validate that the path actually exists
                    if os.path.exists(candidate) or os.path.exists(os.path.join(repo_root, candidate)):
                        return candidate

            return None

        except Exception:
            return None

    # ------------------------------------------------------------------
    # Repo-walk-derived helpers (all use _scan_repo_once data)
    # ------------------------------------------------------------------

    def _scan_repository_structure(self, max_depth: int = 3) -> str:
        """Return the repository file tree structure (cached from single scan)."""
        if self._cached_repo_structure is not None:
            return self._cached_repo_structure
        self._scan_repo_once()
        self._cached_repo_structure = self._repo_structure
        return self._cached_repo_structure

    def _find_reverse_dependencies(self, changed_file: str) -> List[str]:
        """Find files that import or depend on the changed file."""
        reverse_deps = []
        try:
            self._scan_repo_once()

            changed_module = changed_file.replace('/', '.').replace('\\', '.')
            for ext in ['.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.java', '.rb']:
                if changed_module.endswith(ext):
                    changed_module = changed_module[:-len(ext)]
                    break

            changed_filename = os.path.basename(changed_file)
            changed_name = os.path.splitext(changed_filename)[0]

            import_patterns = [
                rf'from\s+{re.escape(changed_module)}\s+import',
                rf'import\s+{re.escape(changed_module)}',
                rf'from\s+.*{re.escape(changed_name)}\s+import',
                rf'import\s+.*{re.escape(changed_name)}',
                rf'require\(["\'].*{re.escape(changed_filename)}["\']\)',
                rf'import\s+.*from\s+["\'].*{re.escape(changed_filename)}["\']',
            ]

            for rel_path, abs_path in self._code_files:
                if rel_path == changed_file:
                    continue

                try:
                    with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    for pattern in import_patterns:
                        if re.search(pattern, content):
                            reverse_deps.append(rel_path)
                            break

                    if len(reverse_deps) >= 10:
                        break
                except Exception:
                    continue

            logger.info(f"Found {len(reverse_deps)} reverse dependencies for {changed_file}")
        except Exception as e:
            logger.debug(f"Error finding reverse dependencies: {str(e)}")

        return reverse_deps

    def _find_function_callers(self, changed_file: str, diff_content: str) -> List[Dict[str, Any]]:
        """Find code that calls functions being modified in this diff."""
        callers = []
        try:
            self._scan_repo_once()

            # Extract function names being added/modified
            changed_functions = set()
            for line in diff_content.split('\n'):
                if line.startswith('+') and not line.startswith('+++'):
                    clean_line = line[1:].strip()
                    if clean_line.startswith('def ') or clean_line.startswith('async def '):
                        func_name = clean_line.split('(')[0].replace('def ', '').replace('async ', '').strip()
                        if func_name:
                            changed_functions.add(func_name)
                    match = re.search(r'function\s+(\w+)', clean_line)
                    if match:
                        changed_functions.add(match.group(1))
                    match = re.search(r'const\s+(\w+)\s*=', clean_line)
                    if match and '=>' in clean_line:
                        changed_functions.add(match.group(1))

            if not changed_functions:
                return []

            logger.debug(f"Looking for callers of functions: {changed_functions}")

            for rel_path, abs_path in self._code_files:
                if rel_path == changed_file:
                    continue

                try:
                    with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        lines = content.split('\n')

                    for func_name in changed_functions:
                        call_pattern = rf'(?<!def\s)(?<!function\s)\b{re.escape(func_name)}\s*\('

                        for i, line in enumerate(lines):
                            if re.search(call_pattern, line):
                                start = max(0, i - 3)
                                end = min(len(lines), i + 4)
                                context_lines = lines[start:end]

                                callers.append({
                                    'file': rel_path,
                                    'function_name': func_name,
                                    'line_number': i + 1,
                                    'calling_code': '\n'.join(context_lines)
                                })

                                if len(callers) >= 15:
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
        """Find definitions of functions that are called in the given file."""
        called_functions = []
        try:
            self._scan_repo_once()

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

                    for rel_path, abs_path in self._code_files:
                        if not rel_path.endswith('.py'):
                            continue
                        if rel_path == file_path:
                            continue

                        try:
                            with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                                search_content = f.read()

                            search_tree = ast.parse(search_content)

                            for node in ast.walk(search_tree):
                                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                    if node.name in function_calls:
                                        lines = search_content.split('\n')
                                        start_line = node.lineno - 1
                                        end_line = min(start_line + 15, len(lines))
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

    def _find_test_files(self, changed_file: str) -> List[str]:
        """Find test files related to the changed file."""
        self._scan_repo_once()
        changed_name = os.path.splitext(os.path.basename(changed_file))[0]

        test_files = []
        # Direct lookup from the test file map
        if changed_name in self._test_file_map:
            test_files.extend(self._test_file_map[changed_name])

        # Also search for partial matches in the map
        for source_name, paths in self._test_file_map.items():
            if len(test_files) >= 5:
                break
            if changed_name in source_name or source_name in changed_name:
                for p in paths:
                    if p not in test_files:
                        test_files.append(p)
                        if len(test_files) >= 5:
                            break

        test_files = test_files[:5]
        if test_files:
            logger.info(f"Found {len(test_files)} test files for {changed_file}")
        return test_files

    def _find_config_files(self) -> List[str]:
        """Find configuration files (cached from single scan)."""
        if self._cached_config_files is not None:
            return self._cached_config_files
        self._scan_repo_once()
        self._cached_config_files = self._config_files
        logger.info(f"Found {len(self._cached_config_files)} config files")
        return self._cached_config_files

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

    # ------------------------------------------------------------------
    # PR-stable helpers
    # ------------------------------------------------------------------

    def _build_repo_mental_model(self) -> str:
        """Build a compact repo-level mental model with key information (cached)."""
        if self._cached_mental_model is not None:
            return self._cached_mental_model

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

                        lines = content.split('\n')
                        summary_lines = []
                        char_count = 0

                        for line in lines[:50]:
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
            self._cached_mental_model = "# Repository Mental Model\n\n" + "\n".join(model_parts)
        else:
            self._cached_mental_model = ""
        return self._cached_mental_model

    def _extract_code_signatures(self, max_files: int = 50) -> str:
        """Extract function and class signatures from code files (cached)."""
        if self._cached_code_signatures is not None:
            return self._cached_code_signatures

        try:
            self._scan_repo_once()
            signatures = ["Code Structure (Functions & Classes):"]
            files_processed = 0

            def extract_python_signatures(file_path: str, rel_path: str) -> List[str]:
                """Extract signatures from Python files with full type annotations."""
                sigs = []
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    tree = ast.parse(content)

                    def format_annotation(ann) -> str:
                        if ann is None:
                            return ""
                        try:
                            return ast.unparse(ann)
                        except Exception:
                            if isinstance(ann, ast.Name):
                                return ann.id
                            elif isinstance(ann, ast.Constant):
                                return repr(ann.value)
                            elif isinstance(ann, ast.Subscript):
                                return f"{format_annotation(ann.value)}[...]"
                            return "?"

                    def format_function_signature(node) -> str:
                        params = []
                        all_args = node.args.args + node.args.kwonlyargs

                        for arg in all_args[:6]:
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

                    for node in ast.iter_child_nodes(tree):
                        if isinstance(node, ast.ClassDef):
                            bases = []
                            for base in node.bases[:3]:
                                bases.append(format_annotation(base))
                            base_str = f"({', '.join(bases)})" if bases else ""

                            methods = []
                            for item in node.body:
                                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                    method_params = []
                                    for arg in item.args.args[1:4]:
                                        if arg.annotation:
                                            method_params.append(f"{arg.arg}: {format_annotation(arg.annotation)}")
                                        else:
                                            method_params.append(arg.arg)
                                    ret = f" -> {format_annotation(item.returns)}" if item.returns else ""
                                    methods.append(f"{item.name}({', '.join(method_params)}){ret}")

                            sigs.append(f"  class {node.name}{base_str}:")
                            for method in methods[:8]:
                                sigs.append(f"    {method}")

                        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            sigs.append(format_function_signature(node))

                except SyntaxError:
                    pass
                except Exception:
                    pass

                return sigs

            def extract_generic_signatures(file_path: str, rel_path: str) -> List[str]:
                """Extract signatures from other code files using regex."""
                sigs = []
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    patterns = [
                        r'function\s+(\w+)\s*\(',
                        r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>',
                        r'export\s+function\s+(\w+)\s*\(',
                        r'class\s+(\w+)',
                        r'func\s+(\w+)\s*\(',
                        r'(?:public|private|protected)\s+(?:static\s+)?(?:\w+)\s+(\w+)\s*\(',
                    ]

                    found_names = set()
                    for pattern in patterns:
                        matches = re.findall(pattern, content)
                        for match in matches[:10]:
                            if match not in found_names:
                                found_names.add(match)
                                sigs.append(f"  {match}")

                except Exception:
                    pass

                return sigs

            # Iterate over pre-scanned code files instead of os.walk
            for rel_path, abs_path in self._code_files:
                if files_processed >= max_files:
                    break

                file_ext = os.path.splitext(rel_path)[1]
                file_sigs = []
                if file_ext == '.py':
                    file_sigs = extract_python_signatures(abs_path, rel_path)
                else:
                    file_sigs = extract_generic_signatures(abs_path, rel_path)

                if file_sigs:
                    signatures.append(f"\n{rel_path}:")
                    signatures.extend(file_sigs[:20])
                    files_processed += 1

            result = "\n".join(signatures[:500])
            self._cached_code_signatures = result if len(signatures) > 1 else ""

        except Exception as e:
            logger.debug(f"Error extracting code signatures: {str(e)}")
            self._cached_code_signatures = ""

        return self._cached_code_signatures

    # ------------------------------------------------------------------
    # Main context builder
    # ------------------------------------------------------------------

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
        max_context_size = self.project_context_budget

        try:
            logger.info("Building comprehensive repository context for code review")

            # Pre-compute diff_content once (used by multiple steps)
            diff_content = "\n".join(
                line for hunk in diff_file.hunks for line in hunk.lines
            )

            # Collect all context sections with priorities for budget-aware assembly
            sections: List[Dict[str, Any]] = []

            # SECTION: Previous review comments (highest priority for follow-ups)
            try:
                prev = self.github_client.get_file_review_comments(
                    pr_details,
                    diff_file.file_info.path,
                    limit=30
                )
                if prev:
                    snippet = prev
                    if len(snippet) > 4000:
                        snippet = snippet[:4000] + "\n... (truncated)"
                    sections.append({
                        'content': f"### Previous review history for {diff_file.file_info.path}\n{snippet}\n",
                        'priority': 1.0,
                        'name': 'previous_comments'
                    })
            except Exception:
                pass

            # SECTION: Related file contents (forward dependencies)
            if related_files:
                scored_files = []
                for related_file in related_files:
                    score = self._score_related_file_relevance(
                        related_file, diff_file.file_info.path, diff_content
                    )
                    scored_files.append((related_file, score))
                scored_files.sort(key=lambda x: x[1], reverse=True)
                logger.debug(f"Prioritized related files: {[(f, f'{s:.2f}') for f, s in scored_files[:5]]}")

                related_parts = []
                for related_file, score in scored_files:
                    content = self._get_file_content_cached(
                        pr_details.owner,
                        pr_details.repo,
                        related_file,
                        pr_details.base_sha or 'main'
                    )
                    if content:
                        if score >= 0.5:
                            max_file_size = 10000
                        elif score >= 0.3:
                            max_file_size = 6000
                        else:
                            max_file_size = 3000
                        if len(content) > max_file_size:
                            content = content[:max_file_size] + "\n... (truncated)"
                        relevance_note = f" [relevance: {score:.2f}]" if score > 0 else ""
                        related_parts.append(
                            f"### Related file (forward dependency): {related_file}{relevance_note}\n```\n{content}\n```\n"
                        )

                if related_parts:
                    sections.append({
                        'content': "\n".join(related_parts),
                        'priority': 0.9,
                        'name': 'related_files'
                    })

            # SECTION: Function callers
            try:
                callers = self._find_function_callers(diff_file.file_info.path, diff_content)
                if callers:
                    caller_parts = ["### Code That Calls Changed Functions\n"]
                    caller_parts.append("These code snippets call functions that are being modified in this PR.\n")
                    caller_parts.append("Changes to function signatures or behavior may affect these callers:\n")

                    for caller in callers[:10]:
                        caller_parts.append(f"\n#### {caller['file']} calls `{caller['function_name']}` (line {caller['line_number']}):")
                        caller_parts.append(f"```\n{caller['calling_code']}\n```")

                    sections.append({
                        'content': "\n".join(caller_parts),
                        'priority': 0.85,
                        'name': 'function_callers'
                    })
                    logger.info(f"Found {len(callers)} function caller contexts")
            except Exception as e:
                logger.debug(f"Error finding function callers: {e}")

            # SECTION: Called function definitions
            try:
                file_content = self._get_file_content_cached(
                    pr_details.owner, pr_details.repo,
                    diff_file.file_info.path,
                    pr_details.head_sha or 'HEAD'
                )
                if file_content:
                    called_funcs = self._find_called_functions(file_content, diff_file.file_info.path)
                    if called_funcs:
                        called_parts = ["### Definitions of Functions Being Called\n"]
                        called_parts.append("These are definitions of functions called in the changed file:\n")

                        for func in called_funcs[:8]:
                            called_parts.append(f"\n#### `{func['function_name']}` from {func['file']}:")
                            called_parts.append(f"```\n{func['definition']}\n```")

                        sections.append({
                            'content': "\n".join(called_parts),
                            'priority': 0.8,
                            'name': 'called_functions'
                        })
                        logger.info(f"Found {len(called_funcs)} called function definitions")
            except Exception as e:
                logger.debug(f"Error finding called functions: {e}")

            # SECTION: Code signatures (PR-stable, cached)
            code_signatures = self._extract_code_signatures()
            if code_signatures:
                sections.append({
                    'content': f"### {code_signatures}\n",
                    'priority': 0.6,
                    'name': 'code_signatures'
                })

            # SECTION: Repo mental model (PR-stable, cached)
            repo_mental_model = self._build_repo_mental_model()
            if repo_mental_model:
                sections.append({
                    'content': repo_mental_model,
                    'priority': 0.5,
                    'name': 'repo_mental_model'
                })

            # SECTION: Dependency-adjacent listing
            dep_adjacent_parts = []
            reverse_deps = self._find_reverse_dependencies(diff_file.file_info.path)
            if reverse_deps:
                dep_adjacent_parts.append(f"#### Reverse Dependencies (files that import {diff_file.file_info.path}):\n" +
                                         "\n".join(f"- {rd}" for rd in reverse_deps))
                logger.info(f"Found {len(reverse_deps)} reverse dependencies")

            test_files = self._find_test_files(diff_file.file_info.path)
            if test_files:
                dep_adjacent_parts.append(f"#### Related Test Files:\n" +
                                         "\n".join(f"- {tf}" for tf in test_files))
                logger.info(f"Found {len(test_files)} test files")

            config_files = self._find_config_files()
            if config_files:
                dep_adjacent_parts.append(f"#### Configuration Files:\n" +
                                         "\n".join(f"- {cf}" for cf in config_files[:15]))
                logger.info(f"Found {len(config_files)} config files")

            if dep_adjacent_parts:
                sections.append({
                    'content': "### Dependency-Adjacent Code\n\n" + "\n\n".join(dep_adjacent_parts) + "\n",
                    'priority': 0.4,
                    'name': 'dependency_adjacent'
                })

            # SECTION: Repo structure tree (PR-stable, cached)
            repo_structure = self._scan_repository_structure()
            if repo_structure:
                sections.append({
                    'content': f"### Full Repository Structure\n{repo_structure}\n",
                    'priority': 0.3,
                    'name': 'repo_structure'
                })

            # Assemble with budget-aware prioritization
            if sections:
                selected = self._prioritize_context_sections(sections, max_context_size)
                if selected:
                    total_size = sum(len(s) for s in selected)
                    logger.info(f"Built project context: {len(selected)} sections, ~{total_size} chars (budget: {max_context_size})")
                    return "\n".join(selected)

        except Exception as e:
            logger.debug(f"Error building project context: {str(e)}")

        return None
