"""
Context builder for the Gemini AI Code Reviewer.

This module is responsible for building analysis context by detecting
related files and gathering project context.
"""

import logging
import os
import re
from typing import List, Optional

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
    
    async def build_project_context(
        self, 
        diff_file: DiffFile, 
        related_files: List[str], 
        pr_details: PRDetails
    ) -> Optional[str]:
        """Build project context including previous comments and related file contents.
        
        Args:
            diff_file: The diff file being analyzed
            related_files: List of related file paths
            pr_details: Pull request details
            
        Returns:
            Formatted project context string or None if no context available
        """
        context_parts = []
        max_context_size = 8000  # Limit total context to avoid token bloat
        current_size = 0
        
        try:
            # Always include previous inline comments on this file (if any)
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
