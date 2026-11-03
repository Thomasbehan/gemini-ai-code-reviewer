"""
Tests for the controls that keep review comments accurate: severity tags, evidence rules,
repository conventions in context, the context-aware verify pass and commit-pinned reviews.
"""

from unittest.mock import Mock, patch

import pytest
from google.genai import types

from gemini_reviewer.comment_processor import tag_with_severity
from gemini_reviewer.config import GeminiConfig, GitHubConfig
from gemini_reviewer.context_builder import ContextBuilder
from gemini_reviewer.gemini_client import GeminiClient, _finish_reason_name
from gemini_reviewer.github_client import GitHubClient
from gemini_reviewer.models import PRDetails, ReviewComment, ReviewPriority
from gemini_reviewer.prompts import ReviewMode, get_review_prompt_template, get_verify_prompt
from gemini_reviewer.utils import strip_leading_tags


class TestSeverityTags:
    @pytest.mark.parametrize(
        "priority, tag",
        [
            (ReviewPriority.CRITICAL, "[BLOCKER]"),
            (ReviewPriority.HIGH, "[MAJOR]"),
            (ReviewPriority.MEDIUM, "[MINOR]"),
            (ReviewPriority.LOW, "[NIT]"),
        ],
    )
    def test_priority_maps_to_tag(self, priority, tag):
        assert priority.severity_tag == tag
        assert tag_with_severity("Body", priority) == f"{tag} Body"

    @pytest.mark.parametrize(
        "body, expected",
        [
            ("[SECURITY] Token is logged", "Token is logged"),
            ("**[BLOCKER]** [BUG] Nil map write", "Nil map write"),
            ("  [NIT]Spacing", "Spacing"),
            ("No tag here", "No tag here"),
            ("Mentions [BLOCKER] mid-sentence", "Mentions [BLOCKER] mid-sentence"),
            (None, ""),
        ],
    )
    def test_strip_leading_tags(self, body, expected):
        assert strip_leading_tags(body) == expected

    def test_model_written_tag_is_replaced_by_priority_tag(self):
        assert tag_with_severity("[BLOCKER] Rename cfg", ReviewPriority.MEDIUM) == "[MINOR] Rename cfg"


class TestFinishReasonName:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (None, None),
            (types.FinishReason.SAFETY, "SAFETY"),
            ("recitation", "RECITATION"),
            (2, "MAX_TOKENS"),
            (999, "999"),
        ],
    )
    def test_normalises_every_representation(self, value, expected):
        assert _finish_reason_name(value) == expected


class TestPromptEvidenceRules:
    def test_review_prompt_includes_evidence_rules(self):
        prompt = get_review_prompt_template(ReviewMode.STANDARD)
        assert "EVIDENCE RULES" in prompt
        assert "Never claim from memory" in prompt
        assert "[BLOCKER]" in prompt

    def test_followup_prompt_is_unchanged_by_evidence_rules(self):
        prompt = get_review_prompt_template(ReviewMode.FOLLOWUP, previous_comments="- earlier point")
        assert "EVIDENCE RULES" not in prompt

    def test_custom_instructions_cannot_override_evidence_rules(self):
        prompt = get_review_prompt_template(ReviewMode.STANDARD, custom_instructions="Label as [SECURITY]")
        assert "never override the EVIDENCE RULES" in prompt

    def test_verify_prompt_carries_context(self):
        prompt = get_verify_prompt("+x = 1", "1. finding", "guard lives here")
        assert "guard lives here" in prompt
        assert "(none available)" in get_verify_prompt("+x = 1", "1. finding")


class TestVerifyPassContext:
    @pytest.fixture
    def client(self):
        config = GeminiConfig(api_key="AIzaSyTestKey123456", max_prompt_length=3000)
        with patch("gemini_reviewer.gemini_client.genai") as mock_genai:
            models = Mock()
            response = Mock(text='{"keep": [1], "notes": ""}', candidates=None)
            models.generate_content.return_value = response
            mock_genai.Client.return_value.models = models
            yield GeminiClient(config), models

    def test_context_is_sent_to_the_verifier(self, client):
        gemini_client, models = client
        kept = gemini_client.verify_findings("+x = 1", ["finding"], "def guard(): ...")
        assert kept == [1]
        assert "def guard(): ..." in models.generate_content.call_args.kwargs["contents"]

    def test_oversized_context_is_truncated_before_the_findings_are(self, client):
        gemini_client, models = client
        gemini_client.verify_findings("+x = 1", ["keep me"], "c" * 10000)
        prompt = models.generate_content.call_args.kwargs["contents"]
        assert "keep me" in prompt
        assert prompt.endswith("...[truncated]\n")
        assert len(prompt) <= gemini_client.config.max_prompt_length + len("\n...[truncated]") + 1


class TestRepositoryConventions:
    @pytest.fixture
    def context_builder(self):
        return ContextBuilder(Mock(), Mock())

    def test_loads_convention_files_as_authoritative(self, context_builder, tmp_path, monkeypatch):
        (tmp_path / "CLAUDE.md").write_text("Keep `err` as the error variable name.")
        (tmp_path / "REVIEW.md").write_text("Coordinates are [lat, lon].")
        monkeypatch.chdir(tmp_path)

        conventions = context_builder._load_repository_conventions()

        assert "Repository Conventions (authoritative)" in conventions
        assert "Keep `err`" in conventions
        assert conventions.index("REVIEW.md") < conventions.index("CLAUDE.md")

    def test_returns_empty_and_caches_without_convention_files(self, context_builder, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert context_builder._load_repository_conventions() == ""
        (tmp_path / "CLAUDE.md").write_text("added later")
        assert context_builder._load_repository_conventions() == ""

    def test_respects_the_budget(self, context_builder, tmp_path, monkeypatch):
        (tmp_path / "REVIEW.md").write_text("r" * 20000)
        (tmp_path / "CLAUDE.md").write_text("never reached")
        monkeypatch.chdir(tmp_path)

        conventions = context_builder._load_repository_conventions()

        assert "... (truncated)" in conventions
        assert "never reached" not in conventions

    def test_unreadable_file_is_skipped(self, context_builder, tmp_path, monkeypatch):
        (tmp_path / "REVIEW.md").write_text("review rules")
        (tmp_path / "CLAUDE.md").write_text("agent rules")
        monkeypatch.chdir(tmp_path)
        real_open = open

        def failing_open(path, *args, **kwargs):
            if str(path).endswith("REVIEW.md"):
                raise OSError("permission denied")
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=failing_open):
            conventions = context_builder._load_repository_conventions()

        assert "agent rules" in conventions
        assert "review rules" not in conventions


class TestCommitPinnedReviews:
    @pytest.fixture
    def github_client(self):
        with (
            patch("gemini_reviewer.github_client.Github") as mock_github,
            patch("gemini_reviewer.github_client.requests.Session"),
        ):
            repository = Mock()
            pull_request = Mock()
            pull_request.create_review.return_value = Mock(id=1)
            repository.get_pull.return_value = pull_request
            mock_github.return_value.get_repo.return_value = repository
            yield GitHubClient(GitHubConfig(token="ghp_" + "a" * 36)), repository, pull_request

    def _comment(self):
        return ReviewComment(body="[MAJOR] Fix", path="main.py", position=1, priority=ReviewPriority.HIGH)

    def test_review_is_anchored_to_the_reviewed_commit(self, github_client):
        client, repository, pull_request = github_client
        details = PRDetails("owner", "repo", 1, "Title", "Desc", head_sha="abc123")

        client.create_review(details, [self._comment()], "REQUEST_CHANGES")

        repository.get_commit.assert_called_once_with("abc123")
        assert pull_request.create_review.call_args.kwargs["commit"] is repository.get_commit.return_value

    def test_review_without_known_head_uses_the_pull_request_head(self, github_client):
        client, repository, pull_request = github_client

        client.create_review(PRDetails("owner", "repo", 1, "Title", "Desc"), [self._comment()], "REQUEST_CHANGES")

        repository.get_commit.assert_not_called()
        assert "commit" not in pull_request.create_review.call_args.kwargs

    def test_signature_ignores_severity_tags(self, github_client):
        client, _, _ = github_client
        assert client._compute_signature("main.py", "[MAJOR] Fix it") == client._compute_signature("main.py", "Fix it")


class TestChangeRequestsNeedFindings:
    @pytest.mark.asyncio
    async def test_no_empty_change_request_when_every_finding_was_a_duplicate(self):
        from gemini_reviewer.code_reviewer import CodeReviewer

        with (
            patch("gemini_reviewer.code_reviewer.GitHubClient") as mock_github,
            patch("gemini_reviewer.code_reviewer.GeminiClient"),
            patch("gemini_reviewer.code_reviewer.DiffParser"),
            patch("gemini_reviewer.code_reviewer.ContextBuilder"),
            patch("gemini_reviewer.code_reviewer.CommentProcessor") as mock_processor,
        ):
            mock_github.return_value.filter_out_existing_comments.return_value = []
            mock_processor.return_value.filter_comments_by_priority.return_value = []
            mock_processor.return_value.apply_comment_limits.return_value = []
            reviewer = CodeReviewer(Mock())

            posted = await reviewer._create_github_review(
                PRDetails("owner", "repo", 1, "Title", "Desc"),
                [ReviewComment(body="dup", path="main.py", position=1)],
                preferred_event="REQUEST_CHANGES",
            )

        assert posted is True
        mock_github.return_value.create_review.assert_not_called()
