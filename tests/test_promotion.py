"""Tests for knowl.context.promotion — cross-project pattern detection."""

import pytest

from knowl.context import store
from knowl.context.promotion import (
    line_overlap,
    scan_for_promotions,
    apply_promotion,
    PromotionSuggestion,
)


@pytest.fixture(autouse=True)
def use_tmp_knowl_dir(tmp_path):
    store.set_knowl_dir(tmp_path / ".knowl")
    store.init_store()
    yield


class TestLineOverlap:
    def test_identical_texts(self):
        text = "- Use TypeScript\n- Use ESLint\n- Prefer const"
        assert line_overlap(text, text) == 1.0

    def test_no_overlap(self):
        a = "- Use Python\n- Use Black"
        b = "- Use Rust\n- Use Clippy"
        assert line_overlap(a, b) == 0.0

    def test_partial_overlap(self):
        a = "- Use TypeScript\n- Use ESLint\n- Prefer const"
        b = "- Use TypeScript\n- Use Prettier\n- Prefer const"
        sim = line_overlap(a, b)
        # 2 shared out of 4 unique = 0.5
        assert 0.4 <= sim <= 0.6

    def test_empty_texts(self):
        assert line_overlap("", "") == 0.0
        assert line_overlap("hello", "") == 0.0

    def test_headers_ignored(self):
        a = "# Coding Style\n- Use TypeScript"
        b = "# Conventions\n- Use TypeScript"
        # Headers are stripped; only "- Use TypeScript" remains in both
        assert line_overlap(a, b) == 1.0


class TestScanForPromotions:
    def test_no_projects(self):
        assert scan_for_promotions() == []

    def test_one_project(self):
        store.create_project("alpha")
        assert scan_for_promotions() == []

    def test_identical_files_across_projects(self):
        content = "# Coding Style\n\n- Use TypeScript\n- Use ESLint\n- Prefer const over let\n"
        store.create_project("alpha")
        store.create_project("beta")
        (store.PROJECTS_DIR / "alpha" / "conventions.md").write_text(content)
        (store.PROJECTS_DIR / "beta" / "conventions.md").write_text(content)

        suggestions = scan_for_promotions()
        assert len(suggestions) >= 1
        match = [s for s in suggestions if s.filename == "conventions.md"]
        assert len(match) == 1
        assert match[0].similarity == 1.0
        assert "alpha" in match[0].projects
        assert "beta" in match[0].projects

    def test_similar_files_above_threshold(self):
        store.create_project("alpha")
        store.create_project("beta")

        content_a = "- Use TypeScript\n- Use ESLint\n- Prefer const\n- Use strict mode\n"
        content_b = "- Use TypeScript\n- Use Prettier\n- Prefer const\n- Use strict mode\n"

        (store.PROJECTS_DIR / "alpha" / "style.md").write_text(content_a)
        (store.PROJECTS_DIR / "beta" / "style.md").write_text(content_b)

        suggestions = scan_for_promotions(similarity_threshold=0.3)
        match = [s for s in suggestions if s.filename == "style.md"]
        assert len(match) == 1
        assert match[0].similarity > 0.3

    def test_dissimilar_files_below_threshold(self):
        store.create_project("alpha")
        store.create_project("beta")

        (store.PROJECTS_DIR / "alpha" / "notes.md").write_text("- Python web app\n- Django backend\n")
        (store.PROJECTS_DIR / "beta" / "notes.md").write_text("- iOS app\n- Swift UI\n")

        suggestions = scan_for_promotions(similarity_threshold=0.3)
        match = [s for s in suggestions if s.filename == "notes.md"]
        assert len(match) == 0

    def test_conflict_detection(self):
        content = "- Use TypeScript\n- Use ESLint\n"
        store.create_project("alpha")
        store.create_project("beta")
        (store.PROJECTS_DIR / "alpha" / "conventions.md").write_text(content)
        (store.PROJECTS_DIR / "beta" / "conventions.md").write_text(content)
        # Create a conflicting global file
        (store.GLOBAL_DIR / "conventions.md").write_text("# Old conventions")

        suggestions = scan_for_promotions()
        match = [s for s in suggestions if s.filename == "conventions.md"]
        assert len(match) == 1
        assert match[0].conflict is True

    def test_skips_context_json(self):
        """context.json should not be considered for promotion."""
        store.create_project("alpha")
        store.create_project("beta")
        # context.json is auto-created by create_project; verify it's excluded
        suggestions = scan_for_promotions()
        json_matches = [s for s in suggestions if s.filename == "context.json"]
        assert len(json_matches) == 0

    def test_three_projects(self):
        content = "- Always use UTC\n- Prefer ISO 8601 dates\n"
        for name in ("alpha", "beta", "gamma"):
            store.create_project(name)
            (store.PROJECTS_DIR / name / "dates.md").write_text(content)

        suggestions = scan_for_promotions()
        match = [s for s in suggestions if s.filename == "dates.md"]
        assert len(match) == 1
        assert len(match[0].projects) == 3

    def test_custom_threshold(self):
        store.create_project("alpha")
        store.create_project("beta")

        content_a = "- Use TypeScript\n- Use ESLint\n- Line A only\n- Line B only\n"
        content_b = "- Use TypeScript\n- Use ESLint\n- Line C only\n- Line D only\n"

        (store.PROJECTS_DIR / "alpha" / "style.md").write_text(content_a)
        (store.PROJECTS_DIR / "beta" / "style.md").write_text(content_b)

        # High threshold should exclude
        high = scan_for_promotions(similarity_threshold=0.9)
        match_high = [s for s in high if s.filename == "style.md"]
        assert len(match_high) == 0

        # Low threshold should include
        low = scan_for_promotions(similarity_threshold=0.1)
        match_low = [s for s in low if s.filename == "style.md"]
        assert len(match_low) == 1


class TestApplyPromotion:
    def test_apply_basic(self):
        content = "- Use TypeScript\n"
        store.create_project("alpha")
        store.create_project("beta")
        (store.PROJECTS_DIR / "alpha" / "style.md").write_text(content)
        (store.PROJECTS_DIR / "beta" / "style.md").write_text(content)

        suggestion = PromotionSuggestion(
            filename="style.md",
            projects=["alpha", "beta"],
            similarity=1.0,
            reason="test",
        )
        result = apply_promotion(suggestion, source_project="alpha")
        assert result is not None
        assert (store.GLOBAL_DIR / "style.md").exists()
        assert (store.GLOBAL_DIR / "style.md").read_text() == content

    def test_apply_with_cleanup(self):
        content = "- Use TypeScript\n"
        store.create_project("alpha")
        store.create_project("beta")
        (store.PROJECTS_DIR / "alpha" / "style.md").write_text(content)
        (store.PROJECTS_DIR / "beta" / "style.md").write_text(content)

        suggestion = PromotionSuggestion(
            filename="style.md",
            projects=["alpha", "beta"],
            similarity=1.0,
            reason="test",
        )
        result = apply_promotion(suggestion, source_project="alpha", remove_from_projects=True)
        assert result is not None
        assert (store.GLOBAL_DIR / "style.md").exists()
        assert not (store.PROJECTS_DIR / "alpha" / "style.md").exists()
        assert not (store.PROJECTS_DIR / "beta" / "style.md").exists()

    def test_apply_defaults_to_first_project(self):
        content = "- hello\n"
        store.create_project("alpha")
        store.create_project("beta")
        (store.PROJECTS_DIR / "alpha" / "test.md").write_text(content)
        (store.PROJECTS_DIR / "beta" / "test.md").write_text("- different\n")

        suggestion = PromotionSuggestion(
            filename="test.md",
            projects=["alpha", "beta"],
            similarity=0.5,
            reason="test",
        )
        result = apply_promotion(suggestion)
        assert result is not None
        assert (store.GLOBAL_DIR / "test.md").read_text() == content

    def test_apply_no_projects(self):
        suggestion = PromotionSuggestion(
            filename="orphan.md",
            projects=[],
            similarity=0.0,
            reason="test",
        )
        assert apply_promotion(suggestion) is None
