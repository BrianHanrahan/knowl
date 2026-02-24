"""Promotion engine — detect cross-project patterns and suggest promotions to global."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from knowl.context import store
from knowl.log import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip markdown headers, collapse whitespace."""
    text = re.sub(r"^#+\s+.*$", "", text, flags=re.MULTILINE)  # strip headings
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _extract_lines(text: str) -> set[str]:
    """Extract non-empty, non-heading lines as a set for comparison."""
    lines: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.add(stripped.lower())
    return lines


def line_overlap(text_a: str, text_b: str) -> float:
    """Return the fraction of overlapping lines between two texts (Jaccard)."""
    lines_a = _extract_lines(text_a)
    lines_b = _extract_lines(text_b)
    if not lines_a or not lines_b:
        return 0.0
    intersection = lines_a & lines_b
    union = lines_a | lines_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Suggestion data
# ---------------------------------------------------------------------------

@dataclass
class PromotionSuggestion:
    """A suggestion to promote project context to global scope."""
    filename: str
    projects: list[str]            # projects that contain similar files
    similarity: float              # 0.0 – 1.0 average similarity score
    reason: str                    # human-readable reason
    conflict: bool = False         # True if a global file with same name exists
    conflicting_global: str = ""   # name of the conflicting global file


# ---------------------------------------------------------------------------
# Analysis engine
# ---------------------------------------------------------------------------

# Minimum Jaccard similarity to consider files "similar"
SIMILARITY_THRESHOLD = 0.3

# Minimum number of projects with similar files to trigger a suggestion
MIN_PROJECTS = 2


def scan_for_promotions(
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    min_projects: int = MIN_PROJECTS,
) -> list[PromotionSuggestion]:
    """Scan all projects for context files that appear across multiple projects.

    Returns a list of PromotionSuggestion objects, sorted by similarity (desc).
    """
    projects = store.list_projects()
    if len(projects) < min_projects:
        return []

    # Collect all project files: {project_name: {filename: content}}
    project_files: dict[str, dict[str, str]] = {}
    for proj in projects:
        files = store.list_project_files(proj)
        file_map: dict[str, str] = {}
        for f in files:
            if f.name in ("context.json", "history.json"):
                continue
            try:
                content = f.read_text(encoding="utf-8")
                file_map[f.name] = content
            except OSError:
                continue
        project_files[proj] = file_map

    # Existing global files for conflict detection
    global_names = {f.name for f in store.list_global_files()}

    # Compare files across projects by name
    all_filenames: set[str] = set()
    for fm in project_files.values():
        all_filenames.update(fm.keys())

    suggestions: list[PromotionSuggestion] = []

    for filename in sorted(all_filenames):
        # Which projects have this filename?
        containing_projects = [
            proj for proj in projects if filename in project_files.get(proj, {})
        ]
        if len(containing_projects) < min_projects:
            continue

        # Compute pairwise similarity
        contents = [project_files[proj][filename] for proj in containing_projects]
        total_sim = 0.0
        pairs = 0
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                total_sim += line_overlap(contents[i], contents[j])
                pairs += 1

        avg_sim = total_sim / pairs if pairs > 0 else 0.0
        if avg_sim < similarity_threshold:
            continue

        conflict = filename in global_names
        reason = (
            f"'{filename}' appears in {len(containing_projects)} projects "
            f"with {avg_sim:.0%} similarity"
        )
        if conflict:
            reason += f" (conflicts with existing global/{filename})"

        suggestions.append(PromotionSuggestion(
            filename=filename,
            projects=containing_projects,
            similarity=avg_sim,
            reason=reason,
            conflict=conflict,
            conflicting_global=filename if conflict else "",
        ))

    # Also check for content similarity across differently-named files
    suggestions.extend(_scan_cross_name_similarities(
        project_files, projects, global_names, similarity_threshold, min_projects
    ))

    # Sort by similarity descending
    suggestions.sort(key=lambda s: s.similarity, reverse=True)
    return suggestions


def _scan_cross_name_similarities(
    project_files: dict[str, dict[str, str]],
    projects: list[str],
    global_names: set[str],
    threshold: float,
    min_projects: int,
) -> list[PromotionSuggestion]:
    """Find files with different names but similar content across projects."""
    suggestions: list[PromotionSuggestion] = []

    # Build a flat list of (project, filename, content) excluding context.json/history.json
    all_entries: list[tuple[str, str, str]] = []
    for proj in projects:
        for fname, content in project_files.get(proj, {}).items():
            all_entries.append((proj, fname, content))

    # Group by normalized content fingerprint to find clusters
    # Use a simpler approach: for each unique filename that only appears once,
    # check if its content matches files with different names in other projects
    seen_filenames: dict[str, list[tuple[str, str]]] = {}  # filename -> [(project, content)]
    for proj, fname, content in all_entries:
        seen_filenames.setdefault(fname, []).append((proj, content))

    # Only look at filenames that appear in exactly one project (unique names)
    unique_files = [
        (fname, entries[0]) for fname, entries in seen_filenames.items()
        if len(entries) == 1
    ]

    # Compare unique files against each other
    for i, (fname_a, (proj_a, content_a)) in enumerate(unique_files):
        matches: list[tuple[str, str]] = [(proj_a, fname_a)]
        for j, (fname_b, (proj_b, content_b)) in enumerate(unique_files):
            if i >= j or proj_a == proj_b:
                continue
            sim = line_overlap(content_a, content_b)
            if sim >= threshold:
                matches.append((proj_b, fname_b))

        if len(matches) >= min_projects:
            match_projects = [m[0] for m in matches]
            match_names = list({m[1] for m in matches})
            canonical = match_names[0]
            avg_sim = threshold  # conservative estimate
            conflict = canonical in global_names

            reason = (
                f"Similar content found across {len(match_projects)} projects "
                f"under names: {', '.join(match_names)}"
            )
            # Avoid duplicate suggestions
            suggestions.append(PromotionSuggestion(
                filename=canonical,
                projects=match_projects,
                similarity=avg_sim,
                reason=reason,
                conflict=conflict,
                conflicting_global=canonical if conflict else "",
            ))

    return suggestions


def apply_promotion(
    suggestion: PromotionSuggestion,
    source_project: str | None = None,
    remove_from_projects: bool = False,
) -> Path | None:
    """Apply a promotion suggestion — copy the best version to global.

    Args:
        suggestion: The suggestion to apply.
        source_project: Which project to use as the source. Defaults to first.
        remove_from_projects: If True, delete the file from all source projects.

    Returns:
        The path to the new global file, or None on failure.
    """
    source = source_project or (suggestion.projects[0] if suggestion.projects else None)
    if not source:
        return None

    result = store.promote_to_global(source, suggestion.filename)

    if result and remove_from_projects:
        for proj in suggestion.projects:
            proj_file = store.PROJECTS_DIR / proj / suggestion.filename
            if proj_file.exists():
                store.delete_context_file(proj_file)
                logger.info("Removed %s/%s after promotion", proj, suggestion.filename)

    return result
