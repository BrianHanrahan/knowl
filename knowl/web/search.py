"""web_search tool — DuckDuckGo search via ddgs API (no browser needed)."""

from __future__ import annotations

from knowl.log import get_logger

logger = get_logger(__name__)


async def web_search(query: str, num_results: int = 5) -> list[dict]:
    """Search DuckDuckGo and return a list of {title, url, snippet} dicts.

    Uses the ddgs library which calls DDG's API directly — no headless
    browser required, so it avoids captcha/bot-detection issues.
    """
    import asyncio
    from ddgs import DDGS

    def _search():
        results = DDGS().text(query, max_results=num_results)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]

    try:
        # Run sync ddgs call in a thread to avoid blocking the event loop
        results = await asyncio.get_event_loop().run_in_executor(None, _search)
        logger.info("web_search(%r) → %d results", query, len(results))
        return results
    except Exception as exc:
        logger.error("web_search failed: %s", exc)
        return [{"title": "Search failed", "url": "", "snippet": str(exc)}]
