"""fetch_page tool — extract readable text from a web page via Playwright."""

from __future__ import annotations

from knowl.log import get_logger
from knowl.web.browser import BrowserPool

logger = get_logger(__name__)

MAX_CONTENT_LENGTH = 12000


async def fetch_page(url: str) -> dict:
    """Navigate to a URL and extract readable text content.

    Returns {url, title, content} with content truncated to MAX_CONTENT_LENGTH.
    """
    pool = BrowserPool.get()
    ctx = await pool.new_context()
    try:
        page = await ctx.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)

        title = await page.title()

        content = await page.evaluate("""() => {
            // Try semantic containers first, fall back to body
            const container = document.querySelector('article')
                || document.querySelector('main')
                || document.querySelector('[role="main"]')
                || document.body;
            if (!container) return '';

            // Remove script, style, nav, footer, header elements
            const clone = container.cloneNode(true);
            for (const tag of ['script', 'style', 'nav', 'footer', 'header', 'aside']) {
                for (const el of clone.querySelectorAll(tag)) {
                    el.remove();
                }
            }
            return clone.innerText || '';
        }""")

        content = content.strip()
        if len(content) > MAX_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH] + "\n\n[Content truncated]"

        logger.info("fetch_page(%r) → %d chars", url, len(content))
        return {"url": url, "title": title, "content": content}
    except Exception as exc:
        logger.error("fetch_page failed: %s", exc)
        return {"url": url, "title": "Error", "content": f"Failed to fetch page: {exc}"}
    finally:
        await ctx.close()
