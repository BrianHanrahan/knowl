"""BrowserPool — singleton managing a persistent headless Playwright browser."""

from __future__ import annotations

import asyncio
from typing import Optional

from knowl.log import get_logger

logger = get_logger(__name__)


class BrowserPool:
    """Lazy-launched singleton for headless Chromium via Playwright async API."""

    _instance: Optional[BrowserPool] = None
    _lock = asyncio.Lock()

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None
        self._startup_lock = asyncio.Lock()

    @classmethod
    def get(cls) -> BrowserPool:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def ensure_browser(self):
        """Launch browser if not already running."""
        if self._browser and self._browser.is_connected():
            return
        async with self._startup_lock:
            if self._browser and self._browser.is_connected():
                return
            logger.info("Launching headless Chromium...")
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=False)
            logger.info("Chromium launched.")

    async def new_context(self):
        """Create an isolated BrowserContext. Caller must close it."""
        await self.ensure_browser()
        return await self._browser.new_context()

    async def shutdown(self):
        """Close browser and stop Playwright."""
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
        logger.info("Browser pool shut down.")
