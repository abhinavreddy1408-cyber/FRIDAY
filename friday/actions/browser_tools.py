"""
F.R.I.D.A.Y. — Browser Tools
==============================
Web automation via Playwright.

Features:
  - Persistent browser session (singleton)
  - Smart element finding (role → text → placeholder → CSS)
  - High-level macros: youtube_play, google_search
  - Self-healing selectors via OCR fallback
  - Headful (visible) for active tasks
  - Headless for background tasks
  - Context pool for parallel execution
  - Auto-recovery on browser crash

Risk levels:
  LOW:    navigate, read_page, screenshot, go_back, scroll
  MEDIUM: search, click
  HIGH:   type (could submit forms or enter passwords)

All functions return {"success": bool, "message": str, ...}
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from friday import config

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Locator, Page, Playwright

# Maximum number of concurrent browser contexts
_MAX_BROWSER_CONTEXTS = 5

logger = logging.getLogger(__name__)
_BROWSER_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="browser-tools")


# ── Exceptions ─────────────────────────────────────────────────────────────


class SelectorHealingRequired(Exception):
    """
    Raised when a target element cannot be found via standard selectors
    and requires visual (VLM) healing to continue.
    """

    def __init__(self, target_description: str):
        self.target_description = target_description
        super().__init__(f"Selector healing required for: {target_description}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _result(success: bool, message: str = "", **kwargs: Any) -> dict[str, Any]:
    return {"success": success, "message": message, **kwargs}


F = TypeVar("F", bound=Callable[..., Any])


def _browser_thread(fn: F) -> F:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return _BROWSER_EXECUTOR.submit(fn, *args, **kwargs).result()

    return wrapper  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# KNOWN SELECTORS (pre-loaded for common sites)
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_SELECTORS = {
    "youtube.com": {
        "search_input": [
            "input#search",
            "input[name='search_query']",
            "[placeholder='Search']",
        ],
        "search_button": [
            "button#search-icon-legacy",
            "[aria-label='Search']",
        ],
        "first_video": [
            "ytd-video-renderer a#video-title",
            "#contents ytd-video-renderer:first-child a",
            "a#video-title",
        ],
    },
    "google.com": {
        "search_input": [
            "textarea[name='q']",
            "input[name='q']",
            "[title='Search']",
        ],
    },
    "gmail.com": {
        "compose": [
            "[gh='cm']",
            "[data-tooltip='Compose']",
            ".T-I.T-I-KE",
        ],
        "search_input": [
            "input[aria-label='Search mail']",
            "#gs",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# BROWSER SESSION MANAGER
# ─────────────────────────────────────────────────────────────────────────────


class _BrowserSession:
    """
    Singleton Playwright browser session.
    Maintains one persistent browser instance.
    Supports multiple named contexts for parallel tasks.
    """

    _instance: _BrowserSession | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._contexts: dict[str, BrowserContext] = {}
        self._pages: dict[str, Page] = {}
        self._ready = False
        self._init_lock = threading.Lock()

    @classmethod
    def get(cls) -> _BrowserSession:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_browser(self, headless: bool = False) -> bool:
        """Start the browser if not already running."""
        with self._init_lock:
            if self._ready and self._browser:
                try:
                    # Quick health check
                    _ = self._browser.contexts
                    return True
                except Exception:
                    logger.warning("[BROWSER] Session crashed. Restarting.")
                    self._ready = False

            try:
                from playwright.sync_api import sync_playwright

                if self._playwright:
                    with contextlib.suppress(Exception):
                        self._playwright.stop()

                self._playwright = sync_playwright().start()
                self._browser = self._playwright.chromium.launch(
                    headless=headless,
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-blink-features=AutomationControlled",
                    ],
                )
                self._ready = True
                logger.info(f"[BROWSER] Started ({'headless' if headless else 'headful'}).")
                return True

            except ImportError:
                logger.error(
                    "[BROWSER] Playwright not installed. "
                    "Run: pip install playwright && "
                    "playwright install chromium"
                )
                return False
            except Exception as e:
                logger.error(f"[BROWSER] Failed to start: {e}")
                return False

    def get_page(
        self,
        task_id: str = "main",
        headless: bool = False,
    ) -> Page | None:
        """
        Get or create a browser page for a task.
        Each task_id gets its own isolated context.
        """
        # Config imported above

        # Enforce context limit
        active = len(self._contexts)
        if task_id not in self._contexts and active >= _MAX_BROWSER_CONTEXTS:
            logger.warning(f"[BROWSER] Context limit ({_MAX_BROWSER_CONTEXTS}) reached.")
            return None

        if not self._ensure_browser(headless):
            return None
        assert self._browser is not None

        try:
            # Reuse existing context
            if task_id in self._contexts:
                try:
                    page = self._pages.get(task_id)
                    if page and not page.is_closed():
                        return page
                except Exception:
                    pass

            # Create new context and page
            context = self._browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                ignore_https_errors=True,
            )
            page = context.new_page()
            page.set_default_timeout(15000)

            self._contexts[task_id] = context
            self._pages[task_id] = page

            logger.debug(f"[BROWSER] Created context for '{task_id}'.")
            return page

        except Exception as e:
            logger.error(f"[BROWSER] Could not get page for '{task_id}': {e}")
            return None

    def release(self, task_id: str) -> None:
        """Close and remove a task's browser context."""
        try:
            if task_id in self._contexts:
                self._contexts[task_id].close()
                del self._contexts[task_id]
            if task_id in self._pages:
                del self._pages[task_id]
            logger.debug(f"[BROWSER] Released context '{task_id}'.")
        except Exception as e:
            logger.warning(f"[BROWSER] Release failed: {e}")

    def release_all(self) -> None:
        """Close all browser contexts and the browser itself."""
        for task_id in list(self._contexts.keys()):
            self.release(task_id)
        try:
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._ready = False
        logger.info("[BROWSER] All contexts released.")

    def current_url(self, task_id: str = "main") -> str | None:
        """Get current URL of a task's page."""
        try:
            page = self._pages.get(task_id)
            return page.url if page else None
        except Exception:
            return None

    def current_title(self, task_id: str = "main") -> str | None:
        """Get current page title of a task's page."""
        try:
            page = self._pages.get(task_id)
            return page.title() if page else None
        except Exception:
            return None


# Global session instance
_session = _BrowserSession.get()


# ─────────────────────────────────────────────────────────────────────────────
# SMART ELEMENT FINDER
# ─────────────────────────────────────────────────────────────────────────────


def _find_element(
    page: Page,
    domain: str,
    purpose: str,
    fallback_selectors: list[str] | None = None,
    description: str = "",
) -> Locator | None:
    """
    Find a page element using multiple strategies in priority order:
    1. Known selectors for this domain/purpose
    2. Accessible role + name
    3. Visible text
    4. Placeholder text
    5. Fallback CSS selectors
    6. OCR healing (if enabled and all else fails)

    Returns Playwright locator or None.
    """

    all_selectors = []

    # 1. Known selectors for this site
    domain_key = next((k for k in KNOWN_SELECTORS if k in domain), None)
    if domain_key and purpose in KNOWN_SELECTORS[domain_key]:
        all_selectors.extend(KNOWN_SELECTORS[domain_key][purpose])

    # 2. Fallback selectors
    if fallback_selectors:
        all_selectors.extend(fallback_selectors)

    # Try each selector
    for selector in all_selectors:
        try:
            locator = page.locator(selector).first
            if locator.count() > 0:
                locator.wait_for(state="visible", timeout=3000)
                logger.debug(f"[BROWSER] Found element with: {selector}")
                return locator
        except Exception:
            continue

    # 3. Try by role if description provided
    if description:
        roles: list[Literal["textbox", "button", "link", "searchbox"]] = [
            "textbox",
            "button",
            "link",
            "searchbox",
        ]
        for role in roles:
            try:
                locator = page.get_by_role(role, name=description).first
                if locator.count() > 0:
                    return locator
            except Exception:
                continue

        # 4. Try by placeholder
        try:
            locator = page.get_by_placeholder(description).first
            if locator.count() > 0:
                return locator
        except Exception:
            pass

        # 5. Try by text
        try:
            locator = page.get_by_text(description).first
            if locator.count() > 0:
                return locator
        except Exception:
            pass

    # 6. Fallback: Visual Healing
    if description:
        logger.info(f"[BROWSER] Selector failure for '{description}'. Triggering VLM healing.")
        raise SelectorHealingRequired(description)

    logger.warning(f"[BROWSER] Element not found: domain={domain} purpose={purpose}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL MACROS
# ─────────────────────────────────────────────────────────────────────────────


@_browser_thread
def youtube_play(
    query: str,
    task_id: str = "main",
) -> dict[str, Any]:
    """
    Search YouTube and play the first result.
    Risk: MEDIUM

    Args:
        query:   Search term e.g. "lofi music"
        task_id: Context ID for parallel execution
    """
    page = _session.get_page(task_id, headless=False)
    if not page:
        return _result(False, _playwright_not_installed())

    try:
        # Navigate to YouTube
        page.goto("https://www.youtube.com", timeout=15000)
        page.wait_for_load_state("domcontentloaded")

        # Find and use search box
        search = _find_element(page, "youtube.com", "search_input", description="Search")
        if not search:
            return _result(False, "Could not find YouTube search bar, Sir.")

        search.click()
        search.fill(query)
        page.keyboard.press("Enter")

        # Wait for results
        page.wait_for_load_state("networkidle", timeout=10000)

        # Click first video
        first = _find_element(page, "youtube.com", "first_video", description="video")
        if first:
            title = first.inner_text().strip()
            first.click()
            page.wait_for_load_state("domcontentloaded")
            return _result(
                True,
                f"Playing '{title}' on YouTube, Sir.",
                video_title=title,
                url=page.url,
            )

        # Fallback: just show search results
        return _result(
            True,
            f"Showing YouTube results for '{query}', Sir.",
            url=page.url,
        )

    except Exception as e:
        logger.error(f"[BROWSER] youtube_play failed: {e}")
        return _result(False, f"YouTube failed: {e}")


@_browser_thread
def google_search(
    query: str,
    task_id: str = "main",
) -> dict[str, Any]:
    """
    Perform a Google search and return top results.
    Risk: MEDIUM

    Args:
        query:   Search query string
        task_id: Context ID for parallel execution
    """
    page = _session.get_page(task_id, headless=False)
    if not page:
        return _result(False, _playwright_not_installed())

    try:
        page.goto("https://www.google.com", timeout=15000)
        page.wait_for_load_state("domcontentloaded")

        # Handle cookie consent if present
        for consent_selector in [
            "button[id='L2AGLb']",
            "[aria-label='Accept all']",
            "button:has-text('Accept')",
        ]:
            try:
                btn = page.locator(consent_selector).first
                if btn.count() > 0:
                    btn.click()
                    time.sleep(0.5)
                    break
            except Exception:
                pass

        # Find search box
        search = _find_element(page, "google.com", "search_input", description="Search")
        if not search:
            return _result(False, "Could not find Google search bar, Sir.")

        search.click()
        search.fill(query)
        page.keyboard.press("Enter")

        page.wait_for_load_state("networkidle", timeout=10000)

        # Extract top results
        results = []
        try:
            result_items = page.locator("div.g").all()[:5]
            for item in result_items:
                try:
                    title = item.locator("h3").first.inner_text()
                    link = item.locator("a").first.get_attribute("href")
                    if title and link and link.startswith("http"):
                        results.append(
                            {
                                "title": title.strip(),
                                "url": link,
                            }
                        )
                except Exception:
                    continue
        except Exception:
            pass

        if not results:
            return _result(
                True,
                f"Search completed for '{query}', but no result links were extracted.",
                url=page.url,
            )

        return _result(
            True,
            f"Found {len(results)} results for '{query}', Sir.",
            results=results,
            url=page.url,
        )

    except Exception as e:
        logger.error(f"[BROWSER] google_search failed: {e}")
        return _result(False, f"Search failed: {e}")


def _playwright_not_installed() -> str:
    return "Playwright is not correctly installed or configured, Sir. Please run 'playwright install chromium'."


# ─────────────────────────────────────────────────────────────────────────────
# CORE BROWSER TOOLS
# ─────────────────────────────────────────────────────────────────────────────


@_browser_thread
def browser_navigate(url: str, task_id: str = "main") -> dict[str, Any]:
    """Navigate to a URL. Risk: LOW"""
    page = _session.get_page(task_id)
    if not page:
        return _result(False, _playwright_not_installed())

    try:
        clean_url = url if url.startswith("http") else f"https://{url}"
        page.goto(clean_url, wait_until="domcontentloaded", timeout=15000)
        return _result(True, f"Navigated to {page.title()}", url=page.url)
    except Exception as e:
        return _result(False, f"Navigation failed: {e}")


@_browser_thread
def browser_click(description: str, task_id: str = "main") -> dict[str, Any]:
    """Find and click an element. Risk: MEDIUM"""
    page = _session.get_page(task_id)
    if not page:
        return _result(False, _playwright_not_installed())

    try:
        domain = _session.current_url(task_id) or ""
        el = _find_element(page, domain, "click", description=description)

        if el:
            el.click()
            page.wait_for_load_state("domcontentloaded", timeout=3000)
            return _result(True, f"Clicked '{description}', Sir.")

        return _result(False, f"Could not find '{description}' to click.")
    except SelectorHealingRequired:
        raise
    except Exception as e:
        return _result(False, f"Click failed: {e}")


@_browser_thread
def browser_type(
    description: str, text: str, task_id: str = "main", press_enter: bool = True
) -> dict[str, Any]:
    """Type into an input field. Risk: HIGH"""
    page = _session.get_page(task_id)
    if not page:
        return _result(False, _playwright_not_installed())

    try:
        domain = _session.current_url(task_id) or ""
        el = _find_element(page, domain, "input", description=description)

        if el:
            el.fill(text)
            if press_enter:
                page.keyboard.press("Enter")
                page.wait_for_load_state("domcontentloaded", timeout=3000)
            return _result(True, f"Typed into '{description}'.")

        return _result(False, f"Could not find input field '{description}'.")
    except SelectorHealingRequired:
        raise
    except Exception as e:
        return _result(False, f"Typing failed: {e}")


@_browser_thread
def browser_read_current_page(task_id: str = "main") -> dict[str, Any]:
    """Read the visible text and title of the page. Risk: LOW"""
    page = _session.get_page(task_id)
    if not page:
        return _result(False, _playwright_not_installed())

    try:
        title = page.title()
        # Simple text extraction
        text = page.evaluate("document.body.innerText")
        # Clean up whitespace
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])[:2000]

        return _result(True, "Page read successfully.", title=title, content=text)
    except Exception as e:
        return _result(False, f"Read failed: {e}")


@_browser_thread
def browser_click_coordinates(x: int, y: int, task_id: str = "main") -> dict[str, Any]:
    """Click specific coordinates on the page. Risk: MEDIUM"""
    page = _session.get_page(task_id)
    if not page:
        return _result(False, _playwright_not_installed())

    try:
        # Playwright mouse actions use logical pixels (ignores DPI scaling)
        page.mouse.click(x, y)
        return _result(True, f"Clicked coordinates [{x}, {y}].")
    except Exception as e:
        return _result(False, f"Coordinate click failed: {e}")


@_browser_thread
def browser_take_viewport_screenshot(task_id: str = "main") -> dict[str, Any]:
    """Capture only the current viewport as a JPEG. Risk: LOW"""
    page = _session.get_page(task_id)
    if not page:
        return _result(False, _playwright_not_installed())

    try:
        # Quality 50 JPEG for VLM performance
        img_bytes = page.screenshot(type="jpeg", quality=50)
        return _result(True, "Viewport screenshot captured.", image_bytes=img_bytes)
    except Exception as e:
        return _result(False, f"Viewport screenshot failed: {e}")


@_browser_thread
def browser_take_screenshot(task_id: str = "main", save_path: str | None = None) -> dict[str, Any]:
    """Capture a screenshot of the current page. Risk: LOW"""
    # Config imported above

    page = _session.get_page(task_id)
    if not page:
        return _result(False, _playwright_not_installed())

    try:
        os.makedirs(config.storage.data_dir / "temp", exist_ok=True)
        path = save_path or str((config.storage.data_dir / "temp") / f"screenshot_{task_id}_{int(time.time())}.png")
        page.screenshot(path=path)
        return _result(True, f"Screenshot saved to {path}", path=path)
    except Exception as e:
        return _result(False, f"Screenshot failed: {e}")


@_browser_thread
def browser_go_back(task_id: str = "main") -> dict[str, Any]:
    """Navigate back in history. Risk: LOW"""
    page = _session.get_page(task_id)
    if not page:
        return _result(False, _playwright_not_installed())
    try:
        page.go_back()
        return _result(True, "Navigated back.")
    except Exception as e:
        return _result(False, f"Go back failed: {e}")


@_browser_thread
def browser_scroll(direction: str = "down", task_id: str = "main") -> dict[str, Any]:
    """Scroll the page. Risk: LOW"""
    page = _session.get_page(task_id)
    if not page:
        return _result(False, _playwright_not_installed())
    try:
        amount = 500 if direction == "down" else -500
        page.evaluate(f"window.scrollBy(0, {amount})")
        return _result(True, f"Scrolled {direction}.")
    except Exception as e:
        return _result(False, f"Scroll failed: {e}")


@_browser_thread
def browser_close(task_id: str = "main") -> dict[str, Any]:
    """Close the current task context."""
    _session.release(task_id)
    return _result(True, f"Browser for '{task_id}' closed.")


@_browser_thread
def browser_get_current_url(task_id: str = "main") -> dict[str, Any]:
    """Get the current URL and title."""
    url = _session.current_url(task_id)
    title = _session.current_title(task_id)
    return _result(True, f"At {url}", url=url, title=title)


@_browser_thread
def browser_shutdown() -> None:
    """Full browser cleanup."""
    _session.release_all()


if __name__ == "__main__":
    # Simple self-test
    logging.basicConfig(level=logging.INFO)
    print("Testing Google search...")
    res = google_search("NVIDIA Nemotron")
    print(f"Success: {res['success']}")
    if res["success"]:
        for r in res.get("results", []):
            print(f" - {r['title']} ({r['url']})")
    browser_shutdown()
