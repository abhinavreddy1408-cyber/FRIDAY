"""
F.R.I.D.A.Y. Browser Skill
Provides Playwright-based browser automation for advanced web interactions
beyond simple URL opening.

Supported commands:
  - "go to [url]"
  - "search [query] on [engine]"
  - "click [element]"
  - "scroll down / up"
  - "fill in [field] with [value]"
  - "read page / what does this page say"
  - "take a page screenshot"
  - "go back / forward"
  - "close tab / browser"
  - "open new tab"
  - "switch to tab [number]"
  - "zoom in / out"

All browser operations run in a managed Playwright context.
The browser launches headfully by default so the user can see activity.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import BaseModel, Field

from friday.actions import Action, ActionResult, ActionStatus
from friday.config import config

try:
    from friday.actions.browser_tools import SelectorHealingRequired
except ImportError:
    class SelectorHealingRequired(Exception):  # type: ignore[no-redef]
        def __init__(self, target_description: str):
            self.target_description = target_description
            super().__init__(f"Selector healing required for: {target_description}")
import contextlib

from friday.memory import MemoryCategory, MemoryManager, MemoryTier, get_memory

if TYPE_CHECKING:
    from collections.abc import Callable

_T = TypeVar("_T")

logger = logging.getLogger(__name__)


# ── Optional Playwright import ────────────────────────────────────────────────

try:
    from playwright.sync_api import (
        Browser,
        BrowserContext,
        Page,
        Playwright,
        sync_playwright,
    )
    from playwright.sync_api import (
        TimeoutError as PlaywrightTimeout,
    )

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False
    logger.warning(
        "[Browser] playwright not installed — browser skill disabled. "
        "Run: pip install playwright && playwright install chromium"
    )


# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_TIMEOUT = 25_000  # ms
_NAVIGATION_TIMEOUT = 45_000  # ms
_SCREENSHOT_DIR = config.storage.base_dir / "screenshots" / "browser"
_MAX_PAGE_TEXT = 2_000  # chars for TTS read-page


# ── Enums ─────────────────────────────────────────────────────────────────────


class BrowserStatus(Enum):
    SUCCESS = auto()
    NOT_READY = auto()
    NAVIGATION_ERROR = auto()
    ELEMENT_ERROR = auto()
    TIMEOUT = auto()
    EXECUTION_ERROR = auto()
    PRIVACY_BLOCKED = auto()


class BrowserEngine(Enum):
    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PageInfo:
    """
    Snapshot of the current browser page state.

    Attributes:
        url:     Current page URL.
        title:   Page title.
        text:    Visible text content (truncated to _MAX_PAGE_TEXT).
        tab_idx: Index of this tab in the browser context.
    """

    url: str
    title: str
    text: str
    tab_idx: int

    def to_speech(self) -> str:
        """Format page info as a natural TTS string."""
        return f"Currently on '{self.title}'. URL: {self.url}."


@dataclass
class BrowserConfig:
    """
    Configuration for the managed browser instance.

    Attributes:
        engine:    Browser engine to use.
        headless:  Run without a visible window.
        slow_mo:   Milliseconds between actions (useful for demos).
        user_agent: Override the default user agent string.
        viewport:  Browser viewport dimensions.
    """

    engine: BrowserEngine = BrowserEngine.CHROMIUM
    headless: bool = True
    slow_mo: int = 0
    user_agent: str | None = None
    viewport: dict[str, int] = field(default_factory=lambda: {"width": 1280, "height": 720})


# ── Agentic Models ────────────────────────────────────────────────────────────


class BrowserAction(BaseModel):
    """A single atomic action to be performed in the browser."""

    tool: Literal["navigate", "click", "fill", "scroll", "read_text", "screenshot", "finish"]
    selector: str | None = Field(None, description="CSS selector or element description")
    value: str | None = Field(None, description="Text value for fill actions")
    url: str | None = Field(None, description="Target URL for navigate actions")
    direction: Literal["up", "down"] | None = Field(None, description="Scroll direction")


class BrowserPlan(BaseModel):
    """A sequence of actions to be performed by the executor."""

    actions: list[BrowserAction]
    reasoning: str | None = Field(None, description="LLM's internal monologue")


# ── Search Engine Registry ────────────────────────────────────────────────────

_SEARCH_ENGINES: dict[str, str] = {
    "google": "https://www.google.com/search?q={}",
    "bing": "https://www.bing.com/search?q={}",
    "duckduckgo": "https://duckduckgo.com/?q={}",
    "ddg": "https://duckduckgo.com/?q={}",
    "yahoo": "https://search.yahoo.com/search?p={}",
    "youtube": "https://www.youtube.com/results?search_query={}",
    "reddit": "https://www.reddit.com/search/?q={}",
    "github": "https://github.com/search?q={}",
    "wikipedia": "https://en.wikipedia.org/wiki/Special:Search?search={}",
}


# ── URL Utilities ─────────────────────────────────────────────────────────────


def _normalise_url(url: str) -> str:
    """
    Ensure a URL has a valid scheme.
    Adds https:// if no scheme is present.

    Args:
        url: Raw URL string from voice input.

    Returns:
        URL with a valid scheme.
    """
    url = url.strip()
    if re.match(r"^https?://", url):
        return url
    if "." in url and " " not in url:
        return f"https://{url}"
    return f"https://www.google.com/search?q={url.replace(' ', '+')}"


def _build_search_url(query: str, engine: str = "google") -> str:
    """
    Build a search URL for the given engine and query.

    Args:
        query:  Search query string.
        engine: Search engine key from _SEARCH_ENGINES.

    Returns:
        Complete search URL.
    """
    template = _SEARCH_ENGINES.get(engine.lower(), _SEARCH_ENGINES["google"])
    return template.format(query.replace(" ", "+"))


# ── Browser Manager ───────────────────────────────────────────────────────────


class _BrowserManager:
    """
    Manages the Playwright browser lifecycle.

    Maintains a single browser instance with a persistent context.
    Pages (tabs) are tracked in a list and accessed by index.
    """

    def __init__(self, config: BrowserConfig, run_in_thread: Callable[..., Any]) -> None:
        self._config: BrowserConfig = config
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._pages: list[Page] = []
        self._active_idx: int = 0
        self._ready: bool = False
        self._run_in_thread_func = run_in_thread

    def _run_in_thread(self, fn: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
        return self._run_in_thread_func(fn, *args, **kwargs)  # type: ignore[no-any-return]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def launch(self) -> bool:
        """
        Start the Playwright instance and launch the browser.

        Returns:
            True if the browser launched successfully.
        """
        if not _PLAYWRIGHT_AVAILABLE:
            return False

        try:
            self._playwright = sync_playwright().start()
            engine_obj = getattr(self._playwright, self._config.engine.value)

            launch_args: dict[str, Any] = {
                "headless": self._config.headless,
                "slow_mo": self._config.slow_mo,
            }

            self._browser = engine_obj.launch(**launch_args)

            context_args: dict[str, Any] = {
                "viewport": self._config.viewport,
            }
            if self._config.user_agent:
                context_args["user_agent"] = self._config.user_agent

            self._context = self._browser.new_context(**context_args)
            self._context.set_default_timeout(_DEFAULT_TIMEOUT)
            self._context.set_default_navigation_timeout(_NAVIGATION_TIMEOUT)

            # Open an initial blank page
            page = self._context.new_page()
            self._pages = [page]
            self._active_idx = 0
            self._ready = True

            logger.info(
                "[Browser] Launched %s (headless=%s).",
                self._config.engine.value,
                self._config.headless,
            )
            return True

        except Exception as exc:
            logger.error("[Browser] Launch failed: %s", exc)
            self._cleanup()
            return False

    def close(self) -> None:
        """Close the browser and release all Playwright resources."""
        self._ready = False
        self._cleanup()

    def _cleanup(self) -> None:
        """Release all resources in reverse order."""
        try:
            if self._context:
                self._context.close()
        except Exception:
            pass
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._playwright = None
        self._browser = None
        self._context = None
        self._pages = []

    # ── Page Access ───────────────────────────────────────────────────────────

    @property
    def active_page(self) -> Page | None:
        """Return the currently active page."""
        if not self._pages or self._active_idx >= len(self._pages):
            return None
        return self._pages[self._active_idx]

    @property
    def active_page_url(self) -> str:
        """Return the URL of the active page."""
        def _get() -> str:
            page = self.active_page
            return page.url if page else "about:blank"
        return self._run_in_thread(_get)

    @property
    def active_page_title(self) -> str:
        """Return the title of the active page."""
        def _get() -> str:
            page = self.active_page
            return page.title() if page else ""
        return self._run_in_thread(_get)

    def type_and_enter(self, value: str) -> None:
        """Type a value on the keyboard and press Enter on the active page."""
        def _type() -> None:
            page = self.active_page
            if page:
                page.keyboard.type(value)
                page.keyboard.press("Enter")
        self._run_in_thread(_type)

    @property
    def page_count(self) -> int:
        return self._run_in_thread(lambda: len(self._pages))

    @property
    def is_ready(self) -> bool:
        return self._run_in_thread(lambda: self._ready and self.active_page is not None)

    # ── Tab Management ────────────────────────────────────────────────────────

    def new_tab(self, url: str | None = None) -> Page | None:
        """
        Open a new browser tab.

        Args:
            url: Optional URL to navigate to immediately.

        Returns:
            The new Page object or None on failure.
        """
        def _new_tab() -> Page | None:
            if not self._context:
                return None
            try:
                page = self._context.new_page()
                self._pages.append(page)
                self._active_idx = len(self._pages) - 1
                if url:
                    page.goto(url, wait_until="domcontentloaded")
                return page
            except Exception as exc:
                logger.warning("[Browser] new_tab failed: %s", exc)
                return None
        return self._run_in_thread(_new_tab)

    def close_tab(self) -> bool:
        """Close the active tab. Returns True if closed."""
        def _close_tab() -> bool:
            page = self.active_page
            if not page:
                return False
            try:
                page.close()
                self._pages.pop(self._active_idx)
                self._active_idx = max(0, self._active_idx - 1)
                return True
            except Exception as exc:
                logger.warning("[Browser] close_tab failed: %s", exc)
                return False
        return self._run_in_thread(_close_tab)

    def switch_tab(self, index: int) -> bool:
        """
        Switch to a tab by 1-based index.

        Args:
            index: 1-based tab number.

        Returns:
            True if switched successfully.
        """
        def _switch_tab() -> bool:
            idx = index - 1
            if 0 <= idx < len(self._pages):
                self._active_idx = idx
                try:
                    self._pages[idx].bring_to_front()
                    return True
                except Exception:
                    return False
            return False
        return self._run_in_thread(_switch_tab)

    # ── Navigation ────────────────────────────────────────────────────────────

    def navigate(self, url: str) -> bool:
        """
        Navigate the active page to a URL.

        Args:
            url: Target URL (must have a valid scheme).

        Returns:
            True on successful navigation.
        """
        def _navigate() -> bool:
            page = self.active_page
            if not page:
                return False
            try:
                page.goto(url, wait_until="domcontentloaded")
                return True
            except PlaywrightTimeout:
                logger.warning("[Browser] Navigation timeout for '%s'.", url)
                return False
            except Exception as exc:
                logger.warning("[Browser] Navigation failed: %s", exc)
                return False
        return self._run_in_thread(_navigate)

    def go_back(self) -> bool:
        def _go_back() -> bool:
            page = self.active_page
            if not page:
                return False
            try:
                page.go_back(wait_until="domcontentloaded")
                return True
            except Exception:
                return False
        return self._run_in_thread(_go_back)

    def go_forward(self) -> bool:
        def _go_forward() -> bool:
            page = self.active_page
            if not page:
                return False
            try:
                page.go_forward(wait_until="domcontentloaded")
                return True
            except Exception:
                return False
        return self._run_in_thread(_go_forward)

    def reload(self) -> bool:
        def _reload() -> bool:
            page = self.active_page
            if not page:
                return False
            try:
                page.reload(wait_until="domcontentloaded")
                return True
            except Exception:
                return False
        return self._run_in_thread(_reload)

    # ── Page Interaction ──────────────────────────────────────────────────────

    def click(self, selector: str) -> bool:
        """
        Click an element matching a CSS selector or text.

        Args:
            selector: CSS selector or text to click.

        Returns:
            True if clicked successfully.
        """
        def _click() -> bool:
            page = self.active_page
            if not page:
                return False
            try:
                # Try text match first, then CSS selector
                if page.get_by_text(selector, exact=False).count() > 0:
                    page.get_by_text(selector, exact=False).first.click()
                else:
                    page.click(selector)
                return True
            except Exception as exc:
                logger.warning("[Browser] Click failed for '%s': %s", selector, exc)
                return False
        return self._run_in_thread(_click)

    def fill(self, field_selector: str, value: str) -> bool:
        """
        Fill a form field with a value.

        Args:
            field_selector: CSS selector, label text, or placeholder.
            value:          Text to fill in.

        Returns:
            True if filled successfully.
        """
        def _fill() -> bool:
            page = self.active_page
            if not page:
                return False
            try:
                # Try label first, then placeholder, then CSS
                locators = [
                    page.get_by_label(field_selector, exact=False),
                    page.get_by_placeholder(field_selector, exact=False),
                ]
                for locator in locators:
                    if locator.count() > 0:
                        locator.first.fill(value)
                        return True
                page.fill(field_selector, value)
                return True
            except Exception as exc:
                logger.warning("[Browser] Fill failed for '%s': %s", field_selector, exc)
                return False
        return self._run_in_thread(_fill)

    def scroll(self, direction: str, amount: int = 300) -> bool:
        """
        Scroll the active page.

        Args:
            direction: "up" or "down".
            amount:    Pixels to scroll.

        Returns:
            True if scrolled successfully.
        """
        def _scroll() -> bool:
            page = self.active_page
            if not page:
                return False
            delta = amount if direction == "down" else -amount
            try:
                page.mouse.wheel(0, delta)
                return True
            except Exception as exc:
                logger.warning("[Browser] Scroll failed: %s", exc)
                return False
        return self._run_in_thread(_scroll)

    def get_page_text(self) -> str:
        """Extract visible text content from the active page."""
        def _get_text() -> str:
            page = self.active_page
            if not page:
                return ""
            try:
                return page.inner_text("body") or ""
            except Exception:
                return ""
        return self._run_in_thread(_get_text)

    def get_page_info(self) -> PageInfo | None:
        """Return a PageInfo snapshot of the current page."""
        def _get_info() -> PageInfo | None:
            page = self.active_page
            if not page:
                return None
            try:
                text = page.inner_text("body") or ""
                return PageInfo(
                    url=page.url,
                    title=page.title() or "Untitled",
                    text=text[:_MAX_PAGE_TEXT],
                    tab_idx=self._active_idx,
                )
            except Exception as exc:
                logger.debug("[Browser] get_page_info failed: %s", exc)
                return None
        return self._run_in_thread(_get_info)

    def screenshot(self, path: str) -> bool:
        """
        Save a screenshot of the active page.

        Args:
            path: File path for the saved PNG.

        Returns:
            True if saved successfully.
        """
        def _screenshot() -> bool:
            page = self.active_page
            if not page:
                return False
            try:
                page.screenshot(path=path, full_page=False)
                return True
            except Exception as exc:
                logger.warning("[Browser] Screenshot failed: %s", exc)
                return False
        return self._run_in_thread(_screenshot)

    def zoom(self, factor: float) -> bool:
        """
        Apply CSS zoom to the page body.

        Args:
            factor: Zoom multiplier (e.g. 1.25 = 125%).

        Returns:
            True if zoom was applied.
        """
        def _zoom() -> bool:
            page = self.active_page
            if not page:
                return False
            try:
                page.evaluate(f"document.body.style.zoom = '{factor}'")
                return True
            except Exception:
                return False
        return self._run_in_thread(_zoom)


# ── Web Agent Prompt ──────────────────────────────────────────────────────────

WEB_AGENT_PROMPT = """
You are F.R.I.D.A.Y.'s Web Automation Specialist.
Your goal is to achieve the user's request by providing a sequence of browser actions.

STRICT OUTPUT FORMAT:
You must output ONLY a valid JSON object matching this schema:
{{
  "reasoning": "Brief explanation of what you are doing",
  "actions": [
    {{"tool": "navigate", "url": "..."}},
    {{"tool": "click", "selector": "..."}},
    {{"tool": "fill", "selector": "...", "value": "..."}},
    {{"tool": "read_text", "selector": "..."}},
    {{"tool": "finish"}}
  ]
}}

CONTEXT:
- Current URL: {current_url}
- User Goal: {user_goal}
- Last Error: {last_error}
- Observed Text: {observed_text}

GUIDELINES:
1. If the goal is reached, include the "finish" tool.
2. Use stable CSS selectors. Avoid volatile IDs if possible.
3. If you see an error, pivot your strategy.
4. Output ONLY the JSON. No markdown formatting, no conversational filler.
"""

# ── Browser Skill ─────────────────────────────────────────────────────────────


class BrowserSkill:
    """
    Playwright-based browser automation skill for F.R.I.D.A.Y.

    Manages a single browser instance and exposes voice-controlled
    navigation, interaction, and content reading capabilities.

    Usage:
        skill = BrowserSkill()
        skill.setup()
        skill.register(action_router)
    """

    def __init__(self, config: BrowserConfig | None = None, llm: Any = None) -> None:
        self._config = config or BrowserConfig()
        self._browser_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="playwright-worker"
        )
        self._browser_loop = None
        self._manager = _BrowserManager(self._config, run_in_thread=self._run_in_browser_thread)
        self._ready = False
        self._zoom_level: float = 1.0
        self._memory: MemoryManager | None = None
        self._llm = llm  # Injected from engine — never create LLMBrain internally

    def _run_in_browser_thread(self, fn: Callable[..., Any], *args: Any) -> Any:
        future = self._browser_executor.submit(fn, *args)
        return future.result(timeout=30)

    @property
    def current_url(self) -> str:
        """Return the URL of the active browser page."""
        return str(self._manager.active_page_url)

    # ── Lifecycle ───────────────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """
        Launch the browser and prepare the skill.

        Returns:
            True if the browser launched successfully.
        """
        if not _PLAYWRIGHT_AVAILABLE:
            logger.warning("[Browser] Skill unavailable — playwright not installed.")
            return False

        self._ready = self._browser_executor.submit(self._manager.launch).result()
        if self._ready:
            _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
            try:
                self._memory = get_memory()
            except RuntimeError:
                logger.warning("[Browser] Memory manager uninitialized.")
            logger.info("[Browser] Browser skill ready.")
        return self._ready

    def teardown(self) -> None:
        """Close the browser and release all resources."""
        self._browser_executor.submit(self._manager.close).result()
        self._browser_executor.shutdown(wait=True)
        self._ready = False
        logger.info("[Browser] Browser closed.")

    def register(self, router: Any) -> None:
        """Register all browser actions with the ActionRouter."""
        actions = [
            Action(
                name="browser_navigate",
                patterns=[
                    r"\b(go\s+to|navigate\s+to|open|visit)\s+(?P<url>[\w./-]+\.\w+\S*)\b",
                    r"\bopen\s+(?P<url>https?://\S+)\b",
                ],
                handler=self._handle_navigate,
                description="Navigate to a URL.",
            ),
            Action(
                name="browser_search",
                patterns=[
                    r"\bsearch\s+(?P<query>.+?)\s+on\s+(?P<engine>google|bing|duckduckgo|ddg|youtube|reddit|github|wikipedia)\b",
                    r"\b(?P<engine>google|bing|duckduckgo|youtube)\s+(?P<query>.+)\b",
                ],
                handler=self._handle_search,
                description="Search using a specific search engine.",
            ),
            Action(
                name="browser_click",
                patterns=[
                    r"\bclick\s+(?:on\s+)?(?:the\s+)?(?P<element>.+)",
                    r"\bpress\s+(?:the\s+)?(?P<element>.+)\s+button\b",
                    r"\bselect\s+(?P<element>.+)",
                ],
                handler=self._handle_click,
                description="Click an element on the page.",
            ),
            Action(
                name="browser_scroll",
                patterns=[
                    r"\bscroll\s+(?P<direction>down|up)(?:\s+(?P<amount>\d+))?\b",
                    r"\bpage\s+(?P<direction>down|up)\b",
                ],
                handler=self._handle_scroll,
                description="Scroll the page up or down.",
            ),
            Action(
                name="browser_fill",
                patterns=[
                    r"\bfill\s+(?:in\s+)?(?:the\s+)?(?P<field>.+?)\s+(?:with|field)\s+(?P<value>.+)",
                    r"\btype\s+(?P<value>.+)\s+in(?:to)?\s+(?:the\s+)?(?P<field>.+)",
                    r"\benter\s+(?P<value>.+)\s+in(?:to)?\s+(?:the\s+)?(?P<field>.+)",
                ],
                handler=self._handle_fill,
                description="Fill a form field.",
            ),
            Action(
                name="browser_read",
                patterns=[
                    r"\b(read|what does this page say|read (the )?page|summarise (this )?page)\b",
                    r"\bwhat(?:'s|\s+is)\s+on\s+(this|the)\s+page\b",
                ],
                handler=self._handle_read,
                description="Read the current page content.",
            ),
            Action(
                name="browser_screenshot",
                patterns=[
                    r"\btake\s+a?\s+(?:page\s+)?screenshot\b",
                    r"\bcapture\s+(?:the\s+)?(?:current\s+)?page\b",
                ],
                handler=self._handle_screenshot,
                description="Take a screenshot of the current page.",
            ),
            Action(
                name="browser_back",
                patterns=[
                    r"\b(go\s+)?back\b",
                    r"\bprevious\s+page\b",
                ],
                handler=self._handle_back,
                description="Navigate back in browser history.",
            ),
            Action(
                name="browser_forward",
                patterns=[
                    r"\b(go\s+)?forward\b",
                    r"\bnext\s+page\b",
                ],
                handler=self._handle_forward,
                description="Navigate forward in browser history.",
            ),
            Action(
                name="browser_reload",
                patterns=[
                    r"\b(reload|refresh)\s*(the\s+)?page\b",
                    r"\brefresh\b",
                ],
                handler=self._handle_reload,
                description="Reload the current page.",
            ),
            Action(
                name="browser_new_tab",
                patterns=[
                    r"\b(open\s+)?(a\s+)?new\s+tab\b",
                    r"\bnew\s+browser\s+tab\b",
                ],
                handler=self._handle_new_tab,
                description="Open a new browser tab.",
            ),
            Action(
                name="browser_close_tab",
                patterns=[
                    r"\bclose\s+(?:this\s+)?tab\b",
                    r"\bclose\s+(?:the\s+)?current\s+tab\b",
                ],
                handler=self._handle_close_tab,
                description="Close the current tab.",
                destructive=True,
            ),
            Action(
                name="browser_switch_tab",
                patterns=[
                    r"\bswitch\s+to\s+tab\s+(?P<number>\d+)\b",
                    r"\bgo\s+to\s+tab\s+(?P<number>\d+)\b",
                    r"\btab\s+(?P<number>\d+)\b",
                ],
                handler=self._handle_switch_tab,
                description="Switch to a specific tab.",
            ),
            Action(
                name="browser_zoom",
                patterns=[
                    r"\bzoom\s+(?P<direction>in|out)(?:\s+(?P<amount>\d+)\s*(?:percent|%)?)?\b",
                    r"\bset\s+zoom\s+to\s+(?P<amount>\d+)\s*(?:percent|%)?\b",
                ],
                handler=self._handle_zoom,
                description="Zoom the browser page in or out.",
            ),
            Action(
                name="browser_page_info",
                patterns=[
                    r"\bwhat\s+(?:page|site|website)\s+(?:am\s+i\s+on|is\s+this)\b",
                    r"\bcurrent\s+(page|url|site)\b",
                    r"\bwhere\s+am\s+i\s+(?:browsing|in\s+the\s+browser)\b",
                ],
                handler=self._handle_page_info,
                description="Report the current page title and URL.",
            ),
        ]

        for action in actions:
            router.register(action)

        # Register the new agentic handler with natural triggers
        router.register(
            Action(
                name="browser_agentic_task",
                patterns=[
                    r"\b(?:research|browse|find\s+info|look\s+up|search\s+the\s+web)\s+(?P<task>.+)",
                    r"\b(?P<task>(?:friday|browser|assistant)\b.*\b(?:do|perform|task|check)\b.*)",
                    # Weather
                    r"(?P<task>what(?:'s|\s+is)\s+(?:the\s+)?weather.*)",
                    r"(?P<task>weather\s+(?:today|forecast|report|update|outside|now|like).*)",
                    r"(?P<task>temperature\s+(?:today|now|outside|right\s+now).*)",
                    r"(?P<task>is\s+it\s+(?:going\s+to\s+)?(?:rain|snow|cold|hot|warm).*)",
                    # News / Events
                    r"(?P<task>(?:latest|today's?|current)\s+(?:news|events|headlines).*)",
                    r"(?P<task>what(?:'s|\s+is)\s+(?:happening|going\s+on)\s+(?:today|now|in\s+the\s+news).*)",
                    # Sports / Scores
                    r"(?P<task>(?:sports\s+score|game\s+score|live\s+score|match\s+result).*)",
                    r"(?P<task>who\s+(?:won|is\s+winning).*)",
                    # Stocks / Market
                    r"(?P<task>(?:stock\s+price|share\s+price|market\s+(?:today|update|status)).*)",
                    # General live data
                    r"(?P<task>when\s+is\s+(?:the\s+)?next.*)",
                    r"(?P<task>where\s+can\s+I\s+(?:watch|buy|find|get).*)",
                    r"(?P<task>how\s+much\s+does\s+.+?\s+(?:cost|price))",
                ],
                handler=self._handle_agentic_task_entry,
                description="Perform a web search or browser task.",
            ),
            priority=True,
        )

        logger.info("[Browser] %d actions registered.", len(actions))

    # ── Agentic Dispatcher ───────────────────────────────────────────────────

    def _handle_agentic_task_entry(self, input_data: re.Match[str] | str) -> ActionResult:
        """Entry point for natural language browser tasks. Checks memory first."""
        if isinstance(input_data, re.Match):
            user_goal = (
                input_data.group("task")
                if "task" in input_data.groupdict()
                else input_data.group(0)
            )
        else:
            user_goal = input_data

        user_goal = user_goal.strip()

        # 1. Semantic Memory Gate
        if self._memory:
            logger.info(f"[Browser/Memory] Checking semantic memory for: {user_goal}")
            relevant = self._memory.recall_relevant(user_goal, limit=2)

            # High-confidence match (ChromaDB distance < 0.3 is usually very close)
            for entry, dist in relevant:
                if dist < 0.3 and entry.category == MemoryCategory.WEB_RESEARCH:
                    logger.info(
                        f"[Browser/Memory] Found cached answer (dist={dist:.3f}). Skipping browser."
                    )
                    return ActionResult(
                        status=ActionStatus.SUCCESS,
                        message=f"I recall looking this up previously, Sir. {entry.content}",
                        action="browser_memory_recall",
                    )

        # 2. Route to quick query or full agentic browse
        # Simple factual queries (weather, news, scores) → quick search + extract
        # Complex tasks (booking, forms, multi-step) → full agentic loop
        word_count = len(user_goal.split())
        if word_count <= 20:
            return self._handle_quick_web_query(user_goal)
        return self._handle_agentic_browse(user_goal)

    def handle_voice_command(self, transcript: str) -> ActionResult:
        """
        Public entry point for voice-driven agentic web tasks.
        """
        # Clean up common wake-word artifacts
        clean_task = re.sub(
            r"^(?:friday|hey friday|ok friday)[,\s]*", "", transcript, flags=re.IGNORECASE
        ).strip()
        return self._handle_agentic_task_entry(clean_task)
    # ── Quick Web Query ──────────────────────────────────────────────────────

    def _handle_quick_web_query(self, query: str) -> ActionResult:
        """
        Fast web query: search Google, read results, extract answer via LLM.
        Does NOT depend on browser_tools — uses _BrowserManager directly.
        Designed for simple factual queries (weather, news, scores, etc.).
        """
        start_time = time.perf_counter()

        if not self._ready and not self.setup():
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="Sir, I couldn't launch the browser.",
                action="browser_quick_query",
            )

        # 1. Navigate to DuckDuckGo (lighter than Google, faster load, less JS)
        search_url = _build_search_url(query, "duckduckgo")
        if not self._manager.navigate(search_url):
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message=f"Sir, I couldn't search for '{query}'.",
                action="browser_quick_query",
            )

        # Wait for results to load
        time.sleep(1.0)

        # 2. Read page text
        page_text = self._manager.get_page_text()
        if not page_text or len(page_text.strip()) < 20:
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message="Sir, the search returned no readable content.",
                action="browser_quick_query",
            )

        # Truncate for LLM context
        page_text = page_text[:3000]

        # 3. Extract answer using ultrafast_8b directly (bypasses full ensemble)
        extraction_prompt = (
            f"The user asked: '{query}'\n\n"
            f"Below is the text from a Google search results page. "
            f"Extract the answer in ONE brief sentence (under 25 words). "
            f"Be precise with numbers, temperatures, and names. "
            f"If you can't find the answer, say 'I couldn't find that information.'\n\n"
            f"PAGE TEXT:\n{page_text}"
        )

        try:
            from friday.async_runner import run_async
            from friday.ensemble import call_canonical_model

            answer = run_async(call_canonical_model(
                "web_extractor",
                "cascade_fast",
                [{"role": "user", "content": extraction_prompt}],
                temperature=0.1,
                max_tokens=80,
                timeout=10.0,
            ))

            if answer and answer.strip():
                answer = answer.strip()
                words = answer.split()
                if len(words) > 40:
                    answer = " ".join(words[:40]) + "."

                # Store in memory for future queries
                if self._memory:
                    with contextlib.suppress(Exception):
                        self._memory.remember(
                            content=f"{query}: {answer}",
                            category=MemoryCategory.WEB_RESEARCH,
                            tier=MemoryTier.LONG_TERM,
                            metadata={"source_url": self.current_url or search_url},
                            semantic=True,
                        )

                return ActionResult(
                    status=ActionStatus.SUCCESS,
                    message=answer,
                    action="browser_quick_query",
                    duration=time.perf_counter() - start_time,
                )
        except Exception as exc:
            logger.error("[Browser] Quick query extraction failed: %s", exc)

        # Fallback: return first 200 chars of page text
        preview = page_text[:200].strip()
        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=preview,
            action="browser_quick_query",
            duration=time.perf_counter() - start_time,
        )

    # ── Agentic Dispatcher ───────────────────────────────────────────────────

    def _handle_agentic_browse(self, user_goal: str) -> ActionResult:
        """
        Agentic entry point for complex web tasks.
        Uses an Observe-and-Pivot loop with LLM planning.
        """
        start_time = time.perf_counter()

        if not self._ready and not self.setup():
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="Sir, I'm having trouble initializing the browser.",
                action="browser_agentic_task",
            )

        # Connect to LLM
        llm = self._llm
        if llm is None or not llm.is_ready:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="Sir, my cognitive systems for web navigation are offline.",
                action="browser_agentic_task",
            )

        current_context = {
            "user_goal": user_goal,
            "last_error": "None",
            "observed_text": "None",
            "current_url": self.current_url or "about:blank",
        }

        max_cycles = 5
        history = []  # Local history for this browsing session

        for cycle in range(max_cycles):
            # 1. Observe and Plan
            prompt = WEB_AGENT_PROMPT.format(**current_context)

            try:
                response = llm.query(prompt)
                if not response.success:
                    return ActionResult(
                        status=ActionStatus.EXECUTION_ERROR,
                        message=f"I encountered an error thinking about the next step: {response.status}",
                        action="browser_agentic_task",
                    )

                try:
                    plan = self._parse_llm_plan(response.text)
                except ValueError as ve:
                    logger.warning(f"[Browser/Agent] JSON Parse Error: {ve}. Retrying...")
                    current_context["last_error"] = (
                        f"Invalid JSON format: {ve}. Please output ONLY valid JSON."
                    )
                    continue

                logger.info(f"[Browser/Agent] Cycle {cycle + 1} Reasoning: {plan.reasoning}")

                if not plan.actions:
                    return ActionResult(
                        status=ActionStatus.SUCCESS,
                        message="Sir, the task is complete.",
                        action="browser_agentic_task",
                        duration=time.perf_counter() - start_time,
                    )

                # 2. Execute plan fragment until pivot boundary
                for action in plan.actions:
                    if action.tool == "finish":
                        return ActionResult(
                            status=ActionStatus.SUCCESS,
                            message="Sir, I've completed the web task.",
                            action="browser_agentic_task",
                            duration=time.perf_counter() - start_time,
                        )

                    last_url = self.current_url
                    result = self._execute_step(action)

                    history.append(
                        {
                            "action": action.tool,
                            "status": result.status.name,
                            "message": result.message,
                        }
                    )

                    # STATE CHANGE: URL changed -> Pivot
                    if action.tool in ["navigate", "click"] and self._did_url_change(last_url):
                        logger.info("[Browser/Agent] State change (URL) detected. Pivoting.")
                        current_context["current_url"] = self.current_url
                        break

                    # STATE CHANGE: Data requested -> Pivot (Let LLM read before next move)
                    if action.tool == "read_text":
                        logger.info("[Browser/Agent] Observation requested. Pivoting.")
                        obs_text = result.message
                        if len(obs_text) > 2000:
                            obs_text = obs_text[:1997] + "..."
                        current_context["observation"] = obs_text

                        # ── Fact Extraction & Memory Storage ──
                        if self._memory:
                            self._extract_and_remember_facts(obs_text, user_goal, self.current_url)

                        break

                    # ERROR DETECTED -> Pivot for auto-correction
                    if result.status != ActionStatus.SUCCESS:
                        logger.warning(
                            "[Browser/Agent] Error detected: %s. Pivoting for correction.",
                            result.message,
                        )
                        current_context["error"] = result.message
                        break
                else:
                    # Plan completed without interruption
                    logger.info("[Browser/Agent] Plan fragment completed successfully.")
                    # In a real scenario, we might ask LLM if we are actually 'done'.
                    # For now, if the plan finishes, we consider the goal met.
                    return ActionResult(
                        status=ActionStatus.SUCCESS,
                        message="I've completed the task.",
                        action="browser_agentic_task",
                        duration=time.perf_counter() - start_time,
                    )

            except Exception as e:
                logger.error("[Browser/Agent] Fatal execution error: %s", e)
                return ActionResult(
                    status=ActionStatus.EXECUTION_ERROR,
                    message=f"I encountered a fatal error: {e}",
                    action="browser_agentic_task",
                )

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message="Task finished.",
            action="browser_agentic_task",
            duration=time.perf_counter() - start_time,
        )

    def _execute_step(self, action: BrowserAction) -> ActionResult:
        """Dispatch a single BrowserAction to the underlying toolset."""
        from friday.actions import browser_tools

        # We reuse existing handlers by spoofing the Regex Match where possible
        # or calling the browser_tools directly.

        if action.tool == "navigate":
            url = _normalise_url(action.url or "")
            if not self._manager.navigate(url):
                return ActionResult(
                    status=ActionStatus.EXECUTION_ERROR,
                    message=f"I couldn't navigate to {url}.",
                    action="browser_navigate",
                )
            info = self._manager.get_page_info()
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Navigated to {info.title}." if info else f"Opened {url}.",
                action="browser_navigate",
            )

        if action.tool == "click":
            try:
                # 1. Standard Attempt
                res = self._run_in_browser_thread(browser_tools.browser_click, action.selector or "")
                return ActionResult(
                    status=ActionStatus.SUCCESS if res["success"] else ActionStatus.EXECUTION_ERROR,
                    message=res["message"],
                    action="browser_click",
                )
            except SelectorHealingRequired as e:
                # 2. Vision Healing Fallback
                logger.info(
                    f"[Browser/Agent] Selector failed for '{e.target_description}'. Attempting VLM healing..."
                )

                # Take viewport screenshot
                shot = self._run_in_browser_thread(browser_tools.browser_take_viewport_screenshot)
                if not shot["success"]:
                    return ActionResult(
                        status=ActionStatus.EXECUTION_ERROR,
                        message="Failed to capture viewport for healing.",
                    )

                # Get coordinates from VLM
                import base64

                img_b64 = base64.b64encode(shot["image_bytes"]).decode("utf-8")
                coords = self._heal_selector_with_vision(e.target_description, img_b64)

                if coords:
                    x, y = coords
                    logger.info(
                        f"[Browser/Agent] VLM found '{e.target_description}' at [{x}, {y}]. Clicking..."
                    )
                    click_res = self._run_in_browser_thread(browser_tools.browser_click_coordinates, x, y)
                    return ActionResult(
                        status=ActionStatus.SUCCESS
                        if click_res["success"]
                        else ActionStatus.EXECUTION_ERROR,
                        message=f"Selector healed via vision. {click_res['message']}",
                        action="browser_click_vision",
                    )

                return ActionResult(
                    status=ActionStatus.NOT_FOUND,
                    message=f"Could not find '{e.target_description}' visually.",
                )

        if action.tool == "fill":
            try:
                # 1. Standard Attempt
                res = self._run_in_browser_thread(browser_tools.browser_type, action.selector or "", action.value or "")
                return ActionResult(
                    status=ActionStatus.SUCCESS if res["success"] else ActionStatus.EXECUTION_ERROR,
                    message=res["message"],
                    action="browser_fill",
                )
            except SelectorHealingRequired as e:
                # 2. Vision Healing Fallback (Click then Type)
                logger.info(
                    f"[Browser/Agent] Input field failed for '{e.target_description}'. Attempting VLM healing..."
                )

                shot = self._run_in_browser_thread(browser_tools.browser_take_viewport_screenshot)
                if shot["success"]:
                    import base64

                    img_b64 = base64.b64encode(shot["image_bytes"]).decode("utf-8")
                    coords = self._heal_selector_with_vision(e.target_description, img_b64)
                    if coords:
                        x, y = coords
                        self._run_in_browser_thread(browser_tools.browser_click_coordinates, x, y)
                        # After clicking the field, use the keyboard to type the value
                        # We use the raw playwright page if possible, or another tool
                        self._manager.type_and_enter(action.value or "")
                        return ActionResult(
                            status=ActionStatus.SUCCESS,
                            message=f"Input field healed via vision. Typed value into [{x}, {y}].",
                            action="browser_fill_vision",
                        )

                return ActionResult(
                    status=ActionStatus.NOT_FOUND,
                    message=f"Could not find input field '{e.target_description}' visually.",
                )

        if action.tool == "read_text":
            res = self._run_in_browser_thread(browser_tools.browser_read_current_page)
            return ActionResult(
                status=ActionStatus.SUCCESS if res["success"] else ActionStatus.SUCCESS,  # noqa: RUF034
                message=res.get("content", ""),
                action="browser_read",
            )

        if action.tool == "screenshot":
            return self._handle_screenshot(None)

        if action.tool == "scroll":
            res = self._run_in_browser_thread(browser_tools.browser_scroll, action.direction or "down")
            return ActionResult(
                status=ActionStatus.SUCCESS if res["success"] else ActionStatus.EXECUTION_ERROR,
                message=res["message"],
                action="browser_scroll",
            )

        return ActionResult(status=ActionStatus.UNSUPPORTED, message=f"Unknown tool: {action.tool}")

    def _did_url_change(self, last_url: str) -> bool:
        """Check if the browser URL has moved from the last known state."""
        current = self.current_url
        # Ignore minor query param differences if needed, but for now exact match
        return current != last_url

    def _heal_selector_with_vision(
        self, description: str, base64_image: str
    ) -> tuple[int, int] | None:
        """
        Ask the VLM to find an element's coordinates on the screen.
        """
        llm = self._llm
        if llm is None or not llm.is_ready:
            return None

        prompt = f"""
        You are a UI coordinate extractor.
        Find the element best matching the description: '{description}'.

        RULES:
        1. Return ONLY a JSON object with 'x' and 'y' coordinates.
        2. Coordinates represent the center of the element relative to the top-left of the image.
        3. Do NOT include any text or markdown formatting.

        Example Output:
        {{ "x": 450, "y": 120 }}
        """

        try:
            response = llm.vision_query(prompt, base64_image)
            if not response.success:
                return None

            # Robust JSON parsing
            import json
            import re

            match = re.search(r"\{.*\}", response.text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return int(data["x"]), int(data["y"])
        except Exception as e:
            logger.error(f"[Browser/Vision] Healing failed: {e}")

        return None

    def _parse_llm_plan(self, text: str) -> BrowserPlan:
        """Parse LLM output into a BrowserPlan."""
        cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
        try:
            return BrowserPlan.model_validate_json(cleaned)
        except Exception as e:
            raise ValueError(f"Failed to parse plan JSON: {e}") from e

    def _extract_and_remember_facts(self, text: str, goal: str, url: str) -> None:
        """
        Extract key facts from page text and store them in semantic memory.
        """
        llm = self._llm
        if llm is None or not llm.is_ready:
            return

        extraction_prompt = f"""
        Extract exactly 1-3 concise, objective facts from the following text that specifically relate to this goal: '{goal}'.
        Respond ONLY with a JSON array of strings.

        TEXT:
        {text}

        JSON OUTPUT:
        """

        try:
            from friday.async_runner import run_async as _run_async
            from friday.ensemble import call_canonical_model as _call_canonical
            resp_text = _run_async(_call_canonical(
                "web_extractor",
                "cascade_fast",
                [{"role": "user", "content": extraction_prompt}],
                temperature=0.1,
                max_tokens=200,
                timeout=10.0,
            ))
            class _FakeResp:
                success = bool(resp_text)
                text = resp_text or ""
            resp = _FakeResp()
            if resp.success:
                import json
                import re

                # Clean and parse
                cleaned = re.sub(r"^```json\s*|\s*```$", "", resp.text.strip(), flags=re.MULTILINE)
                match = re.search(r"\[.*\]", cleaned, re.DOTALL)
                if match:
                    facts = json.loads(match.group(0))
                    for fact in facts:
                        if isinstance(fact, str) and len(fact) > 5:
                            logger.info(f"[Browser/Memory] Storing fact: {fact}")
                            if self._memory:
                                self._memory.remember(
                                    content=fact,
                                    category=MemoryCategory.WEB_RESEARCH,
                                    tier=MemoryTier.LONG_TERM,
                                    metadata={"source_url": url, "original_goal": goal},
                                    semantic=True,
                                )
        except Exception as e:
            logger.error(f"[Browser/Memory] Fact extraction failed: {e}")

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _handle_navigate(self, match: re.Match[str]) -> ActionResult:
        start = time.perf_counter()
        url = _normalise_url((match.group("url") or "").strip())

        if not self._manager.navigate(url):
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message=f"I couldn't navigate to {url}.",
                action="browser_navigate",
            )

        info = self._manager.get_page_info()
        msg = f"Navigated to {info.title}." if info else f"Opened {url}."

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=msg,
            action="browser_navigate",
            duration=time.perf_counter() - start,
        )

    def _handle_search(self, match: re.Match[str]) -> ActionResult:
        start = time.perf_counter()
        query = (match.group("query") or "").strip()
        engine = (match.group("engine") or "google").strip().lower()

        if not query:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="What would you like to search for?",
                action="browser_search",
            )

        url = _build_search_url(query, engine)

        if not self._manager.navigate(url):
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message=f"Search failed on {engine}.",
                action="browser_search",
            )

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=f"Searching {engine} for '{query}'.",
            action="browser_search",
            duration=time.perf_counter() - start,
        )

    def _handle_click(self, match: re.Match[str]) -> ActionResult:
        start = time.perf_counter()
        element = (match.group("element") or "").strip()

        if not element:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="What would you like me to click?",
                action="browser_click",
            )

        if self._manager.click(element):
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Clicked '{element}'.",
                action="browser_click",
                duration=time.perf_counter() - start,
            )

        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message=f"I couldn't find '{element}' on the page.",
            action="browser_click",
        )

    def _handle_scroll(self, match: re.Match[str]) -> ActionResult:
        start = time.perf_counter()
        direction = (match.group("direction") or "down").lower()
        amount_str = match.group("amount") if "amount" in match.groupdict() else None
        amount = int(amount_str) if amount_str else 300

        if self._manager.scroll(direction, amount):
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Scrolled {direction}.",
                action="browser_scroll",
                duration=time.perf_counter() - start,
            )

        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message="I couldn't scroll the page.",
            action="browser_scroll",
        )

    def _handle_fill(self, match: re.Match[str]) -> ActionResult:
        start = time.perf_counter()
        field = (match.group("field") or "").strip()
        value = (match.group("value") or "").strip()

        if not field or not value:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="Please specify both the field and the value to enter.",
                action="browser_fill",
            )

        if self._manager.fill(field, value):
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Filled '{field}' with '{value}'.",
                action="browser_fill",
                duration=time.perf_counter() - start,
            )

        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message=f"I couldn't find the '{field}' field.",
            action="browser_fill",
        )

    def _handle_read(self, match: re.Match[str]) -> ActionResult:
        if config.security.privacy_mode:
            return ActionResult(
                status=ActionStatus.PERMISSION_ERROR,
                message="Page reading is disabled in privacy mode.",
                action="browser_read",
            )

        start = time.perf_counter()
        info = self._manager.get_page_info()

        if not info or not info.text.strip():
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message="The page appears to be empty or I couldn't read its content.",
                action="browser_read",
            )

        # Clean and truncate for TTS
        text = re.sub(r"\s+", " ", info.text).strip()
        preview = text[:500]

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=f"Page '{info.title}' says: {preview}{'...' if len(text) > 500 else ''}",
            action="browser_read",
            duration=time.perf_counter() - start,
        )

    def _handle_screenshot(self, match: re.Match[str] | None = None) -> ActionResult:
        if config.security.privacy_mode:
            return ActionResult(
                status=ActionStatus.PERMISSION_ERROR,
                message="Screenshots are disabled in privacy mode.",
                action="browser_screenshot",
            )

        start = time.perf_counter()
        import datetime

        filename = _SCREENSHOT_DIR / f"browser_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"

        if self._manager.screenshot(str(filename)):
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message="Page screenshot saved.",
                action="browser_screenshot",
                duration=time.perf_counter() - start,
            )

        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message="I couldn't take a screenshot.",
            action="browser_screenshot",
        )

    def _handle_back(self, match: re.Match[str]) -> ActionResult:
        start = time.perf_counter()
        if self._manager.go_back():
            info = self._manager.get_page_info()
            msg = f"Went back to {info.title}." if info else "Went back."
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=msg,
                action="browser_back",
                duration=time.perf_counter() - start,
            )
        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message="Can't go back any further.",
            action="browser_back",
        )

    def _handle_forward(self, match: re.Match[str]) -> ActionResult:
        start = time.perf_counter()
        if self._manager.go_forward():
            info = self._manager.get_page_info()
            msg = f"Went forward to {info.title}." if info else "Went forward."
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=msg,
                action="browser_forward",
                duration=time.perf_counter() - start,
            )
        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message="Can't go forward.",
            action="browser_forward",
        )

    def _handle_reload(self, match: re.Match[str]) -> ActionResult:
        start = time.perf_counter()
        if self._manager.reload():
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message="Page reloaded.",
                action="browser_reload",
                duration=time.perf_counter() - start,
            )
        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message="I couldn't reload the page.",
            action="browser_reload",
        )

    def _handle_new_tab(self, match: re.Match[str]) -> ActionResult:
        start = time.perf_counter()
        page = self._manager.new_tab()
        if page:
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"New tab opened. You now have {self._manager.page_count} tabs.",
                action="browser_new_tab",
                duration=time.perf_counter() - start,
            )
        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message="I couldn't open a new tab.",
            action="browser_new_tab",
        )

    def _handle_close_tab(self, match: re.Match[str]) -> ActionResult:
        start = time.perf_counter()
        if self._manager.close_tab():
            remaining = self._manager.page_count
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Tab closed. {remaining} tab{'s' if remaining != 1 else ''} remaining.",
                action="browser_close_tab",
                duration=time.perf_counter() - start,
            )
        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message="I couldn't close the tab.",
            action="browser_close_tab",
        )

    def _handle_switch_tab(self, match: re.Match[str]) -> ActionResult:
        start = time.perf_counter()
        number = int(match.group("number") or 1)

        if self._manager.switch_tab(number):
            info = self._manager.get_page_info()
            msg = (
                f"Switched to tab {number}: {info.title}." if info else f"Switched to tab {number}."
            )
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=msg,
                action="browser_switch_tab",
                duration=time.perf_counter() - start,
            )

        return ActionResult(
            status=ActionStatus.NOT_FOUND,
            message=f"Tab {number} doesn't exist. You have {self._manager.page_count} tab{'s' if self._manager.page_count != 1 else ''}.",
            action="browser_switch_tab",
        )

    def _handle_zoom(self, match: re.Match[str]) -> ActionResult:
        start = time.perf_counter()
        direction = (match.group("direction") or "in").lower()
        amount_str = match.group("amount") if "amount" in match.groupdict() else None

        if amount_str:
            self._zoom_level = float(amount_str) / 100
        elif direction == "in":
            self._zoom_level = min(3.0, self._zoom_level + 0.25)
        else:
            self._zoom_level = max(0.25, self._zoom_level - 0.25)

        if self._manager.zoom(self._zoom_level):
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Zoom set to {int(self._zoom_level * 100)} percent.",
                action="browser_zoom",
                duration=time.perf_counter() - start,
            )

        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message="I couldn't adjust the zoom.",
            action="browser_zoom",
        )

    def _handle_page_info(self, match: re.Match[str]) -> ActionResult:
        start = time.perf_counter()
        info = self._manager.get_page_info()

        if not info:
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message="No page is currently open.",
                action="browser_page_info",
            )

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=info.to_speech(),
            action="browser_page_info",
            duration=time.perf_counter() - start,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready and self._manager.is_ready

    @property
    def current_title(self) -> str:
        return str(self._manager.active_page_title)

    @property
    def tab_count(self) -> int:
        return int(self._manager.page_count)

    @property
    def zoom_level(self) -> float:
        return self._zoom_level
