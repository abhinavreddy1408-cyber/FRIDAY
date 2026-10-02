"""
F.R.I.D.A.Y.  —  Action Verifier Module
─────────────────────────────────────────────────────────────────────────────
Verifies whether a tool action achieved its intended outcome by combining:

  1. Explicit failure signal scanning  (tool output text)
  2. Typed verification strategies     (URL, OCR, file-system, DOM)
  3. Implicit screen-state fallback    (OCR text match)

Architecture
  ActionVerifier          — public facade; owns the strategy registry
  VerificationStrategy    — abstract base; one subclass per check type
  VerificationResult      — immutable typed result (replaces raw dicts)
  StrategyKind            — enum driving strategy dispatch

Adding a new strategy
  1. Subclass VerificationStrategy and implement `matches()` + `verify()`.
  2. Register an instance in ActionVerifier._strategies.

Thread safety
  ActionVerifier is stateless after construction; all methods are pure
  functions of their arguments and are safe to call concurrently.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# ── Lazy screen imports ───────────────────────────────────────────────────────
# Playwright / OCR may be unavailable in headless or test environments.
# We import lazily and surface a clean error rather than crashing at module load.


def _import_screen():
    try:
        from friday.screen import verify_screen_contains, verify_url_contains

        return verify_url_contains, verify_screen_contains
    except ImportError as exc:
        raise RuntimeError(
            "friday.screen is not available. "
            "Install playwright and its dependencies, or run in a headed environment."
        ) from exc


# ── Result type ───────────────────────────────────────────────────────────────


class VerificationStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    INCONCLUSIVE = "INCONCLUSIVE"  # couldn't determine either way
    STRATEGY_ERROR = "STRATEGY_ERROR"  # verifier itself raised


@dataclass(frozen=True)
class VerificationResult:
    """
    Immutable outcome of a single verification attempt.

    Attributes:
        status:   Terminal classification of the check.
        detail:   Human-readable explanation — shown to the operator / logged.
        strategy: Which strategy produced this result (for diagnostics).
    """

    status: VerificationStatus
    detail: str
    strategy: str = "unknown"

    @property
    def success(self) -> bool:
        return self.status == VerificationStatus.SUCCESS

    @property
    def should_retry(self) -> bool:
        """
        True only for genuine failures — not for INCONCLUSIVE results.
        Use this instead of `not result.success` to avoid retrying on
        actions where verification simply could not be performed.
        """
        return self.status == VerificationStatus.FAILURE

    def to_dict(self) -> dict:
        """Backward-compatible dict for callers expecting the old ``{"success": bool}`` shape."""
        return {
            "success": self.success,
            "detail": self.detail,
            "strategy": self.strategy,
            "status": self.status.value,
        }

    def __str__(self) -> str:
        return f"[{self.status.value}] {self.detail}"


# ── Helpers ───────────────────────────────────────────────────────────────────

# Signals that indicate an explicit tool-level failure in the output text.
_FAILURE_SIGNALS: frozenset[str] = frozenset(
    {
        "error",
        "failed",
        "failure",
        "blocked",
        "exception",
        "timeout",
        "traceback",
        "refused",
    }
)

# Suppress false-positive failure signals when these appear nearby.
_FALSE_POSITIVE_GUARDS: frozenset[str] = frozenset(
    {
        "no error",
        "without error",
        "success",
        "successful",
        "completed",
        "handled",
        "ignored",
    }
)


def _has_failure_signal(text: str) -> bool:
    """
    Return True if *text* contains an explicit failure keyword that is NOT
    neutralised by a nearby success guard.

    We scan the lower-cased text rather than splitting on words so that
    compound tokens like ``timed_out`` are also matched.
    """
    lower = text.lower()
    # Quick exit — no signal at all
    if not any(sig in lower for sig in _FAILURE_SIGNALS):
        return False
    # Check whether a guard phrase appears anywhere (coarse but safe)
    return not any(guard in lower for guard in _FALSE_POSITIVE_GUARDS)


def _extract_quoted_or_trailing(text: str, prefix: str) -> str:
    """
    Given ``text`` and a ``prefix`` to strip, return the remainder with any
    surrounding quotes removed.

    Examples
    --------
    >>> _extract_quoted_or_trailing("url contains 'google.com'", "url contains")
    'google.com'
    >>> _extract_quoted_or_trailing("screen contains results", "screen contains")
    'results'
    """
    remainder = text[len(prefix) :].strip()
    # Remove wrapping single or double quotes
    return remainder.strip("'\"").strip()


# ── Strategy base ─────────────────────────────────────────────────────────────


@dataclass
class VerificationContext:
    """
    All inputs available to a verification strategy.

    Attributes:
        tool_name:        The tool that was executed.
        args:             Arguments passed to the tool.
        tool_output:      The raw string output / result from the tool.
        expected_outcome: Free-text description from the action plan.
        screen_state:     Current screen snapshot (ocr_text, url, dom_hash …).
    """

    tool_name: str
    args: dict
    tool_output: str
    expected_outcome: str
    screen_state: dict

    # Normalised lowercase view — computed once, used by all strategies
    @property
    def expected_lower(self) -> str:
        return self.expected_outcome.lower().strip()

    @property
    def output_lower(self) -> str:
        return self.tool_output.lower().strip()


class VerificationStrategy(ABC):
    """Abstract base for all verification strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in VerificationResult.strategy."""

    @abstractmethod
    def matches(self, ctx: VerificationContext) -> bool:
        """Return True if this strategy should handle the given context."""

    @abstractmethod
    def verify(self, ctx: VerificationContext) -> VerificationResult:
        """Perform the verification and return a result."""

    # Convenience factory methods for consistent result construction
    def _ok(self, detail: str) -> VerificationResult:
        return VerificationResult(VerificationStatus.SUCCESS, detail, self.name)

    def _fail(self, detail: str) -> VerificationResult:
        return VerificationResult(VerificationStatus.FAILURE, detail, self.name)

    def _inconclusive(self, detail: str) -> VerificationResult:
        return VerificationResult(VerificationStatus.INCONCLUSIVE, detail, self.name)

    def _error(self, detail: str) -> VerificationResult:
        return VerificationResult(VerificationStatus.STRATEGY_ERROR, detail, self.name)


# ── Concrete strategies ────────────────────────────────────────────────────────


class URLVerificationStrategy(VerificationStrategy):
    """
    Verifies that the active browser URL contains an expected substring.

    Triggered by expected outcomes like:
      - ``"url contains 'google.com'"``
      - ``"url contains youtube"``
    """

    _PREFIX = "url contains"

    @property
    def name(self) -> str:
        return "url_contains"

    def matches(self, ctx: VerificationContext) -> bool:
        return self._PREFIX in ctx.expected_lower

    def verify(self, ctx: VerificationContext) -> VerificationResult:
        substring = _extract_quoted_or_trailing(ctx.expected_lower, self._PREFIX)

        if not substring:
            return self._inconclusive(
                "expected_outcome contained 'url contains' but no substring to check."
            )

        try:
            verify_url_contains, _ = _import_screen()
            v = verify_url_contains(substring)
        except RuntimeError as exc:
            return self._error(str(exc))
        except Exception as exc:
            logger.exception("[URLVerification] Unexpected error: %s", exc)
            return self._error(f"verify_url_contains raised: {exc}")

        if v.get("found"):
            return self._ok(f"URL contains '{substring}' — confirmed.")
        return self._fail(
            f"URL does not contain '{substring}'. Current URL: {v.get('current_url', 'unknown')}"
        )


class ScreenOCRStrategy(VerificationStrategy):
    """
    Verifies screen content via OCR.

    Triggered by expected outcomes like:
      - ``"screen contains 'Python 3.12'"``
      - ``"text visible"``
      - ``"results appeared"``
    """

    _PREFIXES = ("screen contains", "text contains", "text visible")
    _EXISTENCE = frozenset({"results appeared", "content visible", "text visible", "results"})

    @property
    def name(self) -> str:
        return "screen_ocr"

    def matches(self, ctx: VerificationContext) -> bool:
        exp = ctx.expected_lower
        return any(p in exp for p in self._PREFIXES) or exp in self._EXISTENCE

    def verify(self, ctx: VerificationContext) -> VerificationResult:
        exp = ctx.expected_lower

        # Pure existence check — any visible OCR text is enough
        if exp in self._EXISTENCE or exp == "results appeared":
            ocr = ctx.screen_state.get("ocr_text", "")
            if ocr and ocr.strip():
                return self._ok("Screen contains text content — existence confirmed via OCR.")
            return self._fail("OCR returned empty — screen appears blank.")

        # Substring check
        prefix_used = next((p for p in self._PREFIXES if p in exp), "screen contains")
        substring = _extract_quoted_or_trailing(exp, prefix_used)

        if not substring:
            return self._inconclusive(
                "expected_outcome implied screen check but no target substring found."
            )

        try:
            _, verify_screen_contains = _import_screen()
            v = verify_screen_contains(substring)
        except RuntimeError as exc:
            return self._error(str(exc))
        except Exception as exc:
            logger.exception("[ScreenOCR] Unexpected error: %s", exc)
            return self._error(f"verify_screen_contains raised: {exc}")

        if v.get("found"):
            return self._ok(f"Screen contains '{substring}' — confirmed via OCR.")
        return self._fail(f"OCR could not find '{substring}' on screen.")


class FileWriteStrategy(VerificationStrategy):
    """Verifies that a file exists after a write/create operation."""

    @property
    def name(self) -> str:
        return "file_write"

    def matches(self, ctx: VerificationContext) -> bool:
        return ctx.tool_name in ("file_write", "file_create", "write_file")

    def verify(self, ctx: VerificationContext) -> VerificationResult:
        path = ctx.args.get("path", "").strip()

        if not path:
            return self._inconclusive("No 'path' argument provided to file_write tool.")

        abs_path = os.path.abspath(path)
        if os.path.isfile(abs_path):
            size = os.path.getsize(abs_path)
            return self._ok(f"File verified at '{abs_path}' ({size} bytes).")
        return self._fail(f"File not found after write attempt: '{abs_path}'.")


class FileDeleteStrategy(VerificationStrategy):
    """Verifies that a file no longer exists after a delete operation."""

    @property
    def name(self) -> str:
        return "file_delete"

    def matches(self, ctx: VerificationContext) -> bool:
        return ctx.tool_name in ("file_delete", "delete_file", "remove_file")

    def verify(self, ctx: VerificationContext) -> VerificationResult:
        path = ctx.args.get("path", "").strip()

        if not path:
            return self._inconclusive("No 'path' argument provided to file_delete tool.")

        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            return self._ok(f"File no longer exists — deletion confirmed: '{abs_path}'.")
        return self._fail(f"File still exists after deletion attempt: '{abs_path}'.")


class FileMoveStrategy(VerificationStrategy):
    """Verifies a file move/rename: source gone, destination present."""

    @property
    def name(self) -> str:
        return "file_move"

    def matches(self, ctx: VerificationContext) -> bool:
        return ctx.tool_name in ("file_move", "move_file", "rename_file")

    def verify(self, ctx: VerificationContext) -> VerificationResult:
        src = ctx.args.get("source", ctx.args.get("src", "")).strip()
        dest = ctx.args.get("destination", ctx.args.get("dest", "")).strip()

        if not src or not dest:
            return self._inconclusive("file_move requires 'source' and 'destination' args.")

        src_gone = not os.path.exists(os.path.abspath(src))
        dest_here = os.path.isfile(os.path.abspath(dest))

        if src_gone and dest_here:
            return self._ok(f"File moved: '{src}' → '{dest}'.")
        if not dest_here:
            return self._fail(f"Destination file missing after move: '{dest}'.")
        if not src_gone:
            return self._fail(f"Source file still present after move: '{src}'.")
        return self._fail(f"Move verification failed: src_gone={src_gone}, dest_here={dest_here}.")


class ImplicitScreenStateStrategy(VerificationStrategy):
    """
    Fallback: checks whether expected_outcome text appears in the cached
    OCR snapshot without triggering a live screen capture.

    Only used when no other strategy matches.
    """

    @property
    def name(self) -> str:
        return "implicit_screen_state"

    def matches(self, ctx: VerificationContext) -> bool:
        # This is the catch-all; the orchestrator calls it last
        return True

    def verify(self, ctx: VerificationContext) -> VerificationResult:
        ocr = ctx.screen_state.get("ocr_text", "")
        if ocr and ctx.expected_lower in ocr.lower():
            return self._ok(
                f"Expected outcome '{ctx.expected_outcome}' found in cached OCR snapshot."
            )
        # Cannot confirm or deny — surface as INCONCLUSIVE rather than lying
        return self._inconclusive(
            f"No matching strategy for outcome '{ctx.expected_outcome}'. "
            "Tool did not report failure; treating as inconclusive."
        )


# ── Public facade ─────────────────────────────────────────────────────────────


class ActionVerifier:
    """
    Stateless facade that runs the correct verification strategy for a given
    tool action.

    Usage::

        verifier = ActionVerifier()

        result = verifier.verify(
            tool_name        = "file_write",
            args             = {"path": "/tmp/report.txt"},
            tool_output      = "File written successfully.",
            expected_outcome = "",
            screen_state     = {},
        )

        if result.success:
            friday.speak(result.detail)
        else:
            friday.speak(f"Verification failed — {result.detail}")

        # Backward compat with old dict-based callers:
        legacy_dict = result.to_dict()   # {"success": bool, "detail": str, …}
    """

    def __init__(self) -> None:
        # Order matters: more specific strategies must come before the catch-all.
        self._strategies: list[VerificationStrategy] = [
            URLVerificationStrategy(),
            ScreenOCRStrategy(),
            FileWriteStrategy(),
            FileDeleteStrategy(),
            FileMoveStrategy(),
            ImplicitScreenStateStrategy(),  # must be last
        ]

    def verify(
        self,
        tool_name: str,
        args: dict,
        tool_output: str,
        expected_outcome: str,
        screen_state: dict,
    ) -> VerificationResult:
        """
        Verify a completed tool action.

        Short-circuit checks (explicit failure signal, empty expected_outcome)
        run first. Then the first matching strategy is executed.

        Args:
            tool_name:        Name of the tool that was called (e.g. "file_write").
            args:             Arguments that were passed to the tool.
            tool_output:      The raw result / output string from the tool.
            expected_outcome: Free-text from the action plan describing success.
            screen_state:     Dict with keys like ``ocr_text``, ``url``, ``dom_hash``.

        Returns:
            VerificationResult — use ``.success`` or ``.to_dict()`` on the result.
        """
        # ── Guard 1: explicit failure in tool output ──────────────────────────
        if _has_failure_signal(tool_output):
            logger.warning("[Verifier] Failure signal detected in tool output for '%s'.", tool_name)
            return VerificationResult(
                status=VerificationStatus.FAILURE,
                detail=f"Tool reported failure: {tool_output[:200]}",
                strategy="failure_signal_scan",
            )

        # ── Guard 2: no expected outcome → nothing to verify ──────────────────
        if not expected_outcome or not expected_outcome.strip():
            return VerificationResult(
                status=VerificationStatus.SUCCESS,
                detail="Action completed — no specific outcome to verify.",
                strategy="no_expectation",
            )

        ctx = VerificationContext(
            tool_name=tool_name,
            args=args,
            tool_output=tool_output,
            expected_outcome=expected_outcome,
            screen_state=screen_state,
        )

        # ── Strategy dispatch ─────────────────────────────────────────────────
        for strategy in self._strategies:
            if strategy.matches(ctx):
                logger.debug(
                    "[Verifier] Running strategy '%s' for tool '%s'.",
                    strategy.name,
                    tool_name,
                )
                try:
                    result = strategy.verify(ctx)
                except Exception as exc:
                    logger.exception(
                        "[Verifier] Strategy '%s' raised unexpectedly: %s",
                        strategy.name,
                        exc,
                    )
                    result = VerificationResult(
                        status=VerificationStatus.STRATEGY_ERROR,
                        detail=f"Strategy '{strategy.name}' crashed: {exc}",
                        strategy=strategy.name,
                    )

                logger.info("[Verifier] %s", result)
                return result

        # Unreachable — ImplicitScreenStateStrategy always matches
        return VerificationResult(
            VerificationStatus.INCONCLUSIVE,
            "No strategy matched and no fallback fired.",
            "none",
        )


# ── Module-level convenience function (backward compatibility) ─────────────────

_default_verifier = ActionVerifier()


def verify_action(
    tool_name: str,
    args: dict,
    result: str,
    expected_outcome: str,
    screen_state: dict,
) -> dict:
    """
    Drop-in replacement for the original module-level ``verify_action`` function.

    Returns a ``{"success": bool, "detail": str}`` dict for backward compat.
    Internally delegates to ``ActionVerifier.verify()``.
    """
    vr = _default_verifier.verify(
        tool_name=tool_name,
        args=args,
        tool_output=result,
        expected_outcome=expected_outcome,
        screen_state=screen_state,
    )
    return vr.to_dict()
