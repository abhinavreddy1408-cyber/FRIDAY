"""
F.R.I.D.A.Y. Calendar Skill
Provides calendar querying, event creation, and reminder management
through voice commands.

Supports two backends:
  - Google Calendar API (primary — requires OAuth2 credentials)
  - Local reminder store  (fallback — SQLite-backed, always available)

Supported commands:
  - "what's on my calendar today / this week"
  - "add event [title] on [date] at [time]"
  - "set a reminder for [task] in [N] minutes/hours"
  - "what are my reminders"
  - "cancel reminder [title]"
  - "next event / upcoming events"
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from friday.actions import Action, ActionResult, ActionStatus
from friday.config import config

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


# ── Optional Google Calendar import ──────────────────────────────────────────

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    _GOOGLE_AVAILABLE = True
except ImportError:
    _GOOGLE_AVAILABLE = False
    logger.warning("[Calendar] Google API packages not installed — using local reminders only.")


# ── Constants ─────────────────────────────────────────────────────────────────

_CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar"]
_REMINDER_DB = str(config.storage.data_dir / "reminders.db")
_MAX_EVENTS = 10


# ── Enums ─────────────────────────────────────────────────────────────────────


class CalendarStatus(Enum):
    SUCCESS = auto()
    AUTH_FAILED = auto()
    NOT_FOUND = auto()
    CREATE_FAILED = auto()
    FETCH_FAILED = auto()
    PARSE_FAILED = auto()
    UNAVAILABLE = auto()


class ReminderStatus(Enum):
    PENDING = "pending"
    FIRED = "fired"
    CANCELLED = "cancelled"


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CalendarEvent:
    """
    Immutable representation of a calendar event.

    Attributes:
        id:          Event ID from Google Calendar or local store.
        title:       Event summary / title.
        start:       Start datetime (timezone-aware).
        end:         End datetime (timezone-aware).
        location:    Optional location string.
        description: Optional event description.
        all_day:     True if this is an all-day event.
        source:      "google" or "local".
    """

    id: str
    title: str
    start: datetime
    end: datetime
    location: str
    description: str
    all_day: bool
    source: str

    @property
    def duration_minutes(self) -> int:
        return int((self.end - self.start).total_seconds() / 60)

    @property
    def is_today(self) -> bool:
        today = datetime.now().date()
        return self.start.date() == today

    @property
    def starts_in_minutes(self) -> float:
        delta = self.start - datetime.now(tz=self.start.tzinfo)
        return delta.total_seconds() / 60

    def to_speech(self) -> str:
        """Format as a natural TTS-ready string."""
        if self.all_day:
            return f"{self.title} — all day."

        time_str = self.start.strftime("%I:%M %p").lstrip("0")
        loc_str = f" at {self.location}" if self.location else ""
        return f"{self.title} at {time_str}{loc_str}."


@dataclass(frozen=True)
class Reminder:
    """
    Immutable reminder record.

    Attributes:
        id:         Unique reminder ID.
        title:      What to remind about.
        fire_at:    Unix timestamp when the reminder should fire.
        status:     Current reminder lifecycle status.
        created_at: Unix timestamp of creation.
        metadata:   Optional context data.
    """

    id: str
    title: str
    fire_at: float
    status: ReminderStatus
    created_at: float
    metadata: dict[str, Any]

    @property
    def is_due(self) -> bool:
        return time.time() >= self.fire_at and self.status == ReminderStatus.PENDING

    @property
    def time_until(self) -> float:
        """Seconds until this reminder fires (negative if overdue)."""
        return self.fire_at - time.time()

    def to_speech(self) -> str:
        fire_dt = datetime.fromtimestamp(self.fire_at)
        time_str = fire_dt.strftime("%I:%M %p").lstrip("0")
        today = datetime.now().date()
        if fire_dt.date() == today:
            return f"Reminder: {self.title} at {time_str} today."
        date_str = fire_dt.strftime("%A, %B %d")
        return f"Reminder: {self.title} on {date_str} at {time_str}."


# ── Date / Time Parser ────────────────────────────────────────────────────────


class _DateTimeParser:
    """
    Extracts date and time information from natural language strings.
    Handles relative expressions ("in 30 minutes", "tomorrow at 3pm")
    and absolute dates ("December 25th at 9am").
    """

    _RELATIVE_PATTERNS: list[tuple[re.Pattern[str], str]] = [  # noqa: RUF012
        (re.compile(r"in\s+(\d+)\s+minute", re.I), "minutes"),
        (re.compile(r"in\s+(\d+)\s+hour", re.I), "hours"),
        (re.compile(r"in\s+(\d+)\s+day", re.I), "days"),
        (re.compile(r"tomorrow", re.I), "tomorrow"),
        (re.compile(r"next\s+week", re.I), "next_week"),
        (re.compile(r"tonight", re.I), "tonight"),
        (re.compile(r"this\s+evening", re.I), "tonight"),
    ]

    _TIME_PATTERN = re.compile(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", re.I)
    _WEEKDAY_MAP = {  # noqa: RUF012
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    def parse(self, text: str) -> datetime | None:
        """
        Parse a natural language date/time expression.

        Args:
            text: Input string containing date/time information.

        Returns:
            datetime object or None if parsing fails.
        """
        text = text.lower().strip()
        now = datetime.now()

        # Relative time expressions
        for pattern, kind in self._RELATIVE_PATTERNS:
            match = pattern.search(text)
            if match:
                return self._resolve_relative(match, kind, text, now)

        # Named weekday
        for day_name, day_num in self._WEEKDAY_MAP.items():
            if day_name in text:
                return self._next_weekday(now, day_num, text)

        # Absolute time today (e.g. "at 3pm")
        time_match = self._TIME_PATTERN.search(text)
        if time_match:
            return self._resolve_time(time_match, now)

        return None

    def _resolve_relative(
        self,
        match: re.Match[str],
        kind: str,
        text: str,
        now: datetime,
    ) -> datetime | None:
        """Resolve a relative time expression to a concrete datetime."""
        if kind == "minutes":
            return now + timedelta(minutes=int(match.group(1)))
        if kind == "hours":
            return now + timedelta(hours=int(match.group(1)))
        if kind == "days":
            return now + timedelta(days=int(match.group(1)))
        if kind == "tomorrow":
            base = now + timedelta(days=1)
            time_match = self._TIME_PATTERN.search(text)
            if time_match:
                return self._resolve_time(time_match, base)
            return base.replace(hour=9, minute=0, second=0, microsecond=0)
        if kind == "next_week":
            return now + timedelta(weeks=1)
        if kind == "tonight":
            return now.replace(hour=20, minute=0, second=0, microsecond=0)
        return None

    def _resolve_time(self, match: re.Match[str], base: datetime) -> datetime:
        """Set the time components on a base datetime."""
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        ampm = (match.group(3) or "").lower()

        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        elif not ampm and hour < 8:
            hour += 12  # Assume PM for ambiguous times before 8

        return base.replace(hour=hour, minute=minute, second=0, microsecond=0)

    @staticmethod
    def _next_weekday(now: datetime, weekday: int, text: str) -> datetime:
        """Return the next occurrence of a given weekday."""
        days_ahead = weekday - now.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return (now + timedelta(days=days_ahead)).replace(hour=9, minute=0, second=0, microsecond=0)


# ── Local Reminder Store ──────────────────────────────────────────────────────

_REMINDER_SCHEMA = """
CREATE TABLE IF NOT EXISTS reminders (
    id         TEXT  PRIMARY KEY,
    title      TEXT  NOT NULL,
    fire_at    REAL  NOT NULL,
    status     TEXT  NOT NULL DEFAULT 'pending',
    created_at REAL  NOT NULL,
    metadata   TEXT  NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_reminders_fire_at ON reminders (fire_at);
CREATE INDEX IF NOT EXISTS idx_reminders_status  ON reminders (status);
"""


class _ReminderStore:
    """
    SQLite-backed local reminder persistence.
    Thread-safe via per-thread connections and a write lock.
    """

    def __init__(self, db_path: str = _REMINDER_DB) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._local = threading.local()
        self._init_schema()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self._path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        try:
            yield self._local.conn
        except sqlite3.DatabaseError as exc:
            self._local.conn.rollback()
            logger.error("[Calendar/Store] DB error: %s", exc)
            raise

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_REMINDER_SCHEMA)
            conn.commit()

    def add(self, reminder: Reminder) -> None:
        with self._write_lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO reminders (id, title, fire_at, status, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    reminder.id,
                    reminder.title,
                    reminder.fire_at,
                    reminder.status.value,
                    reminder.created_at,
                    json.dumps(reminder.metadata),
                ),
            )
            conn.commit()

    def get_pending(self) -> list[Reminder]:
        """Return all pending reminders ordered by fire_at."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM reminders WHERE status = 'pending' ORDER BY fire_at ASC"
            ).fetchall()
        return [self._row_to_reminder(r) for r in rows]

    def get_due(self) -> list[Reminder]:
        """Return all reminders that are due right now."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM reminders WHERE status = 'pending' AND fire_at <= ?",
                (time.time(),),
            ).fetchall()
        return [self._row_to_reminder(r) for r in rows]

    def update_status(self, reminder_id: str, status: ReminderStatus) -> bool:
        with self._write_lock, self._conn() as conn:
            cursor = conn.execute(
                "UPDATE reminders SET status = ? WHERE id = ?",
                (status.value, reminder_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def cancel_by_title(self, title: str) -> int:
        """Cancel all pending reminders matching a title substring."""
        with self._write_lock, self._conn() as conn:
            cursor = conn.execute(
                """
                UPDATE reminders SET status = 'cancelled'
                WHERE status = 'pending' AND LOWER(title) LIKE ?
                """,
                (f"%{title.lower()}%",),
            )
            conn.commit()
            return cursor.rowcount

    def count_pending(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM reminders WHERE status = 'pending'").fetchone()
            return row[0] if row else 0

    @staticmethod
    def _row_to_reminder(row: sqlite3.Row) -> Reminder:
        return Reminder(
            id=row["id"],
            title=row["title"],
            fire_at=row["fire_at"],
            status=ReminderStatus(row["status"]),
            created_at=row["created_at"],
            metadata=json.loads(row["metadata"]),
        )


# ── Google Calendar Client ────────────────────────────────────────────────────


class _GoogleCalendarClient:
    """
    Thin wrapper around the Google Calendar API.
    Authentication reuses the Gmail token where possible.
    """

    def __init__(self) -> None:
        self._service: Any = None
        self._ready: bool = False

    def setup(self) -> bool:
        if not _GOOGLE_AVAILABLE:
            return False

        creds_path = Path(config.system.gmail_creds)
        token_path = Path(config.system.gmail_token)

        if not creds_path.exists():
            return False

        try:
            creds = None
            if token_path.exists():
                creds = Credentials.from_authorized_user_file(str(token_path), _CALENDAR_SCOPES)  # type: ignore[no-untyped-call]

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(creds_path), _CALENDAR_SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                token_path.write_text(creds.to_json(), encoding="utf-8")

            self._service = build("calendar", "v3", credentials=creds)
            self._ready = True
            logger.info("[Calendar] Google Calendar client ready.")
            return True

        except Exception as exc:
            logger.warning("[Calendar] Google Calendar setup failed: %s", exc)
            return False

    def list_events(
        self,
        time_min: datetime,
        time_max: datetime,
        max_results: int = _MAX_EVENTS,
    ) -> list[dict[str, Any]]:
        """Fetch events between two datetimes."""
        if not self._ready or not self._service:
            return []

        try:
            result = (
                self._service.events()
                .list(
                    calendarId="primary",
                    timeMin=time_min.isoformat() + "Z",
                    timeMax=time_max.isoformat() + "Z",
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            return cast("list[dict[str, Any]]", result.get("items", []))
        except HttpError as exc:
            logger.error("[Calendar] list_events error: %s", exc)
            return []

    def create_event(self, event_body: dict[str, Any]) -> str | None:
        """
        Create a calendar event.

        Args:
            event_body: Google Calendar event resource dict.

        Returns:
            Created event ID or None on failure.
        """
        if not self._ready or not self._service:
            return None

        try:
            event = (
                self._service.events()
                .insert(
                    calendarId="primary",
                    body=event_body,
                )
                .execute()
            )
            return cast("str | None", event.get("id"))
        except HttpError as exc:
            logger.error("[Calendar] create_event error: %s", exc)
            return None

    @property
    def is_ready(self) -> bool:
        return self._ready


# ── Event Parser ──────────────────────────────────────────────────────────────


class _EventParser:
    """Converts raw Google Calendar API event dicts to CalendarEvent objects."""

    def parse(self, raw: dict[str, Any]) -> CalendarEvent | None:
        try:
            start_raw = raw.get("start", {})
            end_raw = raw.get("end", {})

            all_day = "date" in start_raw and "dateTime" not in start_raw

            start = self._parse_dt(start_raw)
            end = self._parse_dt(end_raw)

            if not start or not end:
                return None

            return CalendarEvent(
                id=raw.get("id", ""),
                title=raw.get("summary", "Untitled Event"),
                start=start,
                end=end,
                location=raw.get("location", ""),
                description=raw.get("description", ""),
                all_day=all_day,
                source="google",
            )
        except Exception as exc:
            logger.debug("[Calendar] Event parse failed: %s", exc)
            return None

    @staticmethod
    def _parse_dt(dt_dict: dict[str, str]) -> datetime | None:
        """Parse a Google Calendar start/end dict to a datetime."""
        if "dateTime" in dt_dict:
            try:
                raw = dt_dict["dateTime"]
                # Handle both offset-aware and naive formats
                if raw.endswith("Z"):
                    raw = raw[:-1] + "+00:00"
                return datetime.fromisoformat(raw)
            except ValueError:
                return None

        if "date" in dt_dict:
            try:
                d = datetime.strptime(dt_dict["date"], "%Y-%m-%d")
                return d.replace(tzinfo=timezone.utc)
            except ValueError:
                return None

        return None


# ── Calendar Skill ────────────────────────────────────────────────────────────


class CalendarSkill:
    """
    Calendar and reminder skill for F.R.I.D.A.Y.

    Provides voice-controlled calendar querying, event creation,
    and local reminder management with background firing.

    Usage:
        skill = CalendarSkill()
        skill.setup()
        skill.register(action_router)
    """

    def __init__(self) -> None:
        self._gcal = _GoogleCalendarClient()
        self._store = _ReminderStore()
        self._dt_parser = _DateTimeParser()
        self._event_parser = _EventParser()
        self._ready = False
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._on_reminder_due: Callable[[Reminder], None] | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(
        self,
        on_reminder_due: Callable[[Reminder], None] | None = None,
    ) -> bool:
        """
        Initialise the calendar skill.

        Args:
            on_reminder_due: Callback invoked with a Reminder when it fires.
                             Signature: (reminder: Reminder) -> None

        Returns:
            True — always succeeds (local reminders always available).
        """
        self._on_reminder_due = on_reminder_due
        self._gcal.setup()  # Non-fatal if Google Calendar unavailable
        self._ready = True
        self._start_reminder_monitor()

        logger.info(
            "[Calendar] Ready. Google Calendar: %s.",
            "online" if self._gcal.is_ready else "offline (local reminders only)",
        )
        return True

    def teardown(self) -> None:
        """Stop the reminder monitor thread."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=3.0)
        self._ready = False

    def register(self, router: Any) -> None:
        """Register all calendar actions with the ActionRouter."""
        if not self._ready:
            return

        actions = [
            Action(
                name="calendar_today",
                patterns=[
                    r"\b(what(?:'s| is) on (my )?calendar|today('s)? (events?|schedule|agenda))\b",
                    r"\bwhat do i have (today|on today)\b",
                    r"\bschedule for today\b",
                ],
                handler=self._handle_today,
                description="List today's calendar events.",
            ),
            Action(
                name="calendar_upcoming",
                patterns=[
                    r"\b(upcoming|next|this week('s)?)\s+(events?|schedule|calendar)\b",
                    r"\bwhat('s| is) (coming up|next on (my )?calendar)\b",
                    r"\bnext event\b",
                ],
                handler=self._handle_upcoming,
                description="List upcoming calendar events.",
            ),
            Action(
                name="calendar_add",
                patterns=[
                    r"\b(add|create|schedule|set up)\s+(an?\s+)?(event|meeting|appointment)\s*(?P<details>.+)?",
                    r"\b(put|block)\s+(?P<details>.+)\s+on (my )?calendar\b",
                ],
                handler=self._handle_add_event,
                description="Add a new calendar event.",
            ),
            Action(
                name="reminder_set",
                patterns=[
                    r"\b(set|add|create)\s+(a\s+)?reminder\s+(for\s+|to\s+)?(?P<task>.+)",
                    r"\bremind\s+me\s+(to\s+|about\s+)?(?P<task>.+)",
                ],
                handler=self._handle_set_reminder,
                description="Set a timed reminder.",
            ),
            Action(
                name="reminder_list",
                patterns=[
                    r"\b(what(?:'s| are)?(my\s+)?reminders?|show\s+reminders?|list\s+reminders?)\b",
                    r"\bdo i have (any )?reminders?\b",
                ],
                handler=self._handle_list_reminders,
                description="List all pending reminders.",
            ),
            Action(
                name="reminder_cancel",
                patterns=[
                    r"\b(cancel|delete|remove)\s+(the\s+)?reminder\s+(for\s+|about\s+)?(?P<title>.+)",
                ],
                handler=self._handle_cancel_reminder,
                description="Cancel a pending reminder.",
            ),
        ]

        for action in actions:
            router.register(action)

        logger.info("[Calendar] %d actions registered.", len(actions))

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _handle_today(self, match: re.Match[str]) -> ActionResult:
        """List today's calendar events."""
        start = time.perf_counter()
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = now.replace(hour=23, minute=59, second=59, microsecond=0)

        events = self._fetch_events(today_start, today_end)

        if not events:
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message="Your calendar is clear today.",
                action="calendar_today",
                duration=time.perf_counter() - start,
            )

        count = len(events)
        event_strs = [e.to_speech() for e in events[:5]]
        msg = f"You have {count} event{'s' if count != 1 else ''} today. " + " ".join(event_strs)

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=msg,
            action="calendar_today",
            duration=time.perf_counter() - start,
        )

    def _handle_upcoming(self, match: re.Match[str]) -> ActionResult:
        """List events for the next 7 days."""
        start = time.perf_counter()
        now = datetime.now()
        week_end = now + timedelta(days=7)

        events = self._fetch_events(now, week_end)

        if not events:
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message="Nothing coming up in the next week.",
                action="calendar_upcoming",
                duration=time.perf_counter() - start,
            )

        count = len(events)
        event_strs = [e.to_speech() for e in events[:5]]
        msg = f"{count} event{'s' if count != 1 else ''} in the next week. " + " ".join(event_strs)

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=msg,
            action="calendar_upcoming",
            duration=time.perf_counter() - start,
        )

    def _handle_add_event(self, match: re.Match[str]) -> ActionResult:
        """Parse and create a new calendar event."""
        start = time.perf_counter()
        details = (match.group("details") or "").strip()

        if not details:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="What event would you like to add? Please include a title and time.",
                action="calendar_add",
            )

        # Parse event datetime
        event_dt = self._dt_parser.parse(details)
        if not event_dt:
            event_dt = datetime.now() + timedelta(hours=1)

        # Extract title (everything before temporal keywords)
        title = self._extract_event_title(details)

        if self._gcal.is_ready:
            event_body = {
                "summary": title,
                "start": {"dateTime": event_dt.isoformat(), "timeZone": "local"},
                "end": {
                    "dateTime": (event_dt + timedelta(hours=1)).isoformat(),
                    "timeZone": "local",
                },
            }
            event_id = self._gcal.create_event(event_body)
            if event_id:
                time_str = event_dt.strftime("%I:%M %p on %A, %B %d").lstrip("0")
                return ActionResult(
                    status=ActionStatus.SUCCESS,
                    message=f"Event '{title}' added to your calendar for {time_str}.",
                    action="calendar_add",
                    duration=time.perf_counter() - start,
                )

        return ActionResult(
            status=ActionStatus.UNSUPPORTED,
            message=(
                "Google Calendar isn't connected. "
                "I can set a local reminder instead — would you like that?"
            ),
            action="calendar_add",
            duration=time.perf_counter() - start,
        )

    def _handle_set_reminder(self, match: re.Match[str]) -> ActionResult:
        """Parse and create a local timed reminder."""
        start = time.perf_counter()
        task = (match.group("task") or "").strip()

        if not task:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="What would you like me to remind you about?",
                action="reminder_set",
            )

        fire_at = self._dt_parser.parse(task)
        if not fire_at:
            # Default: 10 minutes from now
            fire_at = datetime.now() + timedelta(minutes=10)

        # Clean the task title (remove temporal keywords)
        title = self._extract_event_title(task)

        reminder = Reminder(
            id=str(uuid.uuid4())[:8],
            title=title or task,
            fire_at=fire_at.timestamp(),
            status=ReminderStatus.PENDING,
            created_at=time.time(),
            metadata={},
        )

        self._store.add(reminder)

        time_str = fire_at.strftime("%I:%M %p").lstrip("0")
        if fire_at.date() == datetime.now().date():
            when = f"at {time_str} today"
        else:
            when = f"on {fire_at.strftime('%A')} at {time_str}"

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=f"Reminder set: '{title}' {when}.",
            action="reminder_set",
            duration=time.perf_counter() - start,
        )

    def _handle_list_reminders(self, match: re.Match[str]) -> ActionResult:
        """List all pending reminders."""
        start = time.perf_counter()
        pending = self._store.get_pending()

        if not pending:
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message="You have no pending reminders.",
                action="reminder_list",
                duration=time.perf_counter() - start,
            )

        count = len(pending)
        summaries = [r.to_speech() for r in pending[:5]]
        msg = f"You have {count} pending reminder{'s' if count != 1 else ''}. " + " ".join(
            summaries
        )

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=msg,
            action="reminder_list",
            duration=time.perf_counter() - start,
        )

    def _handle_cancel_reminder(self, match: re.Match[str]) -> ActionResult:
        """Cancel a reminder by title match."""
        start = time.perf_counter()
        title = (match.group("title") or "").strip()

        if not title:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="Which reminder would you like to cancel?",
                action="reminder_cancel",
            )

        cancelled = self._store.cancel_by_title(title)

        if cancelled == 0:
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message=f"I couldn't find a reminder matching '{title}'.",
                action="reminder_cancel",
                duration=time.perf_counter() - start,
            )

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=f"Cancelled {cancelled} reminder{'s' if cancelled != 1 else ''} matching '{title}'.",
            action="reminder_cancel",
            duration=time.perf_counter() - start,
        )

    # ── Reminder Monitor ──────────────────────────────────────────────────────

    def _start_reminder_monitor(self) -> None:
        """Start the background thread that polls for due reminders."""
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="reminder-monitor",
        )
        self._monitor_thread.start()
        logger.debug("[Calendar] Reminder monitor started.")

    def _monitor_loop(self) -> None:
        """Poll for due reminders every 30 seconds."""
        while not self._stop_event.is_set():
            try:
                due = self._store.get_due()
                for reminder in due:
                    self._store.update_status(reminder.id, ReminderStatus.FIRED)
                    logger.info("[Calendar] Reminder fired: '%s'", reminder.title)
                    if self._on_reminder_due:
                        try:
                            self._on_reminder_due(reminder)
                        except Exception:
                            logger.debug(
                                "[Calendar] on_reminder_due callback raised.",
                                exc_info=True,
                            )
            except Exception:
                logger.debug("[Calendar] Monitor loop error.", exc_info=True)

            self._stop_event.wait(timeout=30)

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _fetch_events(
        self,
        start: datetime,
        end: datetime,
    ) -> list[CalendarEvent]:
        """
        Fetch events from Google Calendar (if available).

        Args:
            start: Window start datetime.
            end:   Window end datetime.

        Returns:
            List of CalendarEvent objects.
        """
        if not self._gcal.is_ready:
            return []

        raw_events = self._gcal.list_events(
            time_min=start,
            time_max=end,
        )
        events: list[CalendarEvent] = []
        for raw in raw_events:
            parsed = self._event_parser.parse(raw)
            if parsed:
                events.append(parsed)

        return events

    @staticmethod
    def _extract_event_title(text: str) -> str:
        """
        Remove temporal keywords and filler words to extract an event title.

        Args:
            text: Raw command text containing the title and time info.

        Returns:
            Cleaned title string.
        """
        temporal_words = (
            r"\b(in|at|on|this|next|tomorrow|tonight|today|monday|tuesday|"
            r"wednesday|thursday|friday|saturday|sunday|morning|afternoon|"
            r"evening|night|am|pm|\d+\s*(minutes?|hours?|days?|weeks?))\b"
        )
        cleaned = re.sub(temporal_words, "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned if cleaned else text

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def google_calendar_available(self) -> bool:
        return self._gcal.is_ready

    @property
    def pending_reminder_count(self) -> int:
        return self._store.count_pending()
