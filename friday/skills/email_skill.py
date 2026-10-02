"""
F.R.I.D.A.Y. Email Skill
Provides Gmail integration for composing, reading, sending, and managing
email through voice commands.

Authentication uses OAuth2 via the Google API client library.
Credentials are stored locally and refreshed automatically.

Supported commands:
  - "compose/write/send email to [name]"
  - "read my emails / check inbox"
  - "read latest email"
  - "reply to [name]"
  - "search emails about [topic]"
  - "how many unread emails"
"""

from __future__ import annotations

import base64
import logging
import re
import threading
import time
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from friday.actions import Action, ActionResult, ActionStatus
from friday.config import config

logger = logging.getLogger(__name__)


# ── Optional imports ──────────────────────────────────────────────────────────

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    _GOOGLE_AVAILABLE = True
except ImportError:
    _GOOGLE_AVAILABLE = False
    logger.warning("[Email] Google API packages not installed — email skill disabled.")


# ── Constants ─────────────────────────────────────────────────────────────────

_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]

_MAX_RESULTS = 10
_PREVIEW_CHARS = 200


# ── Data Models ───────────────────────────────────────────────────────────────


class EmailStatus(Enum):
    SUCCESS = auto()
    AUTH_FAILED = auto()
    NOT_FOUND = auto()
    SEND_FAILED = auto()
    FETCH_FAILED = auto()
    UNAVAILABLE = auto()


@dataclass(frozen=True)
class EmailMessage:
    """
    Immutable representation of a Gmail message.

    Attributes:
        id:        Gmail message ID.
        thread_id: Thread this message belongs to.
        subject:   Message subject line.
        sender:    Sender display name and address.
        snippet:   Short preview of the message body.
        body:      Full plain-text body (empty if not fetched).
        date:      Human-readable date string.
        unread:    True if the message has not been read.
        labels:    List of Gmail label IDs.
    """

    id: str
    thread_id: str
    subject: str
    sender: str
    snippet: str
    body: str
    date: str
    unread: bool
    labels: list[str]

    def to_speech(self, include_body: bool = False) -> str:
        """Format as a natural TTS-ready description."""
        parts = [
            f"From {self.sender}.",
            f"Subject: {self.subject}.",
        ]
        if include_body and self.body:
            preview = self.body[:_PREVIEW_CHARS].replace("\n", " ")
            parts.append(f"Message: {preview}.")
        elif self.snippet:
            parts.append(f"Preview: {self.snippet[:100]}.")
        return " ".join(parts)


@dataclass
class DraftEmail:
    """
    Mutable draft being composed through multi-turn dialogue.

    Attributes:
        to:      Recipient email address or name.
        subject: Email subject line.
        body:    Email body text.
        cc:      Optional CC recipients.
    """

    to: str = ""
    subject: str = ""
    body: str = ""
    cc: str = ""

    @property
    def is_ready(self) -> bool:
        """True if the draft has enough information to send."""
        return bool(self.to and self.subject and self.body)

    def to_speech_summary(self) -> str:
        """Summarise the draft for user confirmation."""
        return (
            f"Email to {self.to}, "
            f"subject '{self.subject}', "
            f"message: {self.body[:100]}{'...' if len(self.body) > 100 else ''}. "
            "Shall I send it?"
        )


# ── Gmail Client ──────────────────────────────────────────────────────────────


class _GmailClient:
    """
    Handles OAuth2 authentication and raw Gmail API calls.
    Thread-safe via a lock on service creation.
    """

    def __init__(self) -> None:
        self._service: Any | None = None
        self._credentials: Any | None = None
        self._lock = threading.Lock()
        self._ready = False

    def setup(self) -> bool:
        """
        Authenticate with Gmail and build the API service.

        Returns:
            True if authentication succeeded.
        """
        if not _GOOGLE_AVAILABLE:
            return False

        creds_path = Path(config.system.gmail_creds)
        token_path = Path(config.system.gmail_token)

        if not creds_path.exists():
            logger.warning("[Email] Gmail credentials not found at '%s'.", creds_path)
            return False

        try:
            with self._lock:
                creds = self._load_or_refresh_credentials(token_path, creds_path)
                if not creds:
                    return False

                self._credentials = creds
                self._service = build("gmail", "v1", credentials=creds)
                self._ready = True

            logger.info("[Email] Gmail client authenticated.")
            return True

        except Exception as exc:
            logger.error("[Email] Gmail setup failed: %s", exc)
            return False

    def _load_or_refresh_credentials(
        self,
        token_path: Path,
        creds_path: Path,
    ) -> Any | None:
        """
        Load existing token or run the OAuth2 flow to obtain credentials.

        Args:
            token_path: Path to the saved token JSON file.
            creds_path: Path to the OAuth2 credentials JSON file.

        Returns:
            Valid Credentials object or None.
        """
        creds = None

        if token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)  # type: ignore[no-untyped-call]
            except Exception as exc:
                logger.warning("[Email] Could not load token: %s", exc)

        if creds and creds.valid:
            return creds

        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                self._save_token(creds, token_path)
                return creds
            except Exception as exc:
                logger.warning("[Email] Token refresh failed: %s — re-authenticating.", exc)

        # Full OAuth flow
        try:
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), _SCOPES)
            creds = flow.run_local_server(port=0)
            self._save_token(creds, token_path)
            return creds
        except Exception as exc:
            logger.error("[Email] OAuth2 flow failed: %s", exc)
            return None

    @staticmethod
    def _save_token(creds: Any, token_path: Path) -> None:
        """Persist refreshed credentials to disk."""
        try:
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json(), encoding="utf-8")
        except OSError as exc:
            logger.warning("[Email] Could not save token: %s", exc)

    # ── Gmail API calls ───────────────────────────────────────────────────────

    def list_messages(
        self,
        query: str = "",
        max_results: int = _MAX_RESULTS,
        label_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch a list of message metadata objects.

        Args:
            query:       Gmail search query string.
            max_results: Maximum messages to return.
            label_ids:   Filter by Gmail labels (e.g. ["INBOX", "UNREAD"]).

        Returns:
            List of raw message dicts from the Gmail API.
        """
        if not self._ready or not self._service:
            return []

        try:
            params: dict[str, Any] = {
                "userId": "me",
                "maxResults": max_results,
            }
            if query:
                params["q"] = query
            if label_ids:
                params["labelIds"] = label_ids

            response = self._service.users().messages().list(**params).execute()
            return cast("list[dict[str, Any]]", response.get("messages", []))

        except HttpError as exc:
            logger.error("[Email] list_messages API error: %s", exc)
            return []

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        """
        Fetch the full message payload for a given ID.

        Args:
            message_id: Gmail message ID string.

        Returns:
            Full message dict or None on failure.
        """
        if not self._ready or not self._service:
            return None

        try:
            return cast("dict[str, Any] | None", (
                self._service.users()
                .messages()
                .get(
                    userId="me",
                    id=message_id,
                    format="full",
                )
                .execute()
            ))
        except HttpError as exc:
            logger.error("[Email] get_message API error (%s): %s", message_id, exc)
            return None

    def send_message(self, raw_bytes: bytes) -> bool:
        """
        Send a pre-encoded MIME message.

        Args:
            raw_bytes: Base64url-encoded message bytes.

        Returns:
            True if sent successfully.
        """
        if not self._ready or not self._service:
            return False

        try:
            body = {"raw": base64.urlsafe_b64encode(raw_bytes).decode("utf-8")}
            self._service.users().messages().send(userId="me", body=body).execute()
            return True
        except HttpError as exc:
            logger.error("[Email] send_message API error: %s", exc)
            return False

    def mark_as_read(self, message_id: str) -> bool:
        """Remove the UNREAD label from a message."""
        if not self._ready or not self._service:
            return False
        try:
            self._service.users().messages().modify(
                userId="me",
                id=message_id,
                body={"removeLabelIds": ["UNREAD"]},
            ).execute()
            return True
        except HttpError as exc:
            logger.error("[Email] mark_as_read error: %s", exc)
            return False

    def get_unread_count(self) -> int:
        """Return the number of unread messages in the inbox."""
        if not self._ready or not self._service:
            return 0
        try:
            result = (
                self._service.users()
                .messages()
                .list(
                    userId="me",
                    labelIds=["INBOX", "UNREAD"],
                    maxResults=1,
                )
                .execute()
            )
            return int(result.get("resultSizeEstimate", 0))
        except HttpError:
            return 0

    @property
    def is_ready(self) -> bool:
        return self._ready


# ── Message Parser ────────────────────────────────────────────────────────────


class _MessageParser:
    """
    Converts raw Gmail API message dicts into typed EmailMessage objects.
    """

    def parse(self, raw: dict[str, Any]) -> EmailMessage | None:
        """
        Parse a raw Gmail API message into an EmailMessage.

        Args:
            raw: Raw message dict from the Gmail API.

        Returns:
            EmailMessage or None on parse failure.
        """
        try:
            headers = self._extract_headers(raw)
            body = self._extract_body(raw)
            labels = raw.get("labelIds", [])

            return EmailMessage(
                id=raw.get("id", ""),
                thread_id=raw.get("threadId", ""),
                subject=headers.get("subject", "(No subject)"),
                sender=headers.get("from", "Unknown"),
                snippet=raw.get("snippet", ""),
                body=body,
                date=headers.get("date", ""),
                unread="UNREAD" in labels,
                labels=labels,
            )
        except Exception as exc:
            logger.debug("[Email] Message parse failed: %s", exc)
            return None

    @staticmethod
    def _extract_headers(raw: dict[str, Any]) -> dict[str, str]:
        """Extract a name→value header map from the payload."""
        headers: dict[str, str] = {}
        payload = raw.get("payload", {})
        for header in payload.get("headers", []):
            name = header.get("name", "").lower()
            if name in ("from", "to", "subject", "date"):
                headers[name] = header.get("value", "")
        return headers

    @staticmethod
    def _extract_body(raw: dict[str, Any]) -> str:
        """
        Recursively extract plain-text body content from a MIME payload.
        Returns empty string if no plain-text part is found.
        """

        def _recurse(payload: dict[str, Any]) -> str:
            mime_type = payload.get("mimeType", "")

            if mime_type == "text/plain":
                data = payload.get("body", {}).get("data", "")
                if data:
                    try:
                        return base64.urlsafe_b64decode(data + "==").decode(
                            "utf-8", errors="replace"
                        )
                    except Exception:
                        return ""

            parts = payload.get("parts", [])
            for part in parts:
                result = _recurse(part)
                if result:
                    return result

            return ""

        return _recurse(raw.get("payload", {}))


# ── Email Skill ───────────────────────────────────────────────────────────────


class EmailSkill:
    """
    Gmail integration skill for F.R.I.D.A.Y.

    Provides voice-controlled email operations: reading, composing,
    sending, and searching. Handles multi-turn email composition through
    a DraftEmail state object.

    Usage:
        skill = EmailSkill()
        skill.setup()
        skill.register(action_router)
    """

    def __init__(self) -> None:
        self._client = _GmailClient()
        self._parser = _MessageParser()
        self._draft: DraftEmail | None = None
        self._ready = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """
        Authenticate and prepare the Gmail client.

        Returns:
            True if the skill is ready.
        """
        if not _GOOGLE_AVAILABLE:
            logger.warning("[Email] Skill unavailable — Google API packages missing.")
            return False

        self._ready = self._client.setup()
        if self._ready:
            logger.info("[Email] Email skill ready.")
        return self._ready

    def register(self, router: Any) -> None:
        """
        Register all email actions with the ActionRouter.

        Args:
            router: ActionRouter instance.
        """
        if not self._ready:
            logger.warning("[Email] Skipping registration — skill not ready.")
            return

        actions = [
            Action(
                name="email_check",
                patterns=[
                    r"\b(check|read|open|show)\s+(my\s+)?emails?\b",
                    r"\b(any\s+)?new\s+emails?\b",
                    r"\bhow\s+many\s+(unread\s+)?emails?\b",
                ],
                handler=self._handle_check,
                description="Check inbox for new emails.",
            ),
            Action(
                name="email_read_latest",
                patterns=[
                    r"\bread\s+(the\s+)?(latest|newest|last|first)\s+email\b",
                    r"\bopen\s+(the\s+)?latest\s+email\b",
                ],
                handler=self._handle_read_latest,
                description="Read the most recent email.",
            ),
            Action(
                name="email_compose",
                patterns=[
                    r"\b(compose|write|send|draft)\s+(an?\s+)?email\s+(to\s+)?(?P<recipient>.+)",
                    r"\bemail\s+(?P<recipient>.+)",
                ],
                handler=self._handle_compose,
                description="Compose and send an email.",
            ),
            Action(
                name="email_search",
                patterns=[
                    r"\bsearch\s+(my\s+)?emails?\s+(for\s+|about\s+)?(?P<query>.+)",
                    r"\bfind\s+emails?\s+(about\s+|from\s+)?(?P<query>.+)",
                ],
                handler=self._handle_search,
                description="Search emails by keyword.",
            ),
        ]

        for action in actions:
            router.register(action)

        logger.info("[Email] %d actions registered.", len(actions))

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _handle_check(self, match: re.Match[str]) -> ActionResult:
        """Handle inbox check commands."""
        start = time.perf_counter()

        unread_count = self._client.get_unread_count()

        if unread_count == 0:
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message="Your inbox is clear. No unread emails.",
                action="email_check",
                duration=time.perf_counter() - start,
            )

        messages = self._client.list_messages(
            label_ids=["INBOX", "UNREAD"],
            max_results=3,
        )

        if not messages:
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"You have {unread_count} unread email{'s' if unread_count != 1 else ''}.",
                action="email_check",
                duration=time.perf_counter() - start,
            )

        summaries: list[str] = []
        for msg_ref in messages[:3]:
            raw = self._client.get_message(msg_ref["id"])
            if raw:
                email = self._parser.parse(raw)
                if email:
                    summaries.append(f"From {email.sender}: {email.subject}.")

        summary_text = " ".join(summaries) if summaries else ""
        msg = (
            f"You have {unread_count} unread email{'s' if unread_count != 1 else ''}. "
            + summary_text
        )

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=msg,
            action="email_check",
            duration=time.perf_counter() - start,
        )

    def _handle_read_latest(self, match: re.Match[str]) -> ActionResult:
        """Fetch and read the most recent inbox message."""
        start = time.perf_counter()

        messages = self._client.list_messages(
            label_ids=["INBOX"],
            max_results=1,
        )

        if not messages:
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message="Your inbox is empty.",
                action="email_read_latest",
                duration=time.perf_counter() - start,
            )

        raw = self._client.get_message(messages[0]["id"])
        if not raw:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="I couldn't fetch that email.",
                action="email_read_latest",
                duration=time.perf_counter() - start,
            )

        email = self._parser.parse(raw)
        if not email:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="I had trouble reading that email.",
                action="email_read_latest",
                duration=time.perf_counter() - start,
            )

        self._client.mark_as_read(email.id)

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=email.to_speech(include_body=True),
            action="email_read_latest",
            duration=time.perf_counter() - start,
        )

    def _handle_compose(self, match: re.Match[str]) -> ActionResult:
        """Begin composing a new email to a specified recipient."""
        start = time.perf_counter()
        recipient = (match.group("recipient") or "").strip()

        if not recipient:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="Who would you like to email? Please provide a name or address.",
                action="email_compose",
                duration=time.perf_counter() - start,
            )

        self._draft = DraftEmail(to=recipient)

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=(f"Starting an email to {recipient}. What would you like the subject to be?"),
            action="email_compose",
            duration=time.perf_counter() - start,
        )

    def _handle_search(self, match: re.Match[str]) -> ActionResult:
        """Search emails by keyword."""
        start = time.perf_counter()
        query = (match.group("query") or "").strip()

        if not query:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="What would you like to search for in your emails?",
                action="email_search",
            )

        messages = self._client.list_messages(query=query, max_results=5)

        if not messages:
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message=f"No emails found matching '{query}'.",
                action="email_search",
                duration=time.perf_counter() - start,
            )

        count = len(messages)
        msg = f"Found {count} email{'s' if count != 1 else ''} matching '{query}'."

        # Read the first result
        raw = self._client.get_message(messages[0]["id"])
        if raw:
            email = self._parser.parse(raw)
            if email:
                msg += f" Most recent: {email.to_speech()}."

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=msg,
            action="email_search",
            duration=time.perf_counter() - start,
        )

    # ── Draft management ──────────────────────────────────────────────────────

    def set_draft_subject(self, subject: str) -> str:
        """
        Set the subject on the active draft.

        Args:
            subject: Email subject line.

        Returns:
            TTS-ready prompt for the next field.
        """
        if not self._draft:
            return "No email draft in progress."

        self._draft.subject = subject
        return f"Subject set to '{subject}'. What would you like the message to say?"

    def set_draft_body(self, body: str) -> str:
        """
        Set the body on the active draft and prompt for confirmation.

        Args:
            body: Email body text.

        Returns:
            TTS-ready draft summary.
        """
        if not self._draft:
            return "No email draft in progress."

        self._draft.body = body
        return self._draft.to_speech_summary()

    def send_draft(self) -> ActionResult:
        """
        Send the active draft email.

        Returns:
            ActionResult indicating success or failure.
        """
        if not self._draft or not self._draft.is_ready:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="The email draft is incomplete. Please provide a recipient, subject, and message.",
                action="email_send",
            )

        try:
            mime = MIMEMultipart()
            mime["to"] = self._draft.to
            mime["subject"] = self._draft.subject
            if self._draft.cc:
                mime["cc"] = self._draft.cc
            mime.attach(MIMEText(self._draft.body, "plain"))

            raw_bytes = mime.as_bytes()
            success = self._client.send_message(raw_bytes)

            if success:
                recipient = self._draft.to
                self._draft = None
                return ActionResult(
                    status=ActionStatus.SUCCESS,
                    message=f"Email sent to {recipient}.",
                    action="email_send",
                )

            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="I couldn't send the email. Please try again.",
                action="email_send",
            )

        except Exception as exc:
            logger.error("[Email] send_draft failed: %s", exc)
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="Something went wrong sending the email.",
                action="email_send",
            )

    def discard_draft(self) -> str:
        """Discard the active draft."""
        self._draft = None
        return "Email draft discarded."

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def has_draft(self) -> bool:
        return self._draft is not None

    @property
    def draft(self) -> DraftEmail | None:
        return self._draft
