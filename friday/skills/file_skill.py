"""
F.R.I.D.A.Y. File Skill
Provides voice-controlled file system operations including searching,
opening, organising, and inspecting files and directories.

Supported commands:
  - "open file [name]"
  - "find file [name]"
  - "open downloads / documents / desktop"
  - "open recent files"
  - "delete file [name]"         (requires confirmation)
  - "create folder [name]"
  - "move file [name] to [folder]"
  - "how big is [file/folder]"
  - "list files in [folder]"
  - "open [file extension] files"

All destructive operations require explicit confirmation.
Search is restricted to user-accessible directories by default.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

from friday import config
from friday.actions import Action, ActionResult, ActionStatus

logger = logging.getLogger(__name__)


# ── Optional imports ──────────────────────────────────────────────────────────

try:
    import psutil  # noqa: F401

    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

_MAX_SEARCH_RESULTS = 10
_MAX_SEARCH_DEPTH = 5
_MAX_LIST_FILES = 15
_SEARCH_TIMEOUT = 10.0  # seconds


# ── Enums ─────────────────────────────────────────────────────────────────────


class FileStatus(Enum):
    SUCCESS = auto()
    NOT_FOUND = auto()
    PERMISSION_ERROR = auto()
    EXECUTION_ERROR = auto()
    REQUIRES_CONFIRM = auto()
    UNSUPPORTED = auto()


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FileInfo:
    """
    Immutable metadata snapshot for a file or directory.

    Attributes:
        path:         Absolute path to the file/directory.
        name:         File or directory name.
        size_bytes:   Size in bytes (0 for directories without recursion).
        modified_at:  Unix timestamp of last modification.
        is_dir:       True if this is a directory.
        extension:    File extension (empty for directories).
    """

    path: Path
    name: str
    size_bytes: int
    modified_at: float
    is_dir: bool
    extension: str

    @property
    def size_human(self) -> str:
        """Human-readable file size string."""
        return _format_size(self.size_bytes)

    @property
    def modified_ago(self) -> str:
        """Human-readable 'time ago' string for last modification."""
        return _format_time_ago(self.modified_at)

    def to_speech(self) -> str:
        kind = "folder" if self.is_dir else "file"
        size = f", {self.size_human}" if not self.is_dir else ""
        return f"{self.name} ({kind}{size}, modified {self.modified_ago})"


@dataclass
class PendingFileOp:
    """
    A destructive file operation awaiting user confirmation.

    Attributes:
        operation: The operation type ("delete", "move", "overwrite").
        source:    Source path.
        target:    Target path (for move/copy operations).
        created_at: Unix timestamp — expires after 30 seconds.
    """

    operation: str
    source: Path
    target: Path | None = None
    created_at: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > 30.0

    def describe(self) -> str:
        if self.operation == "delete":
            return f"Delete '{self.source.name}'?"
        if self.operation == "move":
            return f"Move '{self.source.name}' to '{self.target}'?"
        return f"Confirm {self.operation} on '{self.source.name}'?"


# ── Known Directories ─────────────────────────────────────────────────────────


def _get_known_dirs() -> dict[str, Path]:
    """
    Return a mapping of common directory aliases to absolute paths.
    Computed at call time so it reflects the current user's home directory.
    """
    home = Path.home()
    return {
        "home": home,
        "desktop": home / "Desktop",
        "downloads": home / "Downloads",
        "documents": home / "Documents",
        "pictures": home / "Pictures",
        "music": home / "Music",
        "videos": home / "Videos",
        "appdata": Path(os.environ.get("APPDATA", home / "AppData" / "Roaming")),
        "temp": Path(os.environ.get("TEMP", "/tmp")),
        "onedrive": home / "OneDrive",
        "friday": Path(__file__).resolve().parent.parent.parent,
    }


# ── Utility Functions ─────────────────────────────────────────────────────────


def _format_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable string."""
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "bytes" else f"{size_bytes} bytes"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _format_time_ago(timestamp: float) -> str:
    """Convert a Unix timestamp to a relative 'N [units] ago' string."""
    delta = time.time() - timestamp
    if delta < 60:
        return "just now"
    if delta < 3600:
        mins = int(delta / 60)
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    if delta < 86400:
        hours = int(delta / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = int(delta / 86400)
    return f"{days} day{'s' if days != 1 else ''} ago"


def _get_file_info(path: Path) -> FileInfo | None:
    """Build a FileInfo from a path, returning None on access errors."""
    try:
        stat = path.stat()
        return FileInfo(
            path=path,
            name=path.name,
            size_bytes=stat.st_size,
            modified_at=stat.st_mtime,
            is_dir=path.is_dir(),
            extension=path.suffix.lower(),
        )
    except (OSError, PermissionError):
        return None


def _open_path(path: Path) -> bool:
    """
    Open a file or directory using the OS default handler.

    Returns:
        True if the open command was dispatched successfully.
    """
    try:
        if os.name == "nt":
            os.startfile(str(path))
        else:
            subprocess.Popen(
                ["xdg-open", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return True
    except Exception as exc:
        logger.warning("[File] Open failed for '%s': %s", path, exc)
        return False


def _safe_search(
    root: Path,
    query: str,
    max_results: int = _MAX_SEARCH_RESULTS,
    max_depth: int = _MAX_SEARCH_DEPTH,
    extensions: list[str] | None = None,
) -> list[Path]:
    """
    Recursively search a directory for files matching a query string.

    Args:
        root:        Directory to search from.
        query:       Substring to match against file names (case-insensitive).
        max_results: Stop after this many matches.
        max_depth:   Maximum directory recursion depth.
        extensions:  If provided, only match files with these extensions.

    Returns:
        List of matching Path objects.
    """
    results: list[Path] = []
    query_lower = query.lower()
    start_time = time.time()

    def _walk(directory: Path, depth: int) -> None:
        if depth > max_depth or len(results) >= max_results:
            return
        if time.time() - start_time > _SEARCH_TIMEOUT:
            return

        try:
            for entry in directory.iterdir():
                if len(results) >= max_results:
                    return
                try:
                    if query_lower in entry.name.lower():
                        if extensions:
                            if entry.suffix.lower() in extensions:
                                results.append(entry)
                        else:
                            results.append(entry)

                    if entry.is_dir() and not entry.name.startswith("."):
                        _walk(entry, depth + 1)
                except PermissionError:
                    continue
        except PermissionError:
            pass

    _walk(root, 0)
    return results


def _dir_size(path: Path) -> int:
    """
    Recursively compute directory size in bytes.
    Returns 0 on permission errors.
    """
    total = 0
    try:
        for entry in path.rglob("*"):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except (OSError, PermissionError):
                continue
    except PermissionError:
        pass
    return total


# ── File Skill ────────────────────────────────────────────────────────────────


class FileSkill:
    """
    File system operations skill for F.R.I.D.A.Y.

    Provides voice-controlled file search, opening, folder navigation,
    organisation, and size inspection. Destructive operations are gated
    behind a confirmation step.

    Usage:
        skill = FileSkill()
        skill.setup()
        skill.register(action_router)
    """

    def __init__(self) -> None:
        self._known_dirs = _get_known_dirs()
        self._pending_op: PendingFileOp | None = None
        self._ready = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """
        Initialise the file skill.

        Returns:
            True — always succeeds (file operations use stdlib only).
        """
        self._ready = True
        logger.info("[File] File skill ready.")
        return True

    def register(self, router: Any) -> None:
        """Register all file actions with the ActionRouter."""
        actions = [
            Action(
                name="file_open_dir",
                patterns=[
                    r"\bopen\s+(?:my\s+)?(?P<dir>downloads?|documents?|desktop|pictures?|music|videos?|home|temp|onedrive|appdata)\b",
                    r"\bgo\s+to\s+(?P<dir>downloads?|documents?|desktop|pictures?|music|videos?|home)\b",
                    r"\bshow\s+(?P<dir>downloads?|documents?|desktop|pictures?|music|videos?)\b",
                ],
                handler=self._handle_open_dir,
                description="Open a known directory in the file explorer.",
            ),
            Action(
                name="file_find",
                patterns=[
                    r"\b(find|search\s+for|look\s+for)\s+(?:file\s+)?(?P<query>.+?)(?:\s+file)?$",
                    r"\bwhere\s+is\s+(?P<query>.+?)(?:\s+file)?$",
                    r"\blocate\s+(?P<query>.+?)(?:\s+file)?$",
                ],
                handler=self._handle_find,
                description="Search for a file by name.",
            ),
            Action(
                name="file_open",
                patterns=[
                    r"\bopen\s+(?:file\s+)?(?P<name>.+?\.[\w]+)\b",
                    r"\blaunch\s+(?P<name>.+?\.[\w]+)\b",
                ],
                handler=self._handle_open_file,
                description="Open a specific file by name.",
            ),
            Action(
                name="file_list",
                patterns=[
                    r"\blist\s+(?:files?\s+in\s+|contents?\s+of\s+)?(?P<dir>.+)",
                    r"\bwhat(?:'s|\s+is)\s+in\s+(?P<dir>.+?)(?:\s+folder)?$",
                    r"\bshow\s+(?:me\s+)?(?:the\s+)?files?\s+in\s+(?P<dir>.+)",
                ],
                handler=self._handle_list,
                description="List files in a directory.",
            ),
            Action(
                name="file_size",
                patterns=[
                    r"\bhow\s+(?:big|large)\s+is\s+(?P<target>.+?)(?:\s+(?:file|folder))?$",
                    r"\bsize\s+of\s+(?P<target>.+?)(?:\s+(?:file|folder))?$",
                    r"\bwhat(?:'s|\s+is)\s+the\s+size\s+of\s+(?P<target>.+)",
                ],
                handler=self._handle_size,
                description="Report the size of a file or folder.",
            ),
            Action(
                name="file_create_folder",
                patterns=[
                    r"\b(create|make|new)\s+(?:a\s+)?folder\s+(?:called\s+|named\s+)?(?P<name>.+)",
                    r"\bmkdir\s+(?P<name>.+)",
                ],
                handler=self._handle_create_folder,
                description="Create a new folder.",
            ),
            Action(
                name="file_delete",
                patterns=[
                    r"\b(delete|remove)\s+(?:the\s+)?(?:file\s+)?(?P<name>.+?)(?:\s+file)?$",
                ],
                handler=self._handle_delete,
                description="Delete a file (requires confirmation).",
                destructive=True,
            ),
            Action(
                name="file_move",
                patterns=[
                    r"\bmove\s+(?P<source>.+?)\s+to\s+(?P<dest>.+)",
                    r"\brelocate\s+(?P<source>.+?)\s+to\s+(?P<dest>.+)",
                ],
                handler=self._handle_move,
                description="Move a file to another location.",
                destructive=True,
            ),
            Action(
                name="file_recent",
                patterns=[
                    r"\b(?:open\s+)?recent(?:ly\s+(?:opened|accessed|modified))?\s+files?\b",
                    r"\bwhat\s+files?\s+did\s+i\s+(?:open|use|access)\s+recently\b",
                ],
                handler=self._handle_recent,
                description="Show recently modified files.",
            ),
        ]

        for action in actions:
            router.register(action)

        logger.info("[File] %d actions registered.", len(actions))

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _handle_open_dir(self, match: re.Match) -> ActionResult:
        """Open a known directory alias in the file explorer."""
        start = time.perf_counter()
        dir_name = (match.group("dir") or "").strip().lower().rstrip("s")

        # Normalise aliases
        alias_map = {
            "download": "downloads",
            "document": "documents",
            "picture": "pictures",
            "video": "videos",
        }
        dir_name = alias_map.get(dir_name, dir_name)

        target = self._known_dirs.get(dir_name)
        if not target:
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message=f"I don't know the location of '{dir_name}'.",
                action="file_open_dir",
            )

        if not target.exists():
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message=f"The {dir_name} folder doesn't exist on this system.",
                action="file_open_dir",
            )

        if _open_path(target):
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Opening {dir_name}.",
                action="file_open_dir",
                duration=time.perf_counter() - start,
            )

        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message=f"I couldn't open the {dir_name} folder.",
            action="file_open_dir",
        )

    def _handle_find(self, match: re.Match) -> ActionResult:
        """Search for files matching a query across common directories."""
        start = time.perf_counter()
        query = (match.group("query") or "").strip()

        if not query:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="What file would you like me to find?",
                action="file_find",
            )

        # Search common user directories
        search_roots = [
            Path.home() / "Desktop",
            Path.home() / "Documents",
            Path.home() / "Downloads",
        ]

        results: list[Path] = []
        for root in search_roots:
            if root.exists():
                found = _safe_search(root, query, max_results=_MAX_SEARCH_RESULTS)
                results.extend(found)
                if len(results) >= _MAX_SEARCH_RESULTS:
                    break

        if not results:
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message=f"I couldn't find any files matching '{query}'.",
                action="file_find",
                duration=time.perf_counter() - start,
            )

        count = len(results)
        first = results[0]
        info = _get_file_info(first)

        msg = f"Found {count} file{'s' if count != 1 else ''} matching '{query}'. "
        if info:
            msg += f"Top result: {info.to_speech()} in {first.parent.name}."

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=msg,
            action="file_find",
            duration=time.perf_counter() - start,
        )

    def _handle_open_file(self, match: re.Match) -> ActionResult:
        """Find and open a specific file by name."""
        start = time.perf_counter()
        file_name = (match.group("name") or "").strip()

        if not file_name:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="Which file would you like to open?",
                action="file_open",
            )

        # Search for the file
        search_roots = [
            Path.home() / "Desktop",
            Path.home() / "Documents",
            Path.home() / "Downloads",
        ]

        for root in search_roots:
            if not root.exists():
                continue
            results = _safe_search(root, file_name, max_results=1)
            if results:
                target = results[0]
                if _open_path(target):
                    return ActionResult(
                        status=ActionStatus.SUCCESS,
                        message=f"Opening {target.name}.",
                        action="file_open",
                        duration=time.perf_counter() - start,
                    )

        return ActionResult(
            status=ActionStatus.NOT_FOUND,
            message=f"I couldn't find a file named '{file_name}'.",
            action="file_open",
            duration=time.perf_counter() - start,
        )

    def _handle_list(self, match: re.Match) -> ActionResult:
        """List files in a directory."""
        start = time.perf_counter()
        dir_name = (match.group("dir") or "").strip().lower()

        # Resolve to a path
        target = self._resolve_dir(dir_name)

        if not target or not target.exists():
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message=f"I couldn't find the folder '{dir_name}'.",
                action="file_list",
            )

        try:
            entries = sorted(
                target.iterdir(),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except PermissionError:
            return ActionResult(
                status=ActionStatus.PERMISSION_ERROR,
                message=f"I don't have permission to list '{target.name}'.",
                action="file_list",
            )

        visible = [e for e in entries if not e.name.startswith(".")]

        if not visible:
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"The {target.name} folder is empty.",
                action="file_list",
                duration=time.perf_counter() - start,
            )

        count = len(visible)
        names = ", ".join(e.name for e in visible[:_MAX_LIST_FILES])
        more = f" and {count - _MAX_LIST_FILES} more" if count > _MAX_LIST_FILES else ""

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=f"{target.name} contains {count} item{'s' if count != 1 else ''}: {names}{more}.",
            action="file_list",
            duration=time.perf_counter() - start,
        )

    def _handle_size(self, match: re.Match) -> ActionResult:
        """Report the size of a file or directory."""
        start = time.perf_counter()
        target_name = (match.group("target") or "").strip()

        target = self._resolve_path(target_name)
        if not target or not target.exists():
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message=f"I couldn't find '{target_name}'.",
                action="file_size",
            )

        try:
            if target.is_dir():
                size = _dir_size(target)
                kind = "folder"
            else:
                size = target.stat().st_size
                kind = "file"

            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"The {kind} '{target.name}' is {_format_size(size)}.",
                action="file_size",
                duration=time.perf_counter() - start,
            )
        except (OSError, PermissionError):
            return ActionResult(
                status=ActionStatus.PERMISSION_ERROR,
                message=f"I couldn't read the size of '{target.name}'.",
                action="file_size",
            )

    def _handle_create_folder(self, match: re.Match) -> ActionResult:
        """Create a new folder on the Desktop."""
        start = time.perf_counter()
        fold_name = (match.group("name") or "").strip()

        if not fold_name:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="What would you like to name the folder?",
                action="file_create_folder",
            )

        # Sanitise folder name
        fold_name = re.sub(r'[<>:"/\\|?*]', "", fold_name).strip()
        if not fold_name:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="That folder name contains invalid characters.",
                action="file_create_folder",
            )

        target = Path.home() / "Desktop" / fold_name

        if target.exists():
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"A folder named '{fold_name}' already exists on the Desktop.",
                action="file_create_folder",
            )

        try:
            target.mkdir(parents=True, exist_ok=True)
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Folder '{fold_name}' created on the Desktop.",
                action="file_create_folder",
                duration=time.perf_counter() - start,
            )
        except (OSError, PermissionError) as exc:
            logger.error("[File] Create folder failed: %s", exc)
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message=f"I couldn't create the folder '{fold_name}'.",
                action="file_create_folder",
            )

    def _handle_delete(self, match: re.Match) -> ActionResult:
        """Locate a file and request deletion confirmation."""
        start = time.perf_counter()
        file_name = (match.group("name") or "").strip()

        if not file_name:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="Which file would you like to delete?",
                action="file_delete",
            )

        target = self._resolve_path(file_name)
        if not target or not target.exists():
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message=f"I couldn't find a file named '{file_name}'.",
                action="file_delete",
            )

        if config.security.confirm_destructive:
            self._pending_op = PendingFileOp(
                operation="delete",
                source=target,
            )
            return ActionResult(
                status=ActionStatus.REQUIRES_CONFIRM,
                message=f"Are you sure you want to permanently delete '{target.name}'? Say yes to confirm.",
                action="file_delete",
                duration=time.perf_counter() - start,
            )

        return self._execute_delete(target)

    def _handle_move(self, match: re.Match) -> ActionResult:
        """Move a file to a new location."""
        start = time.perf_counter()
        source_name = (match.group("source") or "").strip()
        dest_name = (match.group("dest") or "").strip()

        source = self._resolve_path(source_name)
        if not source or not source.exists():
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message=f"I couldn't find '{source_name}'.",
                action="file_move",
            )

        dest = self._resolve_dir(dest_name)
        if not dest:
            dest = Path.home() / dest_name

        if config.security.confirm_destructive:
            self._pending_op = PendingFileOp(
                operation="move",
                source=source,
                target=dest,
            )
            return ActionResult(
                status=ActionStatus.REQUIRES_CONFIRM,
                message=f"Move '{source.name}' to '{dest}'? Say yes to confirm.",
                action="file_move",
                duration=time.perf_counter() - start,
            )

        return self._execute_move(source, dest)

    def _handle_recent(self, match: re.Match) -> ActionResult:
        """List recently modified files across common directories."""
        start = time.perf_counter()

        recent: list[FileInfo] = []
        search_roots = [
            Path.home() / "Desktop",
            Path.home() / "Documents",
            Path.home() / "Downloads",
        ]

        for root in search_roots:
            if not root.exists():
                continue
            try:
                for entry in root.iterdir():
                    if entry.is_file() and not entry.name.startswith("."):
                        info = _get_file_info(entry)
                        if info:
                            recent.append(info)
            except PermissionError:
                continue

        if not recent:
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message="I couldn't find any recent files.",
                action="file_recent",
            )

        recent.sort(key=lambda f: f.modified_at, reverse=True)
        top = recent[:5]

        names = ", ".join(f.name for f in top)
        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=f"Recent files: {names}.",
            action="file_recent",
            duration=time.perf_counter() - start,
        )

    # ── Confirmation Resolution ───────────────────────────────────────────────

    def confirm_pending(self, confirmed: bool) -> ActionResult:
        """
        Resolve a pending destructive operation.

        Args:
            confirmed: True if the user confirmed, False to cancel.

        Returns:
            ActionResult from execution or cancellation.
        """
        if not self._pending_op:
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message="No pending operation to confirm.",
                action="confirm",
            )

        op = self._pending_op
        self._pending_op = None

        if op.is_expired:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="The confirmation window expired. Please repeat the command.",
                action="confirm",
            )

        if not confirmed:
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message="Operation cancelled.",
                action="confirm",
            )

        if op.operation == "delete":
            return self._execute_delete(op.source)

        if op.operation == "move" and op.target:
            return self._execute_move(op.source, op.target)

        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message="Unknown operation type.",
            action="confirm",
        )

    # ── Execution Helpers ─────────────────────────────────────────────────────

    def _execute_delete(self, target: Path) -> ActionResult:
        """Permanently delete a file or directory."""
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            logger.info("[File] Deleted: '%s'", target)
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"'{target.name}' deleted.",
                action="file_delete",
            )
        except PermissionError:
            return ActionResult(
                status=ActionStatus.PERMISSION_ERROR,
                message=f"I don't have permission to delete '{target.name}'.",
                action="file_delete",
            )
        except OSError as exc:
            logger.error("[File] Delete failed for '%s': %s", target, exc)
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message=f"I couldn't delete '{target.name}'.",
                action="file_delete",
            )

    def _execute_move(self, source: Path, dest: Path) -> ActionResult:
        """Move a file or directory to a new location."""
        try:
            dest.mkdir(parents=True, exist_ok=True)
            final = dest / source.name
            shutil.move(str(source), str(final))
            logger.info("[File] Moved '%s' → '%s'", source, final)
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Moved '{source.name}' to '{dest.name}'.",
                action="file_move",
            )
        except PermissionError:
            return ActionResult(
                status=ActionStatus.PERMISSION_ERROR,
                message=f"I don't have permission to move '{source.name}'.",
                action="file_move",
            )
        except (OSError, shutil.Error) as exc:
            logger.error("[File] Move failed: %s", exc)
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message=f"I couldn't move '{source.name}'.",
                action="file_move",
            )

    # ── Path Resolution ───────────────────────────────────────────────────────

    def _resolve_dir(self, name: str) -> Path | None:
        """
        Resolve a directory name to an absolute Path.
        Checks known aliases first, then searches common locations.
        """
        name_lower = name.lower().strip()

        # Check known directory aliases
        for alias, path in self._known_dirs.items():
            if alias in name_lower or name_lower in alias:
                return path

        # Check if it's an absolute path
        candidate = Path(name)
        if candidate.is_absolute() and candidate.exists():
            return candidate

        # Search common parent directories
        for parent in (Path.home(), Path.home() / "Desktop", Path.home() / "Documents"):
            candidate = parent / name
            if candidate.exists() and candidate.is_dir():
                return candidate

        return None

    def _resolve_path(self, name: str) -> Path | None:
        """
        Resolve a file or directory name to an absolute Path.
        Returns None if not found.
        """
        # Absolute path given
        candidate = Path(name)
        if candidate.is_absolute() and candidate.exists():
            return candidate

        # Search common directories
        search_roots = [
            Path.home() / "Desktop",
            Path.home() / "Documents",
            Path.home() / "Downloads",
        ]

        for root in search_roots:
            if not root.exists():
                continue
            results = _safe_search(root, name, max_results=1)
            if results:
                return results[0]

        return None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def has_pending_op(self) -> bool:
        return self._pending_op is not None and not self._pending_op.is_expired
