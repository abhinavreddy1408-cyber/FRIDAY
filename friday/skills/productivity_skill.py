"""
F.R.I.D.A.Y. Productivity Skill
===============================
Manages notes, todos, clipboard content, and quick references.
Supports tagging, search, priorities, and due dates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import pyperclip

from friday.config import config

logger = logging.getLogger(__name__)

# ── Constants & Configuration ─────────────────────────────────────────────────


# Todo priorities
class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TodoStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ARCHIVED = "archived"


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass
class ProductivityResult:
    """Standardized result structure for all productivity operations."""

    success: bool
    message: str
    data: dict[str, Any]
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Note:
    """Represents a single note."""

    note_id: str
    title: str
    content: str
    tags: list[str]
    created_at: float
    updated_at: float
    word_count: int
    path: str


@dataclass
class TodoItem:
    """Represents a single todo item."""

    todo_id: str
    task: str
    status: TodoStatus
    priority: Priority
    created_at: float
    due_date: float | None
    completed_at: float | None
    tags: list[str]
    notes: str


@dataclass
class ClipboardEntry:
    """Represents a clipboard history entry."""

    entry_id: str
    content: str
    content_type: str  # text, url, email, etc.
    copied_at: float
    source: str


# ── Productivity Manager ──────────────────────────────────────────────────────


class ProductivityManager:
    """
    Centralized manager for productivity operations.
    Handles notes, todos, clipboard history, and quick references.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._logger = logging.getLogger("ProductivityManager")

        # Initialize paths from config
        self._notes_dir = config.storage.data_dir / "notes"
        self._todo_file = config.storage.data_dir / "todos.json"
        self._clipboard_file = config.storage.data_dir / "clipboard_history.json"

        # Clipboard history settings
        self._max_clipboard_entries = 50

        self._init_storage()
        self._logger.info("[Productivity] Productivity manager initialized.")

    def _init_storage(self):
        """Initialize storage directories and files."""
        try:
            self._notes_dir.mkdir(parents=True, exist_ok=True)

            if not self._todo_file.exists():
                self._todo_file.write_text(json.dumps([], indent=2))
                self._logger.debug("[Productivity] Created todo file")

            if not self._clipboard_file.exists():
                self._clipboard_file.write_text(json.dumps([], indent=2))
                self._logger.debug("[Productivity] Created clipboard history file")

        except Exception as e:
            self._logger.error(f"[Productivity] Storage initialization failed: {e}")
            raise

    def _generate_id(self, content: str) -> str:
        """Generate a unique ID based on content hash and timestamp."""
        timestamp = int(time.time() * 1000)
        hash_val = hashlib.md5(f"{content}{timestamp}".encode()).hexdigest()[:12]
        return f"{timestamp}_{hash_val}"

    def _sanitize_filename(self, title: str) -> str:
        """Sanitize a title for use as a filename."""
        sanitized = "".join(x for x in title if x.isalnum() or x in " -_").strip()
        return sanitized[:50] if len(sanitized) > 50 else sanitized

    # ── Note Management ───────────────────────────────────────────────────────

    def create_note(
        self, title: str, content: str, tags: list[str] | None = None
    ) -> ProductivityResult:
        """Create a new text note with optional tags."""
        try:
            with self._lock:
                note_id = self._generate_id(title)
                filename = self._sanitize_filename(title) + ".txt"
                path = self._notes_dir / filename

                # Handle duplicate filenames
                counter = 1
                while path.exists():
                    filename = f"{self._sanitize_filename(title)}_{counter}.txt"
                    path = self._notes_dir / filename
                    counter += 1

                now = time.time()

                note = Note(
                    note_id=note_id,
                    title=title,
                    content=content,
                    tags=tags or [],
                    created_at=now,
                    updated_at=now,
                    word_count=len(content.split()),
                    path=str(path),
                )

                # Save note content
                path.write_text(content, encoding="utf-8")

                # Save note metadata
                self._save_note_metadata(note)

                self._logger.info(f"[Productivity] Created note: {title}")

                return ProductivityResult(
                    success=True,
                    message=f"Note '{title}' saved, Sir.",
                    data={"note_id": note_id, "path": str(path), "word_count": note.word_count},
                    timestamp=now,
                )

        except Exception as e:
            self._logger.error(f"[Productivity] Note creation failed: {e}")
            return ProductivityResult(
                success=False, message=f"Failed to create note: {e}", data={}, timestamp=time.time()
            )

    def _save_note_metadata(self, note: Note):
        """Save note metadata to index file."""
        index_file = self._notes_dir / "notes_index.json"

        notes = []
        if index_file.exists():
            try:
                notes = json.loads(index_file.read_text())
            except Exception:
                notes = []

        # Remove existing entry for this note_id if present
        notes = [n for n in notes if n.get("note_id") != note.note_id]
        notes.append(asdict(note))

        index_file.write_text(json.dumps(notes, indent=2))

    def read_note(self, title: str) -> ProductivityResult:
        """Read a note by title (partial match supported)."""
        try:
            with self._lock:
                # Search for matching note
                note_path = None
                note_title = None

                for f in self._notes_dir.glob("*.txt"):
                    if f.name == "notes_index.txt":
                        continue
                    if title.lower() in f.stem.lower():
                        note_path = f
                        note_title = f.stem
                        break

                if not note_path or not note_path.exists():
                    return ProductivityResult(
                        success=False,
                        message=f"Note '{title}' not found.",
                        data={},
                        timestamp=time.time(),
                    )

                content = note_path.read_text(encoding="utf-8")

                self._logger.info(f"[Productivity] Read note: {note_title}")

                return ProductivityResult(
                    success=True,
                    message=f"Found note '{note_title}', Sir.",
                    data={
                        "title": note_title,
                        "content": content,
                        "path": str(note_path),
                        "word_count": len(content.split()),
                    },
                    timestamp=time.time(),
                )

        except Exception as e:
            self._logger.error(f"[Productivity] Note read failed: {e}")
            return ProductivityResult(
                success=False, message=f"Failed to read note: {e}", data={}, timestamp=time.time()
            )

    def update_note(self, title: str, new_content: str) -> ProductivityResult:
        """Update an existing note's content."""
        try:
            with self._lock:
                # Find the note
                note_path = None
                for f in self._notes_dir.glob("*.txt"):
                    if title.lower() in f.stem.lower():
                        note_path = f
                        break

                if not note_path or not note_path.exists():
                    return ProductivityResult(
                        success=False,
                        message=f"Note '{title}' not found.",
                        data={},
                        timestamp=time.time(),
                    )

                # Update content
                note_path.write_text(new_content, encoding="utf-8")

                # Update metadata
                index_file = self._notes_dir / "notes_index.json"
                if index_file.exists():
                    notes = json.loads(index_file.read_text())
                    for note in notes:
                        if title.lower() in note.get("title", "").lower():
                            note["content"] = new_content
                            note["updated_at"] = time.time()
                            note["word_count"] = len(new_content.split())
                    index_file.write_text(json.dumps(notes, indent=2))

                self._logger.info(f"[Productivity] Updated note: {title}")

                return ProductivityResult(
                    success=True,
                    message=f"Note '{title}' updated, Sir.",
                    data={"path": str(note_path), "word_count": len(new_content.split())},
                    timestamp=time.time(),
                )

        except Exception as e:
            self._logger.error(f"[Productivity] Note update failed: {e}")
            return ProductivityResult(
                success=False, message=f"Failed to update note: {e}", data={}, timestamp=time.time()
            )

    def delete_note(self, title: str) -> ProductivityResult:
        """Delete a note by title."""
        try:
            with self._lock:
                note_path = None
                for f in self._notes_dir.glob("*.txt"):
                    if title.lower() in f.stem.lower():
                        note_path = f
                        break

                if not note_path or not note_path.exists():
                    return ProductivityResult(
                        success=False,
                        message=f"Note '{title}' not found.",
                        data={},
                        timestamp=time.time(),
                    )

                note_path.unlink()

                # Update index
                index_file = self._notes_dir / "notes_index.json"
                if index_file.exists():
                    notes = json.loads(index_file.read_text())
                    notes = [n for n in notes if title.lower() not in n.get("title", "").lower()]
                    index_file.write_text(json.dumps(notes, indent=2))

                self._logger.info(f"[Productivity] Deleted note: {title}")

                return ProductivityResult(
                    success=True,
                    message=f"Note '{title}' deleted, Sir.",
                    data={},
                    timestamp=time.time(),
                )

        except Exception as e:
            self._logger.error(f"[Productivity] Note deletion failed: {e}")
            return ProductivityResult(
                success=False, message=f"Failed to delete note: {e}", data={}, timestamp=time.time()
            )

    def list_notes(self, tag_filter: str | None = None) -> ProductivityResult:
        """List all notes, optionally filtered by tag."""
        try:
            with self._lock:
                index_file = self._notes_dir / "notes_index.json"
                notes = []

                if index_file.exists():
                    notes = json.loads(index_file.read_text())

                # Filter by tag if specified
                if tag_filter:
                    notes = [
                        n
                        for n in notes
                        if tag_filter.lower() in [t.lower() for t in n.get("tags", [])]
                    ]

                # Sort by updated_at (newest first)
                notes.sort(key=lambda x: x.get("updated_at", 0), reverse=True)

                self._logger.info(f"[Productivity] Listed {len(notes)} notes")

                return ProductivityResult(
                    success=True,
                    message=f"You have {len(notes)} notes saved, Boss."
                    + (f" (filtered by tag '{tag_filter}')" if tag_filter else ""),
                    data={
                        "count": len(notes),
                        "notes": [
                            {"title": n["title"], "updated_at": n["updated_at"], "tags": n["tags"]}
                            for n in notes[:20]
                        ],  # Limit response
                    },
                    timestamp=time.time(),
                )

        except Exception as e:
            self._logger.error(f"[Productivity] Note list failed: {e}")
            return ProductivityResult(
                success=False, message=f"Failed to list notes: {e}", data={}, timestamp=time.time()
            )

    def search_notes(self, query: str) -> ProductivityResult:
        """Search notes by content."""
        try:
            with self._lock:
                results = []

                for f in self._notes_dir.glob("*.txt"):
                    if f.name == "notes_index.txt":
                        continue
                    try:
                        content = f.read_text(encoding="utf-8")
                        if query.lower() in content.lower():
                            results.append(
                                {
                                    "title": f.stem,
                                    "path": str(f),
                                    "preview": content[:200] + "..."
                                    if len(content) > 200
                                    else content,
                                }
                            )
                    except Exception:
                        continue

                self._logger.info(
                    f"[Productivity] Search found {len(results)} matches for '{query}'"
                )

                return ProductivityResult(
                    success=True,
                    message=f"Found {len(results)} notes matching '{query}', Sir.",
                    data={"query": query, "results": results},
                    timestamp=time.time(),
                )

        except Exception as e:
            self._logger.error(f"[Productivity] Note search failed: {e}")
            return ProductivityResult(
                success=False,
                message=f"Failed to search notes: {e}",
                data={},
                timestamp=time.time(),
            )

    # ── Todo Management ───────────────────────────────────────────────────────

    def add_todo(
        self,
        task: str,
        priority: Priority = Priority.MEDIUM,
        due_date_days: int | None = None,
        tags: list[str] | None = None,
        notes: str = "",
    ) -> ProductivityResult:
        """Add a todo item with optional priority and due date."""
        try:
            with self._lock:
                todos = self._load_todos()

                now = time.time()
                due_date = None
                if due_date_days:
                    due_date = now + (due_date_days * 24 * 60 * 60)

                todo = TodoItem(
                    todo_id=self._generate_id(task),
                    task=task,
                    status=TodoStatus.PENDING,
                    priority=priority,
                    created_at=now,
                    due_date=due_date,
                    completed_at=None,
                    tags=tags or [],
                    notes=notes,
                )

                todos.append(asdict(todo))
                self._save_todos(todos)

                self._logger.info(f"[Productivity] Added todo: {task}")

                msg = f"Added '{task}' to your todo list, Sir."
                if due_date:
                    due_str = datetime.fromtimestamp(due_date).strftime("%Y-%m-%d")
                    msg += f" Due: {due_str}."

                return ProductivityResult(
                    success=True,
                    message=msg,
                    data={"todo_id": todo.todo_id, "priority": priority.value},
                    timestamp=now,
                )

        except Exception as e:
            self._logger.error(f"[Productivity] Todo add failed: {e}")
            return ProductivityResult(
                success=False, message=f"Failed to add todo: {e}", data={}, timestamp=time.time()
            )

    def _load_todos(self) -> list[dict]:
        """Load todos from file."""
        if not self._todo_file.exists():
            return []
        try:
            return json.loads(self._todo_file.read_text())
        except Exception:
            return []

    def _save_todos(self, todos: list[dict]):
        """Save todos to file."""
        self._todo_file.write_text(json.dumps(todos, indent=2))

    def list_todos(
        self,
        status_filter: TodoStatus | None = None,
        priority_filter: Priority | None = None,
        limit: int = 20,
    ) -> ProductivityResult:
        """List todos with optional filters."""
        try:
            with self._lock:
                todos = self._load_todos()

                # Apply filters
                if status_filter:
                    todos = [t for t in todos if t.get("status") == status_filter.value]
                if priority_filter:
                    todos = [t for t in todos if t.get("priority") == priority_filter.value]

                # Sort by priority (critical first) then by created_at
                priority_order = {
                    Priority.CRITICAL.value: 0,
                    Priority.HIGH.value: 1,
                    Priority.MEDIUM.value: 2,
                    Priority.LOW.value: 3,
                }
                todos.sort(
                    key=lambda x: (priority_order.get(x.get("priority"), 2), x.get("created_at", 0))
                )

                pending_count = len(
                    [t for t in todos if t.get("status") == TodoStatus.PENDING.value]
                )

                self._logger.info(f"[Productivity] Listed {len(todos)} todos")

                return ProductivityResult(
                    success=True,
                    message=f"You have {pending_count} pending tasks, Boss.",
                    data={
                        "total": len(todos),
                        "pending": pending_count,
                        "todos": [
                            {
                                "task": t["task"],
                                "priority": t["priority"],
                                "status": t["status"],
                                "due_date": datetime.fromtimestamp(t["due_date"]).strftime(
                                    "%Y-%m-%d"
                                )
                                if t.get("due_date")
                                else None,
                            }
                            for t in todos[:limit]
                        ],
                    },
                    timestamp=time.time(),
                )

        except Exception as e:
            self._logger.error(f"[Productivity] Todo list failed: {e}")
            return ProductivityResult(
                success=False, message=f"Failed to list todos: {e}", data={}, timestamp=time.time()
            )

    def complete_todo(self, task_query: str) -> ProductivityResult:
        """Mark a todo as completed by task name (partial match)."""
        try:
            with self._lock:
                todos = self._load_todos()

                # Find matching todo
                matched_index = None
                for i, todo in enumerate(todos):
                    if task_query.lower() in todo.get("task", "").lower():  # noqa: SIM102
                        if todo.get("status") != TodoStatus.COMPLETED.value:
                            matched_index = i
                            break

                if matched_index is None:
                    return ProductivityResult(
                        success=False,
                        message=f"No pending todo matching '{task_query}' found.",
                        data={},
                        timestamp=time.time(),
                    )

                # Update status
                todos[matched_index]["status"] = TodoStatus.COMPLETED.value
                todos[matched_index]["completed_at"] = time.time()
                self._save_todos(todos)

                self._logger.info(f"[Productivity] Completed todo: {todos[matched_index]['task']}")

                return ProductivityResult(
                    success=True,
                    message=f"Marked task '{todos[matched_index]['task']}' as completed, Sir.",
                    data={"task": todos[matched_index]["task"]},
                    timestamp=time.time(),
                )

        except Exception as e:
            self._logger.error(f"[Productivity] Todo complete failed: {e}")
            return ProductivityResult(
                success=False,
                message=f"Failed to complete todo: {e}",
                data={},
                timestamp=time.time(),
            )

    def delete_todo(self, task_query: str) -> ProductivityResult:
        """Delete a todo item by task name."""
        try:
            with self._lock:
                todos = self._load_todos()

                # Find matching todo
                matched_index = None
                for i, todo in enumerate(todos):
                    if task_query.lower() in todo.get("task", "").lower():
                        matched_index = i
                        break

                if matched_index is None:
                    return ProductivityResult(
                        success=False,
                        message=f"No todo matching '{task_query}' found.",
                        data={},
                        timestamp=time.time(),
                    )

                deleted_task = todos[matched_index]["task"]
                todos.pop(matched_index)
                self._save_todos(todos)

                self._logger.info(f"[Productivity] Deleted todo: {deleted_task}")

                return ProductivityResult(
                    success=True,
                    message=f"Deleted task '{deleted_task}' from your list, Sir.",
                    data={"task": deleted_task},
                    timestamp=time.time(),
                )

        except Exception as e:
            self._logger.error(f"[Productivity] Todo delete failed: {e}")
            return ProductivityResult(
                success=False, message=f"Failed to delete todo: {e}", data={}, timestamp=time.time()
            )

    # ── Clipboard Management ──────────────────────────────────────────────────

    def clipboard_save(self, label: str) -> ProductivityResult:
        """Save current clipboard content as a named note."""
        try:
            content = pyperclip.paste()
            if not content or not content.strip():
                return ProductivityResult(
                    success=False,
                    message="Clipboard is empty, Sir.",
                    data={},
                    timestamp=time.time(),
                )

            # Detect content type
            content_type = "text"
            if content.startswith("http://") or content.startswith("https://"):
                content_type = "url"
            elif "@" in content and "." in content:
                content_type = "email"

            # Save as note
            result = self.create_note(
                title=label, content=content, tags=["clipboard", content_type]
            )

            # Also save to clipboard history
            self._save_clipboard_entry(content, content_type, label)

            return result

        except Exception as e:
            self._logger.error(f"[Productivity] Clipboard save failed: {e}")
            return ProductivityResult(
                success=False,
                message=f"Failed to save clipboard: {e}",
                data={},
                timestamp=time.time(),
            )

    def _save_clipboard_entry(self, content: str, content_type: str, source: str = "manual"):
        """Save clipboard content to history."""
        try:
            history = []
            if self._clipboard_file.exists():
                history = json.loads(self._clipboard_file.read_text())

            entry = ClipboardEntry(
                entry_id=self._generate_id(content),
                content=content[:1000],  # Limit stored content
                content_type=content_type,
                copied_at=time.time(),
                source=source,
            )

            history.insert(0, asdict(entry))

            # Trim to max entries
            history = history[: self._max_clipboard_entries]

            self._clipboard_file.write_text(json.dumps(history, indent=2))

        except Exception as e:
            self._logger.error(f"[Productivity] Clipboard history save failed: {e}")

    def clipboard_history(self, limit: int = 10) -> ProductivityResult:
        """Get recent clipboard history."""
        try:
            with self._lock:
                history = []
                if self._clipboard_file.exists():
                    history = json.loads(self._clipboard_file.read_text())

                recent = history[:limit]

                return ProductivityResult(
                    success=True,
                    message=f"Retrieved {len(recent)} clipboard entries, Sir.",
                    data={
                        "count": len(recent),
                        "entries": [
                            {
                                "content_type": e["content_type"],
                                "copied_at": datetime.fromtimestamp(e["copied_at"]).strftime(
                                    "%H:%M:%S"
                                ),
                                "preview": e["content"][:50] + "..."
                                if len(e["content"]) > 50
                                else e["content"],
                            }
                            for e in recent
                        ],
                    },
                    timestamp=time.time(),
                )

        except Exception as e:
            self._logger.error(f"[Productivity] Clipboard history failed: {e}")
            return ProductivityResult(
                success=False,
                message=f"Failed to get clipboard history: {e}",
                data={},
                timestamp=time.time(),
            )

    def clipboard_copy(self, text: str) -> ProductivityResult:
        """Copy text to system clipboard."""
        try:
            pyperclip.copy(text)

            # Save to history
            self._save_clipboard_entry(text, "text", "copy_command")

            return ProductivityResult(
                success=True,
                message="Copied to clipboard, Sir.",
                data={"length": len(text)},
                timestamp=time.time(),
            )
        except Exception as e:
            self._logger.error(f"[Productivity] Clipboard copy failed: {e}")
            return ProductivityResult(
                success=False,
                message=f"Failed to copy to clipboard: {e}",
                data={},
                timestamp=time.time(),
            )

    # ── Statistics & Cleanup ──────────────────────────────────────────────────

    def get_stats(self) -> ProductivityResult:
        """Get productivity statistics."""
        try:
            with self._lock:
                # Count notes
                note_count = len(list(self._notes_dir.glob("*.txt"))) - 1  # Exclude index
                if note_count < 0:
                    note_count = 0

                # Count todos
                todos = self._load_todos()
                pending = len([t for t in todos if t.get("status") == TodoStatus.PENDING.value])
                completed = len([t for t in todos if t.get("status") == TodoStatus.COMPLETED.value])

                # Count clipboard entries
                clipboard_count = 0
                if self._clipboard_file.exists():
                    clipboard_count = len(json.loads(self._clipboard_file.read_text()))

                return ProductivityResult(
                    success=True,
                    message="Productivity stats retrieved, Sir.",
                    data={
                        "notes": note_count,
                        "todos_pending": pending,
                        "todos_completed": completed,
                        "clipboard_entries": clipboard_count,
                    },
                    timestamp=time.time(),
                )

        except Exception as e:
            self._logger.error(f"[Productivity] Stats failed: {e}")
            return ProductivityResult(
                success=False, message=f"Failed to get stats: {e}", data={}, timestamp=time.time()
            )

    def cleanup(self, days_old: int = 30) -> ProductivityResult:
        """Clean up old completed todos and clipboard entries."""
        try:
            with self._lock:
                cleaned = 0
                cutoff = time.time() - (days_old * 24 * 60 * 60)

                # Clean old completed todos
                todos = self._load_todos()
                original_count = len(todos)
                todos = [
                    t
                    for t in todos
                    if t.get("status") != TodoStatus.COMPLETED.value
                    or (t.get("completed_at") and t["completed_at"] > cutoff)
                ]
                if len(todos) < original_count:
                    self._save_todos(todos)
                    cleaned += original_count - len(todos)

                # Clean old clipboard entries
                if self._clipboard_file.exists():
                    history = json.loads(self._clipboard_file.read_text())
                    original_count = len(history)
                    history = [e for e in history if e["copied_at"] > cutoff]
                    if len(history) < original_count:
                        self._clipboard_file.write_text(json.dumps(history, indent=2))
                        cleaned += original_count - len(history)

                self._logger.info(f"[Productivity] Cleaned up {cleaned} old entries")

                return ProductivityResult(
                    success=True,
                    message=f"Cleaned up {cleaned} old entries, Sir.",
                    data={"cleaned": cleaned},
                    timestamp=time.time(),
                )

        except Exception as e:
            self._logger.error(f"[Productivity] Cleanup failed: {e}")
            return ProductivityResult(
                success=False, message=f"Failed to cleanup: {e}", data={}, timestamp=time.time()
            )


# ── Legacy Function Wrappers ──────────────────────────────────────────────────

# Global productivity manager instance
_productivity_manager: ProductivityManager | None = None


def get_productivity_manager() -> ProductivityManager:
    """Get or create the global productivity manager singleton."""
    global _productivity_manager
    if _productivity_manager is None:
        _productivity_manager = ProductivityManager()
    return _productivity_manager


def note_create(title: str, content: str) -> dict[str, Any]:
    return get_productivity_manager().create_note(title, content).to_dict()


def note_read(title: str) -> dict[str, Any]:
    return get_productivity_manager().read_note(title).to_dict()


def note_list() -> dict[str, Any]:
    return get_productivity_manager().list_notes().to_dict()


def clipboard_save(label: str) -> dict[str, Any]:
    return get_productivity_manager().clipboard_save(label).to_dict()


def todo_add(task: str) -> dict[str, Any]:
    return get_productivity_manager().add_todo(task).to_dict()


def todo_list() -> dict[str, Any]:
    return get_productivity_manager().list_todos().to_dict()


def todo_complete(index: int) -> dict[str, Any]:
    # Legacy used index, new version uses task name
    # This wrapper maintains backward compatibility
    mgr = get_productivity_manager()
    todos = mgr._load_todos()
    if index < len(todos):
        return mgr.complete_todo(todos[index]["task"]).to_dict()
    return {"success": False, "message": "Index out of range."}


# ── Test Suite ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Test Productivity Skill functionality."""
    print("=" * 60)
    print("F.R.I.D.A.Y. Productivity Skill Test Suite")
    print("=" * 60)

    prod_mgr = ProductivityManager()

    # ── Test 1: Create Note ──────────────────────────────────────────────────
    print("\n[TEST 1] Create Note...")
    result = prod_mgr.create_note("Test Note", "This is test content for F.R.I.D.A.Y.")
    print(f"   Success: {result.success}")
    print(f"   Message: {result.message}")

    # ── Test 2: Read Note ────────────────────────────────────────────────────
    print("\n[TEST 2] Read Note...")
    result = prod_mgr.read_note("Test Note")
    print(f"   Success: {result.success}")
    if result.data:
        print(f"   Content Preview: {result.data.get('content', '')[:50]}...")

    # ── Test 3: List Notes ───────────────────────────────────────────────────
    print("\n[TEST 3] List Notes...")
    result = prod_mgr.list_notes()
    print(f"   Success: {result.success}")
    print(f"   Message: {result.message}")

    # ── Test 4: Search Notes ─────────────────────────────────────────────────
    print("\n[TEST 4] Search Notes...")
    result = prod_mgr.search_notes("test")
    print(f"   Success: {result.success}")
    print(f"   Message: {result.message}")

    # ── Test 5: Add Todo ─────────────────────────────────────────────────────
    print("\n[TEST 5] Add Todo...")
    result = prod_mgr.add_todo("Test task", priority=Priority.HIGH, due_date_days=7)
    print(f"   Success: {result.success}")
    print(f"   Message: {result.message}")

    # ── Test 6: List Todos ───────────────────────────────────────────────────
    print("\n[TEST 6] List Todos...")
    result = prod_mgr.list_todos()
    print(f"   Success: {result.success}")
    print(f"   Message: {result.message}")

    # ── Test 7: Complete Todo ────────────────────────────────────────────────
    print("\n[TEST 7] Complete Todo...")
    result = prod_mgr.complete_todo("Test task")
    print(f"   Success: {result.success}")
    print(f"   Message: {result.message}")

    # ── Test 8: Clipboard Operations ─────────────────────────────────────────
    print("\n[TEST 8] Clipboard Operations...")
    result = prod_mgr.clipboard_copy("Test clipboard content")
    print(f"   Copy Success: {result.success}")
    result = prod_mgr.clipboard_history(limit=5)
    print(f"   History Success: {result.success}")
    print(f"   Entries: {result.data.get('count', 0)}")

    # ── Test 9: Statistics ───────────────────────────────────────────────────
    print("\n[TEST 9] Get Statistics...")
    result = prod_mgr.get_stats()
    print(f"   Success: {result.success}")
    print(f"   Data: {result.data}")

    # ── Test 10: Cleanup ─────────────────────────────────────────────────────
    print("\n[TEST 10] Cleanup Old Entries...")
    result = prod_mgr.cleanup(days_old=30)
    print(f"   Success: {result.success}")
    print(f"   Message: {result.message}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Productivity Skill tests completed!")
    print("=" * 60)
