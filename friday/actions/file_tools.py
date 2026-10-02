"""
F.R.I.D.A.Y. File Management Tools
====================================
Safe file operations with path validation.
All write/move/delete operations are HIGH or CRITICAL risk
and require user confirmation before execution.
"""

import os
import shutil

# ── Safety: restricted paths that should never be touched ───

_FORBIDDEN_PATHS = [
    os.environ.get("SYSTEMROOT", r"C:\Windows"),
    r"C:\Program Files",
    r"C:\Program Files (x86)",
]


def _is_safe_path(path: str) -> bool:
    """Check that a path is not inside a protected system directory."""
    abs_path = os.path.abspath(path)
    return all(not abs_path.lower().startswith(forbidden.lower()) for forbidden in _FORBIDDEN_PATHS)


# ── Tool implementations ────────────────────────────────────


def file_exists(path: str = "") -> str:
    """Check if a file or directory exists."""
    if not path:
        return "No path provided."
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        kind = "directory" if os.path.isdir(abs_path) else "file"
        size = os.path.getsize(abs_path) if os.path.isfile(abs_path) else 0
        return f"Yes, {kind} exists: {abs_path} ({size:,} bytes)"
    return f"No, '{abs_path}' does not exist."


def file_list(directory: str = ".") -> str:
    """List files and folders in a directory."""
    abs_dir = os.path.abspath(directory)
    if not os.path.isdir(abs_dir):
        return f"'{abs_dir}' is not a directory."

    try:
        entries = os.listdir(abs_dir)
        if not entries:
            return f"'{abs_dir}' is empty."

        items = []
        for entry in sorted(entries)[:50]:  # Limit to 50 entries
            full = os.path.join(abs_dir, entry)
            if os.path.isdir(full):
                items.append(f"  [DIR]  {entry}")
            else:
                size = os.path.getsize(full)
                items.append(f"  [FILE] {entry} ({size:,} bytes)")

        result = f"Contents of {abs_dir}:\n" + "\n".join(items)
        if len(entries) > 50:
            result += f"\n  ... and {len(entries) - 50} more items"
        return result
    except PermissionError:
        return f"Permission denied: '{abs_dir}'"
    except Exception as e:
        return f"List error: {e}"


def file_read(path: str = "") -> str:
    """Read the contents of a text file."""
    if not path:
        return "No file path provided."

    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        return f"File not found: {abs_path}"

    try:
        size = os.path.getsize(abs_path)
        if size > 100_000:  # 100KB limit
            return f"File too large to read ({size:,} bytes). Max is 100KB."

        with open(abs_path, encoding="utf-8", errors="replace") as f:
            content = f.read()

        if len(content) > 3000:
            content = content[:3000] + "\n... [truncated]"

        return f"[File: {abs_path}]\n{content}"
    except Exception as e:
        return f"Read error: {e}"


def file_write(path: str = "", content: str = "") -> str:
    """Write content to a file. Creates parent directories if needed."""
    if not path:
        return "No file path provided."
    if not content:
        return "No content provided to write."

    abs_path = os.path.abspath(path)
    if not _is_safe_path(abs_path):
        return f"BLOCKED: Cannot write to protected path: {abs_path}"

    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} chars to: {abs_path}"
    except Exception as e:
        return f"Write error: {e}"


def file_move(src: str = "", dst: str = "") -> str:
    """Move or rename a file."""
    if not src or not dst:
        return "Both source and destination paths required."

    abs_src = os.path.abspath(src)
    abs_dst = os.path.abspath(dst)

    if not os.path.exists(abs_src):
        return f"Source not found: {abs_src}"
    if not _is_safe_path(abs_src) or not _is_safe_path(abs_dst):
        return "BLOCKED: Cannot move files in protected system directories."

    try:
        shutil.move(abs_src, abs_dst)
        return f"Moved: {abs_src} -> {abs_dst}"
    except Exception as e:
        return f"Move error: {e}"


def file_delete(path: str = "") -> str:
    """Delete a file. CRITICAL risk — requires secure confirmation."""
    if not path:
        return "No file path provided."

    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return f"File not found: {abs_path}"
    if not _is_safe_path(abs_path):
        return f"BLOCKED: Cannot delete files in protected path: {abs_path}"

    try:
        if os.path.isfile(abs_path):
            os.remove(abs_path)
            return f"Deleted file: {abs_path}"
        elif os.path.isdir(abs_path):
            # Only delete empty directories for safety
            if not os.listdir(abs_path):
                os.rmdir(abs_path)
                return f"Deleted empty directory: {abs_path}"
            else:
                return f"BLOCKED: Directory '{abs_path}' is not empty. Will not delete non-empty directories."
        else:
            return f"BLOCKED: Path is not a regular file or directory: {abs_path}"
    except Exception as e:
        return f"Delete error: {e}"


def file_search(pattern: str = "", directory: str = ".") -> str:
    """Search for files in a directory matching a pattern/substring."""
    if not pattern:
        return "No search pattern provided."
    abs_dir = os.path.abspath(directory)
    if not os.path.isdir(abs_dir):
        return f"'{abs_dir}' is not a directory."
    try:
        matches = []
        for root, dirs, files in os.walk(abs_dir):
            # Safety check on traversed subdirectories
            if not _is_safe_path(root):
                continue
            for file in files:
                if pattern.lower() in file.lower():
                    full_path = os.path.join(root, file)
                    matches.append(full_path)
                    if len(matches) >= 50:
                        break
            if len(matches) >= 50:
                break
        if not matches:
            return f"No files matching '{pattern}' found in '{abs_dir}'."
        res = f"Found {len(matches)} matches in '{abs_dir}':\n"
        res += "\n".join(f"  {m}" for m in matches)
        if len(matches) >= 50:
            res += "\n  ... and possibly more (capped at 50 matches)"
        return res
    except Exception as e:
        return f"Search error: {e}"


def file_create_folder(path: str = "") -> str:
    """Create a new folder (including parent directories if needed)."""
    if not path:
        return "No folder path provided."
    abs_path = os.path.abspath(path)
    if not _is_safe_path(abs_path):
        return f"BLOCKED: Cannot create folder in protected path: {abs_path}"
    if os.path.exists(abs_path):
        if os.path.isdir(abs_path):
            return f"Directory already exists: {abs_path}"
        return f"Path exists but is a file, not a directory: {abs_path}"
    try:
        os.makedirs(abs_path, exist_ok=True)
        return f"Created folder: {abs_path}"
    except Exception as e:
        return f"Create folder error: {e}"


def file_get_info(path: str = "") -> str:
    """Get metadata info for a file or directory."""
    if not path:
        return "No path provided."
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return f"'{abs_path}' does not exist."
    try:
        stat = os.stat(abs_path)
        kind = "directory" if os.path.isdir(abs_path) else "file"
        import datetime
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        ctime = datetime.datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
        size = stat.st_size
        info = [
            f"Path: {abs_path}",
            f"Type: {kind}",
            f"Size: {size:,} bytes" if kind == "file" else f"Size: directory structure",
            f"Last Modified: {mtime}",
            f"Created: {ctime}",
        ]
        return "\n".join(info)
    except Exception as e:
        return f"Error getting file info: {e}"


def file_copy(src: str = "", dst: str = "") -> str:
    """Copy a file from src to dst."""
    if not src or not dst:
        return "Both source and destination paths required."

    abs_src = os.path.abspath(src)
    abs_dst = os.path.abspath(dst)

    if not os.path.isfile(abs_src):
        return f"Source file not found: {abs_src}"
    if not _is_safe_path(abs_src) or not _is_safe_path(abs_dst):
        return "BLOCKED: Cannot copy files in protected system directories."

    try:
        os.makedirs(os.path.dirname(abs_dst), exist_ok=True)
        shutil.copy2(abs_src, abs_dst)
        return f"Copied: {abs_src} -> {abs_dst}"
    except Exception as e:
        return f"Copy error: {e}"
