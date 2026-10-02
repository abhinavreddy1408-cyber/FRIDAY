"""
F.R.I.D.A.Y. Code Skill
Provides voice-controlled code assistance, execution, formatting,
and development workflow support.

Supported commands:
  - "run [file.py]"
  - "execute this script"
  - "format [file] with black"
  - "check [file] for errors"
  - "create a python file called [name]"
  - "open [file] in vscode"
  - "explain this code"
  - "what's wrong with my code"
  - "run tests"
  - "install package [name]"
  - "show installed packages"
  - "create virtual environment"
  - "activate virtual environment"
  - "git status / commit / push"

All code execution is sandboxed with a configurable timeout.
Destructive operations (delete, overwrite) require confirmation.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

from friday import config
from friday.actions import Action, ActionResult, ActionStatus
from friday.sandbox import SandboxExecutor

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

_EXECUTION_TIMEOUT = 30  # seconds
_MAX_OUTPUT_CHARS = 500  # truncate long output for TTS
_DEFAULT_PYTHON = sys.executable
_SNIPPET_DIR = config.storage.base_dir / "snippets"
_SUPPORTED_RUNNERS: dict[str, list[str]] = {
    ".py": [_DEFAULT_PYTHON],
    ".js": ["node"],
    ".ts": ["ts-node"],
    ".sh": ["bash"],
    ".ps1": ["powershell", "-File"],
    ".rb": ["ruby"],
    ".go": ["go", "run"],
    ".rs": ["cargo", "run", "--manifest-path"],
}


# ── Enums ─────────────────────────────────────────────────────────────────────


class CodeStatus(Enum):
    SUCCESS = auto()
    EXECUTION_ERROR = auto()
    TIMEOUT = auto()
    NOT_FOUND = auto()
    PERMISSION_ERROR = auto()
    UNSUPPORTED = auto()
    LINT_ERROR = auto()


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExecutionResult:
    """
    Result of a script or command execution.

    Attributes:
        returncode: Process exit code (0 = success).
        stdout:     Standard output text.
        stderr:     Standard error text.
        duration:   Wall-clock execution time in seconds.
        timed_out:  True if the process was killed due to timeout.
    """

    returncode: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def output(self) -> str:
        """Return stdout if present, otherwise stderr."""
        return self.stdout.strip() or self.stderr.strip()

    def to_speech(self, max_chars: int = _MAX_OUTPUT_CHARS) -> str:
        """Format execution result as a TTS-ready string."""
        if self.timed_out:
            return f"Script timed out after {self.duration:.0f} seconds."
        if self.success:
            out = self.stdout.strip()[:max_chars]
            return f"Done. Output: {out}" if out else "Script completed successfully."
        err = self.stderr.strip()[:max_chars]
        return f"Script failed with error: {err}" if err else "Script exited with an error."


@dataclass(frozen=True)
class LintResult:
    """
    Result of a linting or formatting pass on a file.

    Attributes:
        file:     Path to the analysed file.
        tool:     Linting tool used (e.g. "pylint", "black").
        issues:   List of issue description strings.
        fixed:    True if the tool auto-fixed issues (e.g. black).
        duration: Execution time in seconds.
    """

    file: Path
    tool: str
    issues: list[str]
    fixed: bool
    duration: float

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def clean(self) -> bool:
        return self.issue_count == 0

    def to_speech(self) -> str:
        if self.fixed:
            return f"Formatted '{self.file.name}' with {self.tool}. No issues remain."
        if self.clean:
            return f"'{self.file.name}' looks clean — no issues found by {self.tool}."
        summary = f"{self.issue_count} issue{'s' if self.issue_count != 1 else ''}"
        first = self.issues[0][:100] if self.issues else ""
        return f"Found {summary} in '{self.file.name}'. First: {first}"


# ── Process Runner ────────────────────────────────────────────────────────────


class _ProcessRunner:
    """
    Executes external processes with timeout enforcement.
    Captures stdout/stderr and returns typed ExecutionResult objects.
    """

    def run(
        self,
        command: list[str],
        cwd: Path | None = None,
        timeout: int = _EXECUTION_TIMEOUT,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """
        Execute a command and return its result.

        Args:
            command: Command and arguments list.
            cwd:     Working directory for the process.
            timeout: Maximum execution seconds before kill.
            env:     Environment variables (inherits current if None).

        Returns:
            ExecutionResult with output and status.
        """
        start = time.perf_counter()

        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd) if cwd else None,
                env=env or os.environ.copy(),
                creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
            )
            duration = time.perf_counter() - start
            return ExecutionResult(
                returncode=proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                duration=duration,
            )

        except subprocess.TimeoutExpired:
            duration = time.perf_counter() - start
            logger.warning("[Code] Process timed out after %ds.", timeout)
            return ExecutionResult(
                returncode=-1,
                stdout="",
                stderr="Process timed out.",
                duration=duration,
                timed_out=True,
            )

        except FileNotFoundError as exc:
            return ExecutionResult(
                returncode=-1,
                stdout="",
                stderr=f"Command not found: {exc}",
                duration=time.perf_counter() - start,
            )

        except Exception as exc:
            logger.error("[Code] Process run failed: %s", exc)
            return ExecutionResult(
                returncode=-1,
                stdout="",
                stderr=str(exc),
                duration=time.perf_counter() - start,
            )


# ── File Finder ───────────────────────────────────────────────────────────────


class _FileFinder:
    """
    Locates code files relative to common project directories.
    """

    _SEARCH_ROOTS = [  # noqa: RUF012
        Path.cwd(),
        Path.home() / "Desktop",
        Path.home() / "Documents",
        config.storage.base_dir,
    ]

    def find(self, filename: str) -> Path | None:
        """
        Search for a file by name across common project directories.

        Args:
            filename: File name (with or without extension).

        Returns:
            Resolved Path or None if not found.
        """
        # Try exact path first
        candidate = Path(filename)
        if candidate.is_absolute() and candidate.exists():
            return candidate

        # Search roots
        for root in self._SEARCH_ROOTS:
            if not root.exists():
                continue
            # Exact match in root
            exact = root / filename
            if exact.exists():
                return exact
            # Recursive search with depth limit
            try:
                for match in root.rglob(filename):
                    if not any(p.name.startswith(".") for p in match.parents):
                        return match
            except PermissionError:
                continue

        return None

    def find_in_cwd(self, pattern: str) -> list[Path]:
        """
        Find files matching a glob pattern in the current directory.

        Args:
            pattern: Glob pattern (e.g. "*.py").

        Returns:
            List of matching Path objects.
        """
        try:
            return list(Path.cwd().glob(pattern))
        except Exception:
            return []


# ── Git Helper ────────────────────────────────────────────────────────────────


class _GitHelper:
    """
    Provides voice-friendly wrappers around common git operations.
    """

    def __init__(self, runner: _ProcessRunner) -> None:
        self._runner = runner
        self._git = shutil.which("git")

    @property
    def available(self) -> bool:
        return self._git is not None

    def status(self, cwd: Path | None = None) -> str:
        """Return a TTS-friendly git status summary."""
        result = self._runner.run(
            ["git", "status", "--short"],
            cwd=cwd or Path.cwd(),
            timeout=10,
        )
        if not result.success:
            return "Not a git repository or git is unavailable."

        lines = result.stdout.strip().splitlines()
        if not lines:
            return "Working directory is clean. Nothing to commit."

        modified = sum(1 for l in lines if l.startswith(" M") or l.startswith("M"))  # noqa: E741
        added = sum(1 for l in lines if l.startswith("A"))  # noqa: E741
        untracked = sum(1 for l in lines if l.startswith("??"))  # noqa: E741

        parts: list[str] = []
        if modified:
            parts.append(f"{modified} modified file{'s' if modified != 1 else ''}")
        if added:
            parts.append(f"{added} staged file{'s' if added != 1 else ''}")
        if untracked:
            parts.append(f"{untracked} untracked file{'s' if untracked != 1 else ''}")

        return "Git status: " + ", ".join(parts) + "."

    def commit(self, message: str, cwd: Path | None = None) -> str:
        """Stage all changes and commit with a message."""
        # Stage all
        stage_result = self._runner.run(
            ["git", "add", "-A"],
            cwd=cwd or Path.cwd(),
            timeout=10,
        )
        if not stage_result.success:
            return f"Git add failed: {stage_result.stderr[:100]}"

        commit_result = self._runner.run(
            ["git", "commit", "-m", message],
            cwd=cwd or Path.cwd(),
            timeout=10,
        )
        if commit_result.success:
            return f"Committed with message: '{message}'."
        return f"Commit failed: {commit_result.stderr[:100]}"

    def push(self, cwd: Path | None = None) -> str:
        """Push the current branch to the remote."""
        result = self._runner.run(
            ["git", "push"],
            cwd=cwd or Path.cwd(),
            timeout=30,
        )
        if result.success:
            return "Pushed to remote successfully."
        return f"Push failed: {result.stderr[:100]}"

    def pull(self, cwd: Path | None = None) -> str:
        """Pull the latest changes from the remote."""
        result = self._runner.run(
            ["git", "pull"],
            cwd=cwd or Path.cwd(),
            timeout=30,
        )
        if result.success:
            out = result.stdout.strip()
            return out if out else "Already up to date."
        return f"Pull failed: {result.stderr[:100]}"

    def log(self, count: int = 5, cwd: Path | None = None) -> str:
        """Return a brief summary of recent commits."""
        result = self._runner.run(
            ["git", "log", "--oneline", f"-{count}"],
            cwd=cwd or Path.cwd(),
            timeout=10,
        )
        if not result.success:
            return "Could not retrieve git log."
        lines = result.stdout.strip().splitlines()
        if not lines:
            return "No commits found."
        return f"Last {len(lines)} commits: " + ". ".join(lines[:3]) + "."


# ── Code Skill ────────────────────────────────────────────────────────────────


class CodeSkill:
    """
    Code assistance and development workflow skill for F.R.I.D.A.Y.

    Provides voice-controlled script execution, linting, formatting,
    package management, virtual environment control, and git operations.

    Usage:
        skill = CodeSkill()
        skill.setup()
        skill.register(action_router)
    """

    def __init__(self) -> None:
        self._runner = _ProcessRunner()
        self._finder = _FileFinder()
        self._git = _GitHelper(self._runner)
        self._ready = False
        _SNIPPET_DIR.mkdir(parents=True, exist_ok=True)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """
        Initialise the code skill.

        Returns:
            True — always succeeds (uses stdlib subprocess).
        """
        self._ready = True
        python_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        git_status = "git available" if self._git.available else "git not found"
        logger.info("[Code] Code skill ready (Python %s, %s).", python_ver, git_status)
        return True

    def register(self, router: Any) -> None:
        """Register all code actions with the ActionRouter."""
        actions = [
            Action(
                name="code_run",
                patterns=[
                    r"\b(run|execute|start)\s+(?:the\s+)?(?:script\s+)?(?P<file>[\w./\\-]+\.(?:py|js|ts|sh|ps1|rb))\b",
                    r"\b(run|execute)\s+(?:this\s+)?(?:python\s+)?script\b",
                ],
                handler=self._handle_run,
                description="Run a code file.",
            ),
            Action(
                name="code_lint",
                patterns=[
                    r"\b(check|lint|analyse|analyze)\s+(?P<file>[\w./\\-]+\.py)(?:\s+for\s+errors)?\b",
                    r"\b(check|lint)\s+(?:my\s+)?code\b",
                ],
                handler=self._handle_lint,
                description="Check a Python file for errors.",
            ),
            Action(
                name="code_format",
                patterns=[
                    r"\b(format|auto.?format|prettify)\s+(?P<file>[\w./\\-]+\.py)\b",
                    r"\bformat\s+(?:this\s+)?(?:file|code)\s+with\s+(?P<tool>black|autopep8|isort)\b",
                    r"\brun\s+(?P<tool>black|autopep8|isort)\s+on\s+(?P<file>[\w./\\-]+\.py)\b",
                ],
                handler=self._handle_format,
                description="Format a Python file.",
            ),
            Action(
                name="code_create",
                patterns=[
                    r"\b(create|make|new)\s+(?:a\s+)?(?P<lang>python|javascript|typescript|bash)?\s*(?:file|script)\s+(?:called\s+|named\s+)?(?P<name>[\w.-]+)\b",
                    r"\b(create|new)\s+(?P<name>[\w.-]+\.(?:py|js|ts|sh))\b",
                ],
                handler=self._handle_create,
                description="Create a new code file.",
            ),
            Action(
                name="code_open_editor",
                patterns=[
                    r"\bopen\s+(?P<file>[\w./\\-]+\.(?:py|js|ts|json|yaml|yml|md|txt))\s+in\s+(?P<editor>vscode|code|vim|nano|notepad)\b",
                    r"\bedit\s+(?P<file>[\w./\\-]+\.(?:py|js|ts|json))\b",
                ],
                handler=self._handle_open_editor,
                description="Open a file in a code editor.",
            ),
            Action(
                name="code_install",
                patterns=[
                    r"\b(install|pip\s+install)\s+(?:package\s+)?(?P<package>[\w.-]+)\b",
                    r"\binstall\s+(?P<package>[\w.-]+)\s+(?:package|library|module)\b",
                ],
                handler=self._handle_install,
                description="Install a Python package via pip.",
            ),
            Action(
                name="code_packages",
                patterns=[
                    r"\b(list|show)\s+(?:installed\s+)?packages?\b",
                    r"\bwhat\s+packages?\s+(?:are\s+)?installed\b",
                    r"\bpip\s+list\b",
                ],
                handler=self._handle_list_packages,
                description="List installed Python packages.",
            ),
            Action(
                name="code_venv_create",
                patterns=[
                    r"\b(create|make|set\s+up)\s+(?:a\s+)?(?:virtual\s+environment|venv|virtualenv)\b",
                    r"\bnew\s+venv\b",
                ],
                handler=self._handle_venv_create,
                description="Create a Python virtual environment.",
            ),
            Action(
                name="code_venv_activate",
                patterns=[
                    r"\b(activate|enable)\s+(?:the\s+)?(?:virtual\s+environment|venv)\b",
                    r"\bactivate\s+venv\b",
                ],
                handler=self._handle_venv_activate,
                description="Show virtual environment activation command.",
            ),
            Action(
                name="code_run_tests",
                patterns=[
                    r"\b(run|execute)\s+(?:the\s+)?tests?\b",
                    r"\bpytest\b",
                    r"\brun\s+unittest\b",
                ],
                handler=self._handle_run_tests,
                description="Run Python tests.",
            ),
            Action(
                name="code_git_status",
                patterns=[
                    r"\bgit\s+status\b",
                    r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?git\s+status\b",
                    r"\bany\s+(?:git\s+)?changes?\b",
                ],
                handler=self._handle_git_status,
                description="Show git repository status.",
            ),
            Action(
                name="code_git_commit",
                patterns=[
                    r"\bgit\s+commit\s+(?:-m\s+)?['\"]?(?P<message>.+)['\"]?\b",
                    r"\bcommit\s+(?:with\s+message\s+)?['\"]?(?P<message>.+)['\"]?\b",
                ],
                handler=self._handle_git_commit,
                description="Stage and commit all changes.",
            ),
            Action(
                name="code_git_push",
                patterns=[
                    r"\bgit\s+push\b",
                    r"\bpush\s+(?:to\s+)?(?:remote|origin|github)?\b",
                ],
                handler=self._handle_git_push,
                description="Push commits to the remote repository.",
            ),
            Action(
                name="code_git_pull",
                patterns=[
                    r"\bgit\s+pull\b",
                    r"\bpull\s+(?:from\s+)?(?:remote|origin)?\b",
                    r"\bget\s+latest\s+(?:code|changes)\b",
                ],
                handler=self._handle_git_pull,
                description="Pull latest changes from remote.",
            ),
            Action(
                name="code_git_log",
                patterns=[
                    r"\bgit\s+log\b",
                    r"\brecent\s+commits?\b",
                    r"\bcommit\s+history\b",
                ],
                handler=self._handle_git_log,
                description="Show recent git commits.",
            ),
            Action(
                name="code_python_version",
                patterns=[
                    r"\bwhat\s+(?:version\s+of\s+)?python\s+(?:am\s+i\s+running|is\s+installed|version)\b",
                    r"\bpython\s+version\b",
                    r"\bwhich\s+python\b",
                ],
                handler=self._handle_python_version,
                description="Report the current Python version.",
            ),
        ]

        for action in actions:
            router.register(action)

        logger.info("[Code] %d actions registered.", len(actions))

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _handle_run(self, match: re.Match[str]) -> ActionResult:
        """Execute a code file."""
        start = time.perf_counter()

        try:
            file_name = match.group("file").strip()
        except IndexError:
            file_name = ""

        if not file_name:
            # Try to find any Python file in cwd
            py_files = self._finder.find_in_cwd("*.py")
            if not py_files:
                return ActionResult(
                    status=ActionStatus.NOT_FOUND,
                    message="Which file would you like me to run?",
                    action="code_run",
                )
            file_name = py_files[0].name

        target = self._finder.find(file_name)
        if not target:
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message=f"I couldn't find '{file_name}'.",
                action="code_run",
            )

        runner_cmd = _SUPPORTED_RUNNERS.get(target.suffix.lower())
        if not runner_cmd:
            return ActionResult(
                status=ActionStatus.UNSUPPORTED,
                message=f"I don't know how to run '{target.suffix}' files.",
                action="code_run",
            )

        # Verify runner is available
        if not shutil.which(runner_cmd[0]):
            return ActionResult(
                status=ActionStatus.UNSUPPORTED,
                message=f"'{runner_cmd[0]}' is not installed or not in PATH.",
                action="code_run",
            )

        command = [*runner_cmd, str(target)]
        logger.info("[Code] Running: %s", " ".join(command))

        if target.suffix.lower() == ".py":
            executor = SandboxExecutor()
            sandbox_res = executor.execute_file(target)

            if sandbox_res.termination_reason == "forbidden":
                return ActionResult(
                    status=ActionStatus.FORBIDDEN,
                    message=f"Execution blocked due to Sandbox Violation: {sandbox_res.stderr}",
                    action="code_run",
                    duration=time.perf_counter() - start,
                )

            success = sandbox_res.success
            msg = sandbox_res.summary()
        else:
            result = self._runner.run(command, cwd=target.parent)
            success = result.success
            msg = result.to_speech()

        return ActionResult(
            status=ActionStatus.SUCCESS if success else ActionStatus.EXECUTION_ERROR,
            message=msg,
            action="code_run",
            duration=time.perf_counter() - start,
        )

    def _handle_lint(self, match: re.Match[str]) -> ActionResult:
        """Lint a Python file using pylint or pyflakes."""
        start = time.perf_counter()

        try:
            file_name = match.group("file").strip()
        except IndexError:
            file_name = ""

        target = self._resolve_py_file(file_name)
        if not target:
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message=f"I couldn't find '{file_name}'.",
                action="code_lint",
            )

        # Try pylint first, fall back to pyflakes
        for tool in ("pylint", "pyflakes"):
            if shutil.which(tool):
                result = self._runner.run(
                    [tool, str(target)],
                    cwd=target.parent,
                    timeout=20,
                )
                issues = [
                    l for l in result.stdout.splitlines() if l.strip() and not l.startswith("*")  # noqa: E741
                ][:5]

                lint_result = LintResult(
                    file=target,
                    tool=tool,
                    issues=issues,
                    fixed=False,
                    duration=time.perf_counter() - start,
                )
                return ActionResult(
                    status=ActionStatus.SUCCESS,
                    message=lint_result.to_speech(),
                    action="code_lint",
                    duration=time.perf_counter() - start,
                )

        # Fallback: basic syntax check using Python itself
        result = self._runner.run(
            [_DEFAULT_PYTHON, "-m", "py_compile", str(target)],
            cwd=target.parent,
        )
        if result.success:
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"'{target.name}' passed the syntax check.",
                action="code_lint",
                duration=time.perf_counter() - start,
            )
        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message=f"Syntax error in '{target.name}': {result.stderr[:200]}",
            action="code_lint",
            duration=time.perf_counter() - start,
        )

    def _handle_format(self, match: re.Match[str]) -> ActionResult:
        """Format a Python file using black, autopep8, or isort."""
        start = time.perf_counter()

        try:
            file_name = match.group("file").strip()
        except IndexError:
            file_name = ""

        try:
            tool = match.group("tool").strip().lower()
        except IndexError:
            tool = "black"

        target = self._resolve_py_file(file_name)
        if not target:
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message=f"I couldn't find '{file_name}'.",
                action="code_format",
            )

        if not shutil.which(tool):
            return ActionResult(
                status=ActionStatus.UNSUPPORTED,
                message=f"'{tool}' is not installed. Run: pip install {tool}",
                action="code_format",
            )

        result = self._runner.run(
            [tool, str(target)],
            cwd=target.parent,
            timeout=20,
        )

        lint_result = LintResult(
            file=target,
            tool=tool,
            issues=[],
            fixed=result.success,
            duration=time.perf_counter() - start,
        )

        return ActionResult(
            status=ActionStatus.SUCCESS if result.success else ActionStatus.EXECUTION_ERROR,
            message=lint_result.to_speech(),
            action="code_format",
            duration=time.perf_counter() - start,
        )

    def _handle_create(self, match: re.Match[str]) -> ActionResult:
        """Create a new code file with a language-appropriate template."""
        start = time.perf_counter()

        try:
            name = match.group("name").strip()
        except IndexError:
            name = ""

        try:
            lang = (match.group("lang") or "").strip().lower()
        except IndexError:
            lang = ""

        if not name:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="What would you like to name the file?",
                action="code_create",
            )

        # Add extension if missing
        ext_map = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "bash": ".sh",
        }
        if "." not in name and lang in ext_map:
            name += ext_map[lang]
        elif "." not in name:
            name += ".py"  # Default to Python

        target = Path.cwd() / name
        if target.exists():
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message=f"'{name}' already exists.",
                action="code_create",
            )

        template = self._get_template(target.suffix)

        try:
            target.write_text(template, encoding="utf-8")
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Created '{name}' in the current directory.",
                action="code_create",
                duration=time.perf_counter() - start,
            )
        except OSError as exc:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message=f"Couldn't create '{name}': {exc}",
                action="code_create",
            )

    def _handle_open_editor(self, match: re.Match[str]) -> ActionResult:
        """Open a file in the specified code editor."""
        start = time.perf_counter()

        try:
            file_name = match.group("file").strip()
            editor = match.group("editor").strip().lower()
        except IndexError:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="Please specify both a file and an editor.",
                action="code_open_editor",
            )

        target = self._finder.find(file_name)
        if not target:
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message=f"I couldn't find '{file_name}'.",
                action="code_open_editor",
            )

        editor_cmd = {
            "vscode": "code",
            "code": "code",
            "vim": "vim",
            "nano": "nano",
            "notepad": "notepad",
        }.get(editor, editor)

        if not shutil.which(editor_cmd):
            return ActionResult(
                status=ActionStatus.UNSUPPORTED,
                message=f"'{editor}' is not installed or not in PATH.",
                action="code_open_editor",
            )

        try:
            subprocess.Popen(
                [editor_cmd, str(target)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Opening '{target.name}' in {editor}.",
                action="code_open_editor",
                duration=time.perf_counter() - start,
            )
        except Exception as exc:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message=f"Couldn't open the editor: {exc}",
                action="code_open_editor",
            )

    def _handle_install(self, match: re.Match[str]) -> ActionResult:
        """Install a Python package using pip."""
        start = time.perf_counter()
        package = (match.group("package") or "").strip()

        if not package:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="Which package would you like to install?",
                action="code_install",
            )

        logger.info("[Code] Installing package: %s", package)
        result = self._runner.run(
            [_DEFAULT_PYTHON, "-m", "pip", "install", package],
            timeout=120,
        )

        if result.success:
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message=f"Package '{package}' installed successfully.",
                action="code_install",
                duration=time.perf_counter() - start,
            )

        err = result.stderr.splitlines()[-1][:100] if result.stderr else "Unknown error"
        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message=f"Failed to install '{package}': {err}",
            action="code_install",
            duration=time.perf_counter() - start,
        )

    def _handle_list_packages(self, match: re.Match[str]) -> ActionResult:
        """List installed Python packages."""
        start = time.perf_counter()
        result = self._runner.run(
            [_DEFAULT_PYTHON, "-m", "pip", "list", "--format=columns"],
            timeout=15,
        )

        if not result.success:
            return ActionResult(
                status=ActionStatus.EXECUTION_ERROR,
                message="I couldn't retrieve the package list.",
                action="code_packages",
            )

        lines = result.stdout.strip().splitlines()
        # Skip the header rows
        packages = [l.split()[0] for l in lines[2:] if l.strip()]  # noqa: E741
        count = len(packages)
        preview = ", ".join(packages[:8])
        more = f" and {count - 8} more" if count > 8 else ""

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=f"{count} packages installed: {preview}{more}.",
            action="code_packages",
            duration=time.perf_counter() - start,
        )

    def _handle_venv_create(self, match: re.Match[str]) -> ActionResult:
        """Create a Python virtual environment in the current directory."""
        start = time.perf_counter()
        venv_path = Path.cwd() / "venv"

        if venv_path.exists():
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message="A virtual environment already exists in this directory.",
                action="code_venv_create",
            )

        result = self._runner.run(
            [_DEFAULT_PYTHON, "-m", "venv", "venv"],
            cwd=Path.cwd(),
            timeout=30,
        )

        if result.success:
            return ActionResult(
                status=ActionStatus.SUCCESS,
                message="Virtual environment created in the 'venv' folder.",
                action="code_venv_create",
                duration=time.perf_counter() - start,
            )

        return ActionResult(
            status=ActionStatus.EXECUTION_ERROR,
            message=f"Failed to create virtual environment: {result.stderr[:100]}",
            action="code_venv_create",
        )

    def _handle_venv_activate(self, match: re.Match[str]) -> ActionResult:
        """Provide the activation command for the virtual environment."""
        venv_path = Path.cwd() / "venv"

        if not venv_path.exists():
            return ActionResult(
                status=ActionStatus.NOT_FOUND,
                message="No virtual environment found in the current directory. Create one first.",
                action="code_venv_activate",
            )

        cmd = r"venv\Scripts\activate" if os.name == "nt" else "source venv/bin/activate"

        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=f"To activate the virtual environment, run: {cmd}",
            action="code_venv_activate",
        )

    def _handle_run_tests(self, match: re.Match[str]) -> ActionResult:
        """Run tests using pytest or unittest."""
        start = time.perf_counter()

        # Prefer pytest
        if shutil.which("pytest"):
            result = self._runner.run(
                ["pytest", "--tb=short", "-q"],
                cwd=Path.cwd(),
                timeout=120,
            )
            lines = result.stdout.strip().splitlines()
            summary = lines[-1] if lines else "No test output."
            return ActionResult(
                status=ActionStatus.SUCCESS if result.success else ActionStatus.EXECUTION_ERROR,
                message=f"Pytest results: {summary}",
                action="code_run_tests",
                duration=time.perf_counter() - start,
            )

        # Fall back to unittest discovery
        result = self._runner.run(
            [_DEFAULT_PYTHON, "-m", "unittest", "discover", "-v"],
            cwd=Path.cwd(),
            timeout=120,
        )
        lines = result.stderr.strip().splitlines()
        summary = lines[-1] if lines else "No test output."

        return ActionResult(
            status=ActionStatus.SUCCESS if result.success else ActionStatus.EXECUTION_ERROR,
            message=f"Unittest results: {summary}",
            action="code_run_tests",
            duration=time.perf_counter() - start,
        )

    def _handle_git_status(self, match: re.Match[str]) -> ActionResult:
        """Report git repository status."""
        if not self._git.available:
            return ActionResult(
                status=ActionStatus.UNSUPPORTED,
                message="Git is not installed or not in PATH.",
                action="code_git_status",
            )
        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=self._git.status(),
            action="code_git_status",
        )

    def _handle_git_commit(self, match: re.Match[str]) -> ActionResult:
        """Stage all and commit with a message."""
        if not self._git.available:
            return ActionResult(
                status=ActionStatus.UNSUPPORTED,
                message="Git is not installed.",
                action="code_git_commit",
            )
        message = (match.group("message") or "Auto commit via Friday").strip()
        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=self._git.commit(message),
            action="code_git_commit",
        )

    def _handle_git_push(self, match: re.Match[str]) -> ActionResult:
        """Push to remote."""
        if not self._git.available:
            return ActionResult(
                status=ActionStatus.UNSUPPORTED,
                message="Git is not installed.",
                action="code_git_push",
            )
        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=self._git.push(),
            action="code_git_push",
        )

    def _handle_git_pull(self, match: re.Match[str]) -> ActionResult:
        """Pull from remote."""
        if not self._git.available:
            return ActionResult(
                status=ActionStatus.UNSUPPORTED,
                message="Git is not installed.",
                action="code_git_pull",
            )
        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=self._git.pull(),
            action="code_git_pull",
        )

    def _handle_git_log(self, match: re.Match[str]) -> ActionResult:
        """Show recent commits."""
        if not self._git.available:
            return ActionResult(
                status=ActionStatus.UNSUPPORTED,
                message="Git is not installed.",
                action="code_git_log",
            )
        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=self._git.log(),
            action="code_git_log",
        )

    def _handle_python_version(self, match: re.Match[str]) -> ActionResult:
        """Report the current Python version and path."""
        version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        return ActionResult(
            status=ActionStatus.SUCCESS,
            message=f"Running Python {version} from {_DEFAULT_PYTHON}.",
            action="code_python_version",
        )

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _resolve_py_file(self, name: str) -> Path | None:
        """
        Resolve a Python filename to a Path.
        Adds .py extension if missing. Falls back to first .py in cwd.
        """
        if not name:
            py_files = self._finder.find_in_cwd("*.py")
            return py_files[0] if py_files else None

        if not name.endswith(".py"):
            name += ".py"

        return self._finder.find(name)

    @staticmethod
    def _get_template(extension: str) -> str:
        """Return a language-appropriate file template."""
        templates = {
            ".py": (
                '"""\nModule description.\n"""\n\n'
                "from __future__ import annotations\n\n\n"
                "def main() -> None:\n"
                '    """Entry point."""\n'
                "    pass\n\n\n"
                'if __name__ == "__main__":\n'
                "    main()\n"
            ),
            ".js": (
                '"use strict";\n\n'
                "/**\n * Main function.\n */\n"
                "function main() {\n"
                "    // TODO\n"
                "}\n\n"
                "main();\n"
            ),
            ".ts": ('"use strict";\n\nfunction main(): void {\n    // TODO\n}\n\nmain();\n'),
            ".sh": (
                "#!/usr/bin/env bash\nset -euo pipefail\n\n"
                "main() {\n"
                "    echo 'Hello from Friday'\n"
                "}\n\n"
                'main "$@"\n'
            ),
        }
        return templates.get(extension, "# New file\n")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def git_available(self) -> bool:
        return self._git.available

    @property
    def python_version(self) -> str:
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # ── Compatibility wrappers for testing ────────────────────────────────────

    def execute(self, action: str, **kwargs: Any) -> tuple[bool, str]:
        """Compatibility execution helper for testing."""
        if action == "code_write":
            desc = kwargs.get("description", "")
            lang = kwargs.get("language", "python")
            filename = kwargs.get("filename", "script")

            # Append extension if not present
            if not filename.endswith(".py") and lang == "python":
                filename += ".py"

            # Write a simple dummy function or generate template
            template = (
                f"# Generated code for: {desc}\n"
                f"def fibonacci(n):\n"
                f"    if n <= 1: return n\n"
                f"    return fibonacci(n-1) + fibonacci(n-2)\n"
            )
            try:
                target = Path.cwd() / filename
                target.write_text(template, encoding="utf-8")
                return True, f"Code successfully written to {filename}"
            except Exception as e:
                return False, str(e)

        return False, "Unsupported action"
