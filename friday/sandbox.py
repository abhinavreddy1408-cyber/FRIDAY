"""
F.R.I.D.A.Y. Sandboxed Execution Environment (Phase 5C)
"""

from __future__ import annotations

import ast
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil

try:
    import win32api
    import win32con
    import win32job

    _WIN32JOB_AVAILABLE = True
except ImportError:
    _WIN32JOB_AVAILABLE = False

import contextlib

from friday import config

logger = logging.getLogger("friday.sandbox")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter("%(asctime)s [SANDBOX] %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    _ch = logging.StreamHandler()
    _ch.setFormatter(_fmt)
    logger.addHandler(_ch)


SANDBOX_WARNING = """# [SANDBOX_WARNING]
# This script is executing in the F.R.I.D.A.Y. Sandbox.
# - Resource limits enforced (RAM / CPU / timeout).
# - File I/O is restricted to the sandbox working directory.
# - Network access may be disabled.
# - Violations → immediate termination.
"""


class SandboxValidationError(Exception):
    pass


class ASTGatekeeper(ast.NodeVisitor):
    WHITELISTED_MODULES = {  # noqa: RUF012
        "math",
        "datetime",
        "re",
        "json",
        "time",
        "collections",
        "itertools",
        "typing",
        "random",
        "string",
        "urllib",
        "sqlite3",
    }
    FORBIDDEN_BUILTINS = {"eval", "exec", "open", "compile", "__import__"}  # noqa: RUF012

    def __init__(self):
        self.errors = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_module = alias.name.split(".")[0]
            if base_module not in self.WHITELISTED_MODULES:
                self.errors.append(f"Import of '{alias.name}' is forbidden.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            base_module = node.module.split(".")[0]
            if base_module not in self.WHITELISTED_MODULES:
                self.errors.append(f"Import from '{node.module}' is forbidden.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in self.FORBIDDEN_BUILTINS:
            self.errors.append(f"Call to builtin '{node.func.id}' is forbidden.")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id in self.FORBIDDEN_BUILTINS:
            self.errors.append(f"Reference to builtin '{node.id}' is forbidden.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in ("__globals__", "__code__", "__dict__", "__class__"):
            self.errors.append(f"Access to attribute '{node.attr}' is forbidden.")
        self.generic_visit(node)


TERMINATION_REASONS = {
    "completed",
    "timeout",
    "memory",
    "cpu",
    "error",
    "forbidden",
    "killed",
}


@dataclass
class SandboxResult:
    success: bool
    stdout: str
    stderr: str
    return_code: int
    termination_reason: str
    execution_time: float
    peak_memory_mb: float
    peak_cpu_percent: float
    script_path: str | None = None
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        status = "✅" if self.success else "❌"
        return (
            f"{status} {self.termination_reason} "
            f"({self.execution_time:.2f}s, "
            f"{self.peak_memory_mb:.1f}MB, "
            f"{self.peak_cpu_percent:.0f}% CPU)"
        )


class SandboxExecutor:
    """Handles execution and resource monitoring of untrusted code."""

    def __init__(
        self,
        sandbox_dir: str | Path | None = None,
        max_memory_mb: int | None = None,
        timeout_seconds: int | None = None,
        max_cpu_percent: float | None = None,
        cpu_sustained_seconds: int | None = None,
        allow_network: bool = False,
        block_dangerous: bool = True,
    ):
        self.sandbox_dir = Path(sandbox_dir or config.sandbox.dir).resolve()
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

        self.max_memory = max_memory_mb or config.sandbox.max_memory_mb
        self.timeout = timeout_seconds or config.sandbox.timeout_seconds
        self.max_cpu = max_cpu_percent or config.sandbox.max_cpu_percent
        self.sustained_cpu_limit = cpu_sustained_seconds or config.sandbox.cpu_sustained_seconds

        self.allow_network = allow_network
        self.block_dangerous = block_dangerous

        self._monitor_active = False
        self._termination_reason = "completed"
        self._peak_mem = 0.0
        self._peak_cpu = 0.0
        self._lock = threading.Lock()

    # ── Static analysis ────────────────────────────────────────
    def _scan_code(self, code: str) -> list[str]:
        """Return list of matched forbidden patterns (empty = safe)."""
        try:
            tree = ast.parse(code)
            gatekeeper = ASTGatekeeper()
            gatekeeper.visit(tree)
            return gatekeeper.errors
        except SyntaxError as e:
            return [f"SyntaxError: {e}"]

    # ── Resource monitor ───────────────────────────────────────
    def _monitor_resources(self, process: subprocess.Popen) -> None:
        try:
            p = psutil.Process(process.pid)
            cpu_readings: list[float] = []
            sustained_start: float | None = None

            while process.poll() is None and self._monitor_active:
                try:
                    # Include children (subprocess fork safety)
                    mem = p.memory_info().rss
                    for child in p.children(recursive=True):
                        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                            mem += child.memory_info().rss
                    mem_mb = mem / (1024 * 1024)
                    with self._lock:
                        self._peak_mem = max(self._peak_mem, mem_mb)

                    if mem_mb > self.max_memory:
                        self._termination_reason = "memory"
                        logger.warning(
                            f"🛑 Memory limit exceeded: {mem_mb:.1f}MB > {self.max_memory}MB"
                        )
                        self._kill_tree(p)
                        break

                    cpu = p.cpu_percent(interval=0.1)
                    with self._lock:
                        self._peak_cpu = max(self._peak_cpu, cpu)
                    cpu_readings.append(cpu)
                    if len(cpu_readings) > 50:
                        cpu_readings = cpu_readings[-50:]

                    # Sustained high-CPU detection
                    if len(cpu_readings) >= 10:
                        avg = sum(cpu_readings[-10:]) / 10
                        if avg > self.max_cpu:
                            if sustained_start is None:
                                sustained_start = time.time()
                            elif time.time() - sustained_start > self.sustained_cpu_limit:
                                self._termination_reason = "cpu"
                                logger.warning(
                                    f"🛑 Sustained CPU {avg:.0f}% > {self.max_cpu}% "
                                    f"for >{self.sustained_cpu_limit}s"
                                )
                                self._kill_tree(p)
                                break
                        else:
                            sustained_start = None

                    time.sleep(0.1)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
        except Exception as e:
            logger.error(f"Monitor error: {e}")

    def _kill_tree(self, p: psutil.Process) -> None:
        """Kill a process and all its descendants."""
        if hasattr(self, "_current_job_handle") and self._current_job_handle is not None:
            try:
                win32job.TerminateJobObject(self._current_job_handle, 1)
                win32api.CloseHandle(self._current_job_handle)
                self._current_job_handle = None
                return
            except Exception as e:
                logger.error(f"Job Object termination failed: {e}")

        # Fallback to psutil
        try:
            for child in p.children(recursive=True):
                with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    child.kill()
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # ── Main entry ─────────────────────────────────────────────
    def execute(self, code: str, language: str = "python") -> SandboxResult:
        start_time = time.time()
        self._termination_reason = "completed"
        self._peak_mem = 0.0
        self._peak_cpu = 0.0
        warnings: list[str] = []

        if language != "python":
            return SandboxResult(
                success=False,
                stdout="",
                stderr=f"Unsupported language: {language}",
                return_code=-1,
                termination_reason="error",
                execution_time=0.0,
                peak_memory_mb=0.0,
                peak_cpu_percent=0.0,
            )

        # Static safety check
        if self.block_dangerous:
            hits = self._scan_code(code)
            if hits:
                logger.warning(f"🚫 Forbidden patterns detected: {hits}")
                return SandboxResult(
                    success=False,
                    stdout="",
                    stderr=f"Forbidden patterns: {hits}",
                    return_code=-1,
                    termination_reason="forbidden",
                    execution_time=time.time() - start_time,
                    peak_memory_mb=0.0,
                    peak_cpu_percent=0.0,
                    warnings=hits,
                )

        # Write temp script
        temp_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            dir=self.sandbox_dir,
            suffix=".py",
            mode="w",
            delete=False,
            encoding="utf-8",
        )
        full_code = SANDBOX_WARNING + "\n" + code
        temp_file.write(full_code)
        temp_file.close()
        script_path = temp_file.name

        # Build restricted environment
        env = {
            "PYTHONPATH": str(self.sandbox_dir),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "HOME": str(self.sandbox_dir),
            "TMPDIR": str(self.sandbox_dir),
            "PATH": os.environ.get("PATH", ""),
        }
        if not self.allow_network:
            env["no_proxy"] = "*"
            env["HTTP_PROXY"] = "http://127.0.0.1:1"
            env["HTTPS_PROXY"] = "http://127.0.0.1:1"

        logger.info(
            f"▶ Executing script ({len(code)} chars) "
            f"[mem<{self.max_memory}MB, "
            f"cpu<{self.max_cpu}%, "
            f"timeout={self.timeout}s]"
        )

        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "-B", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.sandbox_dir),
                env=env,
            )

            job_handle = None
            if _WIN32JOB_AVAILABLE:
                try:
                    job_handle = win32job.CreateJobObject(None, "")
                    info = win32job.QueryInformationJobObject(
                        job_handle, win32job.JobObjectExtendedLimitInformation
                    )
                    info["BasicLimitInformation"]["LimitFlags"] |= (
                        win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                    )
                    win32job.SetInformationJobObject(
                        job_handle, win32job.JobObjectExtendedLimitInformation, info
                    )

                    proc_handle = win32api.OpenProcess(
                        win32con.PROCESS_ALL_ACCESS, False, process.pid
                    )
                    win32job.AssignProcessToJobObject(job_handle, proc_handle)
                    win32api.CloseHandle(proc_handle)
                except Exception as e:
                    logger.error(f"Failed to assign process to Windows Job Object: {e}")

            self._current_job_handle = job_handle

            self._monitor_active = True
            monitor_thread = threading.Thread(
                target=self._monitor_resources,
                args=(process,),
                daemon=True,
            )
            monitor_thread.start()

            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                logger.warning(f"🛑 Timeout after {self.timeout}s")
                try:
                    self._kill_tree(psutil.Process(process.pid))
                except psutil.NoSuchProcess:
                    process.kill()
                self._termination_reason = "timeout"
                stdout, stderr = process.communicate()

            self._monitor_active = False
            monitor_thread.join(timeout=1)

            exec_time = time.time() - start_time
            success = process.returncode == 0 and self._termination_reason == "completed"

            if self._termination_reason == "completed" and process.returncode != 0:
                self._termination_reason = "error"

            result = SandboxResult(
                success=success,
                stdout=stdout or "",
                stderr=stderr or "",
                return_code=process.returncode,
                termination_reason=self._termination_reason,
                execution_time=exec_time,
                peak_memory_mb=self._peak_mem,
                peak_cpu_percent=self._peak_cpu,
                script_path=script_path,
                warnings=warnings,
            )
            logger.info(f"  → {result.summary()}")
            return result

        except Exception as e:
            logger.error(f"Execution crashed: {e}")
            return SandboxResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                termination_reason="error",
                execution_time=time.time() - start_time,
                peak_memory_mb=self._peak_mem,
                peak_cpu_percent=self._peak_cpu,
                script_path=script_path,
            )
        finally:
            try:
                if os.path.exists(script_path):
                    os.remove(script_path)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

    def execute_file(self, path: str | Path) -> SandboxResult:
        """Execute an existing script file within the sandbox."""
        path = Path(path)
        if not path.exists():
            return SandboxResult(
                success=False,
                stdout="",
                stderr=f"File not found: {path}",
                return_code=-1,
                termination_reason="error",
                execution_time=0.0,
                peak_memory_mb=0.0,
                peak_cpu_percent=0.0,
            )
        return self.execute(path.read_text(encoding="utf-8"))

    def execute_code_string(self, code: str) -> SandboxResult:
        return self.execute(code)

    def cleanup(self) -> int:
        """Remove all leftover files from the sandbox directory. Returns count."""
        count = 0
        try:
            for item in self.sandbox_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                        count += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        count += 1
                except Exception as e:
                    logger.warning(f"Could not remove {item}: {e}")
            logger.info(f"🧹 Sandbox cleaned ({count} items removed)")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
        return count


if __name__ == "__main__":
    print("Testing SandboxExecutor...\n")
    executor = SandboxExecutor()

    tests = [
        ("Hello", "print('Hello from the Jail!')"),
        ("Memory", "x = [0] * (50 * 1024 * 1024)"),
        ("Timeout", "import time; time.sleep(40)"),
        ("Forbidden", "import os; os.system('echo hacked')"),
        ("Error", "raise ValueError('boom')"),
        ("CPU burn", "while True: pass"),
    ]
    for name, code in tests:
        res = executor.execute(code)
        print(f"[{name:10s}] {res.summary()}")
        if res.stderr.strip():
            print(f"             stderr: {res.stderr.strip()[:80]}")
    print()
    executor.cleanup()
