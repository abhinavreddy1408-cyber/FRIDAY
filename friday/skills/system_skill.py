"""
F.R.I.D.A.Y. System Skill
=========================
Advanced system monitoring, control, and diagnostics.
Provides real-time insights into hardware status, network, and processes.
"""

from __future__ import annotations

import contextlib
import logging
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import psutil

from friday.config import config

logger = logging.getLogger(__name__)

# ── Constants & Safety ────────────────────────────────────────────────────────

# Protected processes that should never be killed accidentally
PROTECTED_PROCESSES = {
    "system",
    "smss",
    "csrss",
    "wininit",
    "services",
    "lsass",
    "svchost",
    "explorer",
    "python",
    "pythonw",
    "friday",
    "engine",
    "code",
    "chrome",
}


class OSType(Enum):
    WINDOWS = "Windows"
    MACOS = "Darwin"
    LINUX = "Linux"
    UNKNOWN = "Unknown"


def get_os_type() -> OSType:
    """Detect current operating system."""
    system = platform.system()
    if system == "Windows":  # noqa: SIM116
        return OSType.WINDOWS
    elif system == "Darwin":
        return OSType.MACOS
    elif system == "Linux":
        return OSType.LINUX
    return OSType.UNKNOWN


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass
class SystemResult:
    """Standardized result structure for all system skills."""

    success: bool
    message: str
    data: dict[str, Any]
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatteryInfo:
    percent: int
    status: str  # charging, discharging, no_battery
    secs_left: int | None


@dataclass
class NetworkInfo:
    ssid: str | None
    signal_strength: str | None
    latency_ms: float | None
    ip_address: str | None


@dataclass
class ResourceUsage:
    cpu_percent: float
    ram_percent: float
    ram_used_gb: float
    ram_total_gb: float


@dataclass
class ProcessInfo:
    pid: int
    name: str
    memory_mb: float
    cpu_percent: float


# ── System Manager ────────────────────────────────────────────────────────────


class SystemManager:
    """
    Centralized manager for system-level operations and monitoring.
    Handles cross-platform differences and safety checks.
    """

    def __init__(self):
        self.os_type = get_os_type()
        self._logger = logging.getLogger("SystemManager")
        self._logger.info(f"[System] System manager initialized on {self.os_type.value}")

    def _run_command(self, command: list[str], timeout: int = 5) -> str | None:
        """Safely run a subprocess command."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=(self.os_type == OSType.WINDOWS),
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            self._logger.error(f"[System] Command timed out: {command}")
            return None
        except Exception as e:
            self._logger.error(f"[System] Command failed: {e}")
            return None

    # ── Power Management ──────────────────────────────────────────────────────

    def get_battery_status(self) -> SystemResult:
        """Get battery level, status, and time remaining."""
        try:
            battery = psutil.sensors_battery()
            if battery is None:
                return SystemResult(
                    success=False,
                    message="No battery detected (likely a desktop PC).",
                    data={"status": "no_battery"},
                    timestamp=time.time(),
                )

            status = "charging" if battery.power_plugged else "discharging"
            if battery.power_plugged and battery.percent == 100:
                status = "fully_charged"

            data = BatteryInfo(percent=battery.percent, status=status, secs_left=battery.secs_left)

            msg = f"Sir, the battery is at {battery.percent}% and is currently {status}."
            if battery.secs_left and battery.secs_left > 0:
                mins = battery.secs_left // 60
                msg += f" Approximately {mins} minutes remaining."

            return SystemResult(success=True, message=msg, data=asdict(data), timestamp=time.time())
        except Exception as e:
            self._logger.error(f"[System] Battery check failed: {e}")
            return SystemResult(
                success=False,
                message=f"Failed to retrieve battery status: {e}",
                data={},
                timestamp=time.time(),
            )

    # ── Network Diagnostics ───────────────────────────────────────────────────

    def get_wifi_status(self) -> SystemResult:
        """Get Wi-Fi network information (SSID, Signal)."""
        ssid = "Unknown"
        signal = "Unknown"

        try:
            if self.os_type == OSType.WINDOWS:
                output = self._run_command(["netsh", "wlan", "show", "interfaces"])
                if output:
                    for line in output.split("\n"):
                        if "SSID" in line and "BSSID" not in line:
                            ssid = line.split(":", 1)[1].strip()
                        if "Signal" in line:
                            signal = line.split(":", 1)[1].strip()

            elif self.os_type == OSType.MACOS:
                output = self._run_command(
                    [
                        "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport",
                        "-I",
                    ]
                )
                if output:
                    for line in output.split("\n"):
                        if "SSID" in line:
                            ssid = line.split(":", 1)[1].strip()
                        if "agrCtlRSSI" in line:
                            signal = line.split(":", 1)[1].strip() + " dBm"

            elif self.os_type == OSType.LINUX:
                output = self._run_command(["iwgetid", "-r"])
                if output:
                    ssid = output.strip()
                # Signal strength requires sudo typically on Linux
                signal = "N/A (requires sudo)"

            return SystemResult(
                success=True,
                message=f"You are connected to '{ssid}' with {signal} signal strength, Boss.",
                data={"ssid": ssid, "signal": signal},
                timestamp=time.time(),
            )
        except Exception as e:
            self._logger.error(f"[System] Wi-Fi check failed: {e}")
            return SystemResult(
                success=False,
                message=f"Could not retrieve Wi-Fi status: {e}",
                data={},
                timestamp=time.time(),
            )

    def get_network_latency(self) -> SystemResult:
        """Check network latency via ping."""
        try:
            # Cross-platform ping command
            if self.os_type == OSType.WINDOWS:
                cmd = ["ping", "-n", "1", "8.8.8.8"]
                pattern = "time="
            else:
                cmd = ["ping", "-c", "1", "8.8.8.8"]
                pattern = "time="

            output = self._run_command(cmd, timeout=5)

            if output and pattern in output:
                # Parse latency (handles both Windows and Unix output formats)
                latency_part = output.split(pattern)[1].split()[0]
                latency_ms = float(latency_part.replace("ms", ""))

                return SystemResult(
                    success=True,
                    message=f"Your network latency is {latency_ms} milliseconds, Boss.",
                    data={"latency_ms": latency_ms, "target": "8.8.8.8"},
                    timestamp=time.time(),
                )

            return SystemResult(
                success=False,
                message="Ping failed. Network may be disconnected.",
                data={},
                timestamp=time.time(),
            )
        except Exception as e:
            self._logger.error(f"[System] Latency check failed: {e}")
            return SystemResult(
                success=False,
                message=f"Network speed test failed: {e}",
                data={},
                timestamp=time.time(),
            )

    # ── Resource Monitoring ───────────────────────────────────────────────────

    def get_resource_usage(self) -> SystemResult:
        """Get current CPU and RAM usage statistics."""
        try:
            # Non-blocking CPU percent (approximate)
            cpu = psutil.cpu_percent(interval=0.5)
            mem = psutil.virtual_memory()

            data = ResourceUsage(
                cpu_percent=cpu,
                ram_percent=mem.percent,
                ram_used_gb=round(mem.used / (1024**3), 2),
                ram_total_gb=round(mem.total / (1024**3), 2),
            )

            msg = f"Status report: CPU is at {cpu}% and RAM is at {mem.percent}% ({data.ram_used_gb}GB/{data.ram_total_gb}GB), Sir."

            # Alert if thresholds exceeded (from config)
            if cpu > config.system.cpu_threshold:
                msg += " WARNING: CPU usage is critical."

            return SystemResult(success=True, message=msg, data=asdict(data), timestamp=time.time())
        except Exception as e:
            self._logger.error(f"[System] Resource check failed: {e}")
            return SystemResult(
                success=False,
                message=f"Failed to retrieve system resources: {e}",
                data={},
                timestamp=time.time(),
            )

    def get_disk_usage(self, drive: str = "") -> SystemResult:
        """Get disk space information."""
        try:
            # Auto-detect primary drive if not specified
            if not drive:
                drive = "C:\\" if self.os_type == OSType.WINDOWS else "/"

            usage = psutil.disk_usage(drive)
            free_gb = round(usage.free / (1024**3), 2)
            total_gb = round(usage.total / (1024**3), 2)

            return SystemResult(
                success=True,
                message=f"On drive {drive}, you have {free_gb} GB free out of {total_gb} GB total ({usage.percent}% used).",
                data={
                    "drive": drive,
                    "free_gb": free_gb,
                    "total_gb": total_gb,
                    "percent_used": usage.percent,
                },
                timestamp=time.time(),
            )
        except Exception as e:
            self._logger.error(f"[System] Disk check failed: {e}")
            return SystemResult(
                success=False,
                message=f"Failed to retrieve disk usage: {e}",
                data={},
                timestamp=time.time(),
            )

    # ── Process Management ────────────────────────────────────────────────────

    def get_top_processes(self, limit: int = 10) -> SystemResult:
        """List top processes by memory usage."""
        try:
            processes = []
            for proc in psutil.process_iter(["pid", "name", "memory_info", "cpu_percent"]):
                with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    processes.append(
                        {
                            "pid": proc.info["pid"],
                            "name": proc.info["name"],
                            "memory_mb": round(proc.info["memory_info"].rss / (1024**2), 2),
                            "cpu_percent": proc.info["cpu_percent"],
                        }
                    )

            # Sort by memory
            processes.sort(key=lambda x: x["memory_mb"], reverse=True)
            top_processes = processes[:limit]

            msg = f"Sir, I've listed the top {limit} processes. {top_processes[0]['name']} is consuming the most resources ({top_processes[0]['memory_mb']} MB)."

            return SystemResult(
                success=True, message=msg, data={"processes": top_processes}, timestamp=time.time()
            )
        except Exception as e:
            self._logger.error(f"[System] Process list failed: {e}")
            return SystemResult(
                success=False,
                message=f"Failed to list processes: {e}",
                data={},
                timestamp=time.time(),
            )

    def terminate_process(self, name: str, force: bool = False) -> SystemResult:
        """
        Force close an application by name.
        Safety: Checks against PROTECTED_PROCESSES list.
        """
        try:
            # Safety Check
            if name.lower() in PROTECTED_PROCESSES:
                msg = f"Sir, I cannot terminate '{name}'. It is a protected system process."
                self._logger.warning(f"[System] Blocked termination attempt: {name}")
                return SystemResult(
                    success=False, message=msg, data={"blocked": True}, timestamp=time.time()
                )

            count = 0
            terminated = []
            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    if name.lower() in proc.info["name"].lower():
                        proc.kill()
                        count += 1
                        terminated.append(proc.info["name"])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            if count > 0:
                msg = f"Terminated {count} instances of {name}, Sir."
                self._logger.warning(f"[System] Terminated {count} processes matching: {name}")
                return SystemResult(
                    success=True,
                    message=msg,
                    data={"count": count, "terminated": terminated},
                    timestamp=time.time(),
                )

            return SystemResult(
                success=False,
                message=f"No running processes found matching '{name}'.",
                data={"count": 0},
                timestamp=time.time(),
            )
        except Exception as e:
            self._logger.error(f"[System] Termination failed: {e}")
            return SystemResult(
                success=False,
                message=f"Failed to terminate process: {e}",
                data={},
                timestamp=time.time(),
            )


# ── Legacy Function Wrappers ──────────────────────────────────────────────────

# Global system manager instance
_system_manager: SystemManager | None = None


def get_system_manager() -> SystemManager:
    """Get or create the global system manager singleton."""
    global _system_manager
    if _system_manager is None:
        _system_manager = SystemManager()
    return _system_manager


def battery_status() -> dict[str, Any]:
    return get_system_manager().get_battery_status().to_dict()


def wifi_status() -> dict[str, Any]:
    return get_system_manager().get_wifi_status().to_dict()


def running_apps() -> dict[str, Any]:
    return get_system_manager().get_top_processes().to_dict()


def kill_app(name: str) -> dict[str, Any]:
    return get_system_manager().terminate_process(name).to_dict()


def disk_usage(drive: str = "C:") -> dict[str, Any]:
    return get_system_manager().get_disk_usage(drive).to_dict()


def cpu_ram_status() -> dict[str, Any]:
    return get_system_manager().get_resource_usage().to_dict()


def network_speed() -> dict[str, Any]:
    return get_system_manager().get_network_latency().to_dict()


# ── Test Suite ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Test System Skill functionality."""
    print("=" * 60)
    print("F.R.I.D.A.Y. System Skill Test Suite")
    print("=" * 60)

    sys_mgr = SystemManager()

    # ── Test 1: Battery ───────────────────────────────────────────────────────
    print("\n[TEST 1] Battery Status...")
    bat = sys_mgr.get_battery_status()
    print(f"   Success: {bat.success}")
    print(f"   Message: {bat.message}")
    if bat.data:
        print(f"   Data: {bat.data}")

    # ── Test 2: Wi-Fi ────────────────────────────────────────────────────────
    print("\n[TEST 2] Wi-Fi Status...")
    wifi = sys_mgr.get_wifi_status()
    print(f"   Success: {wifi.success}")
    print(f"   Message: {wifi.message}")

    # ── Test 3: Resources ────────────────────────────────────────────────────
    print("\n[TEST 3] CPU/RAM Usage...")
    res = sys_mgr.get_resource_usage()
    print(f"   Success: {res.success}")
    print(f"   Message: {res.message}")

    # ── Test 4: Disk ─────────────────────────────────────────────────────────
    print("\n[TEST 4] Disk Usage...")
    disk = sys_mgr.get_disk_usage()
    print(f"   Success: {disk.success}")
    print(f"   Message: {disk.message}")

    # ── Test 5: Network Latency ──────────────────────────────────────────────
    print("\n[TEST 5] Network Latency...")
    net = sys_mgr.get_network_latency()
    print(f"   Success: {net.success}")
    print(f"   Message: {net.message}")

    # ── Test 6: Top Processes ────────────────────────────────────────────────
    print("\n[TEST 6] Top Processes...")
    procs = sys_mgr.get_top_processes(limit=5)
    print(f"   Success: {procs.success}")
    if procs.data and "processes" in procs.data:
        for p in procs.data["processes"]:
            print(f"   - {p['name']} ({p['memory_mb']} MB)")

    # ── Test 7: Safety Check (Protected Process) ─────────────────────────────
    print("\n[TEST 7] Safety Check (Protected Process)...")
    kill = sys_mgr.terminate_process("explorer")
    print(f"   Success: {kill.success}")
    print(f"   Message: {kill.message}")
    assert not kill.success, "Should not be able to kill protected processes"

    print("\n" + "=" * 60)
    print("System Skill tests completed!")
    print("=" * 60)
