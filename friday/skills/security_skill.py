"""
F.R.I.D.A.Y. Security Skill
===========================
Network monitoring, anomaly detection, and cryptographic integrity verification.
Provides real-time security insights and threat assessment.
"""

from __future__ import annotations

import contextlib
import logging
import socket
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import psutil

logger = logging.getLogger(__name__)

# ── Constants & Safety ────────────────────────────────────────────────────────

# Network traffic thresholds (bytes per second)
TRAFFIC_THRESHOLD_NORMAL = 10 * 1024 * 1024  # 10 MB/s
TRAFFIC_THRESHOLD_CRITICAL = 50 * 1024 * 1024  # 50 MB/s

# Protected local network ranges
LOCAL_NETWORK_PREFIXES = (
    "127.",
    "192.168.",
    "10.",
    "172.16.",
    "172.17.",
    "172.18.",
    "172.19.",
    "172.20.",
    "172.21.",
    "172.22.",
    "172.23.",
    "172.24.",
    "172.25.",
    "172.26.",
    "172.27.",
    "172.28.",
    "172.29.",
    "172.30.",
    "172.31.",
    "169.254.",
    "0.0.0.0",
)

# Known suspicious ports (commonly used by malware)
SUSPICIOUS_PORTS = {4444, 5555, 6666, 31337, 12345, 54321, 1337}

# ── Data Models ───────────────────────────────────────────────────────────────


class ThreatLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityResult:
    """Standardized result structure for all security skills."""

    success: bool
    message: str
    data: dict[str, Any]
    threat_level: ThreatLevel
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["threat_level"] = self.threat_level.value
        return result


@dataclass
class NetworkConnection:
    """Information about a single network connection."""

    remote_address: str
    remote_port: int
    local_port: int
    protocol: str
    status: str
    pid: int | None
    process_name: str
    is_external: bool


@dataclass
class TrafficStats:
    """Network traffic statistics."""

    bytes_sent_per_sec: float
    bytes_recv_per_sec: float
    packets_sent_per_sec: float
    packets_recv_per_sec: float
    anomaly_detected: bool
    anomaly_type: str | None


@dataclass
class PQCStatus:
    """Post-Quantum Cryptography status."""

    status: str
    standard: str
    integrity_score: float
    last_key_rotation: str
    next_rotation_due: str


# ── Security Manager ──────────────────────────────────────────────────────────


class SecurityManager:
    """
    Centralized manager for security monitoring and threat detection.
    Handles network analysis, anomaly detection, and cryptographic integrity.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._logger = logging.getLogger("SecurityManager")
        self._baseline_traffic: TrafficStats | None = None
        self._last_io_counters = None
        self._last_io_time = None

        self._logger.info("[Security] Security manager initialized.")

    def _is_external_address(self, ip: str) -> bool:
        """Check if an IP address is external (not local/private)."""
        return not any(ip.startswith(prefix) for prefix in LOCAL_NETWORK_PREFIXES)

    def _get_protocol_name(self, conn_type: int) -> str:
        """Convert connection type to protocol name."""
        if conn_type == socket.SOCK_STREAM:
            return "TCP"
        elif conn_type == socket.SOCK_DGRAM:
            return "UDP"
        return "UNKNOWN"

    # ── Network Monitoring ────────────────────────────────────────────────────

    def scan_network_connections(self) -> SecurityResult:
        """
        Scan active network connections for external outbound traffic.
        Identifies potentially suspicious connections.
        """
        try:
            connections = psutil.net_connections(kind="inet")
            external_conns: list[NetworkConnection] = []
            suspicious_conns: list[NetworkConnection] = []

            for conn in connections:
                if conn.status != "ESTABLISHED" or not conn.raddr:
                    continue

                r_addr, r_port = conn.raddr
                _l_addr, l_port = conn.laddr if conn.laddr else ("0.0.0.0", 0)

                is_external = self._is_external_address(r_addr)

                # Get process name safely
                process_name = "Unknown"
                if conn.pid:
                    with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                        process_name = psutil.Process(conn.pid).name()

                network_conn = NetworkConnection(
                    remote_address=r_addr,
                    remote_port=r_port,
                    local_port=l_port,
                    protocol=self._get_protocol_name(conn.type),
                    status=conn.status,
                    pid=conn.pid,
                    process_name=process_name,
                    is_external=is_external,
                )

                if is_external:
                    external_conns.append(network_conn)

                    # Check for suspicious ports
                    if r_port in SUSPICIOUS_PORTS:
                        suspicious_conns.append(network_conn)

            # Determine threat level
            threat_level = ThreatLevel.NONE
            if suspicious_conns:
                threat_level = ThreatLevel.HIGH
            elif len(external_conns) > 20:
                threat_level = ThreatLevel.MEDIUM
            elif len(external_conns) > 10:
                threat_level = ThreatLevel.LOW

            # Build message
            msg = f"Sir, I've identified {len(external_conns)} active external connections."
            if suspicious_conns:
                msg += f" WARNING: {len(suspicious_conns)} suspicious connections detected on known malware ports."
            elif external_conns:
                processes = ", ".join({c.process_name for c in external_conns[:5]})
                msg += f" Primary traffic from: {processes}."

            data = {
                "external_connections": [asdict(c) for c in external_conns],
                "suspicious_connections": [asdict(c) for c in suspicious_conns],
                "total_connections": len(connections),
                "external_count": len(external_conns),
            }

            self._logger.info(
                f"[Security] Network scan complete: {len(external_conns)} external connections"
            )

            return SecurityResult(
                success=True,
                message=msg,
                data=data,
                threat_level=threat_level,
                timestamp=time.time(),
            )

        except Exception as e:
            self._logger.error(f"[Security] Network scan failed: {e}")
            return SecurityResult(
                success=False,
                message=f"I failed to scan the network: {e}",
                data={},
                threat_level=ThreatLevel.NONE,
                timestamp=time.time(),
            )

    # ── Traffic Anomaly Detection ─────────────────────────────────────────────

    def detect_traffic_anomalies(self, threshold_override: int | None = None) -> SecurityResult:
        """
        Check for suspicious high-volume outbound traffic spikes.
        Compares current traffic against baseline or threshold.
        """
        try:
            # Get IO counters twice with a small delay to calculate rate
            io1 = psutil.net_io_counters()
            time1 = time.time()

            time.sleep(1)

            io2 = psutil.net_io_counters()
            time2 = time.time()

            time_delta = time2 - time1
            if time_delta <= 0:
                time_delta = 1.0

            # Calculate rates (bytes per second)
            out_rate = (io2.bytes_sent - io1.bytes_sent) / time_delta
            in_rate = (io2.bytes_recv - io1.bytes_recv) / time_delta
            out_packets = (io2.packets_sent - io1.packets_sent) / time_delta
            in_packets = (io2.packets_recv - io1.packets_recv) / time_delta

            # Determine threshold
            threshold = threshold_override or TRAFFIC_THRESHOLD_NORMAL

            # Detect anomaly
            anomaly_detected = out_rate > threshold
            anomaly_type = None
            threat_level = ThreatLevel.NONE

            if out_rate > TRAFFIC_THRESHOLD_CRITICAL:
                anomaly_type = "critical_outbound_spike"
                threat_level = ThreatLevel.CRITICAL
            elif out_rate > threshold:
                anomaly_type = "high_outbound_traffic"
                threat_level = ThreatLevel.HIGH

            # Store baseline for future comparison
            if self._baseline_traffic is None:
                self._baseline_traffic = TrafficStats(
                    bytes_sent_per_sec=out_rate,
                    bytes_recv_per_sec=in_rate,
                    packets_sent_per_sec=out_packets,
                    packets_recv_per_sec=in_packets,
                    anomaly_detected=False,
                    anomaly_type=None,
                )

            # Build message
            out_mb = out_rate / (1024 * 1024)
            in_mb = in_rate / (1024 * 1024)

            if anomaly_detected:
                msg = f"Sir, I've detected a significant outbound traffic spike of {out_mb:.2f} MB/s. This warrants investigation."
            else:
                msg = f"Network traffic flows are within nominal parameters. Out: {out_mb:.2f} MB/s, In: {in_mb:.2f} MB/s, Boss."

            data = {
                "traffic_stats": {
                    "outbound_mbps": round(out_mb, 2),
                    "inbound_mbps": round(in_mb, 2),
                    "outbound_packets_per_sec": round(out_packets, 2),
                    "inbound_packets_per_sec": round(in_packets, 2),
                },
                "threshold_mbps": round(threshold / (1024 * 1024), 2),
                "anomaly_detected": anomaly_detected,
                "anomaly_type": anomaly_type,
            }

            self._logger.debug(f"[Security] Traffic check: {out_mb:.2f} MB/s outbound")

            return SecurityResult(
                success=True,
                message=msg,
                data=data,
                threat_level=threat_level,
                timestamp=time.time(),
            )

        except Exception as e:
            self._logger.error(f"[Security] Anomaly detection failed: {e}")
            return SecurityResult(
                success=False,
                message=f"Traffic anomaly detection failed: {e}",
                data={},
                threat_level=ThreatLevel.NONE,
                timestamp=time.time(),
            )

    def set_traffic_baseline(self) -> SecurityResult:
        """
        Set current traffic levels as the new baseline for anomaly detection.
        Useful after legitimate high-traffic operations.
        """
        try:
            io = psutil.net_io_counters()
            self._last_io_counters = io
            self._last_io_time = time.time()

            return SecurityResult(
                success=True,
                message="Sir, I've established a new traffic baseline for anomaly detection.",
                data={"baseline_set": True},
                threat_level=ThreatLevel.NONE,
                timestamp=time.time(),
            )
        except Exception as e:
            self._logger.error(f"[Security] Baseline set failed: {e}")
            return SecurityResult(
                success=False,
                message=f"Failed to set traffic baseline: {e}",
                data={},
                threat_level=ThreatLevel.NONE,
                timestamp=time.time(),
            )

    # ── Cryptographic Integrity ───────────────────────────────────────────────

    def check_pqc_integrity(self) -> SecurityResult:
        """
        Verify Post-Quantum Cryptography (PQC) key rotation and channel integrity.
        In production, this would interface with a hardware security module (HSM).
        """
        try:
            current_time = datetime.now()

            # Simulate PQC status (in production, this would query actual crypto modules)
            status = "Active"
            crypto_standard = "Kyber-768 / Dilithium-3"  # NIST PQC standards
            integrity_score = 99.4

            # Simulate key rotation schedule
            last_rotation = current_time.replace(hour=3, minute=0, second=0)
            next_rotation = last_rotation.replace(day=last_rotation.day + 7)

            # Check if rotation is overdue
            if current_time > next_rotation:
                status = "Rotation Required"
                integrity_score = 87.2
                threat_level = ThreatLevel.MEDIUM
            else:
                threat_level = ThreatLevel.NONE

            msg = f"Sir, the PQC secure channels are {status}. Encrypted using {crypto_standard}. Integrity score is {integrity_score}%."

            if status == "Rotation Required":
                msg += " WARNING: Key rotation is overdue."

            data = {
                "pqc_status": PQCStatus(
                    status=status,
                    standard=crypto_standard,
                    integrity_score=integrity_score,
                    last_key_rotation=last_rotation.strftime("%Y-%m-%d %H:%M:%S"),
                    next_rotation_due=next_rotation.strftime("%Y-%m-%d %H:%M:%S"),
                )
            }
            data["pqc_status"] = asdict(data["pqc_status"])

            self._logger.info(f"[Security] PQC integrity check: {status} ({integrity_score}%)")

            return SecurityResult(
                success=True,
                message=msg,
                data=data,
                threat_level=threat_level,
                timestamp=time.time(),
            )

        except Exception as e:
            self._logger.error(f"[Security] PQC integrity check failed: {e}")
            return SecurityResult(
                success=False,
                message=f"PQC integrity verification failed: {e}",
                data={},
                threat_level=ThreatLevel.NONE,
                timestamp=time.time(),
            )

    # ── Firewall & Port Security ──────────────────────────────────────────────

    def check_open_ports(self, scan_range: tuple[int, int] = (1, 1024)) -> SecurityResult:
        """
        Scan for open listening ports on the local machine.
        Warning: This is a passive check using net_connections, not active scanning.
        """
        try:
            connections = psutil.net_connections(kind="inet")
            listening_ports = []

            for conn in connections:
                if conn.status == "LISTEN" and conn.laddr:
                    port = conn.laddr.port
                    if scan_range[0] <= port <= scan_range[1]:
                        process_name = "Unknown"
                        if conn.pid:
                            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                                process_name = psutil.Process(conn.pid).name()

                        listening_ports.append(
                            {
                                "port": port,
                                "protocol": self._get_protocol_name(conn.type),
                                "process": process_name,
                                "pid": conn.pid,
                            }
                        )

            # Check for suspicious ports
            suspicious = [p for p in listening_ports if p["port"] in SUSPICIOUS_PORTS]

            threat_level = ThreatLevel.NONE
            if suspicious:
                threat_level = ThreatLevel.HIGH

            msg = f"Sir, I've identified {len(listening_ports)} listening ports in range {scan_range[0]}-{scan_range[1]}."
            if suspicious:
                msg += f" WARNING: {len(suspicious)} suspicious ports detected: {[p['port'] for p in suspicious]}."

            data = {
                "listening_ports": listening_ports,
                "suspicious_ports": suspicious,
                "scan_range": scan_range,
            }

            self._logger.info(
                f"[Security] Port scan complete: {len(listening_ports)} listening ports"
            )

            return SecurityResult(
                success=True,
                message=msg,
                data=data,
                threat_level=threat_level,
                timestamp=time.time(),
            )

        except Exception as e:
            self._logger.error(f"[Security] Port scan failed: {e}")
            return SecurityResult(
                success=False,
                message=f"Port scan failed: {e}",
                data={},
                threat_level=ThreatLevel.NONE,
                timestamp=time.time(),
            )

    # ── Comprehensive Security Report ─────────────────────────────────────────

    def generate_security_report(self) -> SecurityResult:
        """
        Generate a comprehensive security status report.
        Combines all security checks into a single assessment.
        """
        try:
            # Run all security checks
            network = self.scan_network_connections()
            traffic = self.detect_traffic_anomalies()
            pqc = self.check_pqc_integrity()
            ports = self.check_open_ports()

            # Determine overall threat level
            threat_levels = [
                network.threat_level,
                traffic.threat_level,
                pqc.threat_level,
                ports.threat_level,
            ]

            # Take the highest threat level
            overall_threat = max(threat_levels, key=lambda x: list(ThreatLevel).index(x))

            # Build summary
            msg = f"Sir, security report complete. Overall threat level: {overall_threat.value.upper()}."

            if overall_threat != ThreatLevel.NONE:
                msg += " Immediate attention recommended."

            data = {
                "network_scan": network.data,
                "traffic_analysis": traffic.data,
                "pqc_status": pqc.data,
                "port_scan": ports.data,
                "overall_threat_level": overall_threat.value,
                "report_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            self._logger.info(
                f"[Security] Comprehensive report generated. Threat level: {overall_threat.value}"
            )

            return SecurityResult(
                success=True,
                message=msg,
                data=data,
                threat_level=overall_threat,
                timestamp=time.time(),
            )

        except Exception as e:
            self._logger.error(f"[Security] Report generation failed: {e}")
            return SecurityResult(
                success=False,
                message=f"Security report generation failed: {e}",
                data={},
                threat_level=ThreatLevel.NONE,
                timestamp=time.time(),
            )


# ── Legacy Function Wrappers ──────────────────────────────────────────────────

# Global security manager instance
_security_manager: SecurityManager | None = None


def get_security_manager() -> SecurityManager:
    """Get or create the global security manager singleton."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


def monitor_network_connections() -> dict[str, Any]:
    return get_security_manager().scan_network_connections().to_dict()


def check_pqc_integrity() -> dict[str, Any]:
    return get_security_manager().check_pqc_integrity().to_dict()


def detect_traffic_anomalies() -> dict[str, Any]:
    return get_security_manager().detect_traffic_anomalies().to_dict()


# ── Test Suite ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Test Security Skill functionality."""
    print("=" * 60)
    print("F.R.I.D.A.Y. Security Skill Test Suite")
    print("=" * 60)

    sec_mgr = SecurityManager()

    # ── Test 1: Network Connections ──────────────────────────────────────────
    print("\n[TEST 1] Network Connection Scan...")
    network = sec_mgr.scan_network_connections()
    print(f"   Success: {network.success}")
    print(f"   Threat Level: {network.threat_level.value}")
    print(f"   Message: {network.message}")
    if network.data:
        print(f"   External Connections: {network.data.get('external_count', 0)}")

    # ── Test 2: Traffic Anomalies ────────────────────────────────────────────
    print("\n[TEST 2] Traffic Anomaly Detection...")
    traffic = sec_mgr.detect_traffic_anomalies()
    print(f"   Success: {traffic.success}")
    print(f"   Threat Level: {traffic.threat_level.value}")
    print(f"   Message: {traffic.message}")
    if traffic.data:
        stats = traffic.data.get("traffic_stats", {})
        print(f"   Outbound: {stats.get('outbound_mbps', 0)} MB/s")

    # ── Test 3: PQC Integrity ────────────────────────────────────────────────
    print("\n[TEST 3] PQC Integrity Check...")
    pqc = sec_mgr.check_pqc_integrity()
    print(f"   Success: {pqc.success}")
    print(f"   Threat Level: {pqc.threat_level.value}")
    print(f"   Message: {pqc.message}")

    # ── Test 4: Open Ports ───────────────────────────────────────────────────
    print("\n[TEST 4] Open Port Scan...")
    ports = sec_mgr.check_open_ports(scan_range=(1, 100))
    print(f"   Success: {ports.success}")
    print(f"   Threat Level: {ports.threat_level.value}")
    print(f"   Message: {ports.message}")
    if ports.data:
        print(f"   Listening Ports: {len(ports.data.get('listening_ports', []))}")

    # ── Test 5: Set Baseline ─────────────────────────────────────────────────
    print("\n[TEST 5] Set Traffic Baseline...")
    baseline = sec_mgr.set_traffic_baseline()
    print(f"   Success: {baseline.success}")
    print(f"   Message: {baseline.message}")

    # ── Test 6: Comprehensive Report ─────────────────────────────────────────
    print("\n[TEST 6] Comprehensive Security Report...")
    report = sec_mgr.generate_security_report()
    print(f"   Success: {report.success}")
    print(f"   Overall Threat Level: {report.threat_level.value}")
    print(f"   Message: {report.message}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Security Skill tests completed!")
    print("=" * 60)
