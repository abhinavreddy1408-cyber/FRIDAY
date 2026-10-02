"""
F.R.I.D.A.Y. IoT Skill
======================
Interface with local smart home devices using MQTT and REST.
Supports Home Assistant, Hue, MQTT brokers, and device state caching.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import requests

try:
    import paho.mqtt.client as mqtt

    _MQTT_AVAILABLE = True
except ImportError:
    _MQTT_AVAILABLE = False
    logging.warning("[IoT] paho-mqtt not installed — MQTT control disabled.")


logger = logging.getLogger(__name__)

# ── Constants & Configuration ─────────────────────────────────────────────────

# MQTT Defaults
DEFAULT_MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
DEFAULT_MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
DEFAULT_MQTT_KEEPALIVE = 60
DEFAULT_MQTT_USERNAME = os.getenv("MQTT_USERNAME")
DEFAULT_MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

# REST Defaults
DEFAULT_HUB_URL = os.getenv("SMART_HOME_HUB_URL", "http://homeassistant.local:8123")
DEFAULT_HUB_TOKEN = os.getenv("SMART_HOME_TOKEN")
REQUEST_TIMEOUT = 5

# State Cache TTL
STATE_CACHE_TTL = 30  # seconds

# ── Data Models ───────────────────────────────────────────────────────────────


class DeviceType(Enum):
    LIGHT = "light"
    SWITCH = "switch"
    SENSOR = "sensor"
    LOCK = "lock"
    CLIMATE = "climate"
    UNKNOWN = "unknown"


class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class IoTResult:
    """Standardized result structure for all IoT operations."""

    success: bool
    message: str
    data: dict[str, Any]
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SmartDevice:
    """Represents a single smart home device."""

    device_id: str
    name: str
    device_type: DeviceType
    state: str  # on, off, locked, unlocked, etc.
    brightness: int | None
    temperature: float | None
    last_updated: float
    reachable: bool


@dataclass
class DeviceStateCache:
    """Cached state for a device to avoid excessive polling."""

    state: dict[str, Any]
    cached_at: float


# ── IoT Manager ───────────────────────────────────────────────────────────────


class IoTManager:
    """
    Centralized manager for IoT operations.
    Handles MQTT connections, REST API calls, and device state caching.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._logger = logging.getLogger("IoTManager")

        # Configuration
        self.mqtt_broker = DEFAULT_MQTT_BROKER
        self.mqtt_port = DEFAULT_MQTT_PORT
        self.hub_url = DEFAULT_HUB_URL
        self.hub_token = DEFAULT_HUB_TOKEN

        # MQTT Client
        self.mqtt_client: mqtt.Client | None = None
        self.mqtt_connected = False
        self._mqtt_thread: threading.Thread | None = None

        # REST Session
        self._session = requests.Session()
        if self.hub_token:
            self._session.headers.update({"Authorization": f"Bearer {self.hub_token}"})

        # State Cache
        self._state_cache: dict[str, DeviceStateCache] = {}

        # Device Registry (Discovered devices)
        self._devices: dict[str, SmartDevice] = {}

        self._logger.info("[IoT] IoT manager initialized.")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """MQTT Connection Callback."""
        if rc == 0:
            self.mqtt_connected = True
            self._logger.info("[IoT] Connected to MQTT broker.")
            # Subscribe to state updates
            self.mqtt_client.subscribe("home/+/+/state")
        else:
            self.mqtt_connected = False
            self._logger.error(f"[IoT] MQTT connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc, properties=None):
        """MQTT Disconnection Callback."""
        self.mqtt_connected = False
        self._logger.warning("[IoT] Disconnected from MQTT broker.")

    def _on_message(self, client, userdata, msg):
        """MQTT Message Callback for state updates."""
        try:
            topic_parts = msg.topic.split("/")
            if len(topic_parts) >= 3:
                topic_parts[1]  # e.g., 'lights'
                device_id = topic_parts[2]  # e.g., 'living_room'
                payload = json.loads(msg.payload.decode())

                # Update cache
                with self._lock:
                    self._state_cache[device_id] = DeviceStateCache(
                        state=payload, cached_at=time.time()
                    )

                self._logger.debug(f"[IoT] State update for {device_id}: {payload}")
        except Exception as e:
            self._logger.error(f"[IoT] Error processing MQTT message: {e}")

    def connect_mqtt(self) -> bool:
        """Establish connection to the MQTT broker."""
        if not _MQTT_AVAILABLE:
            self._logger.warning("[IoT] paho-mqtt not installed.")
            return False

        with self._lock:
            try:
                if self.mqtt_client and self.mqtt_connected:
                    return True

                self.mqtt_client = mqtt.Client(
                    client_id="friday_iot", callback_api_version=mqtt.CallbackAPIVersion.VERSION2
                )
                self.mqtt_client.on_connect = self._on_connect
                self.mqtt_client.on_disconnect = self._on_disconnect
                self.mqtt_client.on_message = self._on_message

                if DEFAULT_MQTT_USERNAME:
                    self.mqtt_client.username_pw_set(DEFAULT_MQTT_USERNAME, DEFAULT_MQTT_PASSWORD)

                self.mqtt_client.connect_async(
                    self.mqtt_broker, self.mqtt_port, DEFAULT_MQTT_KEEPALIVE
                )
                self.mqtt_client.loop_start()

                # Wait for connection
                for _ in range(10):
                    if self.mqtt_connected:
                        return True
                    time.sleep(0.5)

                return self.mqtt_connected

            except Exception as e:
                self._logger.error(f"[IoT] MQTT setup failed: {e}")
                return False

    def disconnect_mqtt(self):
        """Disconnect from MQTT broker."""
        with self._lock:
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
                self.mqtt_client = None
                self.mqtt_connected = False
                self._logger.info("[IoT] MQTT disconnected.")

    def publish_mqtt(self, topic: str, payload: dict[str, Any]) -> bool:
        """Publish a message to an MQTT topic."""
        if not self.mqtt_connected and not self.connect_mqtt():
            return False

        try:
            msg = json.dumps(payload)
            info = self.mqtt_client.publish(topic, msg, qos=1)
            info.wait_for_publish()
            self._logger.debug(f"[IoT] Published to {topic}: {payload}")
            return True
        except Exception as e:
            self._logger.error(f"[IoT] MQTT publish failed: {e}")
            return False

    def rest_request(
        self, endpoint: str, method: str = "GET", data: dict | None = None
    ) -> IoTResult:
        """Make a REST call to the smart home hub."""
        try:
            url = f"{self.hub_url}{endpoint}"

            if method.upper() == "POST":
                resp = self._session.post(url, json=data, timeout=REQUEST_TIMEOUT)
            elif method.upper() == "PUT":
                resp = self._session.put(url, json=data, timeout=REQUEST_TIMEOUT)
            else:
                resp = self._session.get(url, timeout=REQUEST_TIMEOUT)

            if resp.status_code < 300:
                return IoTResult(
                    success=True,
                    message="Command executed successfully.",
                    data=resp.json() if resp.text else {},
                    timestamp=time.time(),
                )
            else:
                return IoTResult(
                    success=False,
                    message=f"Hub returned status {resp.status_code}",
                    data={"status_code": resp.status_code},
                    timestamp=time.time(),
                )
        except Exception as e:
            self._logger.error(f"[IoT] REST call failed: {e}")
            return IoTResult(
                success=False, message=f"REST call failed: {e}", data={}, timestamp=time.time()
            )

    def _get_cached_state(self, device_id: str) -> dict[str, Any] | None:
        """Get device state from cache if valid."""
        with self._lock:
            if device_id in self._state_cache:
                cache = self._state_cache[device_id]
                if (time.time() - cache.cached_at) < STATE_CACHE_TTL:
                    return cache.state
        return None

    def _update_cached_state(self, device_id: str, state: dict[str, Any]):
        """Update device state cache."""
        with self._lock:
            self._state_cache[device_id] = DeviceStateCache(state=state, cached_at=time.time())

    # ── Device Control ────────────────────────────────────────────────────────

    def set_device_state(
        self, device_id: str, state: str, brightness: int | None = None
    ) -> IoTResult:
        """
        Set the state of a device (on/off/dim).
        Tries MQTT first, falls back to REST.
        """
        # 1. Try MQTT
        topic = f"home/devices/{device_id}/set"
        payload = {"state": state.upper()}
        if brightness is not None:
            payload["brightness"] = brightness

        if self.publish_mqtt(topic, payload):
            self._update_cached_state(device_id, payload)
            return IoTResult(
                success=True,
                message=f"Sir, I've set {device_id} to {state}.",
                data={"device_id": device_id, "state": state},
                timestamp=time.time(),
            )

        # 2. Fallback to REST (Home Assistant style)
        entity_id = f"light.{device_id}" if "light" in device_id else f"switch.{device_id}"
        endpoint = f"/api/services/homeassistant/turn_{state.lower()}"
        data = {"entity_id": entity_id}
        if brightness:
            data["brightness"] = brightness
            endpoint = "/api/services/light/turn_on"

        result = self.rest_request(endpoint, method="POST", data=data)
        if result.success:
            self._update_cached_state(device_id, {"state": state})
            result.message = f"Sir, I've set {device_id} to {state} via hub."

        return result

    def get_device_state(self, device_id: str) -> IoTResult:
        """Get the current state of a device."""
        # Check cache first
        cached = self._get_cached_state(device_id)
        if cached:
            return IoTResult(
                success=True,
                message=f"{device_id} is {cached.get('state', 'unknown')}.",
                data=cached,
                timestamp=time.time(),
            )

        # Try REST
        entity_id = f"light.{device_id}" if "light" in device_id else f"switch.{device_id}"
        endpoint = f"/api/states/{entity_id}"
        result = self.rest_request(endpoint)

        if result.success and result.data:
            state = result.data.get("state", "unknown")
            self._update_cached_state(device_id, result.data)
            result.message = f"{device_id} is currently {state}."

        return result

    def read_sensor(self, sensor_id: str) -> IoTResult:
        """Read data from a sensor (temperature, humidity, etc.)."""
        # Check cache
        cached = self._get_cached_state(sensor_id)
        if cached:
            val = cached.get("value", "unknown")
            unit = cached.get("unit", "")
            return IoTResult(
                success=True,
                message=f"The {sensor_id} is reporting {val}{unit}.",
                data=cached,
                timestamp=time.time(),
            )

        # Try REST
        endpoint = f"/api/states/sensor.{sensor_id}"
        result = self.rest_request(endpoint)

        if result.success and result.data:
            val = result.data.get("state", "unknown")
            attrs = result.data.get("attributes", {})
            unit = attrs.get("unit_of_measurement", "")
            self._update_cached_state(sensor_id, {"value": val, "unit": unit, **attrs})
            result.message = f"The {sensor_id} is reporting {val}{unit}, Sir."

        return result

    def discover_devices(self) -> IoTResult:
        """
        Discover available devices on the network.
        In production, this would scan MQTT discovery topics or Hub API.
        """
        # Stub implementation: Query Hub API for all lights
        endpoint = "/api/states"
        result = self.rest_request(endpoint)

        devices = []
        if result.success and result.data:
            for entity in result.data:
                if entity["entity_id"].startswith(("light.", "switch.")):
                    devices.append(
                        {
                            "id": entity["entity_id"].split(".")[1],
                            "type": entity["entity_id"].split(".")[0],
                            "state": entity["state"],
                        }
                    )

        return IoTResult(
            success=True,
            message=f"Found {len(devices)} devices, Sir.",
            data={"devices": devices},
            timestamp=time.time(),
        )

    def get_stats(self) -> IoTResult:
        """Get IoT system statistics."""
        return IoTResult(
            success=True,
            message="IoT stats retrieved.",
            data={
                "mqtt_connected": self.mqtt_connected,
                "cached_devices": len(self._state_cache),
                "hub_url": self.hub_url,
            },
            timestamp=time.time(),
        )


# ── Legacy Function Wrappers ──────────────────────────────────────────────────

# Global IoT manager instance
_iot_manager: IoTManager | None = None


def get_iot_manager() -> IoTManager:
    """Get or create the global IoT manager singleton."""
    global _iot_manager
    if _iot_manager is None:
        _iot_manager = IoTManager()
    return _iot_manager


def toggle_light(device_id: str, state: str) -> dict[str, Any]:
    """Legacy wrapper for toggling lights."""
    result = get_iot_manager().set_device_state(device_id, state)
    return result.to_dict()


def read_sensor(sensor_id: str) -> dict[str, Any]:
    """Legacy wrapper for reading sensors."""
    result = get_iot_manager().read_sensor(sensor_id)
    return result.to_dict()


def discover_devices() -> dict[str, Any]:
    """Legacy wrapper for device discovery."""
    result = get_iot_manager().discover_devices()
    return result.to_dict()


# ── Test Suite ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Test IoT Skill functionality."""
    print("=" * 60)
    print("F.R.I.D.A.Y. IoT Skill Test Suite")
    print("=" * 60)

    iot_mgr = IoTManager()

    # ── Test 1: MQTT Connection ──────────────────────────────────────────────
    print("\n[TEST 1] MQTT Connection...")
    # Note: This will fail if no broker is running, which is expected in test env
    connected = iot_mgr.connect_mqtt()
    print(f"   Connected: {connected}")
    print(f"   Status: {'✅' if connected else '⚠️  (No broker found)'}")

    # ── Test 2: Device Discovery ─────────────────────────────────────────────
    print("\n[TEST 2] Device Discovery...")
    # Note: This will fail if no hub is running
    result = iot_mgr.discover_devices()
    print(f"   Success: {result.success}")
    print(f"   Message: {result.message}")
    if result.data:
        print(f"   Devices Found: {len(result.data.get('devices', []))}")

    # ── Test 3: Set Device State (Mock) ──────────────────────────────────────
    print("\n[TEST 3] Set Device State (Mock)...")
    # We won't actually send this to avoid breaking real devices
    print("   Skipping actual command to prevent hardware interaction.")
    print("   Method: set_device_state(device_id, state)")

    # ── Test 4: Get Device State (Cache) ─────────────────────────────────────
    print("\n[TEST 4] State Cache...")
    iot_mgr._update_cached_state("test_light", {"state": "on", "brightness": 200})
    result = iot_mgr.get_device_state("test_light")
    print(f"   Success: {result.success}")
    print(f"   Cached State: {result.data.get('state')}")

    # ── Test 5: Sensor Read (Cache) ──────────────────────────────────────────
    print("\n[TEST 5] Sensor Read (Cache)...")
    iot_mgr._update_cached_state("temp_sensor", {"value": 22.5, "unit": "°C"})
    result = iot_mgr.read_sensor("temp_sensor")
    print(f"   Success: {result.success}")
    print(f"   Message: {result.message}")

    # ── Test 6: REST Request (Mock) ──────────────────────────────────────────
    print("\n[TEST 6] REST Request Structure...")
    print(f"   Hub URL: {iot_mgr.hub_url}")
    print(f"   Token Set: {bool(iot_mgr.hub_token)}")
    print(f"   Session Headers: {len(iot_mgr._session.headers)}")

    # ── Test 7: Statistics ───────────────────────────────────────────────────
    print("\n[TEST 7] IoT Statistics...")
    stats = iot_mgr.get_stats()
    print(f"   Success: {stats.success}")
    print(f"   Data: {stats.data}")

    # ── Test 8: Cleanup ──────────────────────────────────────────────────────
    print("\n[TEST 8] Cleanup...")
    iot_mgr.disconnect_mqtt()
    print("   MQTT Disconnected.")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("IoT Skill tests completed!")
    print("=" * 60)
