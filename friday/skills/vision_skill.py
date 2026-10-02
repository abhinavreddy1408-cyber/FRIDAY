"""
F.R.I.D.A.Y. Vision Skill
==========================
Capture and process visual information from screens and webcams.
Supports multiple monitors, camera selection, and base64 encoding for LLM vision.
"""

from __future__ import annotations

import base64
import contextlib
import logging
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Optional Dependencies ────────────────────────────────────────────────────

try:
    import mss
    import mss.tools

    _MSS_AVAILABLE = True
except ImportError:
    _MSS_AVAILABLE = False
    logger.warning("[Vision] mss not installed — screen capture disabled.")

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("[Vision] opencv-python not installed — webcam capture disabled.")


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass
class CaptureResult:
    """Standardized result structure for all vision captures."""

    success: bool
    path: str | None
    base64_data: str | None
    message: str
    timestamp: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MonitorInfo:
    """Information about available monitors."""

    monitor_id: int
    width: int
    height: int
    left: int
    top: int
    is_primary: bool


# ── Vision Manager ────────────────────────────────────────────────────────────


class VisionManager:
    """
    Thread-safe vision capture manager for F.R.I.D.A.Y.
    Handles screen captures, webcam frames, and image encoding.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._temp_dir = Path(tempfile.gettempdir()) / "friday_vision"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._camera_cache: dict[int, cv2.VideoCapture] = {}

        # Auto-cleanup old files on initialization
        self._cleanup_old_files(max_age_hours=24)

        logger.info("[Vision] Vision manager initialized.")

    def _cleanup_old_files(self, max_age_hours: int = 24):
        """Remove temporary files older than specified hours."""
        try:
            now = time.time()
            max_age_seconds = max_age_hours * 3600
            for file in self._temp_dir.glob("*"):
                if file.is_file() and (now - file.stat().st_mtime) > max_age_seconds:
                    file.unlink()
                    logger.debug(f"[Vision] Cleaned up old file: {file.name}")
        except Exception as e:
            logger.error(f"[Vision] Cleanup failed: {e}")

    def _generate_filename(self, prefix: str, extension: str = ".png") -> str:
        """Generate a unique filename for captures."""
        timestamp = int(time.time() * 1000)
        return str(self._temp_dir / f"{prefix}_{timestamp}{extension}")

    def get_available_monitors(self) -> list[MonitorInfo]:
        """Get information about all available monitors."""
        if not _MSS_AVAILABLE:
            return []

        try:
            with mss.mss() as sct:
                monitors = []
                for i, mon in enumerate(sct.monitors):
                    if i == 0:  # Skip the virtual "all monitors" entry
                        continue
                    monitors.append(
                        MonitorInfo(
                            monitor_id=i,
                            width=mon["width"],
                            height=mon["height"],
                            left=mon["left"],
                            top=mon["top"],
                            is_primary=(i == 1),
                        )
                    )
                return monitors
        except Exception as e:
            logger.error(f"[Vision] Failed to get monitor info: {e}")
            return []

    def capture_screen(
        self, monitor_id: int = 1, encode_base64: bool = True, compress: bool = False
    ) -> CaptureResult:
        """
        Capture a screenshot of the specified monitor.

        Args:
            monitor_id: Monitor number (1 = primary, 2+ = secondary monitors)
            encode_base64: Whether to return base64 encoded data for LLM vision
            compress: Whether to compress the image (reduces size for LLM tokens)

        Returns:
            CaptureResult with path and optional base64 data
        """
        if not _MSS_AVAILABLE:
            return CaptureResult(
                success=False,
                path=None,
                base64_data=None,
                message="Sir, screen capture module (mss) is not available.",
                timestamp=time.time(),
                metadata={},
            )

        with self._lock:
            try:
                screenshot_path = self._generate_filename("screen", ".png")

                with mss.mss() as sct:
                    # Get monitor bounds
                    monitors = sct.monitors
                    if monitor_id >= len(monitors):
                        monitor_id = 1  # Fallback to primary

                    monitor = monitors[monitor_id]

                    # Capture the monitor
                    screenshot = sct.grab(monitor)

                    # Save to file
                    mss.tools.to_png(screenshot.rgb, screenshot.size, output=str(screenshot_path))

                # Optionally encode to base64 for LLM vision
                base64_data = None
                if encode_base64 and screenshot_path.exists():
                    base64_data = self._encode_image(str(screenshot_path), compress)

                logger.info(
                    f"[Vision] Screen captured (monitor {monitor_id}): {screenshot_path.name}"
                )

                return CaptureResult(
                    success=True,
                    path=str(screenshot_path),
                    base64_data=base64_data,
                    message=f"I've captured monitor {monitor_id}, Sir.",
                    timestamp=time.time(),
                    metadata={
                        "monitor_id": monitor_id,
                        "width": monitor["width"],
                        "height": monitor["height"],
                        "compressed": compress,
                    },
                )

            except Exception as e:
                logger.error(f"[Vision] Screen capture failed: {e}")
                return CaptureResult(
                    success=False,
                    path=None,
                    base64_data=None,
                    message=f"I failed to capture the screen: {e}",
                    timestamp=time.time(),
                    metadata={"error": str(e)},
                )

    def capture_webcam(
        self,
        camera_index: int = 0,
        encode_base64: bool = True,
        compress: bool = False,
        release_immediately: bool = True,
    ) -> CaptureResult:
        """
        Capture a frame from the specified webcam.

        Args:
            camera_index: Camera device index (0 = default)
            encode_base64: Whether to return base64 encoded data for LLM vision
            compress: Whether to compress the image (JPEG quality reduction)
            release_immediately: Whether to release camera after capture

        Returns:
            CaptureResult with path and optional base64 data
        """
        if not _CV2_AVAILABLE:
            return CaptureResult(
                success=False,
                path=None,
                base64_data=None,
                message="Sir, computer vision module (opencv) is not available.",
                timestamp=time.time(),
                metadata={},
            )

        with self._lock:
            cap = None
            try:
                frame_path = self._generate_filename("webcam", ".jpg")

                # Initialize camera
                cap = cv2.VideoCapture(camera_index)

                if not cap.isOpened():
                    return CaptureResult(
                        success=False,
                        path=None,
                        base64_data=None,
                        message=f"I couldn't access webcam {camera_index}, Sir.",
                        timestamp=time.time(),
                        metadata={"camera_index": camera_index},
                    )

                # Configure camera properties for better quality
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

                # Warm up the camera (first frame is often dark)
                for _ in range(3):
                    cap.read()

                # Capture frame
                ret, frame = cap.read()

                if not ret or frame is None:
                    return CaptureResult(
                        success=False,
                        path=None,
                        base64_data=None,
                        message="Sir, the camera returned an empty frame.",
                        timestamp=time.time(),
                        metadata={"camera_index": camera_index},
                    )

                # Save with quality settings
                save_params = [cv2.IMWRITE_JPEG_QUALITY, 85 if compress else 95]
                cv2.imwrite(str(frame_path), frame, save_params)

                # Release camera if requested
                if release_immediately:
                    cap.release()

                # Encode to base64 if requested
                base64_data = None
                if encode_base64 and frame_path.exists():
                    base64_data = self._encode_image(str(frame_path), compress)

                logger.info(
                    f"[Vision] Webcam frame captured (camera {camera_index}): {frame_path.name}"
                )

                return CaptureResult(
                    success=True,
                    path=str(frame_path),
                    base64_data=base64_data,
                    message=f"I've captured a frame from webcam {camera_index}, Sir.",
                    timestamp=time.time(),
                    metadata={
                        "camera_index": camera_index,
                        "width": frame.shape[1] if len(frame.shape) > 1 else 0,
                        "height": frame.shape[0] if len(frame.shape) > 1 else 0,
                        "compressed": compress,
                    },
                )

            except Exception as e:
                logger.error(f"[Vision] Webcam capture failed: {e}")
                return CaptureResult(
                    success=False,
                    path=None,
                    base64_data=None,
                    message=f"I failed to capture the webcam: {e}",
                    timestamp=time.time(),
                    metadata={"error": str(e)},
                )
            finally:
                if cap is not None and release_immediately:
                    cap.release()

    def _encode_image(self, path: str, compress: bool = False) -> str | None:
        """
        Encode an image file to base64 string for LLM vision APIs.

        Args:
            path: Path to the image file
            compress: If True, resize image to reduce token count

        Returns:
            Base64 encoded string or None on failure
        """
        try:
            if not os.path.exists(path):
                return None

            image_data = None

            if compress and _CV2_AVAILABLE:
                # Load, resize, and re-encode for smaller payload
                img = cv2.imread(path)
                if img is not None:
                    # Resize to max 1024px on longest side
                    h, w = img.shape[:2]
                    max_dim = 1024
                    if max(h, w) > max_dim:
                        scale = max_dim / max(h, w)
                        img = cv2.resize(img, (int(w * scale), int(h * scale)))

                    # Encode to JPEG in memory
                    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    image_data = buffer.tobytes()

            if image_data is None:
                # Read original file
                with open(path, "rb") as image_file:
                    image_data = image_file.read()

            return base64.b64encode(image_data).decode("utf-8")

        except Exception as e:
            logger.error(f"[Vision] Image encoding failed: {e}")
            return None

    def get_image_base64(self, path: str, compress: bool = False) -> str | None:
        """Public method to encode any image file to base64."""
        return self._encode_image(path, compress)

    def list_available_cameras(self) -> list[int]:
        """
        Probe system for available camera devices.
        Returns list of working camera indices.
        """
        if not _CV2_AVAILABLE:
            return []

        available = []
        # Test indices 0-9 (covers most systems)
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def cleanup(self):
        """Release all held camera resources and clean temp files."""
        with self._lock:
            for _idx, cap in self._camera_cache.items():
                with contextlib.suppress(BaseException):
                    cap.release()
            self._camera_cache.clear()
            self._cleanup_old_files(max_age_hours=1)

        logger.info("[Vision] Vision manager cleanup completed.")


# ── Legacy Function Wrappers (for backward compatibility) ────────────────────

# Global vision manager instance
_vision_manager: VisionManager | None = None


def get_vision_manager() -> VisionManager:
    """Get or create the global vision manager singleton."""
    global _vision_manager
    if _vision_manager is None:
        _vision_manager = VisionManager()
    return _vision_manager


def capture_screen(monitor_id: int = 1) -> dict[str, Any]:
    """Legacy wrapper for capture_screen."""
    result = get_vision_manager().capture_screen(monitor_id=monitor_id)
    return result.to_dict()


def capture_webcam(camera_index: int = 0) -> dict[str, Any]:
    """Legacy wrapper for capture_webcam."""
    result = get_vision_manager().capture_webcam(camera_index=camera_index)
    return result.to_dict()


def get_encoded_image(path: str, compress: bool = False) -> str | None:
    """Legacy wrapper for image encoding."""
    return get_vision_manager().get_image_base64(path, compress)


# ── Test Suite ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Test Vision Skill functionality."""
    print("=" * 60)
    print("F.R.I.D.A.Y. Vision Skill Test Suite")
    print("=" * 60)

    vision = VisionManager()

    # ── Test 1: Available Monitors ───────────────────────────────────────────
    print("\n[TEST 1] Detecting Monitors...")
    monitors = vision.get_available_monitors()
    if monitors:
        for m in monitors:
            print(
                f"   Monitor {m.monitor_id}: {m.width}x{m.height} {'(Primary)' if m.is_primary else ''}"
            )
        print("✅ Monitor Detection Passed")
    else:
        print("⚠️  No monitors detected (mss may not be installed)")

    # ── Test 2: Screen Capture ───────────────────────────────────────────────
    print("\n[TEST 2] Screen Capture...")
    screen_result = vision.capture_screen(monitor_id=1, encode_base64=True)
    if screen_result.success:
        print(f"   Path: {screen_result.path}")
        print(
            f"   Base64 Length: {len(screen_result.base64_data) if screen_result.base64_data else 0} chars"
        )
        print("✅ Screen Capture Passed")
    else:
        print(f"⚠️  Screen Capture: {screen_result.message}")

    # ── Test 3: Available Cameras ────────────────────────────────────────────
    print("\n[TEST 3] Detecting Cameras...")
    cameras = vision.list_available_cameras()
    if cameras:
        print(f"   Available cameras: {cameras}")
        print("✅ Camera Detection Passed")
    else:
        print("⚠️  No cameras detected (opencv may not be installed)")

    # ── Test 4: Webcam Capture ───────────────────────────────────────────────
    print("\n[TEST 4] Webcam Capture...")
    if cameras:
        webcam_result = vision.capture_webcam(camera_index=cameras[0], encode_base64=True)
        if webcam_result.success:
            print(f"   Path: {webcam_result.path}")
            print(
                f"   Resolution: {webcam_result.metadata.get('width', 0)}x{webcam_result.metadata.get('height', 0)}"
            )
            print("✅ Webcam Capture Passed")
        else:
            print(f"⚠️  Webcam Capture: {webcam_result.message}")
    else:
        print("⚠️  Skipped (no cameras available)")

    # ── Test 5: Base64 Encoding ──────────────────────────────────────────────
    print("\n[TEST 5] Base64 Encoding...")
    if screen_result.success and screen_result.path:
        encoded = vision.get_image_base64(screen_result.path, compress=True)
        if encoded:
            print(f"   Compressed Base64 Length: {len(encoded)} chars")
            print("✅ Base64 Encoding Passed")
        else:
            print("⚠️  Encoding failed")
    else:
        print("⚠️  Skipped (no screen capture available)")

    # ── Test 6: Cleanup ──────────────────────────────────────────────────────
    print("\n[TEST 6] Resource Cleanup...")
    vision.cleanup()
    print("✅ Cleanup Passed")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Vision Skill tests completed!")
    print("=" * 60)
