"""
F.R.I.D.A.Y. Skills Sub-Package
===============================
Exposes all skill classes through a unified registry with lazy loading.

Skills extend F.R.I.D.A.Y.'s capabilities beyond core local actions.
Each skill is self-contained, independently importable, and registers
itself with the ActionRouter via the SkillRegistry.

Architecture:
  - Lazy loading: Skills only import when needed
  - Graceful degradation: Missing dependencies don't crash the system
  - Status tracking: Know which skills are available at runtime
  - Hot reload: Skills can be reloaded without restart (Phase 5)
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field  # noqa: F401
from enum import Enum
from pathlib import Path  # noqa: F401
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple  # noqa: UP035, F401

from friday.config import config  # noqa: F401

if TYPE_CHECKING:
    from friday.actions import ActionRouter

logger = logging.getLogger(__name__)

# ── Skill Status & Metadata ───────────────────────────────────────────────────


class SkillStatus(Enum):
    """Represents the current state of a skill."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class SkillInfo:
    """Metadata about a registered skill."""

    name: str
    module_path: str
    class_name: str
    status: SkillStatus
    loaded_at: float | None
    error_message: str | None
    version: str
    description: str
    dependencies: list[str]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result


# ── Skill Registry Configuration ──────────────────────────────────────────────

# Core skills (always available, no external dependencies)
_CORE_SKILLS: dict[str, tuple[str, str, str, str, list[str]]] = {
    # name: (module_path, class_name, version, description, dependencies)
    "system": (
        "friday.skills.system_skill",
        "SystemManager",
        "5.0.0",
        "System monitoring and control",
        [],
    ),
    "memory": ("friday.memory", "MemoryManager", "5.0.0", "Short/long/semantic memory", []),
    "productivity": (
        "friday.skills.productivity_skill",
        "ProductivityManager",
        "5.0.0",
        "Notes, todos, clipboard",
        [],
    ),
}

# Extended skills (require external APIs or services)
_EXTENDED_SKILLS: dict[str, tuple[str, str, str, str, list[str]]] = {
    "email": (
        "friday.skills.email_skill",
        "EmailSkill",
        "5.0.0",
        "Gmail compose, read, send",
        ["google-auth", "google-api-python-client", "google-auth-oauthlib"],
    ),
    "calendar": (
        "friday.skills.calendar_skill",
        "CalendarSkill",
        "5.0.0",
        "Calendar queries and reminders",
        ["google-auth", "google-api-python-client", "google-auth-oauthlib"],
    ),
    "file": ("friday.skills.file_skill", "FileSkill", "5.0.0", "File search, open, organise", []),
    "browser": (
        "friday.skills.browser_skill",
        "BrowserSkill",
        "5.0.0",
        "Playwright-based browser automation",
        ["playwright"],
    ),
    "task": (
        "friday.skills.task_skill",
        "TaskSkill",
        "5.0.0",
        "Autonomous goal planning and execution",
        [],
    ),
    "weather": (
        "friday.skills.weather_skill",
        "WeatherSkill",
        "5.0.0",
        "Current conditions and forecasts",
        ["requests"],
    ),
    "vision": (
        "friday.skills.vision_skill",
        "VisionManager",
        "5.0.0",
        "Screen capture and webcam",
        ["mss", "opencv-python"],
    ),
    "security": (
        "friday.skills.security_skill",
        "SecurityManager",
        "5.0.0",
        "Network monitoring and threat detection",
        ["psutil"],
    ),
    "research": (
        "friday.skills.research_skill",
        "ResearchManager",
        "5.0.0",
        "Web scraping and PDF processing",
        ["requests", "beautifulsoup4", "PyPDF2"],
    ),
    "iot": (
        "friday.skills.iot_skill",
        "IoTManager",
        "5.0.0",
        "Smart home MQTT and REST control",
        ["paho-mqtt", "requests"],
    ),
}

# Combined registry
_SKILL_REGISTRY: dict[str, tuple[str, str, str, str, list[str]]] = {
    **_CORE_SKILLS,
    **_EXTENDED_SKILLS,
}

# ── Skill Manager ─────────────────────────────────────────────────────────────


class SkillManager:
    """
    Centralized manager for skill loading, registration, and status tracking.
    Implements lazy loading and graceful degradation.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._logger = logging.getLogger("SkillManager")
        self._llm: Any = None

        # Loaded skill instances
        self._loaded_skills: dict[str, Any] = {}

        # Skill metadata and status
        self._skill_info: dict[str, SkillInfo] = {}

        # Initialize metadata for all skills
        self._init_skill_metadata()

        self._logger.info("[Skills] Skill manager initialized.")

    def _init_skill_metadata(self):
        """Initialize metadata for all registered skills."""
        for name, (
            module_path,
            class_name,
            version,
            description,
            dependencies,
        ) in _SKILL_REGISTRY.items():
            self._skill_info[name] = SkillInfo(
                name=name,
                module_path=module_path,
                class_name=class_name,
                status=SkillStatus.UNLOADED,
                loaded_at=None,
                error_message=None,
                version=version,
                description=description,
                dependencies=dependencies,
            )

    def _check_dependencies(self, dependencies: list[str]) -> tuple[bool, list[str]]:
        """
        Check if all dependencies for a skill are available.
        Returns (all_available, missing_list).
        """
        # Map PyPI package names to their actual Python import names
        import_map = {
            "beautifulsoup4": "bs4",
            "google-api-python-client": "googleapiclient",
            "google-auth": "google.auth",
            "opencv-python": "cv2",
            "paho-mqtt": "paho",
        }

        missing = []
        for dep in dependencies:
            import_name = import_map.get(dep.lower(), dep.replace("-", "_"))
            try:
                __import__(import_name)
            except ImportError:
                missing.append(dep)
        return len(missing) == 0, missing

    def get_skill_info(self, name: str) -> SkillInfo | None:
        """Get metadata for a specific skill."""
        return self._skill_info.get(name.lower())

    def get_all_skill_info(self) -> list[SkillInfo]:
        """Get metadata for all skills."""
        return list(self._skill_info.values())

    def load_skill(self, name: str, force: bool = False) -> Any | None:
        """
        Lazily import and instantiate a skill by name.

        Args:
            name: Skill key from the registry (e.g., "email", "weather").
            force: Force reload even if already loaded.

        Returns:
            Instantiated skill object or None if import fails.
        """
        name = name.lower()

        with self._lock:
            # Check if already loaded
            if name in self._loaded_skills and not force:
                return self._loaded_skills[name]

            # Check if skill exists in registry
            if name not in _SKILL_REGISTRY:
                self._logger.warning("[Skills] Unknown skill: '%s'.", name)
                if name in self._skill_info:
                    self._skill_info[name].status = SkillStatus.ERROR
                    self._skill_info[name].error_message = "Unknown skill name"
                return None

            # Update status
            self._skill_info[name].status = SkillStatus.LOADING

            module_path, class_name, version, _description, dependencies = _SKILL_REGISTRY[name]

            # Check dependencies
            deps_ok, missing = self._check_dependencies(dependencies)
            if not deps_ok:
                self._skill_info[name].status = SkillStatus.DISABLED
                self._skill_info[name].error_message = f"Missing dependencies: {', '.join(missing)}"
                if name in _CORE_SKILLS:
                    self._logger.warning("[Skills] Core skill '%s' disabled — missing: %s", name, missing)
                else:
                    self._logger.info("[Skills] Extended skill '%s' disabled — missing: %s", name, missing)
                return None

            # Import and instantiate
            try:
                import importlib

                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)

                # Handle both class-based managers and skill instances
                skill = cls() if hasattr(cls, "__init__") and callable(cls.__init__) else cls

                # Inject llm reference if set
                if self._llm is not None:
                    if hasattr(skill, "_llm"):
                        skill._llm = self._llm
                    elif hasattr(skill, "llm"):
                        skill.llm = self._llm

                self._loaded_skills[name] = skill
                self._skill_info[name].status = SkillStatus.READY
                self._skill_info[name].loaded_at = time.time()

                self._logger.info("[Skills] '%s' loaded successfully (%s).", name, version)
                return skill

            except ImportError as exc:
                self._skill_info[name].status = SkillStatus.ERROR
                self._skill_info[name].error_message = f"Import error: {exc}"
                self._logger.warning("[Skills] '%s' unavailable — import failed: %s", name, exc)
                return None

            except Exception as exc:
                self._skill_info[name].status = SkillStatus.ERROR
                self._skill_info[name].error_message = str(exc)
                self._logger.error("[Skills] Failed to load '%s': %s", name, exc)
                return None

    def unload_skill(self, name: str) -> bool:
        """
        Unload a skill from memory.
        Useful for freeing resources or hot-reloading.
        """
        name = name.lower()

        with self._lock:
            if name not in self._loaded_skills:
                return False

            skill = self._loaded_skills[name]

            # Call cleanup method if available
            if hasattr(skill, "cleanup") and callable(skill.cleanup):
                try:
                    skill.cleanup()
                except Exception as e:
                    self._logger.error("[Skills] Cleanup failed for '%s': %s", name, e)

            del self._loaded_skills[name]
            self._skill_info[name].status = SkillStatus.UNLOADED
            self._skill_info[name].loaded_at = None

            self._logger.info("[Skills] '%s' unloaded.", name)
            return True

    def reload_skill(self, name: str) -> Any | None:
        """
        Reload a skill (unload + load).
        Used for hot-reloading during development.
        """
        self.unload_skill(name)

        # Clear module from cache to force reimport
        if name in _SKILL_REGISTRY:
            module_path = _SKILL_REGISTRY[name][0]
            import sys

            if module_path in sys.modules:
                del sys.modules[module_path]

        return self.load_skill(name, force=True)

    def register_all_skills(self, router: ActionRouter, llm: Any = None) -> dict[str, bool]:
        """
        Attempt to load and register all skills with the ActionRouter.

        Args:
            router: The ActionRouter instance to register skill actions with.
            llm: The LLM instance to inject into skills that need it.

        Returns:
            Dict mapping skill name to registration success (True/False).
        """
        if llm is not None:
            self._llm = llm

        results: dict[str, bool] = {}

        for name in _SKILL_REGISTRY:
            skill = self.load_skill(name)
            if skill is None:
                results[name] = False
                continue

            try:
                # Check if skill has register method
                if hasattr(skill, "register") and callable(skill.register):
                    skill.register(router)
                else:
                    self._logger.debug("[Skills] '%s' loaded but has no register method.", name)

                results[name] = True
                self._logger.info("[Skills] '%s' registered successfully.", name)

            except Exception as exc:
                self._logger.error("[Skills] Failed to register '%s': %s", name, exc)
                results[name] = False

        registered = sum(results.values())
        total = len(_SKILL_REGISTRY)
        self._logger.info("[Skills] %d/%d skills registered.", registered, total)

        return results

    def is_skill_available(self, name: str) -> bool:
        """Check if a skill is loaded and ready."""
        name = name.lower()
        return (
            name in self._loaded_skills
            and self._skill_info.get(name, {}).status == SkillStatus.READY
        )

    def get_available_skills(self) -> list[str]:
        """Return names of all loaded and ready skills."""
        return [name for name, info in self._skill_info.items() if info.status == SkillStatus.READY]

    def get_disabled_skills(self) -> list[str]:
        """Return names of all disabled skills (missing dependencies)."""
        return [
            name for name, info in self._skill_info.items() if info.status == SkillStatus.DISABLED
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get skill system statistics."""
        total = len(_SKILL_REGISTRY)
        loaded = len(self._loaded_skills)
        ready = len(self.get_available_skills())
        disabled = len(self.get_disabled_skills())
        errors = total - loaded - disabled

        return {
            "total_skills": total,
            "loaded": loaded,
            "ready": ready,
            "disabled": disabled,
            "errors": errors,
            "core_skills": len(_CORE_SKILLS),
            "extended_skills": len(_EXTENDED_SKILLS),
        }


# ── Global Skill Manager Singleton ────────────────────────────────────────────

import threading  # noqa: E402

_skill_manager: SkillManager | None = None


def get_skill_manager() -> SkillManager:
    """Get or create the global skill manager singleton."""
    global _skill_manager
    if _skill_manager is None:
        _skill_manager = SkillManager()
    return _skill_manager


# ── Legacy Function Wrappers (Backward Compatibility) ─────────────────────────


def load_skill(name: str) -> Any | None:
    """Legacy wrapper for loading a skill."""
    return get_skill_manager().load_skill(name)


def register_all_skills(router: ActionRouter, llm: Any = None) -> dict[str, bool]:
    """Legacy wrapper for registering all skills."""
    return get_skill_manager().register_all_skills(router, llm)


def list_skills() -> list[str]:
    """Legacy wrapper for listing all skill names."""
    return list(_SKILL_REGISTRY.keys())


def list_available_skills() -> list[str]:
    """List only skills that are loaded and ready."""
    return get_skill_manager().get_available_skills()


def get_skill_status(name: str) -> dict[str, Any] | None:
    """Get the status of a specific skill."""
    info = get_skill_manager().get_skill_info(name)
    return info.to_dict() if info else None


def get_all_skill_status() -> list[dict[str, Any]]:
    """Get status for all skills."""
    return [info.to_dict() for info in get_skill_manager().get_all_skill_info()]


# ── Module Exports ────────────────────────────────────────────────────────────

__all__ = [
    "SkillInfo",
    # Manager
    "SkillManager",
    # Enums & Types
    "SkillStatus",
    "get_all_skill_status",
    "get_skill_manager",
    # Status Functions
    "get_skill_status",
    "list_available_skills",
    "list_skills",
    # Core Functions
    "load_skill",
    "register_all_skills",
]

# ── Test Suite ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Test Skill Registry functionality."""
    print("=" * 60)
    print("F.R.I.D.A.Y. Skill Registry Test Suite")
    print("=" * 60)

    mgr = get_skill_manager()

    # ── Test 1: List All Skills ───────────────────────────────────────────────
    print("\n[TEST 1] List All Registered Skills...")
    all_skills = list_skills()
    print(f"   Total Skills: {len(all_skills)}")
    for skill in all_skills:
        print(f"   - {skill}")

    # ── Test 2: Get Skill Statistics ──────────────────────────────────────────
    print("\n[TEST 2] Skill System Statistics...")
    stats = mgr.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # ── Test 3: Load Core Skill ───────────────────────────────────────────────
    print("\n[TEST 3] Load Core Skill (system)...")
    skill = mgr.load_skill("system")
    print(f"   Loaded: {skill is not None}")
    status = mgr.get_skill_info("system")
    if status:
        print(f"   Status: {status.status.value}")
        print(f"   Version: {status.version}")

    # ── Test 4: Load Extended Skill ───────────────────────────────────────────
    print("\n[TEST 4] Load Extended Skill (weather)...")
    skill = mgr.load_skill("weather")
    print(f"   Loaded: {skill is not None}")
    status = mgr.get_skill_info("weather")
    if status:
        print(f"   Status: {status.status.value}")
        if status.error_message:
            print(f"   Error: {status.error_message}")

    # ── Test 5: Get Available Skills ──────────────────────────────────────────
    print("\n[TEST 5] Get Available Skills...")
    available = mgr.get_available_skills()
    print(f"   Available: {len(available)}")
    for skill in available:
        print(f"   - {skill}")

    # ── Test 6: Get Disabled Skills ───────────────────────────────────────────
    print("\n[TEST 6] Get Disabled Skills...")
    disabled = mgr.get_disabled_skills()
    print(f"   Disabled: {len(disabled)}")
    for skill in disabled:
        info = mgr.get_skill_info(skill)
        print(f"   - {skill}: {info.error_message if info else 'Unknown'}")

    # ── Test 7: Reload Skill ──────────────────────────────────────────────────
    print("\n[TEST 7] Reload Skill (system)...")
    skill = mgr.reload_skill("system")
    print(f"   Reloaded: {skill is not None}")

    # ── Test 8: All Skill Status ──────────────────────────────────────────────
    print("\n[TEST 8] All Skill Status...")
    all_status = get_all_skill_status()
    for status in all_status[:5]:  # Show first 5
        print(f"   {status['name']}: {status['status']} ({status['version']})")

    # ── Test 9: Check Specific Skill ──────────────────────────────────────────
    print("\n[TEST 9] Check Specific Skill Status...")
    status = get_skill_status("productivity")
    if status:
        print(f"   Name: {status['name']}")
        print(f"   Status: {status['status']}")
        print(f"   Dependencies: {status['dependencies']}")

    # ── Test 10: Skill Availability Check ─────────────────────────────────────
    print("\n[TEST 10] Skill Availability Check...")
    for skill_name in ["system", "memory", "vision", "iot"]:
        available = mgr.is_skill_available(skill_name)
        print(f"   {skill_name}: {'✅' if available else '❌'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Skill Registry tests completed!")
    print("=" * 60)
