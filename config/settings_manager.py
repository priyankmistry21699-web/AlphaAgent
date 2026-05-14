"""
AlphaAgent — Settings Manager

Singleton that loads config/settings.yaml and exposes a simple get() API.
Supports hot-reload: call reload() or POST /api/v1/settings to update values
without restarting the server.

Usage anywhere in the codebase:
    from config.settings_manager import settings
    rsi_window = settings.get("technical.rsi_window", default=14)
    settings.set("technical.rsi_window", 21)
    settings.save()
"""

from __future__ import annotations

import copy
import logging
import os
import threading
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_SETTINGS_FILE = Path(__file__).parent / "settings.yaml"


class SettingsManager:
    """Thread-safe singleton settings loader with hot-reload support."""

    _instance: SettingsManager | None = None
    _lock = threading.Lock()

    def __new__(cls) -> SettingsManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._data: dict = {}
                    inst._rw_lock = threading.RLock()
                    inst._load()
                    cls._instance = inst
        return cls._instance

    # ── Load / Save ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            with self._rw_lock:
                self._data = raw
            logger.info("[Settings] Loaded %s", _SETTINGS_FILE)
        except FileNotFoundError:
            logger.warning("[Settings] settings.yaml not found — using empty defaults")
            with self._rw_lock:
                self._data = {}
        except yaml.YAMLError as exc:
            logger.error("[Settings] YAML parse error: %s", exc)

    def reload(self) -> None:
        """Re-read settings.yaml from disk (hot-reload, no restart needed)."""
        self._load()

    def save(self) -> None:
        """Persist current in-memory settings back to settings.yaml."""
        with self._rw_lock:
            snapshot = copy.deepcopy(self._data)
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
            yaml.dump(snapshot, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info("[Settings] Saved to %s", _SETTINGS_FILE)

    # ── Read ─────────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """
        Read a value using dot-notation key.

        Example:
            settings.get("technical.rsi_window")          → 14
            settings.get("fundamental.pb_undervalued")    → 1.5
            settings.get("missing.key", default=42)       → 42
        """
        with self._rw_lock:
            node = self._data
            for part in key.split("."):
                if not isinstance(node, dict) or part not in node:
                    return default
                node = node[part]
            return node

    def get_section(self, section: str) -> dict:
        """Return a full section as a dict copy."""
        with self._rw_lock:
            return copy.deepcopy(self._data.get(section, {}))

    def all(self) -> dict:
        """Return full settings dict (deep copy)."""
        with self._rw_lock:
            return copy.deepcopy(self._data)

    # ── Write ─────────────────────────────────────────────────────────────────

    def set(self, key: str, value: Any) -> None:
        """
        Set a value using dot-notation key. Creates intermediate dicts as needed.

        Example:
            settings.set("technical.rsi_window", 21)
            settings.set("portfolio.max_single_position_pct", 20.0)
        """
        parts = key.split(".")
        with self._rw_lock:
            node = self._data
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], dict):
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value

    def update_section(self, section: str, updates: dict) -> None:
        """Merge a dict of updates into a section."""
        with self._rw_lock:
            if section not in self._data or not isinstance(self._data[section], dict):
                self._data[section] = {}
            self._data[section].update(updates)

    def apply_patch(self, patch: dict) -> None:
        """
        Apply a nested patch dict. Merges recursively.

        Example patch:
            {"technical": {"rsi_window": 21}, "portfolio": {"max_single_position_pct": 20}}
        """
        with self._rw_lock:
            self._deep_merge(self._data, patch)

    @staticmethod
    def _deep_merge(base: dict, patch: dict) -> None:
        for k, v in patch.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                SettingsManager._deep_merge(base[k], v)
            else:
                base[k] = v


# ── Module-level singleton ────────────────────────────────────────────────────
settings = SettingsManager()
