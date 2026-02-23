from __future__ import annotations

import logging
import os

StorageMode = str
_VALID_STORAGE_MODES = {"json", "dual", "sqlite"}
_DEFAULT_STORAGE_MODE = "sqlite"

logger = logging.getLogger("VisionForge.StorageMode")


def get_storage_mode(default: StorageMode = _DEFAULT_STORAGE_MODE) -> StorageMode:
    raw_value = str(os.environ.get("VISIONFORGE_STORAGE_MODE") or "").strip().lower()
    if not raw_value:
        return default
    if raw_value not in _VALID_STORAGE_MODES:
        logger.warning(
            "Invalid VISIONFORGE_STORAGE_MODE=%s. Falling back to %s.",
            raw_value,
            default,
        )
        return default
    return raw_value


def is_json_primary() -> bool:
    mode = get_storage_mode()
    return mode in {"json", "dual"}


def is_sqlite_primary() -> bool:
    return get_storage_mode() == "sqlite"


def is_dual_write_enabled() -> bool:
    return get_storage_mode() == "dual"
