from __future__ import annotations

from engine.db.connection import connect, get_db_path, resolve_db_path
from engine.db.storage_mode import (
    StorageMode,
    get_storage_mode,
    is_dual_write_enabled,
    is_json_primary,
    is_sqlite_primary,
)


def bootstrap_database():
    from engine.db.bootstrap import bootstrap_database as _bootstrap_database

    return _bootstrap_database()


__all__ = [
    "StorageMode",
    "bootstrap_database",
    "connect",
    "get_db_path",
    "get_storage_mode",
    "is_dual_write_enabled",
    "is_json_primary",
    "is_sqlite_primary",
    "resolve_db_path",
]
