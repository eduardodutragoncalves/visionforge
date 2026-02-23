from __future__ import annotations

import contextlib
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Iterator

from engine.io import paths

DEFAULT_DB_FILENAME = "visionforge.db"
DEFAULT_TIMEOUT_SECONDS = 10.0


def resolve_db_path(db_path: str | Path | None = None) -> Path:
    if db_path is None:
        env_value = str(os.environ.get("VISIONFORGE_DB_PATH") or "").strip()
        if env_value:
            resolved = Path(env_value).expanduser()
        else:
            resolved = paths.get_workspace_root() / DEFAULT_DB_FILENAME
    else:
        resolved = Path(db_path).expanduser()

    if not resolved.is_absolute():
        resolved = paths.get_workspace_root() / resolved
    return resolved.resolve()


def get_db_path() -> Path:
    return resolve_db_path()


def configure_connection(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")


def connect(db_path: str | Path | None = None, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> sqlite3.Connection:
    resolved = resolve_db_path(db_path)
    paths.ensure_dirs(resolved.parent)
    conn = sqlite3.connect(str(resolved), timeout=timeout, check_same_thread=False)
    configure_connection(conn)
    return conn


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    savepoint_name: str | None = None
    if conn.in_transaction:
        savepoint_name = f"sp_{uuid.uuid4().hex[:12]}"
        conn.execute(f"SAVEPOINT {savepoint_name}")
    else:
        conn.execute("BEGIN")

    try:
        yield
    except Exception:
        if savepoint_name:
            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
        else:
            conn.rollback()
        raise
    else:
        if savepoint_name:
            conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
        else:
            conn.commit()
