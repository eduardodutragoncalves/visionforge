from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from engine.db.connection import connect, get_db_path
from engine.db.migrator import Migration, apply_pending_migrations


@dataclass(frozen=True)
class DatabaseBootstrapResult:
    db_path: Path
    applied: List[Migration]


def bootstrap_database() -> DatabaseBootstrapResult:
    db_path = get_db_path()
    with connect(db_path=db_path) as conn:
        applied = apply_pending_migrations(conn)
    return DatabaseBootstrapResult(db_path=db_path, applied=applied)
