from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterator

from engine.db.connection import connect, resolve_db_path


class BaseRepository:
    def __init__(self, db_path: str | Path | None = None):
        self._db_path = resolve_db_path(db_path)

    @contextlib.contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        with connect(self._db_path) as conn:
            yield conn

    @staticmethod
    def row_to_dict(row: sqlite3.Row | None) -> Dict[str, Any] | None:
        if row is None:
            return None
        return {key: row[key] for key in row.keys()}
