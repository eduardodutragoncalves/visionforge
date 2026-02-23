from __future__ import annotations

import argparse
import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from engine.db.connection import connect, get_db_path, transaction

logger = logging.getLogger("VisionForge.DBMigrator")

_MIGRATION_FILENAME_RE = r"^(\d{4})_(.+)\.sql$"


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path
    checksum: str


def _migrations_dir() -> Path:
    return Path(__file__).parent / "migrations"


def _compute_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover_migrations(migrations_dir: Path | None = None) -> List[Migration]:
    import re

    root = migrations_dir or _migrations_dir()
    if not root.exists():
        return []

    parsed: List[Migration] = []
    seen_versions = set()
    for path in sorted(root.glob("*.sql")):
        match = re.match(_MIGRATION_FILENAME_RE, path.name)
        if not match:
            continue
        version = int(match.group(1))
        name = match.group(2)
        if version in seen_versions:
            raise ValueError(f"Duplicate migration version: {version}")
        seen_versions.add(version)
        parsed.append(
            Migration(
                version=version,
                name=name,
                path=path,
                checksum=_compute_checksum(path),
            )
        )

    parsed.sort(key=lambda item: item.version)
    return parsed


def ensure_schema_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            applied_at TEXT NOT NULL,
            checksum TEXT
        )
        """
    )


def _fetch_applied_migrations(conn: sqlite3.Connection) -> Dict[int, str]:
    ensure_schema_migrations_table(conn)
    rows = conn.execute(
        """
        SELECT version, checksum
        FROM schema_migrations
        ORDER BY version ASC
        """
    ).fetchall()
    return {int(row["version"]): str(row["checksum"] or "") for row in rows}


def _split_sql_statements(sql_script: str) -> List[str]:
    statements: List[str] = []
    buffer = ""
    for line in sql_script.splitlines():
        striped = line.strip()
        if not striped:
            continue
        buffer += line + "\n"
        if sqlite3.complete_statement(buffer):
            statement = buffer.strip()
            if statement:
                statements.append(statement)
            buffer = ""

    tail = buffer.strip()
    if tail:
        raise ValueError("Migration contains an incomplete SQL statement.")
    return statements


def _apply_single_migration(conn: sqlite3.Connection, migration: Migration) -> None:
    sql_script = migration.path.read_text(encoding="utf-8")
    statements = _split_sql_statements(sql_script)
    if not statements:
        raise ValueError(f"Migration {migration.path.name} has no executable statements.")

    applied_at = datetime.utcnow().isoformat()
    with transaction(conn):
        for statement in statements:
            conn.execute(statement)
        conn.execute(
            """
            INSERT INTO schema_migrations (version, name, applied_at, checksum)
            VALUES (?, ?, ?, ?)
            """,
            (migration.version, migration.name, applied_at, migration.checksum),
        )


def apply_pending_migrations(conn: sqlite3.Connection, migrations_dir: Path | None = None) -> List[Migration]:
    all_migrations = discover_migrations(migrations_dir)
    applied = _fetch_applied_migrations(conn)
    pending: List[Migration] = []

    for migration in all_migrations:
        current_checksum = applied.get(migration.version)
        if current_checksum is None:
            pending.append(migration)
            continue
        if current_checksum and current_checksum != migration.checksum:
            raise ValueError(
                f"Migration checksum changed for version {migration.version}: "
                f"{migration.path.name}"
            )

    for migration in pending:
        logger.info("Applying migration %s (%s)", migration.version, migration.path.name)
        _apply_single_migration(conn, migration)
    return pending


def migrate(db_path: str | Path | None = None, migrations_dir: Path | None = None) -> List[Migration]:
    with connect(db_path=db_path) as conn:
        return apply_pending_migrations(conn=conn, migrations_dir=migrations_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply VisionForge SQLite migrations.")
    parser.add_argument("--db-path", type=str, default=None, help="Optional SQLite path override.")
    parser.add_argument(
        "--migrations-dir",
        type=str,
        default=None,
        help="Optional migrations directory override.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List migration status instead of applying pending migrations.",
    )
    return parser.parse_args()


def _list_status(db_path: str | Path | None = None, migrations_dir: Path | None = None) -> int:
    migrations = discover_migrations(Path(migrations_dir) if migrations_dir else None)
    with connect(db_path=db_path) as conn:
        applied = _fetch_applied_migrations(conn)
    for migration in migrations:
        marker = "x" if migration.version in applied else " "
        print(f"[{marker}] {migration.version:04d} {migration.path.name}")
    print(f"DB: {get_db_path() if db_path is None else Path(db_path).resolve()}")
    return 0


def main() -> int:
    args = _parse_args()
    db_path = args.db_path
    migrations_dir = Path(args.migrations_dir).resolve() if args.migrations_dir else None

    if args.list:
        return _list_status(db_path=db_path, migrations_dir=migrations_dir)

    pending = migrate(db_path=db_path, migrations_dir=migrations_dir)
    if pending:
        print(f"Applied {len(pending)} migration(s).")
    else:
        print("No pending migrations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
