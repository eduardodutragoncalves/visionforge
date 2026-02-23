import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import jsonschema

from engine.db.connection import connect
from engine.io import paths

logger = logging.getLogger("VisionForge.IO")

SCHEMAS_DIR = Path(__file__).parent.parent.parent / "shared" / "schemas"


def _normalize_path(path: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = paths.get_workspace_root() / resolved
    return resolved.resolve()


def _json_doc_key(path: Path) -> str:
    return str(_normalize_path(path))


def _ensure_json_document_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS json_documents (
            path TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            schema_name TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_json_documents_updated_at
        ON json_documents(updated_at)
        """
    )
    conn.commit()


def _fetch_from_sqlite(path: Path) -> Optional[Dict[str, Any]]:
    doc_key = _json_doc_key(path)
    try:
        with connect() as conn:
            _ensure_json_document_table(conn)
            row = conn.execute(
                "SELECT payload_json FROM json_documents WHERE path = ?",
                (doc_key,),
            ).fetchone()
    except Exception as exc:
        logger.warning("SQLite manifest read failed for %s: %s", doc_key, exc)
        return None

    if not row:
        return None
    payload_json = str(row["payload_json"] or "").strip()
    if not payload_json:
        return None
    data = json.loads(payload_json)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest payload stored for {doc_key}: expected object JSON.")
    return data


def _upsert_to_sqlite(path: Path, data: Dict[str, Any], schema_name: Optional[str]) -> None:
    doc_key = _json_doc_key(path)
    payload_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    now = datetime.utcnow().isoformat()
    with connect() as conn:
        _ensure_json_document_table(conn)
        conn.execute(
            """
            INSERT INTO json_documents (path, payload_json, schema_name, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                payload_json = excluded.payload_json,
                schema_name = excluded.schema_name,
                updated_at = excluded.updated_at
            """,
            (doc_key, payload_json, schema_name, now),
        )
        conn.commit()


def _touch_marker(path: Path) -> None:
    normalized = _normalize_path(path)
    paths.ensure_dirs(normalized.parent)
    normalized.touch(exist_ok=True)


def load_schema(schema_name: str) -> Dict[str, Any]:
    schema_path = SCHEMAS_DIR / f"{schema_name}.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema {schema_name} not found at {schema_path}")
    with open(schema_path, "r") as f:
        return json.load(f)


def validate(data: Dict[str, Any], schema_name: str):
    schema = load_schema(schema_name)
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        logger.error(f"Validation failed for {schema_name}: {e.message}")
        raise ValueError(f"Invalid {schema_name}: {e.message}")


def manifest_exists(path: Path) -> bool:
    normalized = _normalize_path(path)
    if normalized.exists():
        return True

    data = _fetch_from_sqlite(normalized)
    if data is None:
        return False

    # Keep filesystem semantics for callers that enumerate/guard by path existence.
    if not normalized.parent.exists():
        return False
    _touch_marker(normalized)
    return True


def save_manifest(path: Path, data: Dict[str, Any], schema_name: Optional[str] = None):
    if schema_name:
        validate(data, schema_name)
    normalized = _normalize_path(path)
    _upsert_to_sqlite(normalized, data, schema_name)
    _touch_marker(normalized)


def load_manifest(path: Path) -> Dict[str, Any]:
    normalized = _normalize_path(path)
    from_sqlite = _fetch_from_sqlite(normalized)
    if from_sqlite is not None:
        if normalized.parent.exists() and not normalized.exists():
            _touch_marker(normalized)
        return from_sqlite

    if not normalized.exists():
        raise FileNotFoundError(f"Manifest not found: {normalized}")

    with open(normalized, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest file {normalized}: expected object JSON.")

    # Lazy migration path: first legacy read imports into SQLite.
    _upsert_to_sqlite(normalized, data, schema_name=None)
    return data


def update_manifest(path: Path, updates: Dict[str, Any], schema_name: Optional[str] = None) -> Dict[str, Any]:
    data = load_manifest(path)
    data.update(updates)
    save_manifest(path, data, schema_name)
    return data


def delete_manifest(path: Path) -> None:
    normalized = _normalize_path(path)
    doc_key = _json_doc_key(normalized)
    with connect() as conn:
        _ensure_json_document_table(conn)
        conn.execute("DELETE FROM json_documents WHERE path = ?", (doc_key,))
        conn.commit()
    if normalized.exists():
        normalized.unlink()


def delete_manifests_under(root: Path) -> None:
    normalized = _normalize_path(root)
    root_key = str(normalized)
    like_key = f"{root_key}/%"
    with connect() as conn:
        _ensure_json_document_table(conn)
        conn.execute(
            "DELETE FROM json_documents WHERE path = ? OR path LIKE ?",
            (root_key, like_key),
        )
        conn.commit()
