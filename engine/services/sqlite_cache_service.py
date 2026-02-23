import hashlib
import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

from engine.io import paths

logger = logging.getLogger("VisionForge.SQLiteCache")

_DB_PATH = paths.get_workspace_root() / "visionforge.db"
_SCHEMA_READY = False
_SCHEMA_LOCK = threading.Lock()


class SQLiteCacheService:
    def _connect(self) -> sqlite3.Connection:
        paths.ensure_dirs(_DB_PATH.parent)
        conn = sqlite3.connect(str(_DB_PATH), timeout=5.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema(conn)
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        global _SCHEMA_READY
        if _SCHEMA_READY:
            return
        with _SCHEMA_LOCK:
            if _SCHEMA_READY:
                return
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS endpoint_cache (
                    cache_key TEXT PRIMARY KEY,
                    signature TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_endpoint_cache_updated_at
                ON endpoint_cache(updated_at)
                """
            )
            conn.commit()
            _SCHEMA_READY = True

    def _safe_stat(self, file_path: Path) -> Tuple[int, int]:
        try:
            stats = file_path.stat()
            return int(stats.st_mtime_ns), int(stats.st_size)
        except Exception:
            return 0, 0

    def _hash_lines(self, lines: Iterable[str]) -> str:
        digest = hashlib.sha1()
        for line in lines:
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def _files_signature(self, file_paths: Iterable[Path]) -> str:
        count = 0
        max_mtime = 0
        sum_mtime = 0
        total_size = 0
        for file_path in sorted(file_paths, key=lambda item: str(item)):
            mtime_ns, size = self._safe_stat(file_path)
            if mtime_ns <= 0 and size <= 0:
                continue
            count += 1
            max_mtime = max(max_mtime, mtime_ns)
            sum_mtime += mtime_ns
            total_size += size
        return f"{count}:{max_mtime}:{sum_mtime}:{total_size}"

    def _run_json_files(self, project_id: str) -> List[Path]:
        runs_dir = paths.get_runs_dir(project_id)
        files: List[Path] = []
        if not runs_dir.exists():
            return files
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            run_json = run_dir / "run.json"
            files.append(run_json)
        return files

    def _variant_manifest_files(self, project_id: str) -> List[Path]:
        variants_root = paths.get_variants_dir(project_id)
        files: List[Path] = []
        if not variants_root.exists():
            return files
        for variant_dir in variants_root.iterdir():
            if not variant_dir.is_dir():
                continue
            files.append(variant_dir / "variant.json")
        return files

    def _variant_onnx_files(self, project_id: str) -> List[Path]:
        variants_root = paths.get_variants_dir(project_id)
        files: List[Path] = []
        if not variants_root.exists():
            return files
        for variant_dir in variants_root.iterdir():
            if not variant_dir.is_dir():
                continue
            files.append(variant_dir / "model.onnx")
            files.append(variant_dir / "exports" / "model.onnx")
        return files

    def _deployment_manifest_files(self, project_id: str) -> List[Path]:
        variants_root = paths.get_variants_dir(project_id)
        files: List[Path] = []
        if not variants_root.exists():
            return files
        for variant_dir in variants_root.iterdir():
            if not variant_dir.is_dir():
                continue
            deploy_root = variant_dir / "deploy"
            if not deploy_root.exists():
                continue
            for target_dir in deploy_root.iterdir():
                if not target_dir.is_dir():
                    continue
                for package_dir in target_dir.iterdir():
                    if not package_dir.is_dir():
                        continue
                    files.append(package_dir / "manifest.json")
        return files

    def projects_signature(self) -> str:
        projects_root = paths.get_projects_root()
        if not projects_root.exists():
            return self._hash_lines(["projects:none"])
        lines: List[str] = []
        for project_dir in sorted(projects_root.iterdir(), key=lambda item: item.name):
            if not project_dir.is_dir():
                continue
            project_id = project_dir.name
            project_manifest_sig = self._files_signature([project_dir / "project.json"])
            dataset_manifest_sig = self._files_signature([project_dir / "dataset.json"])
            runs_sig = self._files_signature(self._run_json_files(project_id))
            lines.append(
                f"{project_id}|project:{project_manifest_sig}|dataset:{dataset_manifest_sig}|runs:{runs_sig}"
            )
        if not lines:
            return self._hash_lines(["projects:empty"])
        return self._hash_lines(lines)

    def runs_signature(self, project_id: str) -> str:
        lines = [
            f"project:{project_id}",
            f"runs:{self._files_signature(self._run_json_files(project_id))}",
        ]
        return self._hash_lines(lines)

    def variants_signature(self, project_id: str) -> str:
        history_sig = self._files_signature([paths.get_project_inference_history_path(project_id)])
        variant_sig = self._files_signature(self._variant_manifest_files(project_id))
        deployment_sig = self._files_signature(self._deployment_manifest_files(project_id))
        onnx_sig = self._files_signature(self._variant_onnx_files(project_id))
        lines = [
            f"project:{project_id}",
            f"history:{history_sig}",
            f"variants:{variant_sig}",
            f"deployments:{deployment_sig}",
            f"onnx:{onnx_sig}",
        ]
        return self._hash_lines(lines)

    def get_payload(self, cache_key: str, signature: str) -> Optional[Any]:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT payload_json
                    FROM endpoint_cache
                    WHERE cache_key = ? AND signature = ?
                    """,
                    (cache_key, signature),
                ).fetchone()
            if not row:
                return None
            payload_json = str(row["payload_json"] or "")
            if not payload_json:
                return None
            return json.loads(payload_json)
        except Exception as exc:
            logger.warning("Cache read failed for %s: %s", cache_key, exc)
            return None

    def set_payload(self, cache_key: str, signature: str, payload: Any) -> None:
        try:
            payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            updated_at = datetime.utcnow().isoformat()
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO endpoint_cache (cache_key, signature, payload_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        signature = excluded.signature,
                        payload_json = excluded.payload_json,
                        updated_at = excluded.updated_at
                    """,
                    (cache_key, signature, payload_json, updated_at),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("Cache write failed for %s: %s", cache_key, exc)


sqlite_cache_service = SQLiteCacheService()
