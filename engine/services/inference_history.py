import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from engine.io import paths, manifests

logger = logging.getLogger("VisionForge.InferenceHistory")


class InferenceHistoryService:
    def _history_path(self, project_id: str) -> Path:
        return paths.get_project_inference_history_path(project_id)

    def _images_dir(self, project_id: str) -> Path:
        return paths.get_project_inference_images_dir(project_id)

    def _load_history(self, project_id: str) -> Dict[str, Any]:
        history_path = self._history_path(project_id)
        if history_path.exists():
            try:
                return manifests.load_manifest(history_path)
            except Exception as exc:
                logger.warning("Failed to load history: %s", exc)
        return {"items": []}

    def _save_history(self, project_id: str, history: Dict[str, Any]) -> None:
        manifests.save_manifest(self._history_path(project_id), history)

    def list_history(self, project_id: str) -> List[Dict[str, Any]]:
        history = self._load_history(project_id)
        items = history.get("items", [])
        return sorted(items, key=lambda item: item.get("last_tested_at", ""), reverse=True)

    def _hash_image(self, image_bytes: bytes) -> str:
        return hashlib.sha1(image_bytes).hexdigest()[:12]

    def record_inference(
        self,
        project_id: str,
        variant_id: str,
        filename: str,
        image_bytes: bytes,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        image_id = self._hash_image(image_bytes)
        ext = Path(filename).suffix.lower() if filename else ".jpg"
        if not ext:
            ext = ".jpg"

        images_dir = self._images_dir(project_id)
        paths.ensure_dirs(images_dir)
        image_file = f"{image_id}{ext}"
        image_path = images_dir / image_file
        if not image_path.exists():
            image_path.write_bytes(image_bytes)

        history = self._load_history(project_id)
        items = history.setdefault("items", [])
        now = datetime.utcnow().isoformat()

        entry = next((item for item in items if item.get("image_id") == image_id), None)
        if not entry:
            entry = {
                "image_id": image_id,
                "image_file": image_file,
                "original_name": filename,
                "created_at": now,
                "last_tested_at": now,
                "variants": {}
            }
            items.append(entry)

        entry["last_tested_at"] = now
        variants = entry.setdefault("variants", {})
        variants[variant_id] = {
            "variant_id": variant_id,
            "tested_at": now,
            "task": result.get("task"),
            "predictions": result.get("predictions", []),
            "latency_ms": result.get("latency_ms"),
            "fps_est": result.get("fps_est"),
            "feedback": variants.get(variant_id, {}).get("feedback")
        }

        self._save_history(project_id, history)
        return entry

    def set_feedback(self, project_id: str, image_id: str, variant_id: str, verdict: str | None) -> Dict[str, Any]:
        history = self._load_history(project_id)
        items = history.get("items", [])
        entry = next((item for item in items if item.get("image_id") == image_id), None)
        if not entry:
            raise ValueError("History item not found")
        variants = entry.setdefault("variants", {})
        if variant_id not in variants:
            raise ValueError("Variant result not found")
        variants[variant_id]["feedback"] = verdict
        self._save_history(project_id, history)
        return variants[variant_id]

    def get_image_path(self, project_id: str, image_id: str) -> Path:
        history = self._load_history(project_id)
        items = history.get("items", [])
        entry = next((item for item in items if item.get("image_id") == image_id), None)
        if not entry:
            raise FileNotFoundError("History image not found")
        image_file = entry.get("image_file")
        if not image_file:
            raise FileNotFoundError("History image not found")
        image_path = self._images_dir(project_id) / image_file
        if not image_path.exists():
            raise FileNotFoundError("History image file missing")
        return image_path


inference_history_service = InferenceHistoryService()
