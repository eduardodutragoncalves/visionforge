import hashlib
import json
import logging
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from engine.io import paths, manifests

logger = logging.getLogger("VisionForge.InferenceHistory")
LABELED_FEEDBACK = {"correct", "incorrect"}


class InferenceHistoryService:
    def _empty_validation_stats(self, total_images: int) -> Dict[str, Any]:
        return {
            "total_images": total_images,
            "tested_images": 0,
            "untested_images": total_images,
            "coverage_pct": 0.0,
            "correct": 0,
            "incorrect": 0,
            "pending": 0,
            "labeled": 0,
            "accuracy_pct": None,
            "last_tested_at": None
        }

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

    def summarize_validation_by_variant(self, project_id: str) -> Dict[str, Any]:
        history = self._load_history(project_id)
        items = history.get("items", [])
        total_images = len(items)

        summary: Dict[str, Dict[str, Any]] = {}
        for item in items:
            variants = item.get("variants") or {}
            for variant_id, variant_data in variants.items():
                if not variant_id:
                    continue
                stats = summary.setdefault(variant_id, self._empty_validation_stats(total_images))
                stats["tested_images"] += 1
                tested_at = variant_data.get("tested_at")
                if tested_at:
                    if not stats["last_tested_at"] or str(tested_at) > str(stats["last_tested_at"]):
                        stats["last_tested_at"] = tested_at

                feedback = variant_data.get("feedback")
                if feedback == "correct":
                    stats["correct"] += 1
                elif feedback == "incorrect":
                    stats["incorrect"] += 1
                else:
                    stats["pending"] += 1

        for stats in summary.values():
            stats["untested_images"] = max(0, total_images - stats["tested_images"])
            stats["labeled"] = stats["correct"] + stats["incorrect"]
            stats["coverage_pct"] = (stats["tested_images"] / total_images * 100.0) if total_images > 0 else 0.0
            if stats["labeled"] > 0:
                stats["accuracy_pct"] = stats["correct"] / stats["labeled"] * 100.0
            else:
                stats["accuracy_pct"] = None

        return {
            "total_images": total_images,
            "by_variant": summary
        }

    def _hash_image(self, image_bytes: bytes) -> str:
        return hashlib.sha1(image_bytes).hexdigest()[:12]

    def _normalize_class_key(self, value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        normalized = unicodedata.normalize("NFD", raw)
        normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
        normalized = normalized.lower()
        normalized = " ".join(normalized.split())
        normalized = "".join(ch for ch in normalized if ch.isalnum() or ch == " ")
        return normalized.strip()

    def _prediction_identity(self, task: str | None, predictions: List[Dict[str, Any]]) -> str:
        if not isinstance(predictions, list) or len(predictions) == 0:
            return "none"

        normalized_task = str(task or "").strip().lower()
        if normalized_task == "classify":
            top = predictions[0] if predictions else None
            if not top:
                return "none"
            normalized_label = self._normalize_class_key(top.get("class_name"))
            if normalized_label:
                return f"cls:{normalized_label}"
            class_id = top.get("class_id")
            return f"id:{class_id}" if class_id is not None else "none"

        ranked = sorted(predictions, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        top_detection = ranked[0] if ranked else None
        if not top_detection:
            return "none"
        normalized_label = self._normalize_class_key(top_detection.get("class_name"))
        if normalized_label:
            return f"det:{normalized_label}"
        class_id = top_detection.get("class_id")
        return f"id:{class_id}" if class_id is not None else "none"

    def _infer_reference_identity(self, entry: Dict[str, Any], exclude_variant_id: str = "") -> str | None:
        variants = entry.get("variants") or {}
        positive_votes: Dict[str, int] = {}

        for variant_id, variant_data in variants.items():
            if exclude_variant_id and variant_id == exclude_variant_id:
                continue
            feedback = variant_data.get("feedback")
            if feedback != "correct":
                continue
            identity = self._prediction_identity(variant_data.get("task"), variant_data.get("predictions") or [])
            if identity == "none":
                continue
            positive_votes[identity] = positive_votes.get(identity, 0) + 1

        if not positive_votes:
            return None
        return sorted(positive_votes.items(), key=lambda item: (-item[1], item[0]))[0][0]

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
        previous_feedback = variants.get(variant_id, {}).get("feedback")
        reference_identity = self._infer_reference_identity(entry, exclude_variant_id=variant_id)
        current_identity = self._prediction_identity(result.get("task"), result.get("predictions", []))
        auto_feedback = None
        if previous_feedback not in LABELED_FEEDBACK and reference_identity and current_identity != "none":
            auto_feedback = "correct" if current_identity == reference_identity else "incorrect"
        variants[variant_id] = {
            "variant_id": variant_id,
            "tested_at": now,
            "task": result.get("task"),
            "predictions": result.get("predictions", []),
            "latency_ms": result.get("latency_ms"),
            "fps_est": result.get("fps_est"),
            "feedback": previous_feedback if previous_feedback in LABELED_FEEDBACK else auto_feedback,
            "feedback_origin": "manual" if previous_feedback in LABELED_FEEDBACK else ("auto" if auto_feedback else None)
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
        variants[variant_id]["feedback_origin"] = "manual" if verdict in LABELED_FEEDBACK else None
        self._save_history(project_id, history)
        return variants[variant_id]

    def set_feedback_batch(self, project_id: str, image_id: str, updates: Dict[str, str | None]) -> Dict[str, Any]:
        history = self._load_history(project_id)
        items = history.get("items", [])
        entry = next((item for item in items if item.get("image_id") == image_id), None)
        if not entry:
            raise ValueError("History item not found")

        variants = entry.setdefault("variants", {})
        updated: Dict[str, Any] = {}
        for variant_id, verdict in updates.items():
            if variant_id not in variants:
                continue
            variants[variant_id]["feedback"] = verdict
            variants[variant_id]["feedback_origin"] = "manual" if verdict in LABELED_FEEDBACK else None
            updated[variant_id] = variants[variant_id]

        self._save_history(project_id, history)
        return {"image_id": image_id, "updated": updated}

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
