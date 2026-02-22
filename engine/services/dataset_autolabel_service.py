import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.core.events import manager
from engine.io import manifests, paths
from engine.services.dataset_service import dataset_service
from engine.services.inference_service import inference_service

logger = logging.getLogger("VisionForge.DatasetAutoLabel")


def _now() -> str:
    return datetime.utcnow().isoformat()


class DatasetAutoLabelService:
    def __init__(self):
        self._manifest_lock = asyncio.Lock()

    def _manifest_path(self, project_id: str) -> Path:
        return paths.get_dataset_autolabel_manifest_path(project_id)

    def _default_manifest(self, project_id: str) -> Dict[str, Any]:
        return {
            "project_id": project_id,
            "updated_at": _now(),
            "classification": {
                "by_variant": {}
            }
        }

    async def _load_manifest(self, project_id: str) -> Dict[str, Any]:
        path = self._manifest_path(project_id)
        if not path.exists():
            return self._default_manifest(project_id)
        try:
            data = manifests.load_manifest(path)
        except Exception:
            logger.exception("Failed to load dataset auto-label manifest. Rebuilding default.")
            return self._default_manifest(project_id)
        if not isinstance(data, dict):
            return self._default_manifest(project_id)
        data.setdefault("project_id", project_id)
        data.setdefault("updated_at", _now())
        classification = data.get("classification")
        if not isinstance(classification, dict):
            classification = {}
            data["classification"] = classification
        by_variant = classification.get("by_variant")
        if not isinstance(by_variant, dict):
            classification["by_variant"] = {}
        return data

    async def _save_manifest(self, project_id: str, manifest: Dict[str, Any]):
        manifest["project_id"] = project_id
        manifest["updated_at"] = _now()
        manifests.save_manifest(self._manifest_path(project_id), manifest)

    def _is_classification_variant(self, variant: Dict[str, Any]) -> bool:
        base_model = str(variant.get("base_model") or "")
        hyper_base = str((variant.get("hyperparameters") or {}).get("base_model") or "")
        name = str(variant.get("name") or "")
        text = f"{base_model} {hyper_base} {name}".lower()
        if "-cls" in text or "cls.pt" in text or "classification" in text:
            return True
        metrics = variant.get("metrics") or {}
        try:
            accuracy = float(metrics.get("accuracy", 0) or 0)
            final_map = float(metrics.get("final_map", 0) or 0)
            if accuracy > 0 and abs(final_map) < 1e-9:
                return True
        except Exception:
            return False
        return False

    def _load_variant_manifest(self, project_id: str, variant_id: str) -> Dict[str, Any]:
        variant_dir = paths.get_variant_dir(project_id, variant_id)
        if not variant_dir.exists():
            raise ValueError(f"Variant {variant_id} not found in project {project_id}")
        manifest_path = variant_dir / "variant.json"
        if not manifest_path.exists():
            raise ValueError("Variant manifest not found")
        data = manifests.load_manifest(manifest_path)
        if not isinstance(data, dict):
            raise ValueError("Invalid variant manifest")
        return data

    def list_classification_models(self, project_id: str) -> List[Dict[str, Any]]:
        variants_root = paths.get_variants_dir(project_id)
        if not variants_root.exists():
            return []
        models: List[Dict[str, Any]] = []
        for variant_dir in variants_root.iterdir():
            if not variant_dir.is_dir():
                continue
            model_path = variant_dir / "model.pt"
            manifest_path = variant_dir / "variant.json"
            if not model_path.exists() or not manifest_path.exists():
                continue
            try:
                variant = manifests.load_manifest(manifest_path)
            except Exception:
                continue
            if not isinstance(variant, dict):
                continue
            if not self._is_classification_variant(variant):
                continue
            models.append({
                "id": variant.get("id") or variant_dir.name,
                "name": variant.get("name") or variant_dir.name,
                "base_model": variant.get("base_model") or (variant.get("hyperparameters") or {}).get("base_model") or "",
                "run_id": variant.get("run_id"),
                "created_at": variant.get("created_at"),
            })
        models.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return models

    async def _broadcast_progress(
        self,
        project_id: str,
        variant_id: str,
        status: str,
        processed: int,
        total: int,
        message: str,
        **extra: Any
    ):
        payload: Dict[str, Any] = {
            "project_id": project_id,
            "variant_id": variant_id,
            "status": status,
            "processed": max(0, int(processed)),
            "total": max(0, int(total)),
            "progress": float((processed / total) * 100.0) if total > 0 else 0.0,
            "message": message,
        }
        if extra:
            payload.update(extra)
        await manager.broadcast("dataset/autolabel_progress", payload)

    async def run_classification_suggestions(
        self,
        project_id: str,
        variant_id: str,
        image_ids: Optional[List[str]] = None,
        only_unlabeled: bool = True,
        min_confidence: float = 0.0,
    ) -> Dict[str, Any]:
        dataset = dataset_service.get_dataset(project_id)
        if str(dataset.get("task") or "classification") != "classification":
            raise ValueError("Auto-classification is available only for classification datasets")

        variant_manifest = self._load_variant_manifest(project_id, variant_id)
        if not self._is_classification_variant(variant_manifest):
            raise ValueError("Selected model is not a classification variant")

        variant_dir = paths.get_variant_dir(project_id, variant_id)
        model_path = variant_dir / "model.pt"
        if not model_path.exists():
            raise ValueError("Model weights not found for selected variant")

        images = dataset.get("images") or []
        by_id = {
            str(image.get("id") or ""): image
            for image in images
            if str(image.get("id") or "")
        }

        if image_ids:
            ordered_ids = [str(image_id) for image_id in image_ids if str(image_id) in by_id]
        else:
            ordered_ids = list(by_id.keys())

        targets = []
        for image_id in ordered_ids:
            image = by_id[image_id]
            labels = image.get("labels") or []
            if only_unlabeled and len(labels) > 0:
                continue
            targets.append(image)

        total = len(targets)
        await self._broadcast_progress(
            project_id,
            variant_id,
            "starting",
            0,
            total,
            f"Preparing auto-classification for {total} image(s)...",
        )

        async with self._manifest_lock:
            manifest = await self._load_manifest(project_id)
            by_variant = (((manifest.get("classification") or {}).get("by_variant")) or {})
            if not isinstance(by_variant, dict):
                by_variant = {}
            variant_suggestions = by_variant.get(variant_id)
            if not isinstance(variant_suggestions, dict):
                variant_suggestions = {}

            suggested = 0
            removed = 0
            errors = 0
            confidence_threshold = max(0.0, min(1.0, float(min_confidence)))

            for index, image in enumerate(targets, start=1):
                image_id = str(image.get("id") or "")
                image_path = Path(str(image.get("path") or ""))
                if not image_id or not image_path.exists():
                    errors += 1
                    continue

                try:
                    image_bytes = await asyncio.to_thread(image_path.read_bytes)
                    infer_result = await asyncio.to_thread(
                        inference_service.run_inference,
                        variant_dir,
                        image_bytes,
                        0.0,
                        0.45,
                        False,
                    )
                    if str(infer_result.get("task") or "") != "classify":
                        raise ValueError("Selected variant did not return classification predictions")
                    predictions = infer_result.get("predictions") or []
                    top = None
                    for prediction in predictions:
                        confidence = float(prediction.get("confidence") or 0.0)
                        if confidence < confidence_threshold:
                            continue
                        top = prediction
                        break

                    if top is None:
                        if image_id in variant_suggestions:
                            del variant_suggestions[image_id]
                            removed += 1
                    else:
                        timestamp = _now()
                        variant_suggestions[image_id] = {
                            "image_id": image_id,
                            "variant_id": variant_id,
                            "run_id": variant_manifest.get("run_id"),
                            "predicted_class_id": int(top.get("class_id", -1)),
                            "predicted_class_name": str(top.get("class_name") or ""),
                            "confidence": float(top.get("confidence") or 0.0),
                            "status": "pending",
                            "created_at": timestamp,
                            "updated_at": timestamp,
                        }
                        suggested += 1
                except Exception:
                    errors += 1
                    logger.exception("Failed to auto-classify image %s for variant %s", image_id, variant_id)

                if index == 1 or index % 10 == 0 or index == total:
                    await self._broadcast_progress(
                        project_id,
                        variant_id,
                        "running",
                        index,
                        total,
                        f"Processed {index}/{total} image(s)...",
                        suggested=suggested,
                        removed=removed,
                        errors=errors,
                    )

            by_variant[variant_id] = variant_suggestions
            manifest.setdefault("classification", {})
            manifest["classification"]["by_variant"] = by_variant
            await self._save_manifest(project_id, manifest)

        await self._broadcast_progress(
            project_id,
            variant_id,
            "completed",
            total,
            total,
            "Auto-classification completed.",
            suggested=suggested,
            removed=removed,
            errors=errors,
        )

        return {
            "project_id": project_id,
            "variant_id": variant_id,
            "processed": total,
            "suggested": suggested,
            "removed": removed,
            "errors": errors,
            "min_confidence": max(0.0, min(1.0, float(min_confidence))),
            "status": "completed",
        }

    async def get_classification_suggestions_map(
        self,
        project_id: str,
        variant_id: str,
        status: Optional[str] = "pending",
        only_unlabeled: bool = True,
    ) -> Dict[str, Any]:
        async with self._manifest_lock:
            manifest = await self._load_manifest(project_id)
            by_variant = (((manifest.get("classification") or {}).get("by_variant")) or {})
            if not isinstance(by_variant, dict):
                return {}
            raw = by_variant.get(variant_id) or {}
            if not isinstance(raw, dict):
                return {}

            dataset = dataset_service.get_dataset(project_id)
            images = dataset.get("images") or []
            image_by_id = {
                str(image.get("id") or ""): image
                for image in images
                if str(image.get("id") or "")
            }

            filtered: Dict[str, Any] = {}
            for image_id, item in raw.items():
                image = image_by_id.get(str(image_id))
                if not image:
                    continue
                if status and str(item.get("status") or "") != status:
                    continue
                if only_unlabeled and len(image.get("labels") or []) > 0:
                    continue
                filtered[str(image_id)] = item
            return filtered

    async def apply_classification_suggestions(
        self,
        project_id: str,
        variant_id: str,
        image_ids: List[str],
        action: str,
    ) -> Dict[str, Any]:
        valid_actions = {"accept", "reject", "clear"}
        if action not in valid_actions:
            raise ValueError(f"Unsupported action: {action}")

        async with self._manifest_lock:
            manifest = await self._load_manifest(project_id)
            by_variant = (((manifest.get("classification") or {}).get("by_variant")) or {})
            if not isinstance(by_variant, dict):
                by_variant = {}
            variant_items = by_variant.get(variant_id)
            if not isinstance(variant_items, dict):
                variant_items = {}

            target_ids = []
            seen = set()
            for image_id in image_ids:
                normalized = str(image_id or "").strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                target_ids.append(normalized)

            assignments: Dict[str, int] = {}
            changed = 0
            missing = 0

            if action == "accept":
                for image_id in target_ids:
                    item = variant_items.get(image_id)
                    if not item:
                        missing += 1
                        continue
                    class_id = item.get("predicted_class_id")
                    try:
                        assignments[image_id] = int(class_id)
                    except Exception:
                        missing += 1
                        continue
                apply_result = dataset_service.apply_classification_assignments(project_id, assignments)
                accepted_ids = set(apply_result.get("updated_ids") or [])
                for image_id in accepted_ids:
                    item = variant_items.get(image_id)
                    if not item:
                        continue
                    item["status"] = "accepted"
                    item["accepted_at"] = _now()
                    item["updated_at"] = _now()
                    changed += 1
                missing += int(apply_result.get("missing", 0)) + int(apply_result.get("invalid_class", 0))
            elif action == "reject":
                for image_id in target_ids:
                    item = variant_items.get(image_id)
                    if not item:
                        missing += 1
                        continue
                    item["status"] = "rejected"
                    item["updated_at"] = _now()
                    changed += 1
            else:
                for image_id in target_ids:
                    if image_id not in variant_items:
                        missing += 1
                        continue
                    del variant_items[image_id]
                    changed += 1

            by_variant[variant_id] = variant_items
            manifest.setdefault("classification", {})
            manifest["classification"]["by_variant"] = by_variant
            await self._save_manifest(project_id, manifest)

        return {
            "project_id": project_id,
            "variant_id": variant_id,
            "action": action,
            "changed": changed,
            "missing": missing,
            "target_count": len(target_ids),
        }


dataset_autolabel_service = DatasetAutoLabelService()

