import logging
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

from engine.io import manifests, paths
from engine.services.dataset_service import dataset_service

logger = logging.getLogger("VisionForge.DatasetVersion")

RESOLUTION_BUCKETS: Dict[str, Tuple[int, int]] = {
    "hd": (1280, 720),
    "fhd": (1920, 1080),
    "qhd": (2560, 1440),
    "uhd": (3840, 2160),
}
VALID_ORIENTATION_FILTERS = {"any", "landscape", "portrait", "square"}
VALID_RESOLUTION_FILTERS = {"any", "hd", "fhd", "qhd", "uhd"}


def _now() -> str:
    return datetime.utcnow().isoformat()


class DatasetVersionService:
    def _manifest_path(self, project_id: str) -> Path:
        return paths.get_dataset_versions_manifest_path(project_id)

    def _default_manifest(self, project_id: str) -> Dict[str, Any]:
        return {
            "project_id": project_id,
            "updated_at": _now(),
            "versions": [],
        }

    def _load_manifest(self, project_id: str) -> Dict[str, Any]:
        manifest_path = self._manifest_path(project_id)
        if not manifest_path.exists():
            return self._default_manifest(project_id)
        try:
            data = manifests.load_manifest(manifest_path)
        except Exception:
            logger.exception("Failed to load dataset version manifest. Rebuilding default.")
            return self._default_manifest(project_id)
        if not isinstance(data, dict):
            return self._default_manifest(project_id)
        data.setdefault("project_id", project_id)
        data.setdefault("updated_at", _now())
        versions = data.get("versions")
        if not isinstance(versions, list):
            data["versions"] = []
        return data

    def _save_manifest(self, project_id: str, manifest: Dict[str, Any]):
        manifest["project_id"] = project_id
        manifest["updated_at"] = _now()
        manifests.save_manifest(self._manifest_path(project_id), manifest)

    def _orientation(self, width: int, height: int) -> str:
        if width > height:
            return "landscape"
        if height > width:
            return "portrait"
        return "square"

    def _resolution_bucket(self, width: int, height: int) -> str:
        if width >= 3840 and height >= 2160:
            return "uhd"
        if width >= 2560 and height >= 1440:
            return "qhd"
        if width >= 1920 and height >= 1080:
            return "fhd"
        if width >= 1280 and height >= 720:
            return "hd"
        return "sd"

    def _resolve_image_size(self, image: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        width = int(image.get("width") or 0)
        height = int(image.get("height") or 0)
        if width > 0 and height > 0:
            return width, height

        image_path = Path(str(image.get("path") or ""))
        if not image_path.exists():
            return None
        try:
            with Image.open(image_path) as img:
                size = img.size
        except Exception:
            logger.warning("Unable to read image dimensions for %s", image_path)
            return None
        if not size or len(size) != 2:
            return None
        width = int(size[0] or 0)
        height = int(size[1] or 0)
        if width <= 0 or height <= 0:
            return None
        return width, height

    def _summarize(self, version: Dict[str, Any], include_image_ids: bool = False) -> Dict[str, Any]:
        summary = dict(version)
        if not include_image_ids:
            summary.pop("image_ids", None)
        return summary

    def list_versions(self, project_id: str) -> List[Dict[str, Any]]:
        manifest = self._load_manifest(project_id)
        versions = manifest.get("versions") or []
        if not isinstance(versions, list):
            return []
        normalized = [self._summarize(item, include_image_ids=False) for item in versions if isinstance(item, dict)]
        normalized.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return normalized

    def get_version(self, project_id: str, version_id: str) -> Optional[Dict[str, Any]]:
        manifest = self._load_manifest(project_id)
        versions = manifest.get("versions") or []
        if not isinstance(versions, list):
            return None
        for version in versions:
            if not isinstance(version, dict):
                continue
            if str(version.get("id") or "") == version_id:
                return version
        return None

    def resolve_selected_image_ids(self, project_id: str, version_id: Optional[str]) -> Optional[Set[str]]:
        normalized_version_id = str(version_id or "").strip()
        if not normalized_version_id:
            return None
        version = self.get_version(project_id, normalized_version_id)
        if not version:
            raise ValueError(f"Dataset version {normalized_version_id} not found")
        image_ids = version.get("image_ids") or []
        if not isinstance(image_ids, list) or len(image_ids) == 0:
            raise ValueError("Selected dataset version has no images")
        return {str(image_id) for image_id in image_ids if str(image_id)}

    def create_classification_version(
        self,
        project_id: str,
        name: Optional[str] = None,
        sample_size: Optional[int] = None,
        balance_classes: bool = False,
        orientation: str = "any",
        resolution: str = "any",
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        dataset = dataset_service.get_dataset(project_id)
        task = str(dataset.get("task") or "classification").lower()
        if task != "classification":
            raise ValueError("Dataset versions with class balancing are currently supported for classification only")

        orientation_filter = str(orientation or "any").strip().lower()
        if orientation_filter not in VALID_ORIENTATION_FILTERS:
            raise ValueError(f"Invalid orientation filter: {orientation_filter}")

        resolution_filter = str(resolution or "any").strip().lower()
        if resolution_filter not in VALID_RESOLUTION_FILTERS:
            raise ValueError(f"Invalid resolution filter: {resolution_filter}")

        if sample_size is not None:
            sample_size = max(1, int(sample_size))

        valid_class_ids = {
            int(cls.get("id"))
            for cls in (dataset.get("classes") or [])
            if isinstance(cls, dict) and "id" in cls
        }
        class_name_by_id = {
            int(cls.get("id")): str(cls.get("name") or f"class_{cls.get('id')}")
            for cls in (dataset.get("classes") or [])
            if isinstance(cls, dict) and "id" in cls
        }
        if not valid_class_ids:
            raise ValueError("Dataset has no classes configured")

        images = dataset.get("images") or []
        candidates: List[Dict[str, Any]] = []
        for image in images:
            if not isinstance(image, dict):
                continue
            image_id = str(image.get("id") or "").strip()
            labels = image.get("labels") or []
            if not image_id or not labels:
                continue
            try:
                class_id = int(labels[0])
            except Exception:
                continue
            if class_id not in valid_class_ids:
                continue
            resolved_size = self._resolve_image_size(image)
            if not resolved_size:
                continue
            width, height = resolved_size
            image_orientation = self._orientation(width, height)
            if orientation_filter != "any" and image_orientation != orientation_filter:
                continue
            image_resolution = self._resolution_bucket(width, height)
            if resolution_filter != "any" and image_resolution != resolution_filter:
                continue
            candidates.append(
                {
                    "image_id": image_id,
                    "class_id": class_id,
                }
            )

        if not candidates:
            raise ValueError("No labeled images matched the selected dataset version filters")

        rng = random.Random(seed)
        by_class: Dict[int, List[Dict[str, Any]]] = {}
        for entry in candidates:
            by_class.setdefault(entry["class_id"], []).append(entry)
        for class_entries in by_class.values():
            rng.shuffle(class_entries)

        selected: List[Dict[str, Any]]
        max_target = len(candidates)
        target_size = min(sample_size, max_target) if sample_size is not None else max_target

        if balance_classes:
            class_ids = sorted(by_class.keys())
            pools = {class_id: list(by_class[class_id]) for class_id in class_ids}
            selected = []
            if sample_size is None:
                per_class = min(len(pools[class_id]) for class_id in class_ids)
                for class_id in class_ids:
                    selected.extend(pools[class_id][:per_class])
            else:
                base_take = target_size // len(class_ids)
                remainder = target_size % len(class_ids)
                for idx, class_id in enumerate(class_ids):
                    take = base_take + (1 if idx < remainder else 0)
                    bucket = pools[class_id]
                    selected.extend(bucket[:take])
                    pools[class_id] = bucket[take:]
                if len(selected) < target_size:
                    leftovers: List[Dict[str, Any]] = []
                    for class_id in class_ids:
                        leftovers.extend(pools[class_id])
                    rng.shuffle(leftovers)
                    selected.extend(leftovers[:target_size - len(selected)])
        else:
            shuffled = list(candidates)
            rng.shuffle(shuffled)
            selected = shuffled[:target_size]

        selected_ids: List[str] = []
        seen: Set[str] = set()
        for entry in selected:
            image_id = str(entry.get("image_id") or "")
            if not image_id or image_id in seen:
                continue
            seen.add(image_id)
            selected_ids.append(image_id)

        class_counts: Dict[int, int] = {}
        for entry in selected:
            class_id = int(entry.get("class_id"))
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

        now = _now()
        display_name = str(name or "").strip() or f"Dataset version {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
        version = {
            "id": f"dsv_{uuid.uuid4().hex[:8]}",
            "name": display_name,
            "task": "classification",
            "created_at": now,
            "updated_at": now,
            "source_count": len(candidates),
            "selected_count": len(selected_ids),
            "rules": {
                "sample_size": sample_size,
                "balance_classes": bool(balance_classes),
                "orientation": orientation_filter,
                "resolution": resolution_filter,
                "seed": seed,
            },
            "class_counts": [
                {
                    "class_id": class_id,
                    "class_name": class_name_by_id.get(class_id, f"class_{class_id}"),
                    "count": count,
                }
                for class_id, count in sorted(class_counts.items(), key=lambda item: item[0])
            ],
            "image_ids": selected_ids,
        }

        manifest = self._load_manifest(project_id)
        versions = manifest.get("versions") or []
        if not isinstance(versions, list):
            versions = []
        versions.append(version)
        manifest["versions"] = versions
        self._save_manifest(project_id, manifest)
        return self._summarize(version, include_image_ids=False)

    def delete_version(self, project_id: str, version_id: str) -> Dict[str, Any]:
        manifest = self._load_manifest(project_id)
        versions = manifest.get("versions") or []
        if not isinstance(versions, list):
            versions = []
        normalized_id = str(version_id or "").strip()
        kept: List[Dict[str, Any]] = []
        removed: Optional[Dict[str, Any]] = None
        for version in versions:
            if not isinstance(version, dict):
                continue
            if str(version.get("id") or "") == normalized_id:
                removed = version
                continue
            kept.append(version)
        if removed is None:
            raise ValueError(f"Dataset version {normalized_id} not found")
        manifest["versions"] = kept
        self._save_manifest(project_id, manifest)
        return {
            "project_id": project_id,
            "version_id": normalized_id,
            "status": "deleted",
        }


dataset_version_service = DatasetVersionService()
