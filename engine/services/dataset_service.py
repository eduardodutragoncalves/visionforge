import os
import asyncio
import hashlib
import json
import logging
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageOps, UnidentifiedImageError

from engine.core.events import manager
from engine.io import paths, manifests

logger = logging.getLogger("VisionForge.Dataset")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_TASK = "classification"
DEFAULT_FORMAT = "yolo"
CLASS_COLOR_POOL = [
    "#2fb8ff",
    "#5fd3ff",
    "#3bd989",
    "#ffc857",
    "#ff5c6f",
    "#7bdff2"
]
THUMBNAIL_MIN_SIZE = 64
THUMBNAIL_MAX_SIZE = 1024
THUMBNAIL_DEFAULT_SIZE = 256

def _now() -> str:
    return datetime.utcnow().isoformat()

def _image_id(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]

def _assign_color(index: int) -> str:
    return CLASS_COLOR_POOL[index % len(CLASS_COLOR_POOL)]

class DatasetService:
    def _normalize_thumbnail_size(self, size: int) -> int:
        safe = int(size or THUMBNAIL_DEFAULT_SIZE)
        return max(THUMBNAIL_MIN_SIZE, min(THUMBNAIL_MAX_SIZE, safe))

    def _thumbnail_cache_root(self, project_id: str) -> Path:
        return paths.get_project_dataset_dir(project_id) / ".thumbnails"

    def _thumbnail_cache_path(self, project_id: str, image_id: str, size: int) -> Path:
        safe_size = self._normalize_thumbnail_size(size)
        return self._thumbnail_cache_root(project_id) / f"s{safe_size}" / f"{image_id}.jpg"

    def _cleanup_thumbnail_cache(self, project_id: str, image_id: str):
        cache_root = self._thumbnail_cache_root(project_id)
        if not cache_root.exists():
            return
        for size_dir in cache_root.iterdir():
            if not size_dir.is_dir():
                continue
            candidate = size_dir / f"{image_id}.jpg"
            if candidate.exists():
                try:
                    candidate.unlink()
                except OSError:
                    logger.warning("Unable to delete cached thumbnail: %s", candidate)
                    continue
            try:
                if not any(size_dir.iterdir()):
                    size_dir.rmdir()
            except OSError:
                pass
        try:
            if cache_root.exists() and not any(cache_root.iterdir()):
                cache_root.rmdir()
        except OSError:
            pass

    def get_or_create_thumbnail_path(self, project_id: str, image_id: str, size: int = THUMBNAIL_DEFAULT_SIZE) -> Path:
        image_path = Path(self.get_image_path(project_id, image_id)).resolve()
        dataset_dir = paths.get_project_dataset_dir(project_id).resolve()
        if not str(image_path).startswith(str(dataset_dir)):
            raise ValueError("Image path not allowed")
        if not image_path.exists():
            raise ValueError("Image file not found")

        safe_size = self._normalize_thumbnail_size(size)
        thumbnail_path = self._thumbnail_cache_path(project_id, image_id, safe_size)
        thumbnail_dir = thumbnail_path.parent
        paths.ensure_dirs(thumbnail_dir)

        source_mtime_ns = image_path.stat().st_mtime_ns
        if thumbnail_path.exists():
            try:
                if thumbnail_path.stat().st_mtime_ns >= source_mtime_ns:
                    return thumbnail_path
            except OSError:
                pass

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
                background = Image.new("RGBA", image.size, (12, 20, 27, 255))
                background.alpha_composite(image.convert("RGBA"))
                image = background.convert("RGB")
            else:
                image = image.convert("RGB")

            resampling = getattr(Image, "Resampling", Image)
            image.thumbnail((safe_size, safe_size), resampling.LANCZOS)
            image.save(thumbnail_path, format="JPEG", quality=82, optimize=True)

        return thumbnail_path

    def _file_content_hash(self, file_path: str, chunk_size: int = 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with open(file_path, "rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _is_annotated_image(self, image: Dict[str, Any]) -> bool:
        labels = image.get("labels") or []
        boxes = image.get("boxes") or []
        detection_done = bool(image.get("detection_done", False))
        return len(labels) > 0 or len(boxes) > 0 or detection_done

    def _load_or_create_dataset(self, project_id: str, root_path: str = "") -> Dict[str, Any]:
        manifest_path = paths.get_dataset_manifest_path(project_id)
        if manifest_path.exists():
            dataset = manifests.load_manifest(manifest_path)
            images = dataset.get("images", [])
            changed = False
            for image in images:
                if "detection_done" not in image:
                    image["detection_done"] = False
                    changed = True
                if "is_augmented" not in image:
                    image["is_augmented"] = False
                    changed = True
                if "augmentation_source_id" not in image:
                    image["augmentation_source_id"] = None
                    changed = True
            if changed:
                dataset["updated_at"] = _now()
                manifests.save_manifest(manifest_path, dataset, "dataset")
            return dataset

        project_manifest_path = paths.get_project_dir(project_id) / "project.json"
        analysis_type = DEFAULT_TASK
        if project_manifest_path.exists():
            try:
                project = manifests.load_manifest(project_manifest_path)
                analysis_type = project.get("analysis_type") or DEFAULT_TASK
            except Exception:
                analysis_type = DEFAULT_TASK

        dataset = {
            "id": f"dataset_{project_id}",
            "project_id": project_id,
            "root_path": root_path,
            "task": analysis_type,
            "format": DEFAULT_FORMAT,
            "created_at": _now(),
            "updated_at": _now(),
            "sources": [],
            "classes": [],
            "images": []
        }
        manifests.save_manifest(manifest_path, dataset, "dataset")
        return dataset

    def get_dataset(self, project_id: str) -> Dict[str, Any]:
        return self._load_or_create_dataset(project_id)

    def initialize_dataset(self, project_id: str, root_path: str, files_list: List[str]) -> Dict[str, Any]:
        dataset = self._load_or_create_dataset(project_id, root_path)
        existing_by_path = {img["path"]: img for img in dataset.get("images", [])}
        images = []

        for file_path in sorted(files_list):
            existing = existing_by_path.get(file_path)
            if existing:
                images.append(existing)
            else:
                images.append({
                    "id": _image_id(file_path),
                    "path": file_path,
                    "split": "unassigned",
                    "labels": [],
                    "boxes": [],
                    "detection_done": False,
                    "is_augmented": False,
                    "augmentation_source_id": None
                })

        dataset["root_path"] = root_path
        dataset["images"] = images
        dataset["updated_at"] = _now()
        manifests.save_manifest(paths.get_dataset_manifest_path(project_id), dataset, "dataset")
        return dataset

    def _ensure_dataset_dirs(self, project_id: str) -> Tuple[Path, Path, Path]:
        dataset_dir = paths.get_project_dataset_dir(project_id)
        images_dir = paths.get_project_dataset_images_dir(project_id)
        frames_dir = paths.get_project_dataset_frames_dir(project_id)
        paths.ensure_dirs(images_dir)
        paths.ensure_dirs(frames_dir)
        return dataset_dir, images_dir, frames_dir

    def _unique_destination(self, dest_dir: Path, filename: str) -> Path:
        base = Path(filename).stem
        ext = Path(filename).suffix
        candidate = dest_dir / f"{base}{ext}"
        counter = 1
        while candidate.exists():
            candidate = dest_dir / f"{base}_{counter}{ext}"
            counter += 1
        return candidate

    def _register_source(self, dataset: Dict[str, Any], source_type: str, source_path: str):
        sources = dataset.get("sources", [])
        sources.append({
            "type": source_type,
            "path": source_path,
            "added_at": _now()
        })
        dataset["sources"] = sources

    def _add_images(self, project_id: str, file_paths: List[str], source_type: str, source_path: str) -> Dict[str, Any]:
        dataset = self._load_or_create_dataset(project_id)
        dataset_dir, images_dir, _ = self._ensure_dataset_dirs(project_id)
        existing_paths = {img["path"] for img in dataset.get("images", [])}
        added = []

        for file_path in file_paths:
            ext = Path(file_path).suffix.lower()
            if ext not in VALID_EXTENSIONS:
                continue
            destination = self._unique_destination(images_dir, Path(file_path).name)
            shutil.copy2(file_path, destination)
            if str(destination) in existing_paths:
                continue
            image_entry = {
                "id": _image_id(str(destination)),
                "path": str(destination),
                "split": "unassigned",
                "labels": [],
                "boxes": [],
                "detection_done": False,
                "is_augmented": False,
                "augmentation_source_id": None
            }
            dataset["images"].append(image_entry)
            added.append(image_entry)
            existing_paths.add(str(destination))

        dataset["root_path"] = str(dataset_dir)
        dataset["updated_at"] = _now()
        self._register_source(dataset, source_type, source_path)
        manifests.save_manifest(paths.get_dataset_manifest_path(project_id), dataset, "dataset")
        return {"added": added, "count": len(added), "dataset": dataset}

    def ingest_directories(self, project_id: str, directories: List[str]) -> Dict[str, Any]:
        files = []
        for directory in directories:
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    ext = Path(filename).suffix.lower()
                    if ext in VALID_EXTENSIONS:
                        files.append(os.path.join(root, filename))
        return self._add_images(project_id, files, "directory", ",".join(directories))

    def ingest_files(self, project_id: str, files: List[str]) -> Dict[str, Any]:
        return self._add_images(project_id, files, "file", ",".join(files))

    def ingest_video(self, project_id: str, video_path: str, frame_interval: int, max_frames: int | None = None) -> Dict[str, Any]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for video ingestion") from exc

        dataset = self._load_or_create_dataset(project_id)
        dataset_dir, _, frames_dir = self._ensure_dataset_dirs(project_id)

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        frame_count = 0
        saved_count = 0
        added_entries = []
        stem = Path(video_path).stem

        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_interval <= 0:
                frame_interval = 1
            if frame_count % frame_interval == 0:
                filename = f"{stem}_{frame_count:06d}.jpg"
                destination = self._unique_destination(frames_dir, filename)
                cv2.imwrite(str(destination), frame)
                image_entry = {
                    "id": _image_id(str(destination)),
                    "path": str(destination),
                    "split": "unassigned",
                    "labels": [],
                    "boxes": [],
                    "detection_done": False,
                    "is_augmented": False,
                    "augmentation_source_id": None
                }
                dataset["images"].append(image_entry)
                added_entries.append(image_entry)
                saved_count += 1
                if max_frames and saved_count >= max_frames:
                    break
            frame_count += 1

        capture.release()
        dataset["root_path"] = str(dataset_dir)
        dataset["updated_at"] = _now()
        self._register_source(dataset, "video", video_path)
        manifests.save_manifest(paths.get_dataset_manifest_path(project_id), dataset, "dataset")
        return {"added": added_entries, "count": saved_count, "dataset": dataset}

    def update_dataset_config(self, project_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        dataset = self._load_or_create_dataset(project_id)
        if "task" in updates and updates["task"]:
            dataset["task"] = updates["task"]
        if "format" in updates and updates["format"]:
            dataset["format"] = updates["format"]
        dataset["updated_at"] = _now()
        manifests.save_manifest(paths.get_dataset_manifest_path(project_id), dataset, "dataset")
        return dataset

    def update_classes(self, project_id: str, classes: List[Dict[str, Any]]) -> Dict[str, Any]:
        dataset = self._load_or_create_dataset(project_id)
        existing = {c["name"]: c for c in dataset.get("classes", [])}
        next_id = max([c["id"] for c in dataset.get("classes", [])] + [-1]) + 1

        normalized = []
        for idx, item in enumerate(classes):
            name = item.get("name", "").strip()
            if not name:
                continue
            existing_class = existing.get(name)
            class_id = item.get("id")
            if class_id is None and existing_class:
                class_id = existing_class["id"]
            if class_id is None:
                class_id = next_id
                next_id += 1
            color = item.get("color") or (existing_class.get("color") if existing_class else None) or _assign_color(idx)
            normalized.append({"id": class_id, "name": name, "color": color})

        dataset["classes"] = normalized
        dataset["updated_at"] = _now()
        manifests.save_manifest(paths.get_dataset_manifest_path(project_id), dataset, "dataset")
        return dataset

    def list_images(self, project_id: str, limit: int = 200, offset: int = 0) -> List[Dict[str, Any]]:
        dataset = self._load_or_create_dataset(project_id)
        images = dataset.get("images", [])
        return images[offset:offset + limit]

    def get_image_path(self, project_id: str, image_id: str) -> str:
        dataset = self._load_or_create_dataset(project_id)
        images = dataset.get("images", [])
        for image in images:
            if image["id"] == image_id:
                return image["path"]
        raise ValueError("Image not found")

    def update_image_labels(self, project_id: str, image_id: str, labels: List[int]) -> Dict[str, Any]:
        dataset = self._load_or_create_dataset(project_id)
        images = dataset.get("images", [])
        for image in images:
            if image["id"] == image_id:
                image["labels"] = sorted(set(labels))
                dataset["updated_at"] = _now()
                manifests.save_manifest(paths.get_dataset_manifest_path(project_id), dataset, "dataset")
                return image
        raise ValueError("Image not found")

    def update_images_labels(self, project_id: str, image_ids: List[str], labels: List[int]) -> Dict[str, Any]:
        dataset = self._load_or_create_dataset(project_id)
        images = dataset.get("images", [])
        updated = 0
        for image in images:
            if image["id"] in image_ids:
                image["labels"] = sorted(set(labels))
                updated += 1
        if updated:
            dataset["updated_at"] = _now()
            manifests.save_manifest(paths.get_dataset_manifest_path(project_id), dataset, "dataset")
        return {"updated": updated}

    def apply_classification_assignments(self, project_id: str, assignments: Dict[str, int]) -> Dict[str, Any]:
        dataset = self._load_or_create_dataset(project_id)
        images = dataset.get("images", [])
        images_by_id = {str(image.get("id") or ""): image for image in images}
        valid_class_ids = {int(cls.get("id")) for cls in (dataset.get("classes") or []) if "id" in cls}

        updated_ids: List[str] = []
        missing = 0
        invalid_class = 0

        for image_id, class_id in assignments.items():
            key = str(image_id or "").strip()
            image = images_by_id.get(key)
            if not image:
                missing += 1
                continue
            try:
                normalized_class_id = int(class_id)
            except Exception:
                invalid_class += 1
                continue
            if normalized_class_id not in valid_class_ids:
                invalid_class += 1
                continue
            next_labels = [normalized_class_id]
            if image.get("labels") == next_labels:
                updated_ids.append(key)
                continue
            image["labels"] = next_labels
            updated_ids.append(key)

        if updated_ids:
            dataset["updated_at"] = _now()
            manifests.save_manifest(paths.get_dataset_manifest_path(project_id), dataset, "dataset")

        return {
            "updated": len(updated_ids),
            "updated_ids": updated_ids,
            "missing": missing,
            "invalid_class": invalid_class,
        }

    def update_image_boxes(self, project_id: str, image_id: str, boxes: List[Dict[str, Any]]) -> Dict[str, Any]:
        dataset = self._load_or_create_dataset(project_id)
        images = dataset.get("images", [])

        def _clamp(value: float) -> float:
            return max(0.0, min(1.0, value))

        normalized_boxes = []
        for box in boxes:
            normalized_boxes.append({
                "class_id": int(box["class_id"]),
                "x": _clamp(float(box["x"])),
                "y": _clamp(float(box["y"])),
                "w": _clamp(float(box["w"])),
                "h": _clamp(float(box["h"]))
            })

        for image in images:
            if image["id"] == image_id:
                image["boxes"] = normalized_boxes
                dataset["updated_at"] = _now()
                manifests.save_manifest(paths.get_dataset_manifest_path(project_id), dataset, "dataset")
                return image
        raise ValueError("Image not found")

    def update_image_detection_status(self, project_id: str, image_id: str, detection_done: bool) -> Dict[str, Any]:
        dataset = self._load_or_create_dataset(project_id)
        images = dataset.get("images", [])
        for image in images:
            if image["id"] == image_id:
                image["detection_done"] = bool(detection_done)
                dataset["updated_at"] = _now()
                manifests.save_manifest(paths.get_dataset_manifest_path(project_id), dataset, "dataset")
                return image
        raise ValueError("Image not found")

    async def augment_classification_images(
        self,
        project_id: str,
        image_ids: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        config = config or {}
        dataset = self._load_or_create_dataset(project_id)
        if dataset.get("task") != "classification":
            raise ValueError("Augmentation is currently available only for classification datasets")

        all_images = dataset.get("images", [])
        if not all_images:
            return {"added": 0, "generated_from": 0}

        target_scope = str(config.get("target_scope", "all")).strip().lower() or "all"
        if target_scope not in {"all", "labeled", "unlabeled"}:
            target_scope = "all"

        def _matches_scope(image: Dict[str, Any]) -> bool:
            labels = image.get("labels", [])
            if target_scope == "labeled":
                return len(labels) > 0
            if target_scope == "unlabeled":
                return len(labels) == 0
            return True

        image_ids_set = set(image_ids or [])
        if image_ids_set:
            source_images = [img for img in all_images if img.get("id") in image_ids_set and _matches_scope(img)]
        else:
            source_images = [img for img in all_images if _matches_scope(img)]
        if not source_images:
            return {"added": 0, "generated_from": 0}

        copies_per_image = max(1, int(config.get("copies_per_image", 1)))
        output_size = max(32, int(config.get("output_size", 640)))

        apply_percent = float(config.get("apply_percent", 100.0))
        apply_ratio = max(0.0, min(1.0, apply_percent / 100.0))

        hflip_percent = float(config.get("horizontal_flip_percent", 50.0))
        vflip_percent = float(config.get("vertical_flip_percent", 0.0))
        erasing_percent = float(config.get("erasing_percent", 0.0))
        hsv_h_percent = float(config.get("hsv_h_percent", 1.5))
        hsv_s_percent = float(config.get("hsv_s_percent", 40.0))
        hsv_v_percent = float(config.get("hsv_v_percent", 40.0))
        auto_augment = config.get("auto_augment")
        if auto_augment in ("none", "", "null"):
            auto_augment = None
        force_color_jitter = bool(config.get("force_color_jitter", False))

        hflip = max(0.0, min(1.0, hflip_percent / 100.0))
        vflip = max(0.0, min(1.0, vflip_percent / 100.0))
        erasing = max(0.0, min(1.0, erasing_percent / 100.0))
        hsv_h = max(0.0, min(1.0, hsv_h_percent / 100.0))
        hsv_s = max(0.0, min(1.0, hsv_s_percent / 100.0))
        hsv_v = max(0.0, min(1.0, hsv_v_percent / 100.0))

        try:
            from ultralytics.data.augment import classify_augmentations
            from torchvision.transforms.functional import to_pil_image
        except ImportError as exc:
            raise RuntimeError("Ultralytics + torchvision are required for classification augmentation") from exc

        request_id = str(config.get("request_id") or "").strip() or None

        transform = classify_augmentations(
            size=output_size,
            hflip=hflip,
            vflip=vflip,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            erasing=erasing,
            auto_augment=auto_augment,
            force_color_jitter=force_color_jitter
        )

        _, images_dir, _ = self._ensure_dataset_dirs(project_id)
        generated_images: List[Dict[str, Any]] = []
        skipped_missing = 0
        planned_operations = max(len(source_images) * copies_per_image, 1)
        processed_operations = 0
        last_reported_processed = -1

        async def emit_progress(status: str, force: bool = False):
            nonlocal last_reported_processed
            if not force and status == "running":
                if processed_operations < planned_operations and (processed_operations - last_reported_processed) < 5:
                    return
            last_reported_processed = processed_operations
            await manager.broadcast("dataset/augment_progress", {
                "project_id": project_id,
                "request_id": request_id,
                "status": status,
                "processed": processed_operations,
                "total": planned_operations,
                "percent": round((processed_operations / planned_operations) * 100, 2),
                "added": len(generated_images)
            })

        await emit_progress("starting", force=True)

        for source in source_images:
            source_path = Path(source.get("path", ""))
            if not source_path.exists() or not source_path.is_file():
                skipped_missing += copies_per_image
                processed_operations += copies_per_image
                await emit_progress("running")
                await asyncio.sleep(0)
                continue
            for _ in range(copies_per_image):
                processed_operations += 1
                if apply_ratio < 1.0 and random.random() > apply_ratio:
                    await emit_progress("running")
                    await asyncio.sleep(0)
                    continue
                try:
                    with Image.open(source_path) as img:
                        pil = img.convert("RGB")
                        augmented = transform(pil)
                        if hasattr(augmented, "detach"):
                            augmented = augmented.detach().clamp(0, 1)
                            augmented_pil = to_pil_image(augmented)
                        else:
                            augmented_pil = augmented
                except Exception:
                    skipped_missing += 1
                    await emit_progress("running")
                    await asyncio.sleep(0)
                    continue

                destination = self._unique_destination(images_dir, f"{source_path.stem}_aug{source_path.suffix.lower()}")
                save_kwargs: Dict[str, Any] = {}
                if destination.suffix.lower() in {".jpg", ".jpeg"}:
                    save_kwargs["quality"] = 95
                    save_kwargs["optimize"] = True
                augmented_pil.save(destination, **save_kwargs)

                image_entry = {
                    "id": _image_id(str(destination)),
                    "path": str(destination),
                    "split": source.get("split", "unassigned") or "unassigned",
                    "labels": sorted(set(source.get("labels", []))),
                    "boxes": [],
                    "detection_done": False,
                    "is_augmented": True,
                    "augmentation_source_id": source.get("id")
                }
                generated_images.append(image_entry)
                await emit_progress("running")
                await asyncio.sleep(0)

        if generated_images:
            dataset["images"].extend(generated_images)
            dataset["updated_at"] = _now()
            manifests.save_manifest(paths.get_dataset_manifest_path(project_id), dataset, "dataset")

        result = {
            "added": len(generated_images),
            "generated_from": len(source_images),
            "skipped": skipped_missing,
            "config": {
                "copies_per_image": copies_per_image,
                "output_size": output_size,
                "apply_percent": apply_percent,
                "horizontal_flip_percent": hflip_percent,
                "vertical_flip_percent": vflip_percent,
                "erasing_percent": erasing_percent,
                "hsv_h_percent": hsv_h_percent,
                "hsv_s_percent": hsv_s_percent,
                "hsv_v_percent": hsv_v_percent,
                "auto_augment": auto_augment,
                "force_color_jitter": force_color_jitter,
                "target_scope": target_scope
            }
        }
        processed_operations = planned_operations
        await emit_progress("completed", force=True)
        return result

    def delete_image(self, project_id: str, image_id: str) -> Dict[str, Any]:
        result = self.delete_images(project_id, [image_id])
        if image_id not in result["deleted_ids"]:
            raise ValueError("Image not found")
        return {"deleted": True, "image_id": image_id}

    def delete_images(self, project_id: str, image_ids: List[str]) -> Dict[str, Any]:
        dataset = self._load_or_create_dataset(project_id)
        images = dataset.get("images", [])
        images_by_id = {image["id"]: image for image in images}
        dataset_dir = paths.get_project_dataset_dir(project_id).resolve()

        deleted_ids = []
        missing_ids = []

        for image_id in image_ids:
            target = images_by_id.get(image_id)
            if not target:
                missing_ids.append(image_id)
                continue
            image_path = Path(target["path"])
            resolved_path = image_path.resolve()
            if str(resolved_path).startswith(str(dataset_dir)) and resolved_path.exists():
                try:
                    resolved_path.unlink()
                except OSError:
                    logger.warning("Unable to delete image file: %s", resolved_path)
            self._cleanup_thumbnail_cache(project_id, image_id)
            deleted_ids.append(image_id)

        if deleted_ids:
            dataset["images"] = [img for img in images if img["id"] not in set(deleted_ids)]
            dataset["updated_at"] = _now()
            manifests.save_manifest(paths.get_dataset_manifest_path(project_id), dataset, "dataset")

        return {
            "deleted": len(deleted_ids),
            "missing": len(missing_ids),
            "deleted_ids": deleted_ids,
            "missing_ids": missing_ids
        }

    async def validate_current_dataset(self, project_id: str):
        dataset = self._load_or_create_dataset(project_id)
        root_path = dataset.get("root_path")
        if not root_path:
            raise ValueError("Dataset root path not configured")
        scan_report, files = await self.scan_dataset(project_id, root_path)
        validation_report = await self.validate_dataset(project_id, files)
        return {
            **validation_report,
            "scan": scan_report
        }

    async def import_dataset(self, project_id: str, directory_path: str):
        path = Path(directory_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        project_dir = paths.get_project_dir(project_id)
        if not project_dir.exists():
            raise ValueError("Project does not exist")

        manifest_path = project_dir / "project.json"
        
        project = manifests.load_manifest(manifest_path)
        if str(path) not in project.get("datasets", []):
            project["datasets"] = list(set(project.get("datasets", []) + [str(path)]))
            manifests.save_manifest(manifest_path, project, "project")
        result = self.ingest_directories(project_id, [str(path)])
        return {"status": "imported", "access_path": str(path), "count": result["count"]}

    async def scan_dataset(self, project_id: str, directory_path: str):
        """Scans directory and reports file counts."""
        path = Path(directory_path)
        
        counts = {ext: 0 for ext in VALID_EXTENSIONS}
        total_files = 0
        
        # Running synchronously in async flow for simplicity (MVP)
        # For large datasets, use run_in_executor
        
        await manager.broadcast("dataset/scan_progress", {
            "phase": "scan",
            "status": "starting",
            "path": str(path),
            "count": 0,
            "progress": 0,
            "message": "Scanning folders and files..."
        })
        
        files_found = []
        
        for root, _, files in os.walk(path):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in VALID_EXTENSIONS:
                    counts[ext] += 1
                    total_files += 1
                    files_found.append(os.path.join(root, file))
                    
                    if total_files % 25 == 0:
                        await manager.broadcast("dataset/scan_progress", {
                            "phase": "scan",
                            "status": "scanning",
                            "count": total_files,
                            "message": f"Found {total_files} image files so far..."
                        })
                        await asyncio.sleep(0) # Yield

        report = {
            "total_images": total_files,
            "distribution": counts,
            "scanned_at": str(path)
        }
        
        await manager.broadcast("dataset/scan_progress", {
            "phase": "scan",
            "status": "completed",
            "count": total_files,
            "progress": 100,
            "message": f"Scan completed with {total_files} image(s). Starting validation..."
        })
        await manager.broadcast("dataset/scan_completed", report)
        return report, files_found

    async def validate_dataset(self, project_id: str, files_list: List[str]):
        """Validates images and computes hash."""
        dataset = self._load_or_create_dataset(project_id)
        images = dataset.get("images", [])
        image_by_path = {}
        for image in images:
            raw_path = image.get("path")
            if not raw_path:
                continue
            image_by_path[raw_path] = image
            try:
                image_by_path[str(Path(raw_path).resolve())] = image
            except Exception:
                pass

        total = len(files_list)
        corrupted = []
        valid_entries = []
        duplicate_buckets: Dict[str, List[Dict[str, Any]]] = {}
        
        dataset_hash = hashlib.sha256()
        
        for idx, file_path in enumerate(files_list):
            try:
                # Validation: Try verifying
                with Image.open(file_path) as img:
                    img.verify()
                
                # Hash calculation (Path + Size for speed, logic can be strictly content later)
                # Sort order matters for deterministic hash, handled by sorted list passed in or sorting here
                # Here we trust the order or should sort
                size = os.path.getsize(file_path)
                content_hash = self._file_content_hash(file_path)
                resolved_file_path = str(Path(file_path).resolve())
                image_meta = image_by_path.get(resolved_file_path) or image_by_path.get(file_path)
                entry = {
                    "path": file_path,
                    "size": size,
                    "content_hash": content_hash,
                    "id": image_meta.get("id") if image_meta else None,
                    "labels_count": len(image_meta.get("labels", [])) if image_meta else 0,
                    "boxes_count": len(image_meta.get("boxes", [])) if image_meta else 0,
                    "detection_done": bool(image_meta.get("detection_done", False)) if image_meta else False,
                    "annotated": self._is_annotated_image(image_meta) if image_meta else False
                }
                valid_entries.append(entry)
                if entry["id"]:
                    duplicate_buckets.setdefault(content_hash, []).append(entry)
                
            except (UnidentifiedImageError, OSError):
                corrupted.append(file_path)
            
            if idx % 10 == 0:
                processed = idx + 1 if total > 0 else 0
                progress = round((processed / total) * 100, 1) if total > 0 else 100.0
                await manager.broadcast("dataset/validate_progress", {
                    "phase": "validate",
                    "status": "running",
                    "progress": progress,
                    "processed": processed,
                    "total": total,
                    "message": f"Validated {processed} / {total} image(s)..."
                })
                await asyncio.sleep(0)

        # Final Hash computation on sorted valid entries
        valid_entries.sort(key=lambda x: x["path"])
        for entry in valid_entries:
            dataset_hash.update(f"{entry['path']}:{entry['size']}".encode('utf-8'))

        duplicate_groups: List[Dict[str, Any]] = []
        for content_hash, entries in duplicate_buckets.items():
            if len(entries) < 2:
                continue
            sorted_entries = sorted(entries, key=lambda item: item["path"])
            duplicate_groups.append({
                "content_hash": content_hash,
                "count": len(sorted_entries),
                "images": sorted_entries
            })

        duplicate_groups.sort(key=lambda group: group["count"], reverse=True)
        duplicate_files_count = sum(group["count"] for group in duplicate_groups)
            
        final_report = {
            "valid_count": len(valid_entries),
            "corrupted_count": len(corrupted),
            "corrupted_files": corrupted[:10], # Truncate for report
            "dataset_hash": dataset_hash.hexdigest(),
            "duplicate_groups_count": len(duplicate_groups),
            "duplicate_files_count": duplicate_files_count,
            "duplicate_groups": duplicate_groups
        }
        
        # Save Report
        project_dir = paths.get_project_dir(project_id)
        manifests.save_manifest(project_dir / "dataset_report.json", final_report)
            
        with open(project_dir / "dataset_hash.txt", "w") as f:
            f.write(final_report["dataset_hash"])

        await manager.broadcast("dataset/validate_progress", {
            "phase": "validate",
            "status": "completed",
            "progress": 100,
            "processed": total,
            "total": total,
            "message": "Validation completed. Finalizing report..."
        })
        await manager.broadcast("dataset/validated", final_report)
        return final_report

dataset_service = DatasetService()
