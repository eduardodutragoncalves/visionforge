import os
import asyncio
import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image, UnidentifiedImageError

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

def _now() -> str:
    return datetime.utcnow().isoformat()

def _image_id(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]

def _assign_color(index: int) -> str:
    return CLASS_COLOR_POOL[index % len(CLASS_COLOR_POOL)]

class DatasetService:
    def _load_or_create_dataset(self, project_id: str, root_path: str = "") -> Dict[str, Any]:
        manifest_path = paths.get_dataset_manifest_path(project_id)
        if manifest_path.exists():
            return manifests.load_manifest(manifest_path)

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
                    "boxes": []
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
                "boxes": []
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
                    "boxes": []
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

    def delete_image(self, project_id: str, image_id: str) -> Dict[str, Any]:
        dataset = self._load_or_create_dataset(project_id)
        images = dataset.get("images", [])
        target = None
        for image in images:
            if image["id"] == image_id:
                target = image
                break
        if not target:
            raise ValueError("Image not found")

        image_path = Path(target["path"])
        dataset_dir = paths.get_project_dataset_dir(project_id).resolve()
        resolved_path = image_path.resolve()
        if str(resolved_path).startswith(str(dataset_dir)) and resolved_path.exists():
            try:
                resolved_path.unlink()
            except OSError:
                logger.warning("Unable to delete image file: %s", resolved_path)

        dataset["images"] = [img for img in images if img["id"] != image_id]
        dataset["updated_at"] = _now()
        manifests.save_manifest(paths.get_dataset_manifest_path(project_id), dataset, "dataset")
        return {"deleted": True, "image_id": image_id}

    async def validate_current_dataset(self, project_id: str):
        dataset = self._load_or_create_dataset(project_id)
        root_path = dataset.get("root_path")
        if not root_path:
            raise ValueError("Dataset root path not configured")
        report, files = await self.scan_dataset(project_id, root_path)
        await self.validate_dataset(project_id, files)
        return report

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
        scan_id = f"scan_{project_id}"
        path = Path(directory_path)
        
        counts = {ext: 0 for ext in VALID_EXTENSIONS}
        total_files = 0
        
        # Running synchronously in async flow for simplicity (MVP)
        # For large datasets, use run_in_executor
        
        await manager.broadcast("dataset/scan_progress", {"status": "starting", "path": str(path)})
        
        files_found = []
        
        for root, _, files in os.walk(path):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in VALID_EXTENSIONS:
                    counts[ext] += 1
                    total_files += 1
                    files_found.append(os.path.join(root, file))
                    
                    if total_files % 100 == 0:
                        await manager.broadcast("dataset/scan_progress", {
                            "status": "scanning",
                            "count": total_files
                        })
                        await asyncio.sleep(0) # Yield

        report = {
            "total_images": total_files,
            "distribution": counts,
            "scanned_at": str(path)
        }
        
        await manager.broadcast("dataset/scan_completed", report)
        return report, files_found

    async def validate_dataset(self, project_id: str, files_list: List[str]):
        """Validates images and computes hash."""
        total = len(files_list)
        corrupted = []
        valid_paths = []
        
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
                valid_paths.append((file_path, size))
                
            except (UnidentifiedImageError, OSError):
                corrupted.append(file_path)
            
            if idx % 50 == 0:
                progress =  round((idx / total) * 100, 1)
                await manager.broadcast("dataset/validate_progress", {
                    "progress": progress,
                    "processed": idx,
                    "total": total
                })
                await asyncio.sleep(0)

        # Final Hash computation on sorted valid entries
        valid_paths.sort(key=lambda x: x[0])
        for p, s in valid_paths:
            dataset_hash.update(f"{p}:{s}".encode('utf-8'))
            
        final_report = {
            "valid_count": len(valid_paths),
            "corrupted_count": len(corrupted),
            "corrupted_files": corrupted[:10], # Truncate for report
            "dataset_hash": dataset_hash.hexdigest()
        }
        
        # Save Report
        project_dir = paths.get_project_dir(project_id)
        with open(project_dir / "dataset_report.json", "w") as f:
            json.dump(final_report, f, indent=2)
            
        with open(project_dir / "dataset_hash.txt", "w") as f:
            f.write(final_report["dataset_hash"])

        await manager.broadcast("dataset/validated", final_report)
        return final_report

dataset_service = DatasetService()
