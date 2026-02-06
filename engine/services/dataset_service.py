import os
import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image, UnidentifiedImageError

from engine.core.events import manager
from engine.io import paths, manifests

logger = logging.getLogger("VisionForge.Dataset")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class DatasetService:
    async def import_dataset(self, project_id: str, directory_path: str):
        path = Path(directory_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Update project manifest
        project_dir = paths.get_project_dir(project_id)
        if not project_dir.exists():
             raise ValueError("Project does not exist")

        manifest_path = project_dir / "project.json"
        
        # Simple Logic: Add path if not present
        project = manifests.load_manifest(manifest_path)
        if str(path) not in project.get("datasets", []):
            project["datasets"] = list(set(project.get("datasets", []) + [str(path)]))
            manifests.save_manifest(manifest_path, project, "project")
        
        return {"status": "imported", "access_path": str(path)}

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
