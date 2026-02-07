from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime
import asyncio

from engine.io import paths, manifests
from engine.core.events import manager

router = APIRouter()

class CreateProjectRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    analysis_type: Optional[str] = "classification"

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    analysis_type: Optional[str] = "classification"
    created_at: str
    updated_at: str
    datasets: List[str]
    images_total: int = 0
    labeled_images: int = 0
    labels_total: int = 0
    classes_total: int = 0
    sources_total: int = 0
    runs_total: int = 0
    last_run_at: Optional[str] = None

def _normalize_project(data: dict) -> dict:
    if not data.get("analysis_type"):
        data["analysis_type"] = "classification"
    return data

def _compute_project_stats(project_id: str, analysis_type: str) -> dict:
    stats = {
        "images_total": 0,
        "labeled_images": 0,
        "labels_total": 0,
        "classes_total": 0,
        "sources_total": 0,
        "runs_total": 0,
        "last_run_at": None
    }

    dataset_path = paths.get_dataset_manifest_path(project_id)
    if dataset_path.exists():
        try:
            dataset = manifests.load_manifest(dataset_path)
            images = dataset.get("images", [])
            stats["images_total"] = len(images)
            stats["classes_total"] = len(dataset.get("classes", []))
            stats["sources_total"] = len(dataset.get("sources", []))
            task = dataset.get("task") or analysis_type
            if task == "detection":
                stats["labeled_images"] = sum(1 for image in images if image.get("boxes"))
                stats["labels_total"] = sum(len(image.get("boxes", [])) for image in images)
            else:
                stats["labeled_images"] = sum(1 for image in images if image.get("labels"))
                stats["labels_total"] = sum(len(image.get("labels", [])) for image in images)
        except Exception:
            pass

    runs_dir = paths.get_runs_dir(project_id)
    if runs_dir.exists():
        run_times = []
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            run_json = run_dir / "run.json"
            if not run_json.exists():
                continue
            try:
                run_data = manifests.load_manifest(run_json)
                stats["runs_total"] += 1
                created_at = run_data.get("created_at")
                if created_at:
                    run_times.append(created_at)
            except Exception:
                continue
        if run_times:
            stats["last_run_at"] = sorted(run_times)[-1]

    return stats

def _attach_stats(data: dict) -> dict:
    stats = _compute_project_stats(data.get("id", ""), data.get("analysis_type", "classification"))
    return {**data, **stats}

@router.get("/projects", response_model=List[ProjectResponse])
async def list_projects():
    projects_root = paths.get_projects_root()
    if not projects_root.exists():
        return []
    
    projects = []
    # Scan directories
    for project_dir in projects_root.iterdir():
        if project_dir.is_dir():
            manifest_path = project_dir / "project.json"
            if manifest_path.exists():
                try:
                    data = manifests.load_manifest(manifest_path)
                    normalized = _normalize_project(data)
                    projects.append(_attach_stats(normalized))
                except Exception as e:
                    print(f"Skipping invalid project {project_dir}: {e}")
    
    return sorted(projects, key=lambda x: x["created_at"], reverse=True)

@router.post("/projects", response_model=ProjectResponse)
async def create_project(request: CreateProjectRequest):
    project_id = str(uuid.uuid4())[:8] # Simple short ID
    now = datetime.now().isoformat()
    
    project_data = {
        "id": project_id,
        "name": request.name,
        "description": request.description,
        "analysis_type": request.analysis_type or "classification",
        "created_at": now,
        "updated_at": now,
        "datasets": []
    }
    
    project_dir = paths.get_project_dir(project_id)
    manifest_path = project_dir / "project.json"
    
    try:
        manifests.save_manifest(manifest_path, project_data, "project")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    normalized = _normalize_project(project_data)
    return _attach_stats(normalized)

@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str):
    project_dir = paths.get_project_dir(project_id)
    manifest_path = project_dir / "project.json"
    
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")
        
    data = manifests.load_manifest(manifest_path)
    normalized = _normalize_project(data)
    return _attach_stats(normalized)

@router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    project_dir = paths.get_project_dir(project_id).resolve()
    projects_root = paths.get_projects_root().resolve()
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    if not str(project_dir).startswith(str(projects_root)):
        raise HTTPException(status_code=403, detail="Invalid project path")

    async def _delete_with_progress():
        paths_list = sorted(project_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True)
        total = len(paths_list) + 1
        deleted = 0

        def should_emit(count: int, total_count: int) -> bool:
            if total_count <= 20:
                return True
            return count % 10 == 0 or count == total_count

        await manager.broadcast("project/delete_progress", {
            "project_id": project_id,
            "status": "starting",
            "total": total,
            "deleted": deleted,
            "progress": 0
        })

        for path in paths_list:
            try:
                if path.is_file() or path.is_symlink():
                    path.unlink()
                else:
                    path.rmdir()
            except Exception:
                pass
            deleted += 1
            if should_emit(deleted, total):
                await manager.broadcast("project/delete_progress", {
                    "project_id": project_id,
                    "status": "deleting",
                    "total": total,
                    "deleted": deleted,
                    "progress": int((deleted / total) * 100)
                })
            await asyncio.sleep(0)

        try:
            project_dir.rmdir()
        except Exception:
            pass
        deleted += 1

        await manager.broadcast("project/delete_progress", {
            "project_id": project_id,
            "status": "completed",
            "total": total,
            "deleted": deleted,
            "progress": 100
        })

    asyncio.create_task(_delete_with_progress())
    return {"status": "delete_started", "project_id": project_id}
