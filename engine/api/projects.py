from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime

from engine.io import paths, manifests

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

def _normalize_project(data: dict) -> dict:
    if not data.get("analysis_type"):
        data["analysis_type"] = "classification"
    return data

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
                    projects.append(_normalize_project(data))
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
    
    return project_data

@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str):
    project_dir = paths.get_project_dir(project_id)
    manifest_path = project_dir / "project.json"
    
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")
        
    data = manifests.load_manifest(manifest_path)
    return _normalize_project(data)
