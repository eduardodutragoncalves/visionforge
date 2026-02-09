from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uuid
import os
import json
from datetime import datetime
import re
from pathlib import Path

from engine.orchestrator.core import orchestrator
from engine.io import paths, manifests

router = APIRouter()

class CreateRunRequest(BaseModel):
    variant_id: Optional[str] = None
    config: Dict[str, Any] = {}

class RunResponse(BaseModel):
    project_id: str
    run_id: str
    status: str
    name: Optional[str] = None

def _slugify(value: str) -> str:
    lowered = (value or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    return lowered.strip("_")

def _build_run_name(project_name: str, config: Dict[str, Any]) -> str:
    epochs = config.get("epochs") or 0
    batch = config.get("batch_size") or config.get("batch") or 0
    model = config.get("base_model") or config.get("model") or "model"
    model_slug = _slugify(Path(str(model)).stem)
    project_slug = _slugify(project_name) or _slugify(str(project_name)) or "project"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{project_slug}_ep_{epochs}_bz_{batch}_modver_{model_slug}_{timestamp}"

@router.post("/projects/{project_id}/runs", response_model=RunResponse)
async def create_run(project_id: str, request: CreateRunRequest):
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    project_name = project_id
    project_manifest = paths.get_project_dir(project_id) / "project.json"
    if project_manifest.exists():
        try:
            project_data = manifests.load_manifest(project_manifest)
            project_name = project_data.get("name") or project_name
        except Exception:
            pass
    run_name = _build_run_name(project_name, request.config or {})
    
    # Initialize run directory and run.json
    # In a real app we'd check if project exists first
    run_dir = os.path.join(orchestrator.workspace_root, "projects", project_id, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    run_metadata = {
        "id": run_id,
        "name": run_name,
        "project_id": project_id,
        "variant_id": request.variant_id,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "config": request.config
    }
    
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump(run_metadata, f, indent=2)
    
    # Submit to Orchestrator
    await orchestrator.submit_job(project_id, run_id, request.config)
    
    return {
        "project_id": project_id,
        "run_id": run_id,
        "status": "queued",
        "name": run_name
    }

@router.get("/projects/{project_id}/runs")
async def list_runs(project_id: str) -> List[Dict[str, Any]]:
    orchestrator.reconcile_stale_runs(project_id=project_id)
    runs_dir = paths.get_runs_dir(project_id)
    if not runs_dir.exists():
        return []
    runs: List[Dict[str, Any]] = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        run_json = run_dir / "run.json"
        if not run_json.exists():
            continue
        try:
            runs.append(manifests.load_manifest(run_json))
        except Exception:
            continue
    return sorted(runs, key=lambda item: item.get("created_at", ""), reverse=True)

@router.get("/projects/{project_id}/runs/{run_id}")
async def get_run(project_id: str, run_id: str):
    run_dir = os.path.join(orchestrator.workspace_root, "projects", project_id, "runs", run_id)
    json_path = os.path.join(run_dir, "run.json")
    
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Run not found")
        
    with open(json_path, "r") as f:
        return json.load(f)
