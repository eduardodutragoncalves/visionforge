from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
import os
import json
from datetime import datetime

from engine.orchestrator.core import orchestrator

router = APIRouter()

class CreateRunRequest(BaseModel):
    variant_id: Optional[str] = None
    config: Dict[str, Any] = {}

class RunResponse(BaseModel):
    project_id: str
    run_id: str
    status: str

@router.post("/projects/{project_id}/runs", response_model=RunResponse)
async def create_run(project_id: str, request: CreateRunRequest):
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    
    # Initialize run directory and run.json
    # In a real app we'd check if project exists first
    run_dir = os.path.join(orchestrator.workspace_root, "projects", project_id, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    run_metadata = {
        "id": run_id,
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
        "status": "queued"
    }

@router.get("/projects/{project_id}/runs/{run_id}")
async def get_run(project_id: str, run_id: str):
    run_dir = os.path.join(orchestrator.workspace_root, "projects", project_id, "runs", run_id)
    json_path = os.path.join(run_dir, "run.json")
    
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Run not found")
        
    with open(json_path, "r") as f:
        return json.load(f)
