from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uuid
import os
import json
import shutil
from datetime import datetime
import re
from pathlib import Path

from engine.orchestrator.core import orchestrator
from engine.io import paths, manifests
from engine.services.sqlite_cache_service import sqlite_cache_service

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
    cache_key = f"api:runs:list:v1:{project_id}"
    signature = sqlite_cache_service.runs_signature(project_id)
    cached_payload = sqlite_cache_service.get_payload(cache_key, signature)
    if isinstance(cached_payload, list):
        return cached_payload

    runs_dir = paths.get_runs_dir(project_id)
    if not runs_dir.exists():
        sqlite_cache_service.set_payload(cache_key, signature, [])
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
    payload = sorted(runs, key=lambda item: item.get("created_at", ""), reverse=True)
    sqlite_cache_service.set_payload(cache_key, signature, payload)
    return payload

@router.get("/projects/{project_id}/runs/{run_id}")
async def get_run(project_id: str, run_id: str):
    run_dir = os.path.join(orchestrator.workspace_root, "projects", project_id, "runs", run_id)
    json_path = os.path.join(run_dir, "run.json")
    
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Run not found")
        
    with open(json_path, "r") as f:
        return json.load(f)

@router.delete("/projects/{project_id}/runs/{run_id}")
async def delete_run(project_id: str, run_id: str):
    """
    Delete a training run and all generated artifacts linked to it:
    - run folder under /runs/{run_id}
    - variants produced from this run (and their exports)
    """
    orchestrator.reconcile_stale_runs(project_id=project_id)

    if run_id in orchestrator.active_jobs or run_id in orchestrator.queued_run_ids:
        raise HTTPException(status_code=409, detail="Run is currently active. Stop it before deleting.")

    run_dir = paths.get_run_dir(project_id, run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    variants_root = paths.get_variants_dir(project_id)
    removed_variants: List[str] = []
    if variants_root.exists():
        for var_dir in variants_root.iterdir():
            if not var_dir.is_dir():
                continue
            manifest_path = var_dir / "variant.json"
            if not manifest_path.exists():
                continue
            try:
                variant = manifests.load_manifest(manifest_path)
            except Exception:
                continue
            if variant.get("run_id") == run_id:
                shutil.rmtree(var_dir, ignore_errors=False)
                removed_variants.append(variant.get("id", var_dir.name))

    # Remove run artifacts last, so we only lose the run once linked artifacts are cleared.
    shutil.rmtree(run_dir, ignore_errors=False)

    project_manifest = paths.get_project_dir(project_id) / "project.json"
    if project_manifest.exists():
        try:
            project = manifests.load_manifest(project_manifest)
            current_primary = project.get("primaryVariantId")
            if current_primary in removed_variants:
                replacement = None
                if variants_root.exists():
                    for var_dir in variants_root.iterdir():
                        if not var_dir.is_dir():
                            continue
                        candidate_manifest = var_dir / "variant.json"
                        if not candidate_manifest.exists():
                            continue
                        try:
                            candidate = manifests.load_manifest(candidate_manifest)
                        except Exception:
                            continue
                        replacement = candidate.get("id", var_dir.name)
                        if replacement:
                            break
                project["primaryVariantId"] = replacement
                project["updated_at"] = datetime.now().isoformat()
                manifests.save_manifest(project_manifest, project, "project")
        except Exception:
            pass

    return {
        "status": "deleted",
        "project_id": project_id,
        "run_id": run_id,
        "variants_deleted": removed_variants
    }
