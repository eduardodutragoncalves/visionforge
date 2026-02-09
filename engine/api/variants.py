from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from engine.io import paths, manifests
from engine.orchestrator.core import orchestrator
from engine.services.inference_history import inference_history_service

router = APIRouter()

TARGET_FORMATS = {
    "edge_jetson": "onnx",
    "edge_raspberry_pi": "onnx",
    "edge_orange_pi": "onnx",
    "edge_generic": "onnx",
    "desktop_linux": "onnx",
    "desktop_windows": "onnx",
    "desktop_macos": "onnx"
}

class DeployRequest(BaseModel):
    target_id: str
    format: Optional[str] = None
    runtime: Optional[str] = "local"
    runtime_config: Optional[Dict[str, Any]] = None

@router.get("/projects/{project_id}/variants")
async def list_variants(project_id: str):
    variants_root = paths.get_variants_dir(project_id)
    if not variants_root.exists():
        return []

    validation_summary = inference_history_service.summarize_validation_by_variant(project_id)
    total_validation_images = int(validation_summary.get("total_images", 0))
    validation_by_variant = validation_summary.get("by_variant", {}) or {}

    variants = []
    for var_dir in variants_root.iterdir():
        if var_dir.is_dir():
            manifest_path = var_dir / "variant.json"
            if manifest_path.exists():
                try:
                    data = manifests.load_manifest(manifest_path)
                    deployments: List[Dict[str, Any]] = []
                    deploy_root = var_dir / "deploy"
                    if deploy_root.exists():
                        for target_dir in deploy_root.iterdir():
                            if not target_dir.is_dir():
                                continue
                            for package_dir in target_dir.iterdir():
                                if not package_dir.is_dir():
                                    continue
                                deployment_manifest = package_dir / "manifest.json"
                                if not deployment_manifest.exists():
                                    continue
                                try:
                                    deployment = manifests.load_manifest(deployment_manifest)
                                    deployments.append(deployment)
                                except Exception:
                                    continue
                    deployments.sort(key=lambda item: item.get("created_at", ""), reverse=True)
                    data["deployments"] = deployments

                    exports_dir = var_dir / "exports"
                    onnx_candidates = [
                        exports_dir / "model.onnx",
                        var_dir / "model.onnx"
                    ]
                    for deployment in deployments:
                        artifact_path = deployment.get("artifact_path")
                        if artifact_path:
                            candidate = Path(artifact_path)
                            if candidate.exists() and candidate.suffix.lower() == ".onnx":
                                onnx_candidates.append(candidate)
                    onnx_path = next((candidate for candidate in onnx_candidates if candidate.exists()), None)
                    data["onnx_exported"] = bool(onnx_path)
                    data["exports"] = {
                        "onnx": bool(onnx_path),
                        "onnx_path": str(onnx_path) if onnx_path else None
                    }
                    data["validation"] = validation_by_variant.get(
                        data.get("id"),
                        {
                            "total_images": total_validation_images,
                            "tested_images": 0,
                            "untested_images": total_validation_images,
                            "coverage_pct": 0.0,
                            "correct": 0,
                            "incorrect": 0,
                            "pending": 0,
                            "labeled": 0,
                            "accuracy_pct": None,
                            "last_tested_at": None
                        }
                    )
                    variants.append(data)
                except Exception:
                    pass
    return sorted(variants, key=lambda item: item.get("created_at", ""), reverse=True)
    
@router.post("/variants/{variant_id}/export")
async def export_variant(variant_id: str, format: str = "onnx", project_id: str = "p1"): # project_id hack if not passed
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id is required")
    return await export_variant_scoped(project_id=project_id, variant_id=variant_id, format=format)

@router.post("/projects/{project_id}/variants/{variant_id}/export")
async def export_variant_scoped(project_id: str, variant_id: str, format: str = "onnx"):
    import uuid
    run_id = f"exp_{uuid.uuid4().hex[:6]}"
    await orchestrator.submit_job(project_id, run_id, {
        "type": "export",
        "variant_id": variant_id,
        "target_id": "edge_generic",
        "format": format
    })
    return {"status": "queued", "job_id": run_id, "target_id": "edge_generic"}

@router.post("/projects/{project_id}/variants/{variant_id}/deploy")
async def deploy_variant(project_id: str, variant_id: str, request: DeployRequest):
    import uuid

    target_id = (request.target_id or "").strip()
    if target_id not in TARGET_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported target_id '{target_id}'")

    variant_dir = paths.get_variant_dir(project_id, variant_id)
    if not variant_dir.exists():
        raise HTTPException(status_code=404, detail="Variant not found")

    resolved_format = (request.format or TARGET_FORMATS[target_id]).strip().lower()
    if not resolved_format:
        raise HTTPException(status_code=400, detail="Export format could not be resolved")

    run_id = f"exp_{uuid.uuid4().hex[:6]}"
    await orchestrator.submit_job(project_id, run_id, {
        "type": "export",
        "variant_id": variant_id,
        "target_id": target_id,
        "format": resolved_format,
        "runtime": request.runtime or "local",
        "runtime_config": request.runtime_config or {},
        "requested_at": datetime.now().isoformat()
    })
    return {
        "status": "queued",
        "job_id": run_id,
        "target_id": target_id,
        "format": resolved_format
    }
