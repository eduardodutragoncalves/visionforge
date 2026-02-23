from datetime import datetime
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from engine.io import paths, manifests
from engine.orchestrator.core import orchestrator
from engine.services.inference_history import inference_history_service
from engine.services.sqlite_cache_service import sqlite_cache_service

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
    cache_key = f"api:variants:list:v1:{project_id}"
    signature = sqlite_cache_service.variants_signature(project_id)
    cached_payload = sqlite_cache_service.get_payload(cache_key, signature)
    if isinstance(cached_payload, list):
        return cached_payload

    variants_root = paths.get_variants_dir(project_id)
    if not variants_root.exists():
        sqlite_cache_service.set_payload(cache_key, signature, [])
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
    payload = sorted(variants, key=lambda item: item.get("created_at", ""), reverse=True)
    sqlite_cache_service.set_payload(cache_key, signature, payload)
    return payload
    
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


@router.post("/projects/{project_id}/deployments/{job_id}/cancel")
async def cancel_deployment(project_id: str, job_id: str):
    result = await orchestrator.cancel_export_job(project_id, job_id)
    status = str(result.get("status") or "")
    if status == "not_found":
        raise HTTPException(status_code=404, detail="Deployment job not found")
    if status == "invalid":
        raise HTTPException(status_code=400, detail=result.get("message", "Invalid request"))
    return result


@router.delete("/projects/{project_id}/variants/{variant_id}/deployments/{deployment_id}")
async def delete_deployment(project_id: str, variant_id: str, deployment_id: str):
    variant_dir = paths.get_variant_dir(project_id, variant_id)
    if not variant_dir.exists():
        raise HTTPException(status_code=404, detail="Variant not found")

    deploy_root = variant_dir / "deploy"
    if not deploy_root.exists():
        raise HTTPException(status_code=404, detail="Deployment not found")

    variant_root_resolved = variant_dir.resolve()
    matched = None

    for target_dir in deploy_root.iterdir():
        if not target_dir.is_dir():
            continue
        for package_dir in target_dir.iterdir():
            if not package_dir.is_dir():
                continue
            manifest_path = package_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                deployment = manifests.load_manifest(manifest_path)
            except Exception:
                continue
            if str(deployment.get("id") or "") != deployment_id:
                continue
            matched = {
                "target_dir": target_dir,
                "package_dir": package_dir,
                "deployment": deployment,
            }
            break
        if matched:
            break

    if not matched:
        raise HTTPException(status_code=404, detail="Deployment not found")

    package_dir = matched["package_dir"]
    deployment = matched["deployment"]

    declared_package_dir = deployment.get("package_dir")
    if declared_package_dir:
        candidate = Path(str(declared_package_dir)).expanduser()
        if candidate.exists():
            try:
                if candidate.resolve().is_relative_to(variant_root_resolved):
                    package_dir = candidate.resolve()
            except Exception:
                pass

    if not package_dir.exists():
        # Already removed, continue with zip cleanup.
        pass
    else:
        try:
            if not package_dir.resolve().is_relative_to(variant_root_resolved):
                raise HTTPException(status_code=400, detail="Unsafe deployment path")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail="Unsafe deployment path")
        shutil.rmtree(package_dir, ignore_errors=True)

    zip_candidates = []
    package_zip_path = deployment.get("package_zip_path")
    if package_zip_path:
        zip_candidates.append(Path(str(package_zip_path)).expanduser())
    zip_candidates.append(package_dir.with_suffix(".zip"))

    for zip_path in zip_candidates:
        try:
            resolved = zip_path.resolve()
            if not resolved.is_relative_to(variant_root_resolved):
                continue
            if resolved.exists():
                resolved.unlink()
        except Exception:
            continue

    target_dir = matched["target_dir"]
    try:
        if target_dir.exists() and not any(target_dir.iterdir()):
            target_dir.rmdir()
    except Exception:
        pass

    try:
        if deploy_root.exists() and not any(deploy_root.iterdir()):
            deploy_root.rmdir()
    except Exception:
        pass

    return {"status": "deleted", "deployment_id": deployment_id, "variant_id": variant_id}
