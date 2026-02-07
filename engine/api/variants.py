from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from engine.io import paths, manifests

router = APIRouter()

@router.get("/projects/{project_id}/variants")
async def list_variants(project_id: str):
    variants_root = paths.get_variants_dir(project_id)
    if not variants_root.exists():
        return []
    
    variants = []
    for var_dir in variants_root.iterdir():
        if var_dir.is_dir():
            manifest_path = var_dir / "variant.json"
            if manifest_path.exists():
                try:
                    data = manifests.load_manifest(manifest_path)
                    variants.append(data)
                except Exception:
                    pass
    return variants
    
@router.post("/variants/{variant_id}/export")
async def export_variant(variant_id: str, format: str = "onnx", project_id: str = "p1"): # project_id hack if not passed
    # We need project_id to resolve paths via Orchestrator, but Orchestrator usually runs within project context
    # API should probably require project_id in params or body
    # Using 'p1' default is risky. Let's ask user.
    # Actually, the user can pass it as query param ?project_id=...
    pass

@router.post("/projects/{project_id}/variants/{variant_id}/export")
async def export_variant_scoped(project_id: str, variant_id: str, format: str = "onnx"):
    from engine.orchestrator import orchestrator
    import uuid
    
    run_id = f"exp_{uuid.uuid4().hex[:6]}" # Ephemeral run ID
    
    await orchestrator.submit_job({
        "project_id": project_id,
        "run_id": run_id,
        "type": "export",
        "config": {
            "variant_id": variant_id,
            "format": format
        }
    })
    
    return {"status": "queued", "job_id": run_id}
