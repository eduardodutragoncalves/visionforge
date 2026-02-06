from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
from engine.io import paths
from engine.services.inference_service import inference_service

router = APIRouter()

@router.post("/variants/{variant_id}/infer")
async def run_infer(
    variant_id: str, 
    project_id: str = Form(...), 
    conf_threshold: float = Form(0.25),
    file: UploadFile = File(...)
):
    """
    Run inference on an uploaded image using a specific variant model.
    """
    
    # Resolve Path
    # Note: We need project_id to locate the variant in the current structure
    variant_dir = paths.get_variant_dir(project_id, variant_id)
    
    if not variant_dir.exists():
        raise HTTPException(status_code=404, detail=f"Variant {variant_id} not found in project {project_id}")

    try:
        content = await file.read()
        result = inference_service.run_inference(variant_dir, content, conf_threshold)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model weights (model.pt) not found for this variant.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
