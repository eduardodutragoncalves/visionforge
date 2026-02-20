from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from typing import Optional
import mimetypes
from pathlib import Path
from pydantic import BaseModel
from typing import List
from engine.io import paths
from engine.services.inference_service import inference_service
from engine.services.inference_history import inference_history_service

router = APIRouter()


class FeedbackUpdateInput(BaseModel):
    variant_id: str
    verdict: Optional[str] = None


class FeedbackBatchRequest(BaseModel):
    updates: List[FeedbackUpdateInput]

@router.post("/variants/{variant_id}/infer")
async def run_infer(
    variant_id: str, 
    project_id: str = Form(...), 
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    agnostic_nms: bool = Form(False),
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
        result = inference_service.run_inference(
            variant_dir,
            content,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            agnostic_nms=agnostic_nms
        )
        history_item = inference_history_service.record_inference(
            project_id,
            variant_id,
            file.filename or "upload.jpg",
            content,
            result
        )
        return {**result, "history_item": history_item}
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model weights (model.pt) not found for this variant.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/projects/{project_id}/variants/{variant_id}/infer/history/{image_id}")
async def run_infer_history(
    project_id: str,
    variant_id: str,
    image_id: str,
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    agnostic_nms: bool = Form(False)
):
    variant_dir = paths.get_variant_dir(project_id, variant_id)
    if not variant_dir.exists():
        raise HTTPException(status_code=404, detail=f"Variant {variant_id} not found in project {project_id}")
    try:
        image_path = inference_history_service.get_image_path(project_id, image_id)
        content = image_path.read_bytes()
        result = inference_service.run_inference(
            variant_dir,
            content,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            agnostic_nms=agnostic_nms
        )
        history_item = inference_history_service.record_inference(
            project_id,
            variant_id,
            image_path.name,
            content,
            result
        )
        return {**result, "history_item": history_item}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects/{project_id}/inference/history")
async def list_inference_history(project_id: str):
    try:
        return inference_history_service.list_history(project_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/projects/{project_id}/inference/history/images/{image_id}")
async def get_inference_history_image(project_id: str, image_id: str):
    try:
        image_path = inference_history_service.get_image_path(project_id, image_id)
        media_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
        return FileResponse(str(image_path), media_type=media_type, filename=image_path.name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/projects/{project_id}/inference/history/{image_id}/feedback")
async def set_inference_feedback(
    project_id: str,
    image_id: str,
    variant_id: str = Form(...),
    verdict: Optional[str] = Form(None)
):
    try:
        return inference_history_service.set_feedback(project_id, image_id, variant_id, verdict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/projects/{project_id}/inference/history/{image_id}/feedback/batch")
async def set_inference_feedback_batch(
    project_id: str,
    image_id: str,
    request: FeedbackBatchRequest
):
    try:
        updates = {item.variant_id: item.verdict for item in request.updates}
        return inference_history_service.set_feedback_batch(project_id, image_id, updates)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
