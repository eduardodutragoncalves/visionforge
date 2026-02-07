from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import mimetypes
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional

from engine.services.dataset_service import dataset_service
from engine.services.preclassify_service import preclassify_service
from engine.io import paths

router = APIRouter()

class ImportDatasetRequest(BaseModel):
    path: str

class ScanDatasetRequest(BaseModel):
    path: str

class DatasetConfigRequest(BaseModel):
    task: Optional[str] = None
    format: Optional[str] = None

class DatasetClassInput(BaseModel):
    id: Optional[int] = None
    name: str
    color: Optional[str] = None

class UpdateClassesRequest(BaseModel):
    classes: List[DatasetClassInput]

class UpdateLabelsRequest(BaseModel):
    labels: List[int]

class BatchUpdateLabelsRequest(BaseModel):
    image_ids: List[str]
    labels: List[int]

class BoundingBox(BaseModel):
    class_id: int
    x: float
    y: float
    w: float
    h: float

class UpdateBoxesRequest(BaseModel):
    boxes: List[BoundingBox]

class IngestDirectoriesRequest(BaseModel):
    paths: List[str]

class IngestFilesRequest(BaseModel):
    paths: List[str]

class IngestVideoRequest(BaseModel):
    path: str
    frame_interval: int
    max_frames: Optional[int] = None

class DeleteImagesRequest(BaseModel):
    image_ids: List[str]

class PreclassifyRequest(BaseModel):
    image_ids: List[str]
    model: Optional[str] = "yolov8n-cls.pt"
    topk: Optional[int] = 3

@router.get("/projects/{project_id}/dataset")
async def get_dataset(project_id: str):
    try:
        return dataset_service.get_dataset(project_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/projects/{project_id}/dataset/images")
async def list_images(project_id: str, limit: int = 200, offset: int = 0):
    try:
        return dataset_service.list_images(project_id, limit=limit, offset=offset)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/projects/{project_id}/dataset/images/{image_id}/file")
async def get_image_file(project_id: str, image_id: str):
    try:
        image_path = dataset_service.get_image_path(project_id, image_id)
        file_path = Path(image_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Image not found on disk")
        dataset_dir = paths.get_project_dataset_dir(project_id).resolve()
        resolved_path = file_path.resolve()
        if not str(resolved_path).startswith(str(dataset_dir)):
            raise HTTPException(status_code=403, detail="Image path not allowed")
        media_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        return FileResponse(str(file_path), media_type=media_type, filename=file_path.name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/projects/{project_id}/dataset/images/{image_id}")
async def delete_image(project_id: str, image_id: str):
    try:
        return dataset_service.delete_image(project_id, image_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/projects/{project_id}/dataset/images/delete")
async def delete_images(project_id: str, request: DeleteImagesRequest):
    try:
        return dataset_service.delete_images(project_id, request.image_ids)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/projects/{project_id}/dataset")
async def update_dataset_config(project_id: str, request: DatasetConfigRequest):
    try:
        return dataset_service.update_dataset_config(project_id, request.dict(exclude_none=True))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/projects/{project_id}/dataset/classes")
async def update_dataset_classes(project_id: str, request: UpdateClassesRequest):
    try:
        payload = [cls.dict(exclude_none=True) for cls in request.classes]
        return dataset_service.update_classes(project_id, payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/projects/{project_id}/dataset/images/{image_id}/labels")
async def update_image_labels(project_id: str, image_id: str, request: UpdateLabelsRequest):
    try:
        return dataset_service.update_image_labels(project_id, image_id, request.labels)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/projects/{project_id}/dataset/images/labels")
async def update_images_labels(project_id: str, request: BatchUpdateLabelsRequest):
    try:
        return dataset_service.update_images_labels(project_id, request.image_ids, request.labels)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/projects/{project_id}/dataset/images/{image_id}/boxes")
async def update_image_boxes(project_id: str, image_id: str, request: UpdateBoxesRequest):
    try:
        payload = [box.dict() for box in request.boxes]
        return dataset_service.update_image_boxes(project_id, image_id, payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/projects/{project_id}/dataset/ingest/directories")
async def ingest_directories(project_id: str, request: IngestDirectoriesRequest):
    try:
        return dataset_service.ingest_directories(project_id, request.paths)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/projects/{project_id}/dataset/ingest/files")
async def ingest_files(project_id: str, request: IngestFilesRequest):
    try:
        return dataset_service.ingest_files(project_id, request.paths)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/projects/{project_id}/dataset/ingest/video")
async def ingest_video(project_id: str, request: IngestVideoRequest):
    try:
        return dataset_service.ingest_video(project_id, request.path, request.frame_interval, request.max_frames)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/projects/{project_id}/dataset/validate")
async def validate_dataset(project_id: str):
    try:
        return await dataset_service.validate_current_dataset(project_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/projects/{project_id}/dataset/preclassify")
async def preclassify_dataset(project_id: str, request: PreclassifyRequest):
    try:
        dataset = dataset_service.get_dataset(project_id)
        images = dataset.get("images", [])
        by_id = {img.get("id"): img.get("path") for img in images}
        targets = [
            (image_id, by_id[image_id])
            for image_id in request.image_ids
            if image_id in by_id
        ]
        results = preclassify_service.preclassify_images(
            targets,
            request.model or "yolov8n-cls.pt",
            request.topk or 3
        )
        return {"model": request.model, "results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/projects/{project_id}/dataset/import")
async def import_dataset(project_id: str, request: ImportDatasetRequest):
    try:
        result = await dataset_service.import_dataset(project_id, request.path)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/projects/{project_id}/dataset/scan")
async def scan_dataset(project_id: str, request: ScanDatasetRequest, background_tasks: BackgroundTasks):
    """
    Starts async scan and then validation immediately for MVP flow.
    """
    async def _sequence():
        report, files = await dataset_service.scan_dataset(project_id, request.path)
        await dataset_service.validate_dataset(project_id, files)

    # Fast API Background tasks run in threadpool usually or post-response
    # But since we need async broadcast, we use create_task loop if not using background_tasks properly with async
    # background_tasks.add_task(_sequence) # This might require _sequence to be sync or properly handled
    
    # Better to just schedule on event loop for this simple engine
    import asyncio
    asyncio.create_task(_sequence())
    
    return {"status": "scan_started", "message": "Progress will be sent via WebSocket"}
