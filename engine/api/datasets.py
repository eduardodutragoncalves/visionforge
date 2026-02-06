from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

from engine.services.dataset_service import dataset_service

router = APIRouter()

class ImportDatasetRequest(BaseModel):
    path: str

class ScanDatasetRequest(BaseModel):
    path: str

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
