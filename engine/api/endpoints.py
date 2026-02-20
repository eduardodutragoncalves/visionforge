from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from engine.services.gpu_stats import get_gpu_stats

router = APIRouter()

class CapabilityResponse(BaseModel):
    gpu_available: bool
    gpu_name: str
    frameworks: List[str]
    version: str

@router.get("/health")
async def get_health():
    """Simple health check"""
    return {"status": "ok"}

@router.get("/capabilities", response_model=CapabilityResponse)
async def get_capabilities():
    """Runtime capabilities detected from the current engine host."""
    gpu_stats = get_gpu_stats()
    primary_gpu = gpu_stats[0] if gpu_stats else None
    return {
        "gpu_available": len(gpu_stats) > 0,
        "gpu_name": (primary_gpu or {}).get("name", "No GPU detected"),
        "frameworks": ["ultralytics"],
        "version": "0.1.0"
    }
