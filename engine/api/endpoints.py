from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

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
    """Mock capabilities for the engine"""
    return {
        "gpu_available": True,
        "gpu_name": "NVIDIA GeForce RTX 4090 (Mock)",
        "frameworks": ["yolo", "tensorflow", "pytorch"],
        "version": "0.1.0"
    }
