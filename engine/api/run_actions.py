from fastapi import APIRouter
from engine.core.events import manager

router = APIRouter()

@router.post("/runs/{run_id}/stop")
async def stop_run(run_id: str):
    await manager.broadcast("train/logs", {"level": "WARNING", "message": f"Stop requested for {run_id}"})
    return {
        "status": "not_supported",
        "detail": "Stop signal is logged, but process interruption is not implemented yet."
    }
