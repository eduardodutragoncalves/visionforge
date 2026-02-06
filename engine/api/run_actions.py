from fastapi import APIRouter
from engine.core.events import manager
from engine.orchestrator import orchestrator

router = APIRouter()

@router.post("/runs/{run_id}/stop")
async def stop_run(run_id: str):
    """
    Mock stop command. Actual implementation would kill the subprocess via orchestrator.
    """
    # In a real app we'd find the job in orchestrator and kill it.
    await manager.broadcast("train/logs", {"level": "WARNING", "message": f"Stop requested for {run_id} (Not implemented in mock yet)"})
    return {"status": "stop_requested"}
