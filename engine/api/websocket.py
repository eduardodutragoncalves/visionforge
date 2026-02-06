from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from engine.core.events import manager
import logging

logger = logging.getLogger("VisionForge.WS")

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Send welcome message
    await manager.broadcast("engine/online", {
        "message": "VisionForge Engine Connected",
        "ready": True
    })
    
    try:
        while True:
            # Keep connection open and listen for client messages (if any)
            # In the future, this could handle client commands via WS
            data = await websocket.receive_text()
            logger.info(f"Received from client: {data}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
