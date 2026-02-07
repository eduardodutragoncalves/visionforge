import json
import logging
from typing import List, Dict, Any
from fastapi import WebSocket

logger = logging.getLogger("VisionForge.Events")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("Client disconnected")

    async def broadcast(self, event: str, payload: Dict[str, Any]):
        """
        Broadcasts a message to all connected clients.
        Format: {"event": str, "data": dict}
        """
        message = {
            "event": event,
            "data": payload
        }
        json_msg = json.dumps(message)
        
        # Iterate over copy to avoid modification issues during iteration if disconnect happens
        for connection in list(self.active_connections):
            try:
                await connection.send_text(json_msg)
            except Exception as e:
                logger.error(f"Error sending to websocket: {e}")
                self.disconnect(connection)

manager = ConnectionManager()
