import asyncio
import logging
import time
from engine.core.events import manager

logger = logging.getLogger("VisionForge.Heartbeat")

async def start_heartbeat_service():
    logger.info("Starting Heartbeat Service...")
    start_time = time.time()
    
    try:
        while True:
            uptime = time.time() - start_time
            await manager.broadcast("engine/heartbeat", {
                "alive": True,
                "uptime_seconds": round(uptime, 2),
                "timestamp": time.time()
            })
            await asyncio.sleep(2)
    except asyncio.CancelledError:
        logger.info("Heartbeat Service Stopping...")
