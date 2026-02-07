import asyncio
import logging
import time
from engine.core.events import manager
from engine.services.gpu_stats import get_gpu_stats

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
            stats = await asyncio.to_thread(get_gpu_stats)
            await manager.broadcast("engine/gpu_stats", {
                "gpus": stats,
                "timestamp": time.time()
            })
            await asyncio.sleep(2)
    except asyncio.CancelledError:
        logger.info("Heartbeat Service Stopping...")
