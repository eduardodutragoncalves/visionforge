import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from engine.api import endpoints, websocket, runs, run_actions, projects, datasets, variants, inference
from engine.services.heartbeat import start_heartbeat_service
from engine.orchestrator import orchestrator

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("VisionForge")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start background tasks
    logger.info("VisionForge Engine Starting...")
    heartbeat_task = asyncio.create_task(start_heartbeat_service())
    await orchestrator.start()
    
    yield
    
    # Shutdown: Cancel tasks
    logger.info("VisionForge Engine Shutting Down...")
    await orchestrator.stop()
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title="VisionForge Engine",
    version="0.1.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to Electron's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(endpoints.router, tags=["System"])
app.include_router(websocket.router, tags=["Realtime"])
app.include_router(projects.router, tags=["Projects"])
app.include_router(datasets.router, tags=["Datasets"])
app.include_router(runs.router, tags=["Runs"])
app.include_router(run_actions.router, tags=["Runs"])
app.include_router(variants.router, tags=["Variants"])
app.include_router(inference.router, tags=["Inference"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
