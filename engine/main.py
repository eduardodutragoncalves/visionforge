import asyncio
import logging
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from engine.api import endpoints, websocket, runs, run_actions, projects, datasets, variants, inference, settings
from engine.db import bootstrap_database, get_storage_mode
from engine.services.heartbeat import start_heartbeat_service
from engine.orchestrator import orchestrator
from engine.io import paths

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("VisionForge")

def _configure_file_logging():
    log_path = os.environ.get("VISIONFORGE_LOG_PATH")
    if log_path:
        log_file = Path(log_path)
        if not log_file.is_absolute():
            log_file = paths.get_workspace_root() / log_file
    else:
        log_file = paths.get_workspace_root() / "logs" / "engine.log"
    log_dir = log_file.parent if log_file.parent else None
    if log_dir:
        paths.ensure_dirs(log_dir)
    handler = logging.FileHandler(str(log_file))
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    logger.info("Logging to %s", log_file)

_configure_file_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start background tasks
    logger.info("VisionForge Engine Starting...")
    logger.info("Workspace root: %s", paths.get_workspace_root())
    logger.info("CWD: %s", os.getcwd())
    logger.info("Python executable: %s", sys.executable)
    bootstrap_result = bootstrap_database()
    logger.info(
        "SQLite ready at %s (storage_mode=%s, migrations_applied=%d)",
        bootstrap_result.db_path,
        get_storage_mode(),
        len(bootstrap_result.applied),
    )
    if bootstrap_result.applied:
        logger.info(
            "Applied migrations: %s",
            ", ".join(str(migration.version) for migration in bootstrap_result.applied),
        )
    heartbeat_task = None
    try:
        heartbeat_task = asyncio.create_task(start_heartbeat_service())
        logger.info("Heartbeat task started.")
        await orchestrator.start()
        logger.info("Orchestrator started.")
    except Exception:
        logger.exception("Startup failure")
        if heartbeat_task:
            heartbeat_task.cancel()
        raise
    
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
app.include_router(settings.router, tags=["Settings"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
