import os
from pathlib import Path

# Base workspace path (relative to engine execution or absolute)
# In production, this might be configurable via env path
WORKSPACE_ROOT = Path("workspace").absolute()

def get_workspace_root() -> Path:
    return WORKSPACE_ROOT

def get_projects_root() -> Path:
    return get_workspace_root() / "projects"

def get_project_dir(project_id: str) -> Path:
    return get_projects_root() / project_id

def get_runs_dir(project_id: str) -> Path:
    return get_project_dir(project_id) / "runs"

def get_run_dir(project_id: str, run_id: str) -> Path:
    return get_runs_dir(project_id) / run_id

def get_variants_dir(project_id: str) -> Path:
    return get_project_dir(project_id) / "variants"

def get_variant_dir(project_id: str, variant_id: str) -> Path:
    return get_variants_dir(project_id) / variant_id

def ensure_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)
