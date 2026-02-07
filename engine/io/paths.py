import os
from pathlib import Path

def _resolve_workspace_root() -> Path:
    env_value = os.environ.get("VISIONFORGE_WORKSPACE")
    if env_value:
        return Path(env_value).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "workspace").resolve()

# Base workspace path (deterministic, not tied to current working directory)
WORKSPACE_ROOT = _resolve_workspace_root()

def get_workspace_root() -> Path:
    return WORKSPACE_ROOT

def get_projects_root() -> Path:
    return get_workspace_root() / "projects"

def get_project_dir(project_id: str) -> Path:
    return get_projects_root() / project_id

def get_project_dataset_dir(project_id: str) -> Path:
    return get_project_dir(project_id) / "dataset"

def get_project_dataset_images_dir(project_id: str) -> Path:
    return get_project_dataset_dir(project_id) / "images"

def get_project_dataset_frames_dir(project_id: str) -> Path:
    return get_project_dataset_dir(project_id) / "frames"

def get_runs_dir(project_id: str) -> Path:
    return get_project_dir(project_id) / "runs"

def get_run_dir(project_id: str, run_id: str) -> Path:
    return get_runs_dir(project_id) / run_id

def get_variants_dir(project_id: str) -> Path:
    return get_project_dir(project_id) / "variants"

def get_variant_dir(project_id: str, variant_id: str) -> Path:
    return get_variants_dir(project_id) / variant_id

def get_dataset_manifest_path(project_id: str) -> Path:
    return get_project_dir(project_id) / "dataset.json"

def ensure_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)
