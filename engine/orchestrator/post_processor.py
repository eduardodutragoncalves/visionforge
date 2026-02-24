import shutil
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from engine.io import paths, manifests
import logging

logger = logging.getLogger("VisionForge.PostProcessor")

def _to_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed

def _resolve_task(config: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    raw_task = str(config.get("task") or "").strip().lower()
    if raw_task in {"classification", "classify"}:
        return "classification"
    if raw_task in {"detection", "detect"}:
        return "detection"

    model_name = str(config.get("base_model") or config.get("model") or "").lower()
    if "-cls" in model_name:
        return "classification"

    map_metric = _to_float(metrics.get("final_map"))
    if map_metric is None:
        map_metric = _to_float(metrics.get("map50"))
    acc_metric = _to_float(metrics.get("accuracy"))
    if map_metric is not None and acc_metric is None:
        return "detection"
    if acc_metric is not None and map_metric is None:
        return "classification"
    return "detection"

def _score_variant(task: str, metrics: Dict[str, Any]) -> float:
    if task == "classification":
        score = _to_float(metrics.get("accuracy"))
        if score is not None:
            return score
        return _to_float(metrics.get("final_map")) or _to_float(metrics.get("map50")) or 0.0
    return _to_float(metrics.get("final_map")) or _to_float(metrics.get("map50")) or _to_float(metrics.get("accuracy")) or 0.0

def handle_training_completion(project_id: str, run_id: str, final_metrics: dict, config: dict):
    """
    Creates a variant, aggregates artifacts, and updates project best-model pointer.
    """
    logger.info(f"Processing completion for Run {run_id}")
    
    run_dir = paths.get_run_dir(project_id, run_id)
    variant_id = f"var_{uuid.uuid4().hex[:8]}"
    variant_dir = paths.get_variant_dir(project_id, variant_id)
    paths.ensure_dirs(variant_dir)
    
    # 1. Archive Checkpoints
    # We copy best.pt to the variant folder to establish it as a "Released" model version
    source_best = run_dir / "checkpoints" / "best.pt"
    if not source_best.exists():
        # Fallback for dummy runner or failed checkpointing
        logger.warning("No best.pt found, skipping artifact copy")
    else:
        shutil.copy(source_best, variant_dir / "model.pt")
        
    # 2. Create Variant Manifest
    variant_data = {
        "id": variant_id,
        "project_id": project_id,
        "name": f"{config.get('base_model', 'Model')} ({run_id[:6]})",
        "base_model": config.get("base_model", "unknown"),
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "hyperparameters": config,
        "metrics": final_metrics
    }
    
    manifests.save_manifest(variant_dir / "variant.json", variant_data, "variant")
    
    # 3. Update Project (Best Model Logic)
    project_dir = paths.get_project_dir(project_id)
    project_manifest = project_dir / "project.json"
    
    if project_manifest.exists():
        project = manifests.load_manifest(project_manifest)
        
        task = _resolve_task(config, final_metrics)
        current_primary = project.get("primaryVariantId")
        should_update = False
        
        if not current_primary:
            should_update = True
        else:
            # Load old variant to compare
            old_var_path = paths.get_variant_dir(project_id, current_primary) / "variant.json"
            if old_var_path.exists():
                old_var = manifests.load_manifest(old_var_path)
                old_metrics = old_var.get("metrics", {}) if isinstance(old_var.get("metrics"), dict) else {}
                old_metric = _score_variant(task, old_metrics)
                new_metric = _score_variant(task, final_metrics)
                if new_metric > old_metric:
                    should_update = True
            else:
                should_update = True # Broken link
        
        if should_update:
            project["primaryVariantId"] = variant_id
            project["updated_at"] = datetime.now().isoformat()
            manifests.save_manifest(project_manifest, project, "project")
            logger.info(f"Updated Project Primary Variant to {variant_id}")

    return variant_id
