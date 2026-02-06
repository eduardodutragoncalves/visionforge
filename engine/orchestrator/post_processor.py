import os
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from engine.io import paths, manifests
import logging

logger = logging.getLogger("VisionForge.PostProcessor")

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
        
        # Simple comparison logic: Maximize mAP or Accuracy
        # If no previous primary, or if this one is better
        current_primary = project.get("primaryVariantId")
        should_update = False
        
        if not current_primary:
            should_update = True
        else:
            # Load old variant to compare
            old_var_path = paths.get_variant_dir(project_id, current_primary) / "variant.json"
            if old_var_path.exists():
                old_var = manifests.load_manifest(old_var_path)
                old_metric = old_var.get("metrics", {}).get("final_map", 0)
                new_metric = final_metrics.get("final_map", 0)
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
