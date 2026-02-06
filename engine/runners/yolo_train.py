import sys
import os
import json
import time
import argparse
import yaml
import traceback
from pathlib import Path
from datetime import datetime

# Redirect stdout to ensure we catch everything, though we will print specifically for events
# We need to ensure we don't mix up Ultralytics logs with our JSON events if possible
# Ultralytics logs to stdout/stderr. We will use a unique prefix VF_EVENT:

def emit_event(event_type, payload):
    envelope = {
        "type": event_type,
        "payload": payload,
        "timestamp": time.time()
    }
    print(f"VF_EVENT: {json.dumps(envelope)}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    emit_event("train/started", {
        "run_id": args.run_id,
        "config": vars(args)
    })

    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        emit_event("train/failed", {"error": "Ultralytics/Torch not installed"})
        sys.exit(1)

    # 1. Prepare Dataset YAML
    # We assume 'dataset-path' has images/train, images/val or similar structure
    # For this implementation, we'll create a simple data.yaml pointing validation to train if val doesn't exist ( MVP )
    
    dataset_path = Path(args.dataset_path)
    # Simple auto-detection of structure
    train_path = dataset_path / "train" / "images"
    val_path = dataset_path / "valid" / "images"
    
    # Fallback/Simplification for MVP: Just use the root or search
    if not train_path.exists():
        train_path = dataset_path # Assume flat or custom
    if not val_path.exists():
        val_path = train_path 

    data_yaml_content = {
        "path": str(dataset_path.absolute()),
        "train": str(train_path.absolute()),
        "val": str(val_path.absolute()),
        "names": {0: "object"}, # Mock names if classes.txt not found, ideally read classes.txt
        "nc": 1
    }
    
    # Try to find classes.txt to populate names properly
    classes_file = dataset_path / "classes.txt" # Common format
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            data_yaml_content["names"] = {i: name for i, name in enumerate(lines)}
            data_yaml_content["nc"] = len(lines)

    data_yaml_path = run_dir / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml_content, f)

    # 2. Callback Definition
    def on_train_epoch_end(trainer):
        """Called at end of training epoch"""
        metrics = trainer.metrics
        epoch = trainer.epoch + 1
        
        # Extract relevant metrics
        # keys like 'metrics/mAP50(B)', 'val/box_loss', etc.
        # Ultralytics 8 structure:
        
        parsed_metrics = {
            "epoch": epoch,
            "total_epochs": args.epochs,
            "loss": float(trainer.loss_items[0]) if hasattr(trainer, "loss_items") else 0.0, # Approximate box loss
            # Detailed metrics from trainer.metrics which is populated after validation
            # For intermediate, we might only have loss
        }
        
        # If validation ran
        if hasattr(trainer, "metrics") and trainer.metrics:
             parsed_metrics.update({
                 "map50": trainer.metrics.get("metrics/mAP50(B)", 0),
                 "map5095": trainer.metrics.get("metrics/mAP50-95(B)", 0),
                 "precision": trainer.metrics.get("metrics/precision(B)", 0),
                 "recall": trainer.metrics.get("metrics/recall(B)", 0)
             })

        emit_event("train/metrics", parsed_metrics)
        
        # Log entry
        emit_event("train/logs", {
            "message": f"Epoch {epoch} completed. Loss: {parsed_metrics['loss']:.4f}",
            "level": "INFO"
        })

    # 3. Training
    try:
        model = YOLO(args.model)
        
        # Register hooks
        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        print("Starting Ultralytics Training...", flush=True)
        
        results = model.train(
            data=str(data_yaml_path),
            project=str(run_dir),
            name="yolo_run",
            epochs=args.epochs,
            batch=args.batch,
            exist_ok=True,
            verbose=True
        )
        
        # Move/Copy specific checkpoints if needed, or rely on 'yolo_run' folder
        # Ultralytics saves to project/name/weights
        weights_dir = run_dir / "yolo_run" / "weights"
        if weights_dir.exists():
            import shutil
            if (weights_dir / "best.pt").exists():
                shutil.copy(weights_dir / "best.pt", checkpoints_dir / "best.pt")
            if (weights_dir / "last.pt").exists():
                shutil.copy(weights_dir / "last.pt", checkpoints_dir / "last.pt")

        emit_event("train/completed", {
            "run_id": args.run_id,
            "final_map": results.box.map50 if hasattr(results, 'box') else 0
        })

    except torch.cuda.OutOfMemoryError:
        emit_event("train/oom_detected", {
            "batch_size": args.batch,
            "suggtestions": [
                "Reduce batch size (currently " + str(args.batch) + ")",
                "Reduce image size (imgsz)",
                "Enable Automatic Mixed Precision (AMP)"
            ]
        })
        emit_event("train/logs", {"message": "CUDA OOM Detected!", "level": "ERROR"})
        emit_event("train/failed", {"error": "CUDA Out Of Memory"})
        sys.exit(1)
        
    except Exception as e:
        traceback.print_exc()
        emit_event("train/failed", {"error": str(e)})
        sys.exit(1)

if __name__ == "__main__":
    main()
