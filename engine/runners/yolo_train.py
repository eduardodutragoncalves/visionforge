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
    parser.add_argument("--dataset-manifest", required=False)
    parser.add_argument("--task", type=str, default="detect")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    task = (args.task or "detect").lower()
    if task == "classification":
        task = "classify"

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

    # 1. Prepare Dataset
    dataset_path = Path(args.dataset_path)
    data_yaml_path = run_dir / "data.yaml"

    def build_classification_dataset(manifest_path: Path, output_dir: Path) -> Path:
        if not manifest_path.exists():
            raise ValueError("Dataset manifest not found for classification")
        with open(manifest_path, "r") as f:
            dataset = json.load(f)
        classes = dataset.get("classes", [])
        images = dataset.get("images", [])
        if not classes:
            raise ValueError("No classes found in dataset")
        class_map = {cls["id"]: cls["name"] for cls in classes}
        labeled_images = []
        for image in images:
            labels = image.get("labels", [])
            if not labels:
                continue
            label_id = labels[0]
            if label_id not in class_map:
                continue
            labeled_images.append((image, label_id))
        if not labeled_images:
            raise ValueError("No labeled images found for classification")

        labeled_images.sort(key=lambda item: item[0].get("id", ""))
        split_index = max(1, int(len(labeled_images) * 0.8))
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, (image, label_id) in enumerate(labeled_images):
            split = "train" if idx < split_index else "val"
            class_name = class_map[label_id]
            target_dir = output_dir / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            source_path = Path(image["path"])
            suffix = source_path.suffix or ".jpg"
            filename = f"{image.get('id', source_path.stem)}{suffix}"
            target_path = target_dir / filename
            if not target_path.exists():
                try:
                    os.link(source_path, target_path)
                except OSError:
                    import shutil
                    shutil.copy2(source_path, target_path)
        return output_dir

    if task == "classify":
        classification_root = dataset_path
        train_dir = classification_root / "train"
        if not train_dir.exists():
            manifest_path = Path(args.dataset_manifest) if args.dataset_manifest else None
            classification_root = build_classification_dataset(manifest_path, run_dir / "cls_dataset")
        data_root = classification_root
    else:
        # Detection dataset YAML
        train_path = dataset_path / "train" / "images"
        val_path = dataset_path / "valid" / "images"
        if not train_path.exists():
            train_path = dataset_path
        if not val_path.exists():
            val_path = train_path

        data_yaml_content = {
            "path": str(dataset_path.absolute()),
            "train": str(train_path.absolute()),
            "val": str(val_path.absolute()),
            "names": {0: "object"},
            "nc": 1
        }

        classes_file = dataset_path / "classes.txt"
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                data_yaml_content["names"] = {i: name for i, name in enumerate(lines)}
                data_yaml_content["nc"] = len(lines)

        with open(data_yaml_path, "w") as f:
            yaml.dump(data_yaml_content, f)

    # 2. Callback Definition
    def _safe_loss_value(trainer):
        loss_items = getattr(trainer, "loss_items", None)
        if loss_items is None:
            return None
        if isinstance(loss_items, torch.Tensor):
            if loss_items.ndim == 0:
                return float(loss_items.item())
            flat = loss_items.detach().cpu().flatten()
            return float(flat[0].item()) if flat.numel() > 0 else None
        if isinstance(loss_items, (list, tuple)):
            if not loss_items:
                return None
            first = loss_items[0]
            return float(first.item()) if isinstance(first, torch.Tensor) else float(first)
        try:
            return float(loss_items)
        except Exception:
            return None

    def on_train_epoch_end(trainer):
        """Called at end of training epoch"""
        metrics = trainer.metrics
        epoch = trainer.epoch + 1
        
        # Extract relevant metrics
        # keys like 'metrics/mAP50(B)', 'val/box_loss', etc.
        # Ultralytics 8 structure:
        
        loss_value = _safe_loss_value(trainer)
        parsed_metrics = {
            "epoch": epoch,
            "total_epochs": args.epochs,
            "loss": loss_value if loss_value is not None else 0.0,
            # Detailed metrics from trainer.metrics which is populated after validation
            # For intermediate, we might only have loss
        }
        
        # If validation ran
        if hasattr(trainer, "metrics") and trainer.metrics:
            if task == "classify":
                parsed_metrics.update({
                    "accuracy": trainer.metrics.get("metrics/accuracy_top1", trainer.metrics.get("metrics/accuracy", 0)),
                    "accuracy_top5": trainer.metrics.get("metrics/accuracy_top5", 0)
                })
            else:
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
        model_name = args.model
        if task == "classify" and "-cls" not in model_name:
            model_name = model_name.replace(".pt", "-cls.pt")
        model = YOLO(model_name)
        
        # Register hooks
        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        print("Starting Ultralytics Training...", flush=True)
        
        results = model.train(
            data=str(data_root) if task == "classify" else str(data_yaml_path),
            project=str(run_dir),
            name="yolo_run",
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            task=task,
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
