import sys
import os
import json
import time
import argparse
import yaml
import traceback
import shutil
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
    parser.add_argument("--split-train", type=float, default=None)
    parser.add_argument("--split-val", type=float, default=None)
    parser.add_argument("--split-test", type=float, default=None)
    parser.add_argument("--dataset-version-id", type=str, default=None)
    args = parser.parse_args()
    task = (args.task or "detect").lower()
    task_aliases = {
        "classification": "classify",
        "detection": "detect",
        "segmentation": "segment",
    }
    task = task_aliases.get(task, task)

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

    def resolve_split_ratios():
        train_ratio = args.split_train
        val_ratio = args.split_val
        test_ratio = args.split_test

        if train_ratio is None and val_ratio is None and test_ratio is None:
            train_ratio, val_ratio, test_ratio = 80, 20, 0

        ratios = [train_ratio or 0, val_ratio or 0, test_ratio or 0]
        total_ratio = sum(ratios)
        if total_ratio <= 0:
            ratios = [80, 20, 0]
            total_ratio = 100
        if total_ratio > 1.5:
            ratios = [r / 100 for r in ratios]
        ratio_total = sum(ratios)
        if ratio_total <= 0:
            ratios = [0.8, 0.2, 0.0]
            ratio_total = 1.0
        return [r / ratio_total for r in ratios]

    def resolve_split_counts(total_images: int):
        ratios = resolve_split_ratios()
        train_count = int(total_images * ratios[0])
        val_count = int(total_images * ratios[1])
        test_count = int(total_images * ratios[2])
        if total_images > 0 and train_count == 0:
            train_count = 1
        if total_images > 1 and ratios[1] > 0 and val_count == 0:
            if train_count > 1:
                train_count -= 1
                val_count = 1
            elif test_count > 0:
                test_count -= 1
                val_count = 1
        remaining = total_images - (train_count + val_count + test_count)
        if remaining > 0:
            train_count += remaining
        elif remaining < 0:
            overflow = -remaining
            if test_count > 0:
                take = min(test_count, overflow)
                test_count -= take
                overflow -= take
            if overflow > 0 and val_count > 0:
                take = min(val_count, overflow)
                val_count -= take
                overflow -= take
            if overflow > 0 and train_count > 1:
                train_count = max(1, train_count - overflow)
        return train_count, val_count, test_count

    def load_selected_image_ids(manifest_path: Path, version_id: str):
        versions_manifest_path = manifest_path.parent / "dataset_versions.json"
        if not versions_manifest_path.exists():
            raise ValueError(f"Dataset version {version_id} was selected but dataset_versions.json was not found")
        with open(versions_manifest_path, "r") as f:
            payload = json.load(f)
        versions = payload.get("versions", []) if isinstance(payload, dict) else []
        for version in versions:
            if not isinstance(version, dict):
                continue
            if str(version.get("id") or "") != version_id:
                continue
            image_ids = {
                str(image_id)
                for image_id in (version.get("image_ids") or [])
                if str(image_id)
            }
            if not image_ids:
                raise ValueError(f"Dataset version {version_id} has no images")
            emit_event("train/logs", {
                "level": "INFO",
                "message": (
                    f"Using dataset version {version_id} "
                    f"({version.get('name') or 'unnamed'}) with {len(image_ids)} image(s)."
                )
            })
            return image_ids
        raise ValueError(f"Dataset version {version_id} not found")

    def build_classification_dataset(manifest_path: Path, output_dir: Path, selected_image_ids=None) -> Path:
        if not manifest_path.exists():
            raise ValueError("Dataset manifest not found for classification")
        with open(manifest_path, "r") as f:
            dataset = json.load(f)
        classes = dataset.get("classes", [])
        images = dataset.get("images", [])
        if selected_image_ids:
            images = [
                image
                for image in images
                if str(image.get("id", "")) in selected_image_ids
            ]
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
        total_images = len(labeled_images)
        train_count, val_count, test_count = resolve_split_counts(total_images)
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, (image, label_id) in enumerate(labeled_images):
            if idx < train_count:
                split = "train"
            elif idx < train_count + val_count:
                split = "val"
            else:
                split = "test"
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
                    shutil.copy2(source_path, target_path)
        return output_dir

    def build_detection_dataset(manifest_path: Path, output_dir: Path, selected_image_ids=None):
        if not manifest_path.exists():
            raise ValueError("Dataset manifest not found for detection")
        with open(manifest_path, "r") as f:
            dataset = json.load(f)

        classes = dataset.get("classes", [])
        images = dataset.get("images", [])
        if not images:
            raise ValueError("No images found in dataset manifest")

        normalized_classes = []
        for idx, cls in enumerate(sorted(classes, key=lambda item: item.get("id", 0))):
            class_id = cls.get("id", idx)
            class_name = (cls.get("name") or f"class_{idx}").strip()
            normalized_classes.append((class_id, class_name))
        if not normalized_classes:
            normalized_classes = [(0, "object")]

        class_index_by_id = {class_id: index for index, (class_id, _) in enumerate(normalized_classes)}
        class_names = [name for _, name in normalized_classes]

        valid_images = []
        for image in images:
            if selected_image_ids and str(image.get("id", "")) not in selected_image_ids:
                continue
            source_path = Path(image.get("path", ""))
            if source_path.exists() and source_path.is_file():
                valid_images.append(image)

        if not valid_images:
            raise ValueError("No valid image files found on disk for detection dataset")

        # Prefer finalized images to avoid training with partially-labeled data.
        done_images = [image for image in valid_images if bool(image.get("detection_done"))]
        if done_images:
            selected_images = done_images
            selection_reason = "detection_done=true"
        else:
            labeled_images = [image for image in valid_images if image.get("boxes")]
            selected_images = labeled_images if labeled_images else valid_images
            selection_reason = "images_with_boxes_only" if labeled_images else "all_images_fallback"

        if not selected_images:
            raise ValueError("No images available for detection training after filtering")

        # Infer which box coordinate convention is stored in manifest.
        # VisionForge UI stores top-left xywh; YOLO expects center xywh.
        def infer_box_mode(candidates):
            center_valid = 0
            top_left_valid = 0
            total_boxes = 0
            for image in candidates:
                for box in image.get("boxes", []):
                    try:
                        x = float(box.get("x", 0.0))
                        y = float(box.get("y", 0.0))
                        w = float(box.get("w", 0.0))
                        h = float(box.get("h", 0.0))
                    except Exception:
                        continue
                    if w <= 0 or h <= 0:
                        continue
                    total_boxes += 1
                    if 0.0 <= x - (w / 2) and 0.0 <= y - (h / 2) and x + (w / 2) <= 1.0 and y + (h / 2) <= 1.0:
                        center_valid += 1
                    if 0.0 <= x and 0.0 <= y and x + w <= 1.0 and y + h <= 1.0:
                        top_left_valid += 1
            if total_boxes == 0:
                return "top_left"
            # Require a clear margin before switching away from the UI-native convention.
            if center_valid > top_left_valid * 1.25:
                return "center"
            return "top_left"

        box_mode = infer_box_mode(selected_images)

        selected_images.sort(key=lambda item: item.get("id", ""))
        split_buckets = {"train": [], "val": [], "test": []}
        explicitly_split = any(image.get("split") in split_buckets for image in selected_images)

        if explicitly_split:
            for image in selected_images:
                split_name = image.get("split")
                if split_name in split_buckets:
                    split_buckets[split_name].append(image)
                else:
                    split_buckets["train"].append(image)
        else:
            train_count, val_count, test_count = resolve_split_counts(len(selected_images))
            for idx, image in enumerate(selected_images):
                if idx < train_count:
                    split_name = "train"
                elif idx < train_count + val_count:
                    split_name = "val"
                elif idx < train_count + val_count + test_count:
                    split_name = "test"
                else:
                    split_name = "train"
                split_buckets[split_name].append(image)

        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name in split_buckets:
            (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
            (output_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)

        image_counts = {"train": 0, "val": 0, "test": 0}
        box_counts = {"train": 0, "val": 0, "test": 0}

        for split_name, split_images in split_buckets.items():
            for image in split_images:
                source_path = Path(image["path"])
                suffix = source_path.suffix or ".jpg"
                image_stem = image.get("id", source_path.stem)
                filename = f"{image_stem}{suffix}"
                target_image_path = output_dir / split_name / "images" / filename
                if not target_image_path.exists():
                    try:
                        os.link(source_path, target_image_path)
                    except OSError:
                        shutil.copy2(source_path, target_image_path)

                label_lines = []
                for box in image.get("boxes", []):
                    try:
                        class_id = int(box.get("class_id"))
                    except Exception:
                        continue
                    if class_id not in class_index_by_id:
                        continue
                    try:
                        x_raw = max(0.0, min(1.0, float(box.get("x", 0.0))))
                        y_raw = max(0.0, min(1.0, float(box.get("y", 0.0))))
                        w_raw = max(0.0, min(1.0, float(box.get("w", 0.0))))
                        h_raw = max(0.0, min(1.0, float(box.get("h", 0.0))))
                    except Exception:
                        continue
                    if w_raw <= 0 or h_raw <= 0:
                        continue

                    if box_mode == "top_left":
                        x_center = x_raw + (w_raw / 2.0)
                        y_center = y_raw + (h_raw / 2.0)
                        w = w_raw
                        h = h_raw
                    else:
                        x_center = x_raw
                        y_center = y_raw
                        w = w_raw
                        h = h_raw

                    # Clamp by corners to guarantee YOLO-valid center boxes.
                    x1 = max(0.0, x_center - (w / 2.0))
                    y1 = max(0.0, y_center - (h / 2.0))
                    x2 = min(1.0, x_center + (w / 2.0))
                    y2 = min(1.0, y_center + (h / 2.0))
                    w = x2 - x1
                    h = y2 - y1
                    if w <= 0 or h <= 0:
                        continue
                    x_center = (x1 + x2) / 2.0
                    y_center = (y1 + y2) / 2.0

                    class_index = class_index_by_id[class_id]
                    label_lines.append(f"{class_index} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

                target_label_path = output_dir / split_name / "labels" / f"{Path(filename).stem}.txt"
                if label_lines:
                    with open(target_label_path, "w") as f:
                        f.write("\n".join(label_lines) + "\n")
                    box_counts[split_name] += len(label_lines)
                elif target_label_path.exists():
                    target_label_path.unlink()
                image_counts[split_name] += 1

        classes_file = output_dir / "classes.txt"
        with open(classes_file, "w") as f:
            f.write("\n".join(class_names) + "\n")

        emit_event("train/logs", {
            "level": "INFO",
            "message": (
                "Detection dataset prepared from manifest "
                f"(mode={box_mode}, selection={selection_reason}, selected={len(selected_images)}/{len(valid_images)}): "
                f"train={image_counts['train']} ({box_counts['train']} boxes), "
                f"val={image_counts['val']} ({box_counts['val']} boxes), "
                f"test={image_counts['test']} ({box_counts['test']} boxes)"
            )
        })
        return output_dir, class_names

    selected_image_ids = None
    manifest_path = Path(args.dataset_manifest) if args.dataset_manifest else None
    selected_version_id = str(args.dataset_version_id or "").strip()
    if selected_version_id:
        if not manifest_path or not manifest_path.exists():
            raise ValueError("Dataset version requires a valid dataset manifest")
        selected_image_ids = load_selected_image_ids(manifest_path, selected_version_id)

    if task == "classify":
        classification_root = dataset_path
        train_dir = classification_root / "train"
        if selected_image_ids or not train_dir.exists():
            if not manifest_path:
                raise ValueError("Dataset manifest is required to build a versioned classification dataset")
            classification_root = build_classification_dataset(
                manifest_path,
                run_dir / "cls_dataset",
                selected_image_ids=selected_image_ids
            )
        data_root = classification_root
    else:
        # Detection dataset YAML (manifest-first build for VisionForge projects)
        detection_root = dataset_path
        detection_class_names = ["object"]
        train_path = detection_root / "train" / "images"
        val_path = detection_root / "valid" / "images"
        test_path = detection_root / "test" / "images"

        if manifest_path and manifest_path.exists():
            detection_root, detection_class_names = build_detection_dataset(
                manifest_path,
                run_dir / "det_dataset",
                selected_image_ids=selected_image_ids
            )
            train_path = detection_root / "train" / "images"
            val_path = detection_root / "val" / "images"
            test_path = detection_root / "test" / "images"
        else:
            if not train_path.exists():
                train_path = detection_root / "images"
            if not train_path.exists():
                train_path = detection_root
            if not val_path.exists():
                val_path = detection_root / "val" / "images"
            if not val_path.exists():
                val_path = train_path

            classes_file = detection_root / "classes.txt"
            if classes_file.exists():
                with open(classes_file, "r") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                    if lines:
                        detection_class_names = lines

        data_yaml_content = {
            "path": str(detection_root.absolute()),
            "train": str(train_path.absolute()),
            "val": str(val_path.absolute()),
            "names": {i: name for i, name in enumerate(detection_class_names)},
            "nc": len(detection_class_names)
        }

        if test_path.exists():
            data_yaml_content["test"] = str(test_path.absolute())

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
        model_task_raw = str(getattr(model, "task", "") or "").strip().lower()
        model_task_normalized = task_aliases.get(model_task_raw, model_task_raw)
        if model_task_normalized and model_task_normalized != model_task_raw:
            emit_event("train/logs", {
                "level": "INFO",
                "message": f"Normalizing loaded model task from '{model_task_raw}' to '{model_task_normalized}'."
            })
            setattr(model, "task", model_task_normalized)
        
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
