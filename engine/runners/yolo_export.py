import argparse
import json
import shutil
import sys
import textwrap
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ultralytics import YOLO


TARGET_LABELS = {
    "edge_jetson": "NVIDIA Jetson",
    "edge_raspberry_pi": "Raspberry Pi",
    "edge_orange_pi": "Orange Pi",
    "edge_generic": "Generic Edge Device",
    "desktop_linux": "Desktop Linux",
    "desktop_windows": "Desktop Windows",
    "desktop_macos": "Desktop macOS",
}

TARGET_PROFILES: Dict[str, Dict[str, Any]] = {
    "edge_jetson": {
        "shell": "bash",
        "venv_create": "python3.8 -m venv .venv 2>/dev/null || python3.10 -m venv .venv 2>/dev/null || python3 -m venv .venv",
        "venv_activate": "source .venv/bin/activate",
        "install_cmd": (
            "python -m pip install --upgrade pip setuptools wheel && \\\n"
            "python -m pip install ultralytics onnx opencv-python numpy flask && \\\n"
            "(python -m pip install onnxruntime-gpu || python -m pip install onnxruntime || \\\n"
            "  (echo 'Install NVIDIA ONNX Runtime wheel compatible with your JetPack version before inference.' && false))"
        ),
        "onnx_opset": 19,
        "single_infer_cmd": "bash run_linux.sh validation/images/<image_file>",
        "validation_cmd": "bash validate_linux.sh",
        "script_files": ["run_linux.sh", "validate_linux.sh"],
        "notes": [
            "JetPack version and CUDA stack must match the target device.",
            "Most Jetson deployments use Python 3.8; use 3.10 only on newer JetPack images that support it.",
            "Do not rely on PyPI `onnxruntime-gpu` on Jetson; install NVIDIA's JetPack-matched ONNX Runtime wheel when required.",
            "This package exports ONNX with opset 19 for broader Jetson ONNX Runtime compatibility.",
            "For lower latency, consider converting ONNX to TensorRT after functional validation.",
        ],
    },
    "edge_raspberry_pi": {
        "shell": "bash",
        "venv_create": "python3 -m venv .venv",
        "venv_activate": "source .venv/bin/activate",
        "install_cmd": "python -m pip install --upgrade pip && python -m pip install ultralytics onnxruntime opencv-python numpy flask",
        "single_infer_cmd": "bash run_linux.sh validation/images/<image_file>",
        "validation_cmd": "bash validate_linux.sh",
        "script_files": ["run_linux.sh", "validate_linux.sh"],
        "notes": [
            "Use a small model and reduced image size when RAM is limited.",
            "Prefer active cooling for long-running inference sessions.",
        ],
    },
    "edge_orange_pi": {
        "shell": "bash",
        "venv_create": "python3 -m venv .venv",
        "venv_activate": "source .venv/bin/activate",
        "install_cmd": "python -m pip install --upgrade pip && python -m pip install ultralytics onnxruntime opencv-python numpy flask",
        "single_infer_cmd": "bash run_linux.sh validation/images/<image_file>",
        "validation_cmd": "bash validate_linux.sh",
        "script_files": ["run_linux.sh", "validate_linux.sh"],
        "notes": [
            "Current package uses ONNX Runtime CPU baseline.",
            "For Rockchip variants, an RKNN pipeline can be added in a future profile.",
        ],
    },
    "edge_generic": {
        "shell": "bash",
        "venv_create": "python3 -m venv .venv",
        "venv_activate": "source .venv/bin/activate",
        "install_cmd": "python -m pip install --upgrade pip && python -m pip install ultralytics onnxruntime opencv-python numpy flask",
        "single_infer_cmd": "bash run_linux.sh validation/images/<image_file>",
        "validation_cmd": "bash validate_linux.sh",
        "script_files": ["run_linux.sh", "validate_linux.sh"],
        "notes": [
            "This profile is a Linux baseline for edge CPU inference.",
            "Tune confidence/IoU thresholds after validating outputs on device.",
        ],
    },
    "desktop_linux": {
        "shell": "bash",
        "venv_create": "python3 -m venv .venv",
        "venv_activate": "source .venv/bin/activate",
        "install_cmd": "python -m pip install --upgrade pip && python -m pip install ultralytics onnxruntime opencv-python numpy flask",
        "single_infer_cmd": "bash run_linux.sh validation/images/<image_file>",
        "validation_cmd": "bash validate_linux.sh",
        "script_files": ["run_linux.sh", "validate_linux.sh"],
        "notes": [
            "This profile targets Linux desktop/server CPU baseline.",
            "Use a GPU runtime profile only when your desktop stack is validated for it.",
        ],
    },
    "desktop_macos": {
        "shell": "bash",
        "venv_create": "python3 -m venv .venv",
        "venv_activate": "source .venv/bin/activate",
        "install_cmd": "python -m pip install --upgrade pip && python -m pip install ultralytics onnxruntime opencv-python numpy flask",
        "single_infer_cmd": "bash run_macos.sh validation/images/<image_file>",
        "validation_cmd": "bash validate_macos.sh",
        "script_files": ["run_macos.sh", "validate_macos.sh"],
        "notes": [
            "This profile uses ONNX Runtime baseline on macOS.",
            "CoreML conversion can be evaluated later for Apple Silicon optimization.",
        ],
    },
    "desktop_windows": {
        "shell": "powershell",
        "venv_create": "python -m venv .venv",
        "venv_activate": ".\\.venv\\Scripts\\Activate.ps1",
        "install_cmd": "python -m pip install --upgrade pip; python -m pip install ultralytics onnxruntime opencv-python numpy flask",
        "single_infer_cmd": ".\\run_windows.ps1 -ImagePath validation\\images\\<image_file>",
        "validation_cmd": ".\\validate_windows.ps1",
        "script_files": ["run_windows.ps1", "validate_windows.ps1"],
        "notes": [
            "This profile targets Windows baseline with PowerShell commands.",
            "If you need GPU acceleration on Windows, evaluate ONNX Runtime DirectML after baseline validation.",
        ],
    },
}


def emit_event(event_type, payload):
    envelope = {
        "type": event_type,
        "payload": payload,
        "timestamp": time.time()
    }
    print(f"VF_EVENT: {json.dumps(envelope)}", flush=True)


def _target_profile(target_id: str) -> Dict[str, Any]:
    fallback = TARGET_PROFILES["edge_generic"]
    return TARGET_PROFILES.get(target_id, fallback)


def _normalize_class_key(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    return " ".join("".join(ch for ch in raw if ch.isalnum() or ch == " ").split())


def _parse_timestamp(value: Any) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except Exception:
        return 0.0


def _extract_identity(task: Any, predictions: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(predictions, list) or len(predictions) == 0:
        return None

    normalized_task = str(task or "").strip().lower()
    if normalized_task == "classify":
        top = predictions[0] if predictions else None
        if not isinstance(top, dict):
            return None
        class_name = str(top.get("class_name") or "").strip()
        class_id = top.get("class_id")
        identity = f"cls:{_normalize_class_key(class_name)}" if class_name else (f"id:{class_id}" if class_id is not None else "none")
        if identity == "none":
            return None
        return {
            "identity": identity,
            "task": "classify",
            "class_name": class_name or None,
            "class_id": int(class_id) if class_id is not None else None,
            "confidence": float(top.get("confidence", 0.0) or 0.0)
        }

    # Detection: compare by top-confidence class label for consistency with validation flow.
    ranked = sorted(
        [item for item in predictions if isinstance(item, dict)],
        key=lambda item: float(item.get("confidence", 0.0) or 0.0),
        reverse=True
    )
    top_det = ranked[0] if ranked else None
    if not top_det:
        return None
    class_name = str(top_det.get("class_name") or "").strip()
    class_id = top_det.get("class_id")
    identity = f"det:{_normalize_class_key(class_name)}" if class_name else (f"id:{class_id}" if class_id is not None else "none")
    if identity == "none":
        return None
    return {
        "identity": identity,
        "task": "detect",
        "class_name": class_name or None,
        "class_id": int(class_id) if class_id is not None else None,
        "confidence": float(top_det.get("confidence", 0.0) or 0.0)
    }


def _derive_expected_from_history_item(item: Dict[str, Any], variant_id: str) -> Optional[Dict[str, Any]]:
    variants = item.get("variants") or {}
    if not isinstance(variants, dict):
        return None

    # Prefer manually/automatically validated "correct" references from any model.
    candidates: List[Dict[str, Any]] = []
    for source_variant_id, data in variants.items():
        if not isinstance(data, dict):
            continue
        if str(data.get("feedback") or "").lower() != "correct":
            continue
        identity_payload = _extract_identity(data.get("task"), data.get("predictions"))
        if not identity_payload:
            continue
        candidates.append({
            "source_variant_id": source_variant_id,
            "tested_at": data.get("tested_at"),
            **identity_payload
        })

    if not candidates:
        return None

    # Select the most common identity, then the latest/highest-confidence candidate.
    votes: Dict[str, int] = {}
    for candidate in candidates:
        identity = str(candidate.get("identity") or "")
        votes[identity] = votes.get(identity, 0) + 1
    top_identity = sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0]
    scoped = [candidate for candidate in candidates if candidate.get("identity") == top_identity]
    scoped.sort(
        key=lambda candidate: (
            _parse_timestamp(candidate.get("tested_at")),
            float(candidate.get("confidence", 0.0) or 0.0)
        ),
        reverse=True
    )
    selected = scoped[0]
    supporting_variants = sorted({str(candidate.get("source_variant_id")) for candidate in scoped if candidate.get("source_variant_id")})

    variant_feedback = None
    if variant_id in variants and isinstance(variants[variant_id], dict):
        variant_feedback = variants[variant_id].get("feedback")

    return {
        "identity": selected.get("identity"),
        "task": selected.get("task"),
        "class_name": selected.get("class_name"),
        "class_id": selected.get("class_id"),
        "confidence": float(selected.get("confidence", 0.0) or 0.0),
        "source_variant_ids": supporting_variants,
        "variant_feedback": variant_feedback
    }


def _collect_validation_assets(project_id: str, variant_id: str, variant_path: Path, package_dir: Path) -> Dict[str, Any]:
    validation_dir = package_dir / "validation"
    validation_images_dir = validation_dir / "images"
    validation_dir.mkdir(parents=True, exist_ok=True)
    validation_images_dir.mkdir(parents=True, exist_ok=True)

    # variant_path = .../projects/<project_id>/variants/<variant_id>
    project_dir = variant_path.parent.parent
    inference_dir = project_dir / "inference"
    history_path = inference_dir / "history.json"
    source_images_dir = inference_dir / "images"

    manifest: Dict[str, Any] = {
        "project_id": project_id,
        "variant_id": variant_id,
        "generated_at": datetime.utcnow().isoformat(),
        "source": str(history_path),
        "entries": []
    }

    if not history_path.exists() or not source_images_dir.exists():
        manifest["total_images"] = 0
        manifest["images_with_expected"] = 0
        manifest["images_without_expected"] = 0
        output_path = validation_dir / "expected_results.json"
        output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {
            "validation_manifest_path": output_path,
            "total_images": 0,
            "images_with_expected": 0,
            "images_without_expected": 0
        }

    try:
        history_data = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception:
        history_data = {"items": []}
    items = history_data.get("items", [])
    if not isinstance(items, list):
        items = []

    entries: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        image_file = str(item.get("image_file") or "").strip()
        image_id = str(item.get("image_id") or "").strip()
        if not image_file or not image_id:
            continue
        src = source_images_dir / image_file
        if not src.exists():
            continue

        dst = validation_images_dir / image_file
        if not dst.exists():
            shutil.copy2(str(src), str(dst))

        expected = _derive_expected_from_history_item(item, variant_id)
        entry = {
            "image_id": image_id,
            "image_file": image_file,
            "original_name": item.get("original_name"),
            "last_tested_at": item.get("last_tested_at"),
            "expected_identity": expected.get("identity") if expected else None,
            "expected": {
                "task": expected.get("task"),
                "class_name": expected.get("class_name"),
                "class_id": expected.get("class_id")
            } if expected else None,
            "reference_confidence": expected.get("confidence") if expected else None,
            "source_variant_ids": expected.get("source_variant_ids") if expected else [],
            "variant_feedback": expected.get("variant_feedback") if expected else None
        }
        entries.append(entry)

    entries.sort(key=lambda item: str(item.get("image_file") or ""))
    images_with_expected = sum(1 for item in entries if item.get("expected_identity"))
    manifest["entries"] = entries
    manifest["total_images"] = len(entries)
    manifest["images_with_expected"] = images_with_expected
    manifest["images_without_expected"] = max(0, len(entries) - images_with_expected)

    output_path = validation_dir / "expected_results.json"
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "validation_manifest_path": output_path,
        "total_images": len(entries),
        "images_with_expected": images_with_expected,
        "images_without_expected": max(0, len(entries) - images_with_expected)
    }


def _build_readme(
    target_id: str,
    model_filename: str,
    fmt: str,
    validation_summary: Dict[str, Any]
) -> str:
    label = TARGET_LABELS.get(target_id, target_id)
    profile = _target_profile(target_id)
    shell = profile.get("shell", "bash")
    scripts = profile.get("script_files", [])
    total_images = int(validation_summary.get("total_images", 0))
    labeled_images = int(validation_summary.get("images_with_expected", 0))
    sample_image = "<image_file>"
    validation_manifest_path = Path(str(validation_summary.get("validation_manifest_path") or "validation/expected_results.json"))
    try:
        data = json.loads(validation_manifest_path.read_text(encoding="utf-8"))
        entries = data.get("entries", [])
        if isinstance(entries, list) and entries:
            sample_image = str(entries[0].get("image_file") or sample_image)
    except Exception:
        pass

    command_lang = "powershell" if shell == "powershell" else "bash"
    notes_lines = "\n".join(f"- {line}" for line in profile.get("notes", []))
    script_list = "\n".join(f"- `{name}`" for name in scripts) if scripts else "- No target wrapper scripts generated."

    return (
        f"# Deployment Guide - {label}\n\n"
        f"This package is specific to **{label}** and target id `{target_id}`.\n"
        "It contains model artifacts, validation images, and target-focused execution steps.\n\n"
        "## Package Contents\n"
        f"- `artifacts/{model_filename}` ({fmt})\n"
        "- `validation/images/` (images used in project validation history)\n"
        "- `validation/expected_results.json` (reference labels derived from validated history)\n"
        "- `validation/validation_report.json` (generated after running batch validation)\n"
        "- `validation/predictions/` (annotated validation images generated by `validate_batch.py`)\n"
        "- `run_local.py`\n"
        "- `validate_batch.py`\n"
        "- `integration_example.py` (example script for image/video inference)\n"
        "- `demo_app.py` (simple web UI for image/video testing)\n"
        "- `assets/visionforge-logo.png`\n"
        f"{script_list}\n"
        "- `manifest.json`\n\n"
        "## Validation Dataset Snapshot\n"
        f"- Total validation images packaged: **{total_images}**\n"
        f"- Images with expected reference label: **{labeled_images}**\n"
        f"- Images without expected label: **{max(0, total_images - labeled_images)}**\n\n"
        "## Step-by-Step Setup\n"
        "### 1) Open this folder on the target device\n"
        "- Keep the package structure unchanged.\n"
        "- Confirm `artifacts/` and `validation/` folders are present.\n\n"
        "### 2) Create and activate a Python environment\n"
        f"```{command_lang}\n"
        f"{profile['venv_create']}\n"
        f"{profile['venv_activate']}\n"
        "```\n\n"
        "### 3) Install runtime dependencies\n"
        f"```{command_lang}\n"
        f"{profile['install_cmd']}\n"
        "```\n\n"
        "### 4) Run inference on one image\n"
        f"```{command_lang}\n"
        f"{profile['single_infer_cmd'].replace('<image_file>', sample_image)}\n"
        "```\n\n"
        "### 5) Run packaged batch validation\n"
        f"```{command_lang}\n"
        f"{profile['validation_cmd']}\n"
        "```\n\n"
        "### 6) Inspect results\n"
        "- Validation report is saved to `validation/validation_report.json`.\n"
        "- Annotated validation images are saved to `validation/predictions/`.\n"
        "- Batch output includes evaluated image count and top-1 agreement with reference labels.\n"
        "- Use per-image results in the report to investigate errors directly on device.\n\n"
        "### 7) Use Python integration examples (image and video)\n"
        f"```{command_lang}\n"
        "python integration_example.py --source validation/images/<image_file>\n"
        "python integration_example.py --source /path/to/video.mp4\n"
        "```\n\n"
        "### 8) Launch a simple local demo UI\n"
        f"```{command_lang}\n"
        "python demo_app.py --host 0.0.0.0 --port 7860\n"
        "```\n"
        "- Open `http://127.0.0.1:7860` (or the target device IP) in your browser.\n\n"
        "## Target-Specific Notes\n"
        f"{notes_lines}\n\n"
        "## Troubleshooting\n"
        "- If import errors happen, confirm the virtual environment is active.\n"
        "- If inference is slow, reduce input resolution and batch size.\n"
        "- If classes look wrong, verify class mapping against your training dataset.\n"
    )


def _write_runner_scripts(package_dir: Path, model_filename: str, target_id: str):
    run_py = package_dir / "run_local.py"
    validate_py = package_dir / "validate_batch.py"
    example_py = package_dir / "integration_example.py"
    demo_py = package_dir / "demo_app.py"

    run_script = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        from ultralytics import YOLO


        def main():
            if len(sys.argv) < 2:
                print("Usage: python run_local.py <image_path>")
                sys.exit(1)
            image_path = Path(sys.argv[1]).expanduser().resolve()
            if not image_path.exists():
                print(f"Image not found: {image_path}")
                sys.exit(1)
            model_path = Path(__file__).resolve().parent / "artifacts" / "__MODEL_FILENAME__"
            model = YOLO(str(model_path))
            results = model.predict(source=str(image_path), verbose=False)
            if not results:
                print("No results produced")
                return
            result = results[0]
            if getattr(result, "probs", None) is not None:
                print("Task: classify")
                probs = result.probs
                if hasattr(probs, "top5") and hasattr(probs, "top5conf"):
                    for cls_id, conf in zip(probs.top5, probs.top5conf):
                        print(f"  class={result.names[int(cls_id)]} conf={float(conf):.4f}")
                else:
                    data = getattr(probs, "data", probs)
                    values = data.tolist() if hasattr(data, "tolist") else list(data)
                    ranked = sorted(enumerate(values), key=lambda item: item[1], reverse=True)[:5]
                    for cls_id, conf in ranked:
                        print(f"  class={result.names[int(cls_id)]} conf={float(conf):.4f}")
                return
            print("Task: detect")
            boxes = list(getattr(result, "boxes", []) or [])
            if not boxes:
                print("  no detections")
                return
            ranked = sorted(boxes, key=lambda box: float(box.conf), reverse=True)
            for box in ranked[:10]:
                class_id = int(box.cls)
                conf = float(box.conf)
                print(f"  class={result.names[class_id]} conf={conf:.4f}")


        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"
    run_py.write_text(run_script.replace("__MODEL_FILENAME__", model_filename), encoding="utf-8")

    validate_script = textwrap.dedent(
        """
        import argparse
        import json
        import shutil
        from pathlib import Path
        from typing import Any, Dict, List, Optional

        import cv2
        from ultralytics import YOLO


        def normalize_class(value: Any) -> str:
            raw = str(value or "").strip().lower()
            return " ".join("".join(ch for ch in raw if ch.isalnum() or ch == " ").split())


        def identity_from_result(result: Any) -> Optional[Dict[str, Any]]:
            if getattr(result, "probs", None) is not None:
                probs = result.probs
                if hasattr(probs, "top5") and hasattr(probs, "top5conf") and probs.top5:
                    cls_id = int(probs.top5[0])
                    conf = float(probs.top5conf[0])
                else:
                    data = getattr(probs, "data", probs)
                    values = data.tolist() if hasattr(data, "tolist") else list(data)
                    if not values:
                        return None
                    cls_id = int(max(range(len(values)), key=lambda idx: values[idx]))
                    conf = float(values[cls_id])
                class_name = result.names[int(cls_id)]
                return {
                    "task": "classify",
                    "identity": f"cls:{normalize_class(class_name)}",
                    "class_name": class_name,
                    "class_id": cls_id,
                    "confidence": conf,
                }
            boxes = list(getattr(result, "boxes", []) or [])
            if not boxes:
                return None
            top = sorted(boxes, key=lambda box: float(box.conf), reverse=True)[0]
            class_id = int(top.cls)
            class_name = result.names[class_id]
            return {
                "task": "detect",
                "identity": f"det:{normalize_class(class_name)}",
                "class_name": class_name,
                "class_id": class_id,
                "confidence": float(top.conf),
            }


        def to_relative(base: Path, path: Path) -> str:
            try:
                return str(path.relative_to(base))
            except ValueError:
                return str(path)


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--model", default="artifacts/__MODEL_FILENAME__")
            parser.add_argument("--expected", default="validation/expected_results.json")
            parser.add_argument("--images-dir", default="validation/images")
            parser.add_argument("--out", default="validation/validation_report.json")
            parser.add_argument("--viz-dir", default="validation/predictions")
            args = parser.parse_args()

            package_dir = Path(__file__).resolve().parent
            model_path = (package_dir / args.model).resolve()
            expected_path = (package_dir / args.expected).resolve()
            images_dir = (package_dir / args.images_dir).resolve()
            out_path = (package_dir / args.out).resolve()
            viz_dir = (package_dir / args.viz_dir).resolve()

            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            if not expected_path.exists():
                raise FileNotFoundError(f"Expected manifest not found: {expected_path}")
            if not images_dir.exists():
                raise FileNotFoundError(f"Validation images directory not found: {images_dir}")
            viz_dir.mkdir(parents=True, exist_ok=True)

            expected_data = json.loads(expected_path.read_text(encoding="utf-8"))
            entries = expected_data.get("entries", []) if isinstance(expected_data, dict) else []
            if not isinstance(entries, list):
                entries = []

            model = YOLO(str(model_path))
            report_entries: List[Dict[str, Any]] = []
            evaluated = 0
            correct = 0
            skipped = 0
            missing = 0
            prediction_errors = 0

            for item in entries:
                if not isinstance(item, dict):
                    continue
                image_file = str(item.get("image_file") or "").strip()
                expected_identity = item.get("expected_identity")
                image_path = (images_dir / image_file).resolve()
                viz_path: Optional[Path] = None
                viz_rel: Optional[str] = None
                if image_file:
                    viz_path = (viz_dir / image_file).resolve()
                    viz_path.parent.mkdir(parents=True, exist_ok=True)
                    viz_rel = to_relative(package_dir, viz_path)

                if not image_file or not image_path.exists():
                    missing += 1
                    report_entries.append(
                        {
                            "image_file": image_file,
                            "status": "missing_image",
                            "expected_identity": expected_identity,
                            "visualization_image": viz_rel,
                        }
                    )
                    continue

                if not expected_identity:
                    skipped += 1
                    if viz_path is not None:
                        shutil.copy2(str(image_path), str(viz_path))
                    report_entries.append(
                        {
                            "image_file": image_file,
                            "status": "skipped_no_expected",
                            "expected_identity": None,
                            "visualization_image": viz_rel,
                        }
                    )
                    continue

                try:
                    results = model.predict(source=str(image_path), verbose=False)
                except Exception as exc:
                    prediction_errors += 1
                    if viz_path is not None:
                        shutil.copy2(str(image_path), str(viz_path))
                    report_entries.append(
                        {
                            "image_file": image_file,
                            "status": "prediction_error",
                            "error": str(exc),
                            "expected_identity": expected_identity,
                            "visualization_image": viz_rel,
                        }
                    )
                    continue

                if not results:
                    if viz_path is not None:
                        shutil.copy2(str(image_path), str(viz_path))
                    report_entries.append(
                        {
                            "image_file": image_file,
                            "status": "no_result",
                            "expected_identity": expected_identity,
                            "visualization_image": viz_rel,
                        }
                    )
                    continue

                result = results[0]
                if viz_path is not None:
                    annotated = result.plot()
                    cv2.imwrite(str(viz_path), annotated)
                pred = identity_from_result(result)
                pred_identity = pred.get("identity") if pred else None
                is_match = bool(pred_identity) and str(pred_identity) == str(expected_identity)
                evaluated += 1
                if is_match:
                    correct += 1
                report_entries.append(
                    {
                        "image_file": image_file,
                        "status": "evaluated",
                        "expected_identity": expected_identity,
                        "predicted_identity": pred_identity,
                        "predicted_task": pred.get("task") if pred else None,
                        "predicted_class_name": pred.get("class_name") if pred else None,
                        "predicted_confidence": pred.get("confidence") if pred else None,
                        "match": is_match,
                        "visualization_image": viz_rel,
                    }
                )

            accuracy_pct = (correct / evaluated * 100.0) if evaluated > 0 else None
            report = {
                "generated_at": __import__("datetime").datetime.utcnow().isoformat(),
                "model_path": str(model_path),
                "expected_path": str(expected_path),
                "images_dir": str(images_dir),
                "visualizations_dir": str(viz_dir),
                "total_manifest_entries": len(entries),
                "evaluated": evaluated,
                "correct": correct,
                "incorrect": max(0, evaluated - correct),
                "skipped_no_expected": skipped,
                "missing_images": missing,
                "prediction_errors": prediction_errors,
                "accuracy_pct": accuracy_pct,
                "entries": report_entries,
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Validation report saved: {out_path}")
            print(f"Annotated images saved: {viz_dir}")
            print(f"Evaluated: {evaluated} | Correct: {correct} | Incorrect: {max(0, evaluated - correct)}")
            if prediction_errors > 0:
                print(f"Prediction errors: {prediction_errors}")
            if accuracy_pct is None:
                print("Accuracy: n/a (no expected labels available)")
            else:
                print(f"Accuracy: {accuracy_pct:.2f}%")


        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"
    validate_py.write_text(validate_script.replace("__MODEL_FILENAME__", model_filename), encoding="utf-8")

    integration_script = textwrap.dedent(
        """
        import argparse
        import time
        from pathlib import Path

        import cv2
        from ultralytics import YOLO


        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


        def summarize(result):
            if getattr(result, "probs", None) is not None:
                probs = result.probs
                print("Task: classify")
                if hasattr(probs, "top5") and hasattr(probs, "top5conf"):
                    for cls_id, conf in zip(probs.top5, probs.top5conf):
                        print(f"  class={result.names[int(cls_id)]} conf={float(conf):.4f}")
                return
            boxes = list(getattr(result, "boxes", []) or [])
            print("Task: detect")
            if not boxes:
                print("  no detections")
                return
            ranked = sorted(boxes, key=lambda box: float(box.conf), reverse=True)
            for box in ranked[:10]:
                print(f"  class={result.names[int(box.cls)]} conf={float(box.conf):.4f}")


        def infer_image(model, image_path: Path, out_dir: Path, conf: float):
            results = model.predict(source=str(image_path), verbose=False, conf=conf)
            if not results:
                raise RuntimeError("No result produced for image")
            result = results[0]
            annotated = result.plot()
            out_path = out_dir / f"{image_path.stem}_pred.jpg"
            cv2.imwrite(str(out_path), annotated)
            summarize(result)
            return out_path


        def infer_video(model, video_path: Path, out_dir: Path, conf: float):
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
            if width <= 0 or height <= 0:
                cap.release()
                raise RuntimeError("Invalid video dimensions")

            out_path = out_dir / f"{video_path.stem}_pred.mp4"
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                out_path = out_dir / f"{video_path.stem}_pred.avi"
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError("Could not create output video writer")

            total_frames = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                results = model.predict(source=frame, verbose=False, conf=conf)
                if results:
                    writer.write(results[0].plot())
                else:
                    writer.write(frame)
                total_frames += 1
                if total_frames % 60 == 0:
                    print(f"Processed {total_frames} frames...")

            cap.release()
            writer.release()
            return out_path, total_frames


        def main():
            parser = argparse.ArgumentParser(description="VisionForge integration example for image and video inference")
            parser.add_argument("--source", required=True, help="Image or video file path")
            parser.add_argument("--model", default="artifacts/__MODEL_FILENAME__", help="Relative model path inside package")
            parser.add_argument("--out-dir", default="example_outputs", help="Output directory for annotated files")
            parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
            args = parser.parse_args()

            package_dir = Path(__file__).resolve().parent
            source_path = Path(args.source).expanduser().resolve()
            model_path = (package_dir / args.model).resolve()
            out_dir = (package_dir / args.out_dir).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            if not source_path.exists():
                raise FileNotFoundError(f"Source not found: {source_path}")
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            model = YOLO(str(model_path))
            suffix = source_path.suffix.lower()
            started = time.time()

            if suffix in IMAGE_EXTENSIONS:
                out_path = infer_image(model, source_path, out_dir, args.conf)
                print(f"Annotated image saved: {out_path}")
            elif suffix in VIDEO_EXTENSIONS:
                out_path, frames = infer_video(model, source_path, out_dir, args.conf)
                print(f"Annotated video saved: {out_path}")
                print(f"Frames processed: {frames}")
            else:
                raise ValueError(f"Unsupported source extension: {suffix}")

            print(f"Done in {time.time() - started:.2f}s")


        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"
    example_py.write_text(integration_script.replace("__MODEL_FILENAME__", model_filename), encoding="utf-8")

    demo_script = textwrap.dedent(
        """
        import argparse
        import time
        from pathlib import Path
        from typing import Optional

        import cv2
        from flask import Flask, render_template_string, request, send_from_directory
        from ultralytics import YOLO


        PACKAGE_DIR = Path(__file__).resolve().parent
        MODEL_PATH = PACKAGE_DIR / "artifacts" / "__MODEL_FILENAME__"
        ASSET_DIR = PACKAGE_DIR / "assets"
        UPLOAD_DIR = PACKAGE_DIR / "demo_uploads"
        OUTPUT_DIR = PACKAGE_DIR / "demo_outputs"
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if (ASSET_DIR / "visionforge-logo.png").exists():
            LOGO_FILE = "visionforge-logo.png"
        elif (ASSET_DIR / "visionforge-logo.svg").exists():
            LOGO_FILE = "visionforge-logo.svg"
        else:
            LOGO_FILE = None

        MODEL = YOLO(str(MODEL_PATH))
        APP = Flask(__name__)

        TEMPLATE = \"\"\"
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>VisionForge Demo</title>
            <style>
                :root { color-scheme: light; }
                body { margin: 0; font-family: "Segoe UI", "Noto Sans", sans-serif; background: radial-gradient(circle at top, #e6f1ff, #f7f9fc 40%, #ffffff 100%); color: #0f172a; }
                .container { max-width: 980px; margin: 0 auto; padding: 28px 18px 36px; }
                .hero { display: flex; gap: 14px; align-items: center; padding: 16px; border: 1px solid #cbd5e1; border-radius: 14px; background: rgba(255, 255, 255, 0.85); }
                .logo { width: 56px; height: 56px; object-fit: contain; }
                .title { margin: 0; font-size: 24px; letter-spacing: 0.01em; }
                .subtitle { margin: 6px 0 0; color: #334155; font-size: 14px; }
                .notice { margin-top: 14px; padding: 10px 12px; border: 1px solid #94a3b8; border-radius: 10px; background: #f8fafc; font-size: 14px; }
                .grid { margin-top: 18px; display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
                .card { border: 1px solid #cbd5e1; border-radius: 14px; padding: 14px; background: rgba(255, 255, 255, 0.92); }
                .card h3 { margin: 0 0 10px; font-size: 16px; }
                .card p { margin: 0 0 10px; color: #475569; font-size: 13px; }
                .button { margin-top: 10px; padding: 10px 14px; border: 0; border-radius: 10px; background: #0f172a; color: #fff; cursor: pointer; font-weight: 600; }
                .button:hover { background: #1e293b; }
                .result { margin-top: 18px; border: 1px solid #cbd5e1; border-radius: 14px; padding: 14px; background: #fff; }
                .result h3 { margin: 0 0 10px; }
                .result img, .result video { width: 100%; border-radius: 10px; background: #020617; }
                .hint { margin-top: 12px; color: #475569; font-size: 12px; }
            </style>
        </head>
        <body>
            <main class="container">
                <section class="hero">
                    {% if logo_url %}
                        <img class="logo" src="{{ logo_url }}" alt="VisionForge logo" />
                    {% endif %}
                    <div>
                        <h1 class="title">VisionForge Deployment Demo</h1>
                        <p class="subtitle">Test custom images and videos directly on this target package.</p>
                    </div>
                </section>

                {% if message %}
                    <div class="notice">{{ message }}</div>
                {% endif %}

                <section class="grid">
                    <form class="card" action="/predict/image" method="post" enctype="multipart/form-data">
                        <h3>Image Inference</h3>
                        <p>Upload one image and get an annotated output.</p>
                        <input type="file" name="image_file" accept="image/*" required />
                        <button class="button" type="submit">Run Image Inference</button>
                    </form>

                    <form class="card" action="/predict/video" method="post" enctype="multipart/form-data">
                        <h3>Video Inference</h3>
                        <p>Upload one video and get an annotated output video.</p>
                        <input type="file" name="video_file" accept="video/*" required />
                        <button class="button" type="submit">Run Video Inference</button>
                    </form>
                </section>

                {% if image_output %}
                    <section class="result">
                        <h3>Image Output</h3>
                        <img src="/outputs/{{ image_output }}" alt="Annotated image" />
                    </section>
                {% endif %}

                {% if video_output %}
                    <section class="result">
                        <h3>Video Output</h3>
                        <video controls>
                            <source src="/outputs/{{ video_output }}" type="video/mp4" />
                        </video>
                    </section>
                {% endif %}

                <p class="hint">Output files are stored in <code>demo_outputs/</code>.</p>
            </main>
        </body>
        </html>
        \"\"\"


        def render_page(message: str = "", image_output: Optional[str] = None, video_output: Optional[str] = None):
            logo_url = f"/assets/{LOGO_FILE}" if LOGO_FILE else None
            return render_template_string(
                TEMPLATE,
                logo_url=logo_url,
                message=message,
                image_output=image_output,
                video_output=video_output,
            )


        def infer_image_file(image_path: Path) -> str:
            results = MODEL.predict(source=str(image_path), verbose=False)
            if not results:
                raise RuntimeError("No inference result for image")
            annotated = results[0].plot()
            out_name = f"{image_path.stem}_pred_{int(time.time() * 1000)}.jpg"
            out_path = OUTPUT_DIR / out_name
            cv2.imwrite(str(out_path), annotated)
            return out_name


        def infer_video_file(video_path: Path) -> str:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
            if width <= 0 or height <= 0:
                cap.release()
                raise RuntimeError("Invalid video dimensions")

            base_name = f"{video_path.stem}_pred_{int(time.time() * 1000)}"
            out_name = f"{base_name}.mp4"
            out_path = OUTPUT_DIR / out_name
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                out_name = f"{base_name}.avi"
                out_path = OUTPUT_DIR / out_name
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError("Could not create output video writer")

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                results = MODEL.predict(source=frame, verbose=False)
                writer.write(results[0].plot() if results else frame)

            cap.release()
            writer.release()
            return out_name


        @APP.get("/")
        def home():
            return render_page()


        @APP.get("/assets/<path:filename>")
        def assets(filename: str):
            return send_from_directory(str(ASSET_DIR), filename)


        @APP.get("/outputs/<path:filename>")
        def outputs(filename: str):
            return send_from_directory(str(OUTPUT_DIR), filename)


        @APP.post("/predict/image")
        def predict_image():
            file = request.files.get("image_file")
            if file is None or not file.filename:
                return render_page(message="Select an image file first.")
            suffix = Path(file.filename).suffix.lower() or ".jpg"
            input_path = UPLOAD_DIR / f"upload_image_{int(time.time() * 1000)}{suffix}"
            file.save(str(input_path))
            try:
                out_name = infer_image_file(input_path)
            except Exception as exc:
                return render_page(message=f"Image inference failed: {exc}")
            return render_page(message="Image inference completed.", image_output=out_name)


        @APP.post("/predict/video")
        def predict_video():
            file = request.files.get("video_file")
            if file is None or not file.filename:
                return render_page(message="Select a video file first.")
            suffix = Path(file.filename).suffix.lower() or ".mp4"
            input_path = UPLOAD_DIR / f"upload_video_{int(time.time() * 1000)}{suffix}"
            file.save(str(input_path))
            try:
                out_name = infer_video_file(input_path)
            except Exception as exc:
                return render_page(message=f"Video inference failed: {exc}")
            return render_page(message="Video inference completed.", video_output=out_name)


        def main():
            parser = argparse.ArgumentParser(description="VisionForge demo UI")
            parser.add_argument("--host", default="127.0.0.1")
            parser.add_argument("--port", type=int, default=7860)
            args = parser.parse_args()
            print(f"VisionForge demo running at http://{args.host}:{args.port}")
            APP.run(host=args.host, port=args.port, debug=False)


        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"
    demo_py.write_text(demo_script.replace("__MODEL_FILENAME__", model_filename), encoding="utf-8")

    profile = _target_profile(target_id)
    script_files = profile.get("script_files", [])
    for filename in script_files:
        target_path = package_dir / filename
        if filename.endswith(".ps1"):
            if filename.startswith("run_"):
                target_path.write_text(
                    "param(\n"
                    "  [Parameter(Mandatory=$true)][string]$ImagePath\n"
                    ")\n"
                    "python .\\run_local.py $ImagePath\n",
                    encoding="utf-8",
                )
            else:
                target_path.write_text(
                    "python .\\validate_batch.py\n",
                    encoding="utf-8",
                )
            continue

        if filename.startswith("run_"):
            target_path.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "if [ $# -lt 1 ]; then\n"
                "  echo 'Usage: bash " + filename + " <image_path>'\n"
                "  exit 1\n"
                "fi\n"
                "python run_local.py \"$1\"\n",
                encoding="utf-8",
            )
        else:
            target_path.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "python validate_batch.py\n",
                encoding="utf-8",
            )
        target_path.chmod(0o755)


def _write_branding_assets(package_dir: Path):
    assets_dir = package_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "apps" / "desktop" / "public" / "branding" / "ai-brain-icon-256.png",
        repo_root / "apps" / "desktop" / "public" / "branding" / "ai-brain-icon.png",
    ]

    for source in candidates:
        if source.exists():
            shutil.copy2(str(source), str(assets_dir / "visionforge-logo.png"))
            return

    fallback_logo = assets_dir / "visionforge-logo.svg"
    fallback_logo.write_text(
        textwrap.dedent(
            """
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 96">
              <rect width="320" height="96" rx="16" fill="#0f172a"/>
              <text x="160" y="58" text-anchor="middle" font-size="32" font-family="Segoe UI, Arial" fill="#f8fafc">VisionForge</text>
            </svg>
            """
        ).strip() + "\n",
        encoding="utf-8",
    )


def _build_package_zip(package_dir: Path) -> Path:
    # Creates <package_dir>.zip next to the package folder.
    zip_path = Path(shutil.make_archive(str(package_dir), "zip", root_dir=str(package_dir)))
    return zip_path


def _write_package_metadata(
    package_dir: Path,
    project_id: str,
    variant_id: str,
    target_id: str,
    fmt: str,
    artifact_path: Path,
    zip_path: Path,
    validation_summary: Dict[str, Any]
):
    manifest = {
        "id": f"deploy_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "project_id": project_id,
        "variant_id": variant_id,
        "target_id": target_id,
        "target_label": TARGET_LABELS.get(target_id, target_id),
        "format": fmt,
        "package_dir": str(package_dir),
        "artifact_path": str(artifact_path),
        "package_zip_path": str(zip_path),
        "readme_path": str(package_dir / "README.md"),
        "validation_manifest_path": str(validation_summary.get("validation_manifest_path", package_dir / "validation" / "expected_results.json")),
        "validation_total_images": int(validation_summary.get("total_images", 0)),
        "validation_images_with_expected": int(validation_summary.get("images_with_expected", 0)),
        "created_at": datetime.now().isoformat(),
        "status": "completed",
    }
    (package_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-path", required=True)
    parser.add_argument("--variant-id", required=False, default="")
    parser.add_argument("--project-id", required=False, default="")
    parser.add_argument("--target-id", required=False, default="edge_generic")
    parser.add_argument("--format", default="onnx")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--package-dir", required=True)
    args = parser.parse_args()

    variant_path = Path(args.variant_path)
    output_dir = Path(args.output_dir)
    package_dir = Path(args.package_dir)
    profile = _target_profile(args.target_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    package_dir.mkdir(parents=True, exist_ok=True)
    model_path = variant_path / "model.pt"
    fmt = args.format.lower()

    emit_event("export/started", {
        "variant_id": args.variant_id,
        "project_id": args.project_id,
        "target_id": args.target_id,
        "format": fmt,
    })

    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        emit_event("export/progress", {"message": "Loading model...", "stage": "init", "progress": 10})
        model = YOLO(str(model_path))

        emit_event("export/progress", {"message": f"Exporting to {fmt}...", "stage": "export", "progress": 45})
        start_t = time.time()
        export_kwargs: Dict[str, Any] = {"format": fmt, "dynamic": True}
        if fmt == "onnx":
            onnx_opset = profile.get("onnx_opset")
            if isinstance(onnx_opset, int) and onnx_opset > 0:
                export_kwargs["opset"] = onnx_opset
                emit_event(
                    "export/progress",
                    {
                        "message": f"Using ONNX opset {onnx_opset} for target compatibility...",
                        "stage": "export",
                        "progress": 48,
                    },
                )
        exported_path = model.export(**export_kwargs)
        duration = time.time() - start_t

        final_path = Path(exported_path)
        target_path = output_dir / f"model.{fmt}"
        if final_path.resolve() != target_path.resolve():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if final_path.exists():
                shutil.copy2(str(final_path), str(target_path))
        else:
            target_path = final_path

        emit_event("export/progress", {"message": "Preparing deployment package...", "stage": "package", "progress": 70})
        model_filename = target_path.name

        validation_summary = _collect_validation_assets(
            project_id=args.project_id,
            variant_id=args.variant_id,
            variant_path=variant_path,
            package_dir=package_dir
        )

        (package_dir / "README.md").write_text(
            _build_readme(args.target_id, model_filename, fmt, validation_summary),
            encoding="utf-8"
        )

        emit_event("export/progress", {"message": "Writing runtime scripts...", "stage": "scripts", "progress": 82})
        _write_runner_scripts(package_dir, model_filename, args.target_id)

        emit_event("export/progress", {"message": "Copying branding assets...", "stage": "assets", "progress": 86})
        _write_branding_assets(package_dir)

        emit_event("export/progress", {"message": "Creating deployment zip...", "stage": "archive", "progress": 90})
        package_zip_path = _build_package_zip(package_dir)

        emit_event("export/progress", {"message": "Writing manifest...", "stage": "manifest", "progress": 96})
        _write_package_metadata(
            package_dir=package_dir,
            project_id=args.project_id,
            variant_id=args.variant_id,
            target_id=args.target_id,
            fmt=fmt,
            artifact_path=target_path,
            zip_path=package_zip_path,
            validation_summary=validation_summary
        )

        emit_event("export/completed", {
            "path": str(target_path),
            "package_dir": str(package_dir),
            "package_zip_path": str(package_zip_path),
            "duration": duration,
            "format": fmt,
            "variant_id": args.variant_id,
            "project_id": args.project_id,
            "target_id": args.target_id,
            "validation_images": int(validation_summary.get("total_images", 0)),
            "validation_images_with_expected": int(validation_summary.get("images_with_expected", 0)),
        })
    except Exception as e:
        traceback.print_exc()
        emit_event("export/failed", {
            "error": str(e),
            "variant_id": args.variant_id,
            "project_id": args.project_id,
            "target_id": args.target_id,
        })
        sys.exit(1)


if __name__ == "__main__":
    main()
