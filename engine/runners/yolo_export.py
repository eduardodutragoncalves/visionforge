import argparse
import json
import shutil
import sys
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

    run_py.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "from ultralytics import YOLO\n\n"
        "def main():\n"
        "    if len(sys.argv) < 2:\n"
        "        print('Usage: python run_local.py <image_path>')\n"
        "        sys.exit(1)\n"
        "    image_path = Path(sys.argv[1]).expanduser().resolve()\n"
        "    if not image_path.exists():\n"
        "        print(f'Image not found: {image_path}')\n"
        "        sys.exit(1)\n"
        f"    model_path = Path(__file__).resolve().parent / 'artifacts' / '{model_filename}'\n"
        "    model = YOLO(str(model_path))\n"
        "    results = model.predict(source=str(image_path), verbose=False)\n"
        "    if not results:\n"
        "        print('No results produced')\n"
        "        return\n"
        "    result = results[0]\n"
        "    if getattr(result, 'probs', None) is not None:\n"
        "        print('Task: classify')\n"
        "        probs = result.probs\n"
        "        if hasattr(probs, 'top5') and hasattr(probs, 'top5conf'):\n"
        "            for cls_id, conf in zip(probs.top5, probs.top5conf):\n"
        "                print(f'  class={result.names[int(cls_id)]} conf={float(conf):.4f}')\n"
        "        else:\n"
        "            data = getattr(probs, 'data', probs)\n"
        "            values = data.tolist() if hasattr(data, 'tolist') else list(data)\n"
        "            ranked = sorted(enumerate(values), key=lambda x: x[1], reverse=True)[:5]\n"
        "            for cls_id, conf in ranked:\n"
        "                print(f'  class={result.names[int(cls_id)]} conf={float(conf):.4f}')\n"
        "        return\n"
        "    print('Task: detect')\n"
        "    boxes = list(getattr(result, 'boxes', []) or [])\n"
        "    if not boxes:\n"
        "        print('  no detections')\n"
        "        return\n"
        "    ranked = sorted(boxes, key=lambda b: float(b.conf), reverse=True)\n"
        "    for box in ranked[:10]:\n"
        "        class_id = int(box.cls)\n"
        "        conf = float(box.conf)\n"
        "        print(f'  class={result.names[class_id]} conf={conf:.4f}')\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8"
    )

    validate_py.write_text(
        "import argparse\n"
        "import json\n"
        "from pathlib import Path\n"
        "from typing import Any, Dict, List, Optional\n"
        "from ultralytics import YOLO\n\n"
        "def normalize_class(value: Any) -> str:\n"
        "    raw = str(value or '').strip().lower()\n"
        "    return ' '.join(''.join(ch for ch in raw if ch.isalnum() or ch == ' ').split())\n\n"
        "def identity_from_result(result: Any) -> Optional[Dict[str, Any]]:\n"
        "    if getattr(result, 'probs', None) is not None:\n"
        "        probs = result.probs\n"
        "        if hasattr(probs, 'top5') and hasattr(probs, 'top5conf') and probs.top5:\n"
        "            cls_id = int(probs.top5[0])\n"
        "            conf = float(probs.top5conf[0])\n"
        "        else:\n"
        "            data = getattr(probs, 'data', probs)\n"
        "            values = data.tolist() if hasattr(data, 'tolist') else list(data)\n"
        "            if not values:\n"
        "                return None\n"
        "            cls_id = int(max(range(len(values)), key=lambda i: values[i]))\n"
        "            conf = float(values[cls_id])\n"
        "        class_name = result.names[int(cls_id)]\n"
        "        return {\n"
        "            'task': 'classify',\n"
        "            'identity': f\"cls:{normalize_class(class_name)}\",\n"
        "            'class_name': class_name,\n"
        "            'class_id': cls_id,\n"
        "            'confidence': conf,\n"
        "        }\n"
        "    boxes = list(getattr(result, 'boxes', []) or [])\n"
        "    if not boxes:\n"
        "        return None\n"
        "    top = sorted(boxes, key=lambda b: float(b.conf), reverse=True)[0]\n"
        "    class_id = int(top.cls)\n"
        "    class_name = result.names[class_id]\n"
        "    return {\n"
        "        'task': 'detect',\n"
        "        'identity': f\"det:{normalize_class(class_name)}\",\n"
        "        'class_name': class_name,\n"
        "        'class_id': class_id,\n"
        "        'confidence': float(top.conf),\n"
        "    }\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--model', default='artifacts/model.onnx')\n"
        "    parser.add_argument('--expected', default='validation/expected_results.json')\n"
        "    parser.add_argument('--images-dir', default='validation/images')\n"
        "    parser.add_argument('--out', default='validation/validation_report.json')\n"
        "    args = parser.parse_args()\n\n"
        "    package_dir = Path(__file__).resolve().parent\n"
        "    model_path = (package_dir / args.model).resolve()\n"
        "    expected_path = (package_dir / args.expected).resolve()\n"
        "    images_dir = (package_dir / args.images_dir).resolve()\n"
        "    out_path = (package_dir / args.out).resolve()\n\n"
        "    if not model_path.exists():\n"
        "        raise FileNotFoundError(f'Model not found: {model_path}')\n"
        "    if not expected_path.exists():\n"
        "        raise FileNotFoundError(f'Expected manifest not found: {expected_path}')\n"
        "    if not images_dir.exists():\n"
        "        raise FileNotFoundError(f'Validation images directory not found: {images_dir}')\n\n"
        "    expected_data = json.loads(expected_path.read_text(encoding='utf-8'))\n"
        "    entries = expected_data.get('entries', []) if isinstance(expected_data, dict) else []\n"
        "    if not isinstance(entries, list):\n"
        "        entries = []\n\n"
        "    model = YOLO(str(model_path))\n"
        "    report_entries: List[Dict[str, Any]] = []\n"
        "    evaluated = 0\n"
        "    correct = 0\n"
        "    skipped = 0\n"
        "    missing = 0\n\n"
        "    for item in entries:\n"
        "        if not isinstance(item, dict):\n"
        "            continue\n"
        "        image_file = str(item.get('image_file') or '').strip()\n"
        "        expected_identity = item.get('expected_identity')\n"
        "        image_path = (images_dir / image_file).resolve()\n"
        "        if not image_file or not image_path.exists():\n"
        "            missing += 1\n"
        "            report_entries.append({\n"
        "                'image_file': image_file,\n"
        "                'status': 'missing_image',\n"
        "                'expected_identity': expected_identity,\n"
        "            })\n"
        "            continue\n"
        "        if not expected_identity:\n"
        "            skipped += 1\n"
        "            report_entries.append({\n"
        "                'image_file': image_file,\n"
        "                'status': 'skipped_no_expected',\n"
        "                'expected_identity': None,\n"
        "            })\n"
        "            continue\n"
        "        results = model.predict(source=str(image_path), verbose=False)\n"
        "        if not results:\n"
        "            report_entries.append({\n"
        "                'image_file': image_file,\n"
        "                'status': 'no_result',\n"
        "                'expected_identity': expected_identity,\n"
        "            })\n"
        "            continue\n"
        "        pred = identity_from_result(results[0])\n"
        "        pred_identity = pred.get('identity') if pred else None\n"
        "        is_match = bool(pred_identity) and str(pred_identity) == str(expected_identity)\n"
        "        evaluated += 1\n"
        "        if is_match:\n"
        "            correct += 1\n"
        "        report_entries.append({\n"
        "            'image_file': image_file,\n"
        "            'status': 'evaluated',\n"
        "            'expected_identity': expected_identity,\n"
        "            'predicted_identity': pred_identity,\n"
        "            'predicted_task': pred.get('task') if pred else None,\n"
        "            'predicted_class_name': pred.get('class_name') if pred else None,\n"
        "            'predicted_confidence': pred.get('confidence') if pred else None,\n"
        "            'match': is_match,\n"
        "        })\n\n"
        "    accuracy_pct = (correct / evaluated * 100.0) if evaluated > 0 else None\n"
        "    report = {\n"
        "        'generated_at': __import__('datetime').datetime.utcnow().isoformat(),\n"
        "        'model_path': str(model_path),\n"
        "        'expected_path': str(expected_path),\n"
        "        'images_dir': str(images_dir),\n"
        "        'total_manifest_entries': len(entries),\n"
        "        'evaluated': evaluated,\n"
        "        'correct': correct,\n"
        "        'incorrect': max(0, evaluated - correct),\n"
        "        'skipped_no_expected': skipped,\n"
        "        'missing_images': missing,\n"
        "        'accuracy_pct': accuracy_pct,\n"
        "        'entries': report_entries,\n"
        "    }\n"
        "    out_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')\n"
        "    print(f\"Validation report saved: {out_path}\")\n"
        "    print(f\"Evaluated: {evaluated} | Correct: {correct} | Incorrect: {max(0, evaluated - correct)}\")\n"
        "    if accuracy_pct is None:\n"
        "        print('Accuracy: n/a (no expected labels available)')\n"
        "    else:\n"
        "        print(f\"Accuracy: {accuracy_pct:.2f}%\")\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8"
    )

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
