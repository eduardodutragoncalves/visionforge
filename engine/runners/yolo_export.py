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
        "conda_env_name": "vf_deploy",
        "conda_python": "3.8",
        "docker_base_image": "nvcr.io/nvidia/l4t-ml:r35.4.1-py3",
        "venv_create": "python3.8 -m venv .venv 2>/dev/null || python3.10 -m venv .venv 2>/dev/null || python3 -m venv .venv",
        "venv_activate": "source .venv/bin/activate",
        "install_cmd": (
            "python3 -m pip install --upgrade pip setuptools wheel && \\\n"
            "python3 -m pip install ultralytics onnx opencv-python numpy flask requests && \\\n"
            "(python3 -m pip install onnxruntime-gpu || python3 -m pip install onnxruntime || \\\n"
            "  (echo 'Install NVIDIA ONNX Runtime wheel compatible with your JetPack version before inference.' && false))"
        ),
        "onnx_opset": 19,
        "single_infer_cmd": "bash run_linux.sh validation/images/<image_file>",
        "validation_cmd": "bash validate_linux.sh",
        "serve_cmd": "bash serve_linux.sh",
        "curl_image_cmd": "curl -X POST http://127.0.0.1:8090/predict -F file=@validation/images/<image_file>",
        "curl_video_cmd": "curl -X POST http://127.0.0.1:8090/predict -F file=@/path/to/video.mp4",
        "example_image_cmd": "bash example_linux.sh validation/images/<image_file>",
        "example_video_cmd": "bash example_linux.sh /path/to/video.mp4",
        "demo_cmd": "bash demo_linux.sh",
        "script_files": ["run_linux.sh", "validate_linux.sh", "serve_linux.sh", "example_linux.sh", "demo_linux.sh"],
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
        "conda_env_name": "vf_deploy",
        "conda_python": "3.10",
        "docker_base_image": "python:3.10-slim",
        "venv_create": "python3 -m venv .venv",
        "venv_activate": "source .venv/bin/activate",
        "install_cmd": "python3 -m pip install --upgrade pip && python3 -m pip install ultralytics onnxruntime opencv-python numpy flask requests",
        "single_infer_cmd": "bash run_linux.sh validation/images/<image_file>",
        "validation_cmd": "bash validate_linux.sh",
        "serve_cmd": "bash serve_linux.sh",
        "curl_image_cmd": "curl -X POST http://127.0.0.1:8090/predict -F file=@validation/images/<image_file>",
        "curl_video_cmd": "curl -X POST http://127.0.0.1:8090/predict -F file=@/path/to/video.mp4",
        "example_image_cmd": "bash example_linux.sh validation/images/<image_file>",
        "example_video_cmd": "bash example_linux.sh /path/to/video.mp4",
        "demo_cmd": "bash demo_linux.sh",
        "script_files": ["run_linux.sh", "validate_linux.sh", "serve_linux.sh", "example_linux.sh", "demo_linux.sh"],
        "notes": [
            "Use a small model and reduced image size when RAM is limited.",
            "Prefer active cooling for long-running inference sessions.",
        ],
    },
    "edge_orange_pi": {
        "shell": "bash",
        "conda_env_name": "vf_deploy",
        "conda_python": "3.10",
        "docker_base_image": "python:3.10-slim",
        "venv_create": "python3 -m venv .venv",
        "venv_activate": "source .venv/bin/activate",
        "install_cmd": "python3 -m pip install --upgrade pip && python3 -m pip install ultralytics onnxruntime opencv-python numpy flask requests",
        "single_infer_cmd": "bash run_linux.sh validation/images/<image_file>",
        "validation_cmd": "bash validate_linux.sh",
        "serve_cmd": "bash serve_linux.sh",
        "curl_image_cmd": "curl -X POST http://127.0.0.1:8090/predict -F file=@validation/images/<image_file>",
        "curl_video_cmd": "curl -X POST http://127.0.0.1:8090/predict -F file=@/path/to/video.mp4",
        "example_image_cmd": "bash example_linux.sh validation/images/<image_file>",
        "example_video_cmd": "bash example_linux.sh /path/to/video.mp4",
        "demo_cmd": "bash demo_linux.sh",
        "script_files": ["run_linux.sh", "validate_linux.sh", "serve_linux.sh", "example_linux.sh", "demo_linux.sh"],
        "notes": [
            "Current package uses ONNX Runtime CPU baseline.",
            "For Rockchip variants, an RKNN pipeline can be added in a future profile.",
        ],
    },
    "edge_generic": {
        "shell": "bash",
        "conda_env_name": "vf_deploy",
        "conda_python": "3.10",
        "docker_base_image": "python:3.10-slim",
        "venv_create": "python3 -m venv .venv",
        "venv_activate": "source .venv/bin/activate",
        "install_cmd": "python3 -m pip install --upgrade pip && python3 -m pip install ultralytics onnxruntime opencv-python numpy flask requests",
        "single_infer_cmd": "bash run_linux.sh validation/images/<image_file>",
        "validation_cmd": "bash validate_linux.sh",
        "serve_cmd": "bash serve_linux.sh",
        "curl_image_cmd": "curl -X POST http://127.0.0.1:8090/predict -F file=@validation/images/<image_file>",
        "curl_video_cmd": "curl -X POST http://127.0.0.1:8090/predict -F file=@/path/to/video.mp4",
        "example_image_cmd": "bash example_linux.sh validation/images/<image_file>",
        "example_video_cmd": "bash example_linux.sh /path/to/video.mp4",
        "demo_cmd": "bash demo_linux.sh",
        "script_files": ["run_linux.sh", "validate_linux.sh", "serve_linux.sh", "example_linux.sh", "demo_linux.sh"],
        "notes": [
            "This profile is a Linux baseline for edge CPU inference.",
            "Tune confidence/IoU thresholds after validating outputs on device.",
        ],
    },
    "desktop_linux": {
        "shell": "bash",
        "conda_env_name": "vf_deploy",
        "conda_python": "3.10",
        "docker_base_image": "python:3.10-slim",
        "venv_create": "python3 -m venv .venv",
        "venv_activate": "source .venv/bin/activate",
        "install_cmd": "python3 -m pip install --upgrade pip && python3 -m pip install ultralytics onnxruntime opencv-python numpy flask requests",
        "single_infer_cmd": "bash run_linux.sh validation/images/<image_file>",
        "validation_cmd": "bash validate_linux.sh",
        "serve_cmd": "bash serve_linux.sh",
        "curl_image_cmd": "curl -X POST http://127.0.0.1:8090/predict -F file=@validation/images/<image_file>",
        "curl_video_cmd": "curl -X POST http://127.0.0.1:8090/predict -F file=@/path/to/video.mp4",
        "example_image_cmd": "bash example_linux.sh validation/images/<image_file>",
        "example_video_cmd": "bash example_linux.sh /path/to/video.mp4",
        "demo_cmd": "bash demo_linux.sh",
        "script_files": ["run_linux.sh", "validate_linux.sh", "serve_linux.sh", "example_linux.sh", "demo_linux.sh"],
        "notes": [
            "This profile targets Linux desktop/server CPU baseline.",
            "Use a GPU runtime profile only when your desktop stack is validated for it.",
        ],
    },
    "desktop_macos": {
        "shell": "bash",
        "conda_env_name": "vf_deploy",
        "conda_python": "3.10",
        "docker_base_image": "python:3.10-slim",
        "venv_create": "python3 -m venv .venv",
        "venv_activate": "source .venv/bin/activate",
        "install_cmd": "python3 -m pip install --upgrade pip && python3 -m pip install ultralytics onnxruntime opencv-python numpy flask requests",
        "single_infer_cmd": "bash run_macos.sh validation/images/<image_file>",
        "validation_cmd": "bash validate_macos.sh",
        "serve_cmd": "bash serve_macos.sh",
        "curl_image_cmd": "curl -X POST http://127.0.0.1:8090/predict -F file=@validation/images/<image_file>",
        "curl_video_cmd": "curl -X POST http://127.0.0.1:8090/predict -F file=@/path/to/video.mp4",
        "example_image_cmd": "bash example_macos.sh validation/images/<image_file>",
        "example_video_cmd": "bash example_macos.sh /path/to/video.mp4",
        "demo_cmd": "bash demo_macos.sh",
        "script_files": ["run_macos.sh", "validate_macos.sh", "serve_macos.sh", "example_macos.sh", "demo_macos.sh"],
        "notes": [
            "This profile uses ONNX Runtime baseline on macOS.",
            "CoreML conversion can be evaluated later for Apple Silicon optimization.",
        ],
    },
    "desktop_windows": {
        "shell": "powershell",
        "conda_env_name": "vf_deploy",
        "conda_python": "3.10",
        "docker_base_image": "python:3.10-slim",
        "venv_create": "python -m venv .venv",
        "venv_activate": ".\\.venv\\Scripts\\Activate.ps1",
        "install_cmd": "python -m pip install --upgrade pip; python -m pip install ultralytics onnxruntime opencv-python numpy flask requests",
        "single_infer_cmd": ".\\run_windows.ps1 -ImagePath validation\\images\\<image_file>",
        "validation_cmd": ".\\validate_windows.ps1",
        "serve_cmd": ".\\serve_windows.ps1 -Host 0.0.0.0 -Port 8090",
        "curl_image_cmd": "curl.exe -X POST http://127.0.0.1:8090/predict -F \"file=@validation/images/<image_file>\"",
        "curl_video_cmd": "curl.exe -X POST http://127.0.0.1:8090/predict -F \"file=@C:\\path\\to\\video.mp4\"",
        "example_image_cmd": ".\\example_windows.ps1 -SourcePath validation\\images\\<image_file>",
        "example_video_cmd": ".\\example_windows.ps1 -SourcePath C:\\path\\to\\video.mp4",
        "demo_cmd": ".\\demo_windows.ps1 -Host 0.0.0.0 -Port 7860",
        "script_files": ["run_windows.ps1", "validate_windows.ps1", "serve_windows.ps1", "example_windows.ps1", "demo_windows.ps1"],
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


def _venv_python_cmd(shell: str) -> str:
    return ".\\.venv\\Scripts\\python.exe" if shell == "powershell" else "./.venv/bin/python"


def _venv_install_cmd(profile: Dict[str, Any], shell: str) -> str:
    install_cmd = str(profile.get("install_cmd", "python -m pip install --upgrade pip"))
    venv_python = _venv_python_cmd(shell)
    install_cmd = install_cmd.replace("python3 -m pip", f"{venv_python} -m pip")
    install_cmd = install_cmd.replace("python -m pip", f"{venv_python} -m pip")
    return install_cmd


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
    venv_install_cmd = _venv_install_cmd(profile, shell)
    notes_lines = "\n".join(f"- {line}" for line in profile.get("notes", []))
    script_list = "\n".join(f"- `{name}`" for name in scripts) if scripts else "- No target wrapper scripts generated."
    conda_env_name = str(profile.get("conda_env_name", "vf_deploy"))
    conda_python = str(profile.get("conda_python", "3.10"))
    conda_create_cmd = str(profile.get("conda_create_cmd", f"conda create -n {conda_env_name} python={conda_python} -y"))
    conda_activate_cmd = str(profile.get("conda_activate_cmd", f"conda activate {conda_env_name}"))
    docker_base_image = str(profile.get("docker_base_image", "python:3.10-slim"))
    docker_build_cmd = str(
        profile.get(
            "docker_build_cmd",
            f"docker build --build-arg BASE_IMAGE={docker_base_image} -t visionforge-deploy:local .",
        )
    )
    docker_run_cmd = str(
        profile.get(
            "docker_run_cmd",
            "docker run --rm -p 8090:8090 visionforge-deploy:local "
            "python inference_server.py --host 0.0.0.0 --port 8090",
        )
    )
    compose_up_cmd = str(profile.get("compose_up_cmd", "docker compose up --build"))
    compose_down_cmd = str(profile.get("compose_down_cmd", "docker compose down"))
    example_image_cmd = str(
        profile.get("example_image_cmd", "python integration_example.py --source validation/images/<image_file>")
    ).replace("<image_file>", sample_image)
    example_video_cmd = str(
        profile.get("example_video_cmd", "python integration_example.py --source /path/to/video.mp4")
    )
    example_server_cmd = f"python integration_example.py --source validation/images/{sample_image} --server-url http://127.0.0.1:8090"
    demo_cmd = str(profile.get("demo_cmd", "python demo_app.py --host 0.0.0.0 --port 7860"))
    serve_cmd = str(profile.get("serve_cmd", "python inference_server.py --host 0.0.0.0 --port 8090"))
    curl_image_cmd = str(
        profile.get("curl_image_cmd", "curl -X POST http://127.0.0.1:8090/predict -F file=@validation/images/<image_file>")
    ).replace("<image_file>", sample_image)
    curl_video_cmd = str(
        profile.get("curl_video_cmd", "curl -X POST http://127.0.0.1:8090/predict -F file=@/path/to/video.mp4")
    )

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
        "- `inference_server.py` (HTTP inference endpoint)\n"
        "- `integration_example.py` (example script for image/video inference)\n"
        "- `demo_app.py` (simple web UI for image/video testing)\n"
        "- `requirements.runtime.txt` (runtime deps for Conda/Docker)\n"
        "- `environment.conda.yml` (Conda environment template)\n"
        "- `Dockerfile`\n"
        "- `docker-compose.yml` (server + demo UI together)\n"
        "- `.dockerignore`\n"
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
        "### 2) Choose one runtime setup\n"
        "#### Option A: Python venv\n"
        f"```{command_lang}\n"
        f"{profile['venv_create']}\n"
        f"{profile['venv_activate']}\n"
        f"{venv_install_cmd}\n"
        "```\n\n"
        "#### Option B: Conda (Anaconda/Miniconda)\n"
        f"```{command_lang}\n"
        f"{conda_create_cmd}\n"
        f"{conda_activate_cmd}\n"
        "python -m pip install --upgrade pip setuptools wheel\n"
        "python -m pip install -r requirements.runtime.txt\n"
        "```\n\n"
        "#### Option C: Dockerfile (server only)\n"
        "- If needed, override base image with `VISIONFORGE_BASE_IMAGE`/`--build-arg`.\n"
        f"```{command_lang}\n"
        f"{docker_build_cmd}\n"
        f"{docker_run_cmd}\n"
        "```\n\n"
        "#### Option D: Docker Compose (server + demo UI together)\n"
        f"```{command_lang}\n"
        f"{compose_up_cmd}\n"
        "```\n"
        "- Demo UI: `http://127.0.0.1:7860`\n"
        "- Inference server: `http://127.0.0.1:8090`\n"
        f"To stop:\n```{command_lang}\n{compose_down_cmd}\n```\n\n"
        "### 3) Run inference on one image (venv/conda)\n"
        f"```{command_lang}\n"
        f"{profile['single_infer_cmd'].replace('<image_file>', sample_image)}\n"
        "```\n\n"
        "### 4) Run packaged batch validation (venv/conda)\n"
        f"```{command_lang}\n"
        f"{profile['validation_cmd']}\n"
        "```\n\n"
        "### 5) Inspect results\n"
        "- Validation report is saved to `validation/validation_report.json`.\n"
        "- Annotated validation images are saved to `validation/predictions/`.\n"
        "- Batch output includes evaluated image count and top-1 agreement with reference labels.\n"
        "- Use per-image results in the report to investigate errors directly on device.\n\n"
        "### 6) Use Python integration examples (image and video)\n"
        f"```{command_lang}\n"
        f"{example_image_cmd}\n"
        f"{example_video_cmd}\n"
        f"{example_server_cmd}\n"
        "```\n\n"
        "### 7) Start the inference server endpoint (venv/conda)\n"
        f"```{command_lang}\n"
        f"{serve_cmd}\n"
        "```\n"
        "- Health check endpoint: `GET /health`\n"
        "- Inference endpoint: `POST /predict` (multipart form with `file`)\n\n"
        "### 8) Test server via curl\n"
        f"```{command_lang}\n"
        f"{curl_image_cmd}\n"
        f"{curl_video_cmd}\n"
        "```\n\n"
        "### 9) Launch a simple local demo UI (venv/conda)\n"
        f"```{command_lang}\n"
        f"{demo_cmd}\n"
        "```\n"
        "- Open `http://127.0.0.1:7860` (or the target device IP) in your browser.\n\n"
        "## Target-Specific Notes\n"
        f"{notes_lines}\n\n"
        "## Troubleshooting\n"
        "- If import errors happen, confirm the virtual environment is active.\n"
        "- If pip warns about `~/.local/bin` script location, dependencies were installed outside `.venv`; run Option A again and keep the `.venv` commands exactly as documented.\n"
        "- If inference is slow, reduce input resolution and batch size.\n"
        "- If classes look wrong, verify class mapping against your training dataset.\n"
    )


def _write_runner_scripts(package_dir: Path, model_filename: str, target_id: str):
    run_py = package_dir / "run_local.py"
    validate_py = package_dir / "validate_batch.py"
    server_py = package_dir / "inference_server.py"
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

    server_script = textwrap.dedent(
        """
        import argparse
        import json
        import time
        from pathlib import Path
        from typing import Any, Dict, List

        import cv2
        from flask import Flask, jsonify, request, send_from_directory
        from ultralytics import YOLO


        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

        PACKAGE_DIR = Path(__file__).resolve().parent
        MODEL_PATH = PACKAGE_DIR / "artifacts" / "__MODEL_FILENAME__"
        UPLOAD_DIR = PACKAGE_DIR / "server_uploads"
        OUTPUT_DIR = PACKAGE_DIR / "server_outputs"
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        MODEL = YOLO(str(MODEL_PATH))
        APP = Flask(__name__)
        STARTED_AT = time.time()


        def summarize_result(result: Any) -> Dict[str, Any]:
            if getattr(result, "probs", None) is not None:
                probs = result.probs
                top_predictions: List[Dict[str, Any]] = []
                if hasattr(probs, "top5") and hasattr(probs, "top5conf"):
                    for cls_id, conf in zip(probs.top5, probs.top5conf):
                        top_predictions.append(
                            {
                                "class_id": int(cls_id),
                                "class_name": str(result.names[int(cls_id)]),
                                "confidence": float(conf),
                            }
                        )
                return {"task": "classify", "predictions": top_predictions}

            boxes = list(getattr(result, "boxes", []) or [])
            detections: List[Dict[str, Any]] = []
            for box in boxes:
                cls_id = int(box.cls)
                xyxy = box.xyxy[0].tolist() if hasattr(box, "xyxy") else [0, 0, 0, 0]
                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": str(result.names.get(cls_id, cls_id) if isinstance(result.names, dict) else result.names[cls_id]),
                        "confidence": float(box.conf),
                        "bbox_xyxy": [float(v) for v in xyxy],
                    }
                )
            detections.sort(key=lambda item: item["confidence"], reverse=True)
            return {"task": "detect", "predictions": detections}


        def infer_image(path: Path, conf: float, iou: float) -> Dict[str, Any]:
            started = time.time()
            results = MODEL.predict(source=str(path), verbose=False, conf=conf, iou=iou)
            if not results:
                raise RuntimeError("No inference result produced")
            result = results[0]
            summary = summarize_result(result)
            annotated = result.plot()
            output_name = f"{path.stem}_pred_{int(time.time() * 1000)}.jpg"
            output_path = OUTPUT_DIR / output_name
            cv2.imwrite(str(output_path), annotated)
            return {
                "status": "ok",
                "source_type": "image",
                "source_file": path.name,
                "output_file": output_name,
                "output_url": f"/outputs/{output_name}",
                "inference_ms": round((time.time() - started) * 1000, 2),
                **summary,
            }


        def infer_video(path: Path, conf: float, iou: float) -> Dict[str, Any]:
            started = time.time()
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {path}")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
            if width <= 0 or height <= 0:
                cap.release()
                raise RuntimeError("Invalid video dimensions")

            base_name = f"{path.stem}_pred_{int(time.time() * 1000)}"
            output_name = f"{base_name}.mp4"
            output_path = OUTPUT_DIR / output_name
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                output_name = f"{base_name}.avi"
                output_path = OUTPUT_DIR / output_name
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError("Could not create output video writer")

            frames = 0
            sample_predictions: List[Dict[str, Any]] = []
            sample_task = "detect"
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                results = MODEL.predict(source=frame, verbose=False, conf=conf, iou=iou)
                if results:
                    first = results[0]
                    if frames == 0:
                        summary = summarize_result(first)
                        sample_predictions = summary.get("predictions", [])
                        sample_task = str(summary.get("task") or "detect")
                    writer.write(first.plot())
                else:
                    writer.write(frame)
                frames += 1

            cap.release()
            writer.release()
            return {
                "status": "ok",
                "source_type": "video",
                "source_file": path.name,
                "output_file": output_name,
                "output_url": f"/outputs/{output_name}",
                "frames_processed": frames,
                "inference_ms": round((time.time() - started) * 1000, 2),
                "task": sample_task,
                "predictions": sample_predictions[:10],
            }


        @APP.get("/")
        def root():
            return jsonify(
                {
                    "service": "visionforge-inference-server",
                    "status": "ok",
                    "health_url": "/health",
                    "predict_url": "/predict",
                }
            )


        @APP.get("/health")
        def health():
            return jsonify(
                {
                    "status": "ok",
                    "uptime_sec": round(time.time() - STARTED_AT, 3),
                    "model_path": str(MODEL_PATH),
                }
            )


        @APP.get("/outputs/<path:filename>")
        def outputs(filename: str):
            return send_from_directory(str(OUTPUT_DIR), filename)


        @APP.post("/predict")
        def predict():
            file = request.files.get("file")
            if file is None or not file.filename:
                return jsonify({"status": "error", "error": "Missing multipart file field `file`"}), 400

            suffix = Path(file.filename).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS and suffix not in VIDEO_EXTENSIONS:
                return jsonify({"status": "error", "error": f"Unsupported extension: {suffix}"}), 400

            conf = float(request.form.get("conf", "0.25") or 0.25)
            iou = float(request.form.get("iou", "0.7") or 0.7)
            uploaded_name = f"upload_{int(time.time() * 1000)}{suffix}"
            uploaded_path = UPLOAD_DIR / uploaded_name
            file.save(str(uploaded_path))

            try:
                if suffix in IMAGE_EXTENSIONS:
                    payload = infer_image(uploaded_path, conf=conf, iou=iou)
                else:
                    payload = infer_video(uploaded_path, conf=conf, iou=iou)
            except Exception as exc:
                return jsonify({"status": "error", "error": str(exc)}), 500
            return jsonify(payload)


        def main():
            parser = argparse.ArgumentParser(description="VisionForge inference server")
            parser.add_argument("--host", default="0.0.0.0")
            parser.add_argument("--port", type=int, default=8090)
            args = parser.parse_args()
            print(f"VisionForge inference server running at http://{args.host}:{args.port}")
            APP.run(host=args.host, port=args.port, debug=False)


        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"
    server_py.write_text(server_script.replace("__MODEL_FILENAME__", model_filename), encoding="utf-8")

    integration_script = textwrap.dedent(
        """
        import argparse
        import json
        import time
        from pathlib import Path
        from urllib.parse import urljoin

        import cv2
        import requests
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


        def infer_via_server(server_url: str, source_path: Path, out_dir: Path, conf: float, iou: float, timeout: float):
            base = server_url.rstrip("/") + "/"
            health_url = urljoin(base, "health")
            predict_url = urljoin(base, "predict")
            health_res = requests.get(health_url, timeout=timeout)
            health_res.raise_for_status()
            with source_path.open("rb") as fp:
                response = requests.post(
                    predict_url,
                    files={"file": (source_path.name, fp, "application/octet-stream")},
                    data={"conf": str(conf), "iou": str(iou)},
                    timeout=max(timeout, 60.0),
                )
            response.raise_for_status()
            payload = response.json()
            print(json.dumps(payload, indent=2, ensure_ascii=False))

            output_url = payload.get("output_url")
            if output_url:
                download_url = output_url if str(output_url).startswith("http") else urljoin(base, str(output_url).lstrip("/"))
                target_name = str(payload.get("output_file") or Path(str(output_url)).name or f"{source_path.stem}_server_pred")
                target_path = out_dir / target_name
                with requests.get(download_url, timeout=max(timeout, 60.0), stream=True) as download:
                    download.raise_for_status()
                    with target_path.open("wb") as out_fp:
                        for chunk in download.iter_content(chunk_size=1024 * 64):
                            if chunk:
                                out_fp.write(chunk)
                print(f"Downloaded server output: {target_path}")


        def main():
            parser = argparse.ArgumentParser(description="VisionForge integration example for image/video inference")
            parser.add_argument("--source", required=True, help="Image or video file path")
            parser.add_argument("--model", default="artifacts/__MODEL_FILENAME__", help="Relative model path inside package")
            parser.add_argument("--out-dir", default="example_outputs", help="Output directory for annotated files")
            parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
            parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold")
            parser.add_argument("--server-url", default="", help="Optional inference server URL, e.g. http://127.0.0.1:8090")
            parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout in seconds for server mode")
            args = parser.parse_args()

            package_dir = Path(__file__).resolve().parent
            source_path = Path(args.source).expanduser().resolve()
            out_dir = (package_dir / args.out_dir).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            if not source_path.exists():
                raise FileNotFoundError(f"Source not found: {source_path}")

            started = time.time()
            if args.server_url.strip():
                infer_via_server(
                    server_url=args.server_url.strip(),
                    source_path=source_path,
                    out_dir=out_dir,
                    conf=args.conf,
                    iou=args.iou,
                    timeout=args.timeout,
                )
                print(f"Done via server in {time.time() - started:.2f}s")
                return

            model_path = (package_dir / args.model).resolve()
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            model = YOLO(str(model_path))
            suffix = source_path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                out_path = infer_image(model, source_path, out_dir, args.conf)
                print(f"Annotated image saved: {out_path}")
            elif suffix in VIDEO_EXTENSIONS:
                out_path, frames = infer_video(model, source_path, out_dir, args.conf)
                print(f"Annotated video saved: {out_path}")
                print(f"Frames processed: {frames}")
            else:
                raise ValueError(f"Unsupported source extension: {suffix}")
            print(f"Done locally in {time.time() - started:.2f}s")


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
        from urllib.parse import urljoin

        import cv2
        import requests
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

        MODEL = None
        APP = Flask(__name__)
        SETTINGS = {
            "server_url": "http://127.0.0.1:8090",
            "use_server": True,
            "http_timeout": 15.0,
            "conf": 0.25,
            "iou": 0.7,
        }

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
                .server { margin-top: 10px; padding: 10px 12px; border: 1px solid #bfdbfe; border-radius: 10px; background: #eff6ff; font-size: 13px; color: #1e3a8a; }
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
                <div class="server">{{ server_status }}</div>

                <section class="grid">
                    <form class="card" action="/predict/image" method="post" enctype="multipart/form-data">
                        <h3>Image Inference</h3>
                        <p>Upload one image and get an annotated output (server when available).</p>
                        <input type="file" name="image_file" accept="image/*" required />
                        <button class="button" type="submit">Run Image Inference</button>
                    </form>

                    <form class="card" action="/predict/video" method="post" enctype="multipart/form-data">
                        <h3>Video Inference</h3>
                        <p>Upload one video and get an annotated output (server when available).</p>
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


        def get_model():
            global MODEL
            if MODEL is None:
                MODEL = YOLO(str(MODEL_PATH))
            return MODEL


        def check_server():
            if not SETTINGS.get("use_server"):
                return False, "Inference server disabled. Using local model."
            base = str(SETTINGS.get("server_url") or "").strip()
            if not base:
                return False, "No server URL configured. Using local model."
            try:
                res = requests.get(urljoin(base.rstrip("/") + "/", "health"), timeout=float(SETTINGS.get("http_timeout", 15.0)))
                if not res.ok:
                    return False, f"Inference server unavailable ({res.status_code}). Using local model."
                return True, f"Inference server online at {base}."
            except Exception as exc:
                return False, f"Inference server offline ({exc}). Using local model."


        def server_predict(input_path: Path) -> dict:
            base = str(SETTINGS.get("server_url") or "").rstrip("/") + "/"
            predict_url = urljoin(base, "predict")
            with input_path.open("rb") as fp:
                response = requests.post(
                    predict_url,
                    files={"file": (input_path.name, fp, "application/octet-stream")},
                    data={
                        "conf": str(SETTINGS.get("conf", 0.25)),
                        "iou": str(SETTINGS.get("iou", 0.7)),
                    },
                    timeout=max(float(SETTINGS.get("http_timeout", 15.0)), 60.0),
                )
            response.raise_for_status()
            return response.json()


        def download_server_output(payload: dict) -> Optional[str]:
            output_url = payload.get("output_url")
            if not output_url:
                return None
            base = str(SETTINGS.get("server_url") or "").rstrip("/") + "/"
            url = output_url if str(output_url).startswith("http") else urljoin(base, str(output_url).lstrip("/"))
            target_name = str(payload.get("output_file") or Path(str(output_url)).name or f"server_pred_{int(time.time() * 1000)}")
            target_path = OUTPUT_DIR / target_name
            with requests.get(url, timeout=max(float(SETTINGS.get("http_timeout", 15.0)), 60.0), stream=True) as response:
                response.raise_for_status()
                with target_path.open("wb") as out_fp:
                    for chunk in response.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            out_fp.write(chunk)
            return target_name


        def render_page(
            message: str = "",
            image_output: Optional[str] = None,
            video_output: Optional[str] = None,
            server_status: Optional[str] = None,
        ):
            logo_url = f"/assets/{LOGO_FILE}" if LOGO_FILE else None
            if server_status is None:
                _, server_status = check_server()
            return render_template_string(
                TEMPLATE,
                logo_url=logo_url,
                message=message,
                image_output=image_output,
                video_output=video_output,
                server_status=server_status,
            )


        def infer_image_file(image_path: Path) -> str:
            results = get_model().predict(source=str(image_path), verbose=False)
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
                results = get_model().predict(source=frame, verbose=False)
                writer.write(results[0].plot() if results else frame)

            cap.release()
            writer.release()
            return out_name


        @APP.get("/")
        def home():
            return render_page(server_status=check_server()[1])


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
            server_ok, server_status = check_server()
            try:
                if server_ok:
                    payload = server_predict(input_path)
                    out_name = download_server_output(payload)
                    if not out_name:
                        raise RuntimeError("Server did not return output file URL")
                    return render_page(
                        message="Image inference completed via server.",
                        image_output=out_name,
                        server_status=server_status,
                    )
                out_name = infer_image_file(input_path)
            except Exception as exc:
                return render_page(message=f"Image inference failed: {exc}", server_status=server_status)
            return render_page(message="Image inference completed locally.", image_output=out_name, server_status=server_status)


        @APP.post("/predict/video")
        def predict_video():
            file = request.files.get("video_file")
            if file is None or not file.filename:
                return render_page(message="Select a video file first.")
            suffix = Path(file.filename).suffix.lower() or ".mp4"
            input_path = UPLOAD_DIR / f"upload_video_{int(time.time() * 1000)}{suffix}"
            file.save(str(input_path))
            server_ok, server_status = check_server()
            try:
                if server_ok:
                    payload = server_predict(input_path)
                    out_name = download_server_output(payload)
                    if not out_name:
                        raise RuntimeError("Server did not return output file URL")
                    return render_page(
                        message="Video inference completed via server.",
                        video_output=out_name,
                        server_status=server_status,
                    )
                out_name = infer_video_file(input_path)
            except Exception as exc:
                return render_page(message=f"Video inference failed: {exc}", server_status=server_status)
            return render_page(message="Video inference completed locally.", video_output=out_name, server_status=server_status)


        def main():
            parser = argparse.ArgumentParser(description="VisionForge demo UI")
            parser.add_argument("--host", default="127.0.0.1")
            parser.add_argument("--port", type=int, default=7860)
            parser.add_argument("--server-url", default="http://127.0.0.1:8090")
            parser.add_argument("--disable-server", action="store_true")
            parser.add_argument("--timeout", type=float, default=15.0)
            args = parser.parse_args()
            SETTINGS["server_url"] = str(args.server_url).strip()
            SETTINGS["use_server"] = not args.disable_server
            SETTINGS["http_timeout"] = max(1.0, float(args.timeout))
            print(f"VisionForge demo running at http://{args.host}:{args.port}")
            if SETTINGS["use_server"]:
                print(f"Server endpoint target: {SETTINGS['server_url']}")
            APP.run(host=args.host, port=args.port, debug=False)


        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n"
    demo_py.write_text(demo_script.replace("__MODEL_FILENAME__", model_filename), encoding="utf-8")

    profile = _target_profile(target_id)
    script_files = profile.get("script_files", [])
    bash_python_bootstrap = (
        "PYTHON_BIN=\"${PYTHON_BIN:-python}\"\n"
        "if ! command -v \"$PYTHON_BIN\" >/dev/null 2>&1; then\n"
        "  if command -v python3 >/dev/null 2>&1; then\n"
        "    PYTHON_BIN=\"python3\"\n"
        "  else\n"
        "    echo 'Python interpreter not found. Install python3 or set PYTHON_BIN.'\n"
        "    exit 1\n"
        "  fi\n"
        "fi\n"
    )
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
            elif filename.startswith("validate_"):
                target_path.write_text(
                    "python .\\validate_batch.py\n",
                    encoding="utf-8",
                )
            elif filename.startswith("serve_"):
                target_path.write_text(
                    "param(\n"
                    "  [string]$Host='0.0.0.0',\n"
                    "  [int]$Port=8090\n"
                    ")\n"
                    "python .\\inference_server.py --host $Host --port $Port\n",
                    encoding="utf-8",
                )
            elif filename.startswith("example_"):
                target_path.write_text(
                    "param(\n"
                    "  [Parameter(Mandatory=$true)][string]$SourcePath,\n"
                    "  [double]$Conf=0.25,\n"
                    "  [string]$ServerUrl=''\n"
                    ")\n"
                    "if ([string]::IsNullOrWhiteSpace($ServerUrl)) {\n"
                    "  python .\\integration_example.py --source $SourcePath --conf $Conf\n"
                    "} else {\n"
                    "  python .\\integration_example.py --source $SourcePath --conf $Conf --server-url $ServerUrl\n"
                    "}\n",
                    encoding="utf-8",
                )
            elif filename.startswith("demo_"):
                target_path.write_text(
                    "param(\n"
                    "  [string]$Host='0.0.0.0',\n"
                    "  [int]$Port=7860,\n"
                    "  [string]$ServerUrl='http://127.0.0.1:8090',\n"
                    "  [switch]$DisableServer\n"
                    ")\n"
                    "$disableFlag = if ($DisableServer.IsPresent) { '--disable-server' } else { '' }\n"
                    "python .\\demo_app.py --host $Host --port $Port --server-url $ServerUrl $disableFlag\n",
                    encoding="utf-8",
                )
            else:
                target_path.write_text(
                    "python .\\run_local.py\n",
                    encoding="utf-8",
                )
            continue

        if filename.startswith("run_"):
            target_path.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                + bash_python_bootstrap
                + "if [ $# -lt 1 ]; then\n"
                "  echo 'Usage: bash " + filename + " <image_path>'\n"
                "  exit 1\n"
                "fi\n"
                "\"$PYTHON_BIN\" run_local.py \"$1\"\n",
                encoding="utf-8",
            )
        elif filename.startswith("validate_"):
            target_path.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                + bash_python_bootstrap
                + "\"$PYTHON_BIN\" validate_batch.py\n",
                encoding="utf-8",
            )
        elif filename.startswith("serve_"):
            target_path.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                + bash_python_bootstrap
                + "HOST=\"${1:-0.0.0.0}\"\n"
                "PORT=\"${2:-8090}\"\n"
                "\"$PYTHON_BIN\" inference_server.py --host \"$HOST\" --port \"$PORT\"\n",
                encoding="utf-8",
            )
        elif filename.startswith("example_"):
            target_path.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                + bash_python_bootstrap
                + "if [ $# -lt 1 ]; then\n"
                "  echo 'Usage: bash " + filename + " <image_or_video_path> [conf] [server_url]'\n"
                "  exit 1\n"
                "fi\n"
                "CONF=\"${2:-0.25}\"\n"
                "SERVER_URL=\"${3:-}\"\n"
                "if [ -n \"$SERVER_URL\" ]; then\n"
                "  \"$PYTHON_BIN\" integration_example.py --source \"$1\" --conf \"$CONF\" --server-url \"$SERVER_URL\"\n"
                "else\n"
                "  \"$PYTHON_BIN\" integration_example.py --source \"$1\" --conf \"$CONF\"\n"
                "fi\n",
                encoding="utf-8",
            )
        elif filename.startswith("demo_"):
            target_path.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                + bash_python_bootstrap
                + "HOST=\"${1:-0.0.0.0}\"\n"
                "PORT=\"${2:-7860}\"\n"
                "SERVER_URL=\"${3:-http://127.0.0.1:8090}\"\n"
                "\"$PYTHON_BIN\" demo_app.py --host \"$HOST\" --port \"$PORT\" --server-url \"$SERVER_URL\"\n",
                encoding="utf-8",
            )
        else:
            target_path.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                + bash_python_bootstrap
                + "\"$PYTHON_BIN\" run_local.py\n",
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


def _write_runtime_assets(package_dir: Path, target_id: str):
    profile = _target_profile(target_id)
    conda_env_name = str(profile.get("conda_env_name") or "vf_deploy")
    conda_python = str(profile.get("conda_python") or "3.10")
    docker_base_image = str(profile.get("docker_base_image") or "python:3.10-slim")

    requirements_runtime = textwrap.dedent(
        """
        ultralytics
        opencv-python-headless
        numpy
        flask
        requests
        onnx
        onnxruntime
        """
    ).strip() + "\n"
    (package_dir / "requirements.runtime.txt").write_text(requirements_runtime, encoding="utf-8")

    conda_env = textwrap.dedent(
        f"""
        name: {conda_env_name}
        channels:
          - conda-forge
        dependencies:
          - python={conda_python}
          - pip
          - pip:
              - -r requirements.runtime.txt
        """
    ).strip() + "\n"
    (package_dir / "environment.conda.yml").write_text(conda_env, encoding="utf-8")

    dockerfile_content = textwrap.dedent(
        f"""
        ARG BASE_IMAGE={docker_base_image}
        FROM ${{BASE_IMAGE}}

        ENV PYTHONDONTWRITEBYTECODE=1 \\
            PYTHONUNBUFFERED=1 \\
            PIP_NO_CACHE_DIR=1

        WORKDIR /opt/visionforge
        COPY requirements.runtime.txt ./requirements.runtime.txt

        RUN python -m pip install --upgrade pip setuptools wheel \\
            && python -m pip install -r requirements.runtime.txt

        COPY . .
        EXPOSE 8090 7860

        CMD ["python", "inference_server.py", "--host", "0.0.0.0", "--port", "8090"]
        """
    ).strip() + "\n"
    (package_dir / "Dockerfile").write_text(dockerfile_content, encoding="utf-8")

    docker_compose = textwrap.dedent(
        f"""
        services:
          vf-server:
            build:
              context: .
              dockerfile: Dockerfile
              args:
                BASE_IMAGE: ${{VISIONFORGE_BASE_IMAGE:-{docker_base_image}}}
            command: ["python", "inference_server.py", "--host", "0.0.0.0", "--port", "8090"]
            ports:
              - "8090:8090"
            restart: unless-stopped
            volumes:
              - ./server_outputs:/opt/visionforge/server_outputs
              - ./server_uploads:/opt/visionforge/server_uploads

          vf-demo:
            build:
              context: .
              dockerfile: Dockerfile
              args:
                BASE_IMAGE: ${{VISIONFORGE_BASE_IMAGE:-{docker_base_image}}}
            command: ["python", "demo_app.py", "--host", "0.0.0.0", "--port", "7860", "--server-url", "http://vf-server:8090"]
            ports:
              - "7860:7860"
            depends_on:
              - vf-server
            restart: unless-stopped
            volumes:
              - ./demo_outputs:/opt/visionforge/demo_outputs
              - ./demo_uploads:/opt/visionforge/demo_uploads
        """
    ).strip() + "\n"
    (package_dir / "docker-compose.yml").write_text(docker_compose, encoding="utf-8")

    dockerignore = textwrap.dedent(
        """
        .git
        .gitignore
        .venv
        __pycache__/
        *.pyc
        *.pyo
        *.pyd
        demo_outputs/
        demo_uploads/
        server_outputs/
        server_uploads/
        example_outputs/
        validation/predictions/
        """
    ).strip() + "\n"
    (package_dir / ".dockerignore").write_text(dockerignore, encoding="utf-8")


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

        emit_event("export/progress", {"message": "Writing runtime environment assets...", "stage": "runtime", "progress": 84})
        _write_runtime_assets(package_dir, args.target_id)

        emit_event("export/progress", {"message": "Copying branding assets...", "stage": "assets", "progress": 87})
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
