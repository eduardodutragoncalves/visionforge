import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

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


def emit_event(event_type, payload):
    envelope = {
        "type": event_type,
        "payload": payload,
        "timestamp": time.time()
    }
    print(f"VF_EVENT: {json.dumps(envelope)}", flush=True)


def _build_readme(target_id: str, model_filename: str, fmt: str) -> str:
    label = TARGET_LABELS.get(target_id, target_id)
    install_cmd = "pip install ultralytics onnxruntime opencv-python numpy"
    if target_id == "desktop_windows":
        install_cmd = "pip install ultralytics onnxruntime-directml opencv-python numpy"
    elif target_id == "edge_jetson":
        install_cmd = "pip install ultralytics onnxruntime-gpu opencv-python numpy"

    notes = {
        "edge_jetson": (
            "Optional optimization:\n"
            "- Install TensorRT/JetPack runtime matching your Jetson release.\n"
            "- Convert ONNX to TensorRT engine for lower latency when needed."
        ),
        "edge_raspberry_pi": (
            "Optional optimization:\n"
            "- Use smaller input resolution for lower RAM usage.\n"
            "- Consider NCNN/OpenVINO profile in future versions."
        ),
        "edge_orange_pi": (
            "Optional optimization:\n"
            "- For Rockchip variants, use a future RKNN profile.\n"
            "- Current package uses ONNX Runtime CPU baseline."
        ),
        "desktop_macos": (
            "Optional optimization:\n"
            "- Evaluate CoreML path for Apple Silicon in later profiles.\n"
            "- Current package uses ONNX baseline."
        ),
    }.get(target_id, "Optional optimization:\n- Tune image size and post-processing thresholds.")

    return (
        f"# Deployment Guide - {label}\n\n"
        f"This package was generated for target `{target_id}`.\n\n"
        "## What is in this package\n"
        f"- `artifacts/{model_filename}` ({fmt})\n"
        "- `manifest.json`\n"
        "- `run_local.py`\n"
        "- `run_linux.sh`\n"
        "- `run_windows.ps1`\n\n"
        "## Step-by-step setup\n"
        "### 1. Open this package directory\n"
        "- Confirm you can see the files listed above.\n"
        "- Keep this folder structure unchanged.\n\n"
        "### 2. Check runtime prerequisites\n"
        "- Python >= 3.9\n"
        "- Pip available in terminal\n"
        "- GPU drivers/runtime correctly installed for your target (if applicable)\n\n"
        "### 3. Create a dedicated Python environment\n"
        "- Linux/macOS:\n"
        "  ```bash\n"
        "  python3 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  ```\n"
        "- Windows (PowerShell):\n"
        "  ```powershell\n"
        "  python -m venv .venv\n"
        "  .\\.venv\\Scripts\\Activate.ps1\n"
        "  ```\n\n"
        "### 4. Install dependencies\n"
        "```bash\n"
        f"{install_cmd}\n"
        "```\n\n"
        "### 5. Run a first inference test\n"
        "- Linux/macOS:\n"
        "  ```bash\n"
        "  bash run_linux.sh <image_path>\n"
        "  ```\n"
        "- Windows:\n"
        "  ```powershell\n"
        "  ./run_windows.ps1 <image_path>\n"
        "  ```\n\n"
        "### 6. Validate output\n"
        "- Confirm predicted class names are expected.\n"
        "- Confirm confidence values look coherent.\n"
        "- Measure latency and memory usage on target hardware.\n\n"
        "## Troubleshooting checklist\n"
        "- If module import fails, confirm environment is activated.\n"
        "- If inference is very slow, reduce input resolution.\n"
        "- If output classes are wrong, verify class order from training dataset.\n\n"
        "## Target-specific optimization notes\n"
        f"{notes}\n"
    )


def _write_runner_scripts(package_dir: Path, model_filename: str):
    run_py = package_dir / "run_local.py"
    run_sh = package_dir / "run_linux.sh"
    run_ps1 = package_dir / "run_windows.ps1"

    run_py.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "from ultralytics import YOLO\n\n"
        "def main():\n"
        "    if len(sys.argv) < 2:\n"
        "        print('Usage: python run_local.py <image_path>')\n"
        "        sys.exit(1)\n"
        "    image_path = sys.argv[1]\n"
        f"    model_path = Path(__file__).resolve().parent / 'artifacts' / '{model_filename}'\n"
        "    model = YOLO(str(model_path))\n"
        "    results = model.predict(source=image_path, verbose=False)\n"
        "    if not results:\n"
        "        print('No results produced')\n"
        "        return\n"
        "    top = results[0]\n"
        "    print(top)\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8"
    )

    run_sh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "python run_local.py \"$1\"\n",
        encoding="utf-8"
    )
    run_sh.chmod(0o755)

    run_ps1.write_text(
        "param(\n"
        "  [Parameter(Mandatory=$true)][string]$ImagePath\n"
        ")\n"
        "python .\\run_local.py $ImagePath\n",
        encoding="utf-8"
    )


def _write_package_metadata(
    package_dir: Path,
    project_id: str,
    variant_id: str,
    target_id: str,
    fmt: str,
    artifact_path: Path
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
        "readme_path": str(package_dir / "README.md"),
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
        exported_path = model.export(format=fmt, dynamic=True)
        duration = time.time() - start_t

        final_path = Path(exported_path)
        target_path = output_dir / f"model.{fmt}"
        if final_path.resolve() != target_path.resolve():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if final_path.exists():
                import shutil
                shutil.copy2(str(final_path), str(target_path))
        else:
            target_path = final_path

        emit_event("export/progress", {"message": "Preparing deployment package...", "stage": "package", "progress": 70})
        model_filename = target_path.name
        (package_dir / "README.md").write_text(
            _build_readme(args.target_id, model_filename, fmt),
            encoding="utf-8"
        )
        emit_event("export/progress", {"message": "Writing runtime scripts...", "stage": "scripts", "progress": 85})
        _write_runner_scripts(package_dir, model_filename)
        emit_event("export/progress", {"message": "Writing manifest...", "stage": "manifest", "progress": 95})
        _write_package_metadata(
            package_dir=package_dir,
            project_id=args.project_id,
            variant_id=args.variant_id,
            target_id=args.target_id,
            fmt=fmt,
            artifact_path=target_path
        )

        emit_event("export/completed", {
            "path": str(target_path),
            "package_dir": str(package_dir),
            "duration": duration,
            "format": fmt,
            "variant_id": args.variant_id,
            "project_id": args.project_id,
            "target_id": args.target_id,
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
