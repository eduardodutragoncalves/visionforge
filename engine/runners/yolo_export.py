import sys
import argparse
import time
import json
import traceback
from pathlib import Path
from ultralytics import YOLO

def emit_event(event_type, payload):
    envelope = {
        "type": event_type,
        "payload": payload,
        "timestamp": time.time()
    }
    print(f"VF_EVENT: {json.dumps(envelope)}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-path", required=True)
    parser.add_argument("--format", default="onnx")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    variant_path = Path(args.variant_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = variant_path / "model.pt"

    emit_event("export/started", {
        "variant": str(variant_path),
        "format": args.format
    })

    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        emit_event("export/progress", {"message": "Loading model..."})
        model = YOLO(str(model_path))

        emit_event("export/progress", {"message": f"Exporting to {args.format}..."})
        
        # Run export
        # Ultralytics export returns the filename of the exported model
        start_t = time.time()
        exported_path = model.export(format=args.format, dynamic=True) # dynamic for safe onnx
        duration = time.time() - start_t
        
        # The exported file is usually saved alongside the pt file or in cwd. 
        # YOLO export path behavior: usually returns absolute path.
        # We might want to move it to output_dir if it's not there.
        
        final_path = Path(exported_path)
        
        # If possible, move to our artifacts folder if it's not there
        # but usually user wants it in the variant folder or separate export folder
        # For now, we leave it where YOLO puts it or move if needed.
        # Let's ensure strict location: `output_dir/model.onnx`
        
        target_path = output_dir / f"model.{args.format}"
        if final_path != target_path:
             import shutil
             # shutil.move(str(final_path), str(target_path)) # Optional: Enforce names
             pass

        emit_event("export/completed", {
            "path": str(final_path),
            "duration": duration,
            "format": args.format
        })

    except Exception as e:
        traceback.print_exc()
        emit_event("export/failed", {"error": str(e)})
        sys.exit(1)

if __name__ == "__main__":
    main()
