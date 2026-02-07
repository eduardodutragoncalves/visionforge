import time
import io
import logging
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger("VisionForge.Inference")

class InferenceService:
    def __init__(self):
        self._loaded_model = None
        self._loaded_variant_path = None

    def load_model(self, variant_path: Path):
        model_path = variant_path / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        
        # Load custom model
        logger.info(f"Loading model from {model_path}")
        self._loaded_model = YOLO(str(model_path))
        self._loaded_variant_path = str(variant_path)

    def run_inference(self, variant_path: Path, image_bytes: bytes, conf_threshold=0.25):
        # Check cache
        if self._loaded_variant_path != str(variant_path):
            self.load_model(variant_path)
            
        # Run inference
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception:
            raise ValueError("Invalid image data")
        
        start_time = time.time()
        # Ultralytics inference
        results = self._loaded_model(image, conf=conf_threshold)[0]
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        predictions = []
        task = "detect"
        if getattr(results, "probs", None) is not None:
            task = "classify"
            probs = results.probs
            if hasattr(probs, "top5") and hasattr(probs, "top5conf"):
                for cls_id, conf in zip(probs.top5, probs.top5conf):
                    predictions.append({
                        "confidence": float(conf),
                        "class_id": int(cls_id),
                        "class_name": results.names[int(cls_id)]
                    })
            else:
                data = getattr(probs, "data", probs)
                values = data.tolist() if hasattr(data, "tolist") else list(data)
                ranked = sorted(enumerate(values), key=lambda x: x[1], reverse=True)[:5]
                for cls_id, conf in ranked:
                    predictions.append({
                        "confidence": float(conf),
                        "class_id": int(cls_id),
                        "class_name": results.names[int(cls_id)]
                    })
        else:
            # results.boxes is a Boxes object
            for box in results.boxes:
                # xyxy format: x1, y1, x2, y2
                b = box.xyxy[0].tolist()
                predictions.append({
                    "box": b,
                    "confidence": float(box.conf),
                    "class_id": int(box.cls),
                    "class_name": results.names[int(box.cls)]
                })
            
        return {
            "task": task,
            "predictions": predictions,
            "latency_ms": latency_ms,
            "fps_est": 1000 / latency_ms if latency_ms > 0 else 0,
            "model_path": str(variant_path)
        }

inference_service = InferenceService()
