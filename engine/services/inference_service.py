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
            "predictions": predictions,
            "latency_ms": latency_ms,
            "fps_est": 1000 / latency_ms if latency_ms > 0 else 0,
            "model_path": str(variant_path)
        }

inference_service = InferenceService()
