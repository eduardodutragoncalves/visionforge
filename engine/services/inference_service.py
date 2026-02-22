import time
import io
import logging
import threading
from pathlib import Path
from PIL import Image, ImageOps

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None

logger = logging.getLogger("VisionForge.Inference")

TASK_ALIASES = {
    "detection": "detect",
    "classification": "classify",
    "segmentation": "segment",
}

class InferenceService:
    def __init__(self):
        self._loaded_model = None
        self._loaded_variant_path = None
        self._infer_lock = threading.Lock()

    def _normalize_model_task(self, model) -> None:
        raw_task = str(getattr(model, "task", "") or "").strip().lower()
        if not raw_task:
            return
        normalized_task = TASK_ALIASES.get(raw_task, raw_task)
        if normalized_task != raw_task:
            logger.info("Normalizing model task alias from '%s' to '%s' for inference", raw_task, normalized_task)
            setattr(model, "task", normalized_task)

    def load_model(self, variant_path: Path):
        if YOLO is None:
            raise RuntimeError("Ultralytics is not installed in the engine environment.")
        model_path = variant_path / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        
        # Load custom model
        logger.info(f"Loading model from {model_path}")
        self._loaded_model = YOLO(str(model_path))
        self._normalize_model_task(self._loaded_model)
        self._loaded_variant_path = str(variant_path)

    def _box_iou_xyxy(self, a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _dedupe_predictions(self, predictions, iou_threshold=0.85):
        if not predictions:
            return predictions
        ordered = sorted(predictions, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        kept = []
        for candidate in ordered:
            box = candidate.get("box")
            class_id = candidate.get("class_id")
            if not box or class_id is None:
                kept.append(candidate)
                continue
            duplicated = False
            for existing in kept:
                if existing.get("class_id") != class_id:
                    continue
                existing_box = existing.get("box")
                if not existing_box:
                    continue
                if self._box_iou_xyxy(box, existing_box) >= iou_threshold:
                    duplicated = True
                    break
            if not duplicated:
                kept.append(candidate)
        return kept

    def run_inference(
        self,
        variant_path: Path,
        image_bytes: bytes,
        conf_threshold=0.25,
        iou_threshold=0.45,
        agnostic_nms=False
    ):
        with self._infer_lock:
            # Check cache
            if self._loaded_variant_path != str(variant_path):
                self.load_model(variant_path)
            else:
                # Keep compatibility in case model.task was mutated externally.
                self._normalize_model_task(self._loaded_model)

            # Run inference
            try:
                image = Image.open(io.BytesIO(image_bytes))
                # Keep inference orientation consistent with what users see in the browser.
                image = ImageOps.exif_transpose(image).convert("RGB")
            except Exception:
                raise ValueError("Invalid image data")

            start_time = time.time()
            # Ultralytics inference
            results = self._loaded_model(
                image,
                conf=float(conf_threshold),
                iou=float(iou_threshold),
                agnostic_nms=bool(agnostic_nms),
                verbose=False
            )[0]
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
                predictions = self._dedupe_predictions(predictions, iou_threshold=0.85)

            return {
                "task": task,
                "predictions": predictions,
                "latency_ms": latency_ms,
                "fps_est": 1000 / latency_ms if latency_ms > 0 else 0,
                "model_path": str(variant_path)
            }

inference_service = InferenceService()
