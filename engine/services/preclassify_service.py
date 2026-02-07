import logging
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO

logger = logging.getLogger("VisionForge.Preclassify")


class PreclassifyService:
    def __init__(self):
        self._models: Dict[str, YOLO] = {}

    def _get_model(self, model_name: str) -> YOLO:
        if model_name not in self._models:
            logger.info("Loading preclassify model: %s", model_name)
            self._models[model_name] = YOLO(model_name)
        return self._models[model_name]

    def _parse_predictions(self, results, topk: int) -> List[Dict[str, Any]]:
        predictions: List[Dict[str, Any]] = []
        probs = getattr(results, "probs", None)
        if probs is None:
            return predictions
        k = max(1, min(topk, 10))
        if hasattr(probs, "top5") and hasattr(probs, "top5conf"):
            ids = list(probs.top5)[:k]
            confs = list(probs.top5conf)[:k]
            for cls_id, conf in zip(ids, confs):
                predictions.append({
                    "class_id": int(cls_id),
                    "class_name": results.names[int(cls_id)],
                    "confidence": float(conf)
                })
            return predictions
        data = getattr(probs, "data", probs)
        values = data.tolist() if hasattr(data, "tolist") else list(data)
        ranked = sorted(enumerate(values), key=lambda x: x[1], reverse=True)[:k]
        for cls_id, conf in ranked:
            predictions.append({
                "class_id": int(cls_id),
                "class_name": results.names[int(cls_id)],
                "confidence": float(conf)
            })
        return predictions

    def preclassify_images(
        self,
        image_paths: List[Tuple[str, str]],
        model_name: str,
        topk: int
    ) -> Dict[str, Any]:
        model = self._get_model(model_name)
        output: Dict[str, Any] = {}
        for image_id, image_path in image_paths:
            try:
                results = model(image_path)[0]
                output[image_id] = {
                    "predictions": self._parse_predictions(results, topk)
                }
            except Exception as exc:
                output[image_id] = {"error": str(exc), "predictions": []}
        return output


preclassify_service = PreclassifyService()
