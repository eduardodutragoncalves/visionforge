# Python Recipes (Ultralytics API)

## Load + Predict
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.predict(source="image.jpg", imgsz=640, conf=0.25)

Python usage is supported by Ultralytics docs. :contentReference[oaicite:13]{index=13}

## Train
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=100, imgsz=640, batch=16, device=0)

Train mode exists for training custom models. :contentReference[oaicite:14]{index=14}

## Export
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
model.export(format="onnx")

Export formats and export mode are documented. :contentReference[oaicite:15]{index=15}
