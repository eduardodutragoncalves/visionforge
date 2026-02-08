# CLI Recipes

## Train (detect example)
yolo detect train model=yolov8n.pt data=data.yaml imgsz=640 epochs=100 batch=16 device=0

Train is one of the core modes. :contentReference[oaicite:7]{index=7}

## Validate
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml imgsz=640

Val is a core mode. :contentReference[oaicite:8]{index=8}

## Predict
yolo detect predict model=runs/detect/train/weights/best.pt source=./images conf=0.25 save=true

Predict is a core mode. :contentReference[oaicite:9]{index=9}

## Export (ONNX / TensorRT)
yolo export model=runs/detect/train/weights/best.pt format=onnx
yolo export model=runs/detect/train/weights/best.pt format=engine

Export is a core mode and supports multiple formats. :contentReference[oaicite:10]{index=10}

## Track
yolo track model=runs/detect/train/weights/best.pt source=0

Track is a core mode. :contentReference[oaicite:11]{index=11}

## Benchmark
yolo benchmark model=runs/detect/train/weights/best.pt imgsz=640
Benchmark is a core mode. :contentReference[oaicite:12]{index=12}
