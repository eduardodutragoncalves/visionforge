# Export & Deploy

## Why export
Export makes models deployable to different runtimes (e.g., ONNX/TensorRT). :contentReference[oaicite:16]{index=16}

## Post-export usage
You can run predict/val on exported artifacts via CLI (e.g., .onnx). :contentReference[oaicite:17]{index=17}

## Deployment checklist
- confirm dynamic/static shapes
- verify NMS behavior (if applicable)
- measure latency on target hardware
