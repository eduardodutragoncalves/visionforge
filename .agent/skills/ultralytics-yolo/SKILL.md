---
name: ultralytics-yolo
description: Ultralytics YOLO playbook: tasks, modes, CLI/Python usage, dataset YAML, training/validation, export/deployment, tracking, benchmarking, and troubleshooting.
---

# Ultralytics YOLO Skill

Use this skill whenever the user is working with Ultralytics YOLO (Ultralytics framework), including training, inference, dataset setup, exporting, tracking, or benchmarking.

## Canonical Concepts (must apply)
### Tasks (what you do)
- detect, segment, classify, pose, obb (task argument in CLI)

### Modes (lifecycle stage)
- train, val, predict, export, track, benchmark :contentReference[oaicite:3]{index=3}

### CLI signature (canonical)
`yolo TASK MODE ARGS` :contentReference[oaicite:4]{index=4}

Where:
- TASK is optional (detect/segment/classify/pose/obb)
- MODE is required (train/val/predict/export/track/benchmark)
- ARGS are `arg=value` overrides

## Standard deliverables
When user asks for help, output:
1) the exact CLI command(s) and/or Python snippet
2) required files (dataset YAML, folder layout)
3) expected outputs (runs/…, weights/best.pt, metrics)
4) a short verification checklist

## Sections
- See `sections/quickstart.md`
- See `sections/cli-recipes.md`
- See `sections/python-recipes.md`
- See `sections/datasets.md`
- See `sections/training-playbook.md`
- See `sections/export-deploy.md`
- See `sections/tracking-benchmark.md`
- See `sections/troubleshooting.md`
