# Troubleshooting (systematic)

## Common failure classes
- CUDA / driver mismatch
- wrong dataset YAML paths
- class mismatch (names vs labels)
- OOM (batch/imgsz too high)
- slow dataloader (workers/pin_memory/storage)

## Minimal debug loop
1) run a tiny train: epochs=1 imgsz=320 batch=2 device=0
2) confirm it produces runs/... and a weights file
3) only then scale up
