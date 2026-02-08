# Datasets

## Dataset YAML (high-level expectations)
A dataset YAML typically defines:
- path (dataset root)
- train and val (relative paths or lists)
- names (class names)
- nc (optional but recommended)

Key: task-specific formats differ (detect/segment/pose/classify).

Tip: Always verify:
- number of classes == len(names)
- labels exist for every image (or expected subset)
- train/val splits are correct
