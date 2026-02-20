# VisionForge

Monorepo for VisionForge - Local CV Training Desktop App.

## Structure

- `apps/desktop`: Electron + React + Vite + TS
- `engine`: Python + FastAPI
- `shared`: Shared assets/schemas

## What you need

- Node.js 18+ (npm or pnpm)
- Python 3.10+ (use pyenv, conda, or system Python)
- Optional: Anaconda or Miniconda
- Optional: Docker Desktop/Engine (only for Docker runtime in training)

## Install JS dependencies

```bash
npm install
```

## Python setup (choose ONE)

### Option A: pyenv + venv (recommended for local dev)

1. Install pyenv: https://github.com/pyenv/pyenv#installation
2. Install and select a Python version:
   ```bash
   pyenv install 3.10.13
   pyenv local 3.10.13
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv engine/venv
   source engine/venv/bin/activate  # or engine\venv\Scripts\activate on Windows
   ```
4. Install engine dependencies:
   ```bash
   pip install -r engine/requirements.txt
   ```

### Option B: conda environment (Anaconda/Miniconda)

1. Create and activate a conda env:
   ```bash
   conda create -n visionforge python=3.10 -y
   conda activate visionforge
   ```
2. Install engine dependencies:
   ```bash
   pip install -r engine/requirements.txt
   ```

### Option C: system Python + venv

1. Create and activate a virtual environment:
   ```bash
   python -m venv engine/venv
   source engine/venv/bin/activate  # or engine\venv\Scripts\activate on Windows
   ```
2. Install engine dependencies:
   ```bash
   pip install -r engine/requirements.txt
   ```

## Run in development

1. Activate your Python environment (pyenv/conda/venv).
2. Run both Desktop and Engine:
   ```bash
   npm run dev
   ```

This will launch:
- Uvicorn server on http://localhost:8000
- Vite server for Electron renderer
- Electron window

## Validation

1. **Frontend:** Confirm the Electron window opens.
2. **Backend:** Check http://localhost:8000/health returns `{"status": "ok"}`.

## Automatic commits (autosync)

Automatic commit/push can run in the background and create compact, intelligent commit messages based on the current diff.

Commands:

```bash
npm run autosync:start    # start background watcher
npm run autosync:status   # check watcher status
npm run autosync:stop     # stop watcher
npm run autosync:once     # commit now (single run)
```

Behavior:
- Watches repository changes and waits for a short stable window before committing.
- Uses generated commit subjects with type + scope + short implementation topics.
- Existing `post-commit` hook pushes to `origin/<current-branch>` when configured.

Optional env vars:
- `AUTO_SYNC_ENABLED=0` disables autosync.
- `AUTO_SYNC_INTERVAL_SECONDS` controls watcher poll interval (default `8`).
- `AUTO_SYNC_STABLE_SECONDS` controls stable window before commit (default `20`).

## Training runtime options (inside the app)

The Desktop app can run training jobs using different runtimes. This does not change how you start the app; it only affects training.

### Anaconda

- Requirement: `conda` must be in your PATH and the env must already exist.
- Usage: In the Train Setup page, select **Anaconda** and provide your env name (example: `visionforge`).

### Docker

- Requirement: Docker daemon must be running.
- Usage: In the Train Setup page, select **Docker** and provide an image name (example: `ultralytics/ultralytics:latest`).
- Custom image (runner):
  ```bash
  docker build -t vf-runner -f docker/Dockerfile.runner .
  ```
