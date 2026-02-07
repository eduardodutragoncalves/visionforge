#!/usr/bin/env bash
set -euo pipefail

if [[ "${AUTO_SYNC_ENABLED:-1}" != "1" ]]; then
  exit 0
fi

git add -A

if git diff --cached --quiet; then
  echo "[auto-sync] No changes to commit."
  exit 0
fi

timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
message="${AUTO_SYNC_MESSAGE:-chore: autosync ${timestamp}}"

git commit -m "$message"
