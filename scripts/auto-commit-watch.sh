#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_FILE="${ROOT_DIR}/.agent/auto-commit-watch.pid"
LOG_FILE="${ROOT_DIR}/.agent/auto-commit-watch.log"

INTERVAL_SECONDS="${AUTO_SYNC_INTERVAL_SECONDS:-8}"
STABLE_SECONDS="${AUTO_SYNC_STABLE_SECONDS:-20}"

is_running() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

current_signature() {
  git -C "${ROOT_DIR}" status --porcelain | sha1sum | awk '{print $1}'
}

is_dirty() {
  ! git -C "${ROOT_DIR}" diff --quiet || ! git -C "${ROOT_DIR}" diff --cached --quiet
}

start_daemon() {
  if [[ -f "${PID_FILE}" ]]; then
    local existing_pid
    existing_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if is_running "${existing_pid}"; then
      echo "[auto-sync] Watcher already running (PID ${existing_pid})."
      exit 0
    fi
    rm -f "${PID_FILE}"
  fi

  mkdir -p "$(dirname "${PID_FILE}")"
  touch "${LOG_FILE}"
  if command -v setsid >/dev/null 2>&1; then
    setsid bash "${BASH_SOURCE[0]}" _run </dev/null >>"${LOG_FILE}" 2>&1 &
  else
    nohup bash "${BASH_SOURCE[0]}" _run </dev/null >>"${LOG_FILE}" 2>&1 &
  fi
  local pid=$!
  echo "${pid}" >"${PID_FILE}"
  echo "[auto-sync] Watcher started (PID ${pid})."
}

stop_daemon() {
  if [[ ! -f "${PID_FILE}" ]]; then
    echo "[auto-sync] Watcher is not running."
    exit 0
  fi

  local pid
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if is_running "${pid}"; then
    kill "${pid}" >/dev/null 2>&1 || true
    sleep 0.2
  fi
  rm -f "${PID_FILE}"
  echo "[auto-sync] Watcher stopped."
}

status_daemon() {
  if [[ ! -f "${PID_FILE}" ]]; then
    echo "[auto-sync] Watcher status: stopped."
    return
  fi
  local pid
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if is_running "${pid}"; then
    echo "[auto-sync] Watcher status: running (PID ${pid})."
  else
    echo "[auto-sync] Watcher status: stale PID file."
  fi
}

run_loop() {
  local last_sig=""
  local stable_since=0

  echo "[auto-sync] Loop running (interval=${INTERVAL_SECONDS}s, stable=${STABLE_SECONDS}s)."

  while true; do
    if [[ "${AUTO_SYNC_ENABLED:-1}" != "1" ]]; then
      sleep "${INTERVAL_SECONDS}"
      continue
    fi

    if is_dirty; then
      local sig
      sig="$(current_signature)"
      local now
      now="$(date +%s)"

      if [[ "${sig}" != "${last_sig}" ]]; then
        last_sig="${sig}"
        stable_since="${now}"
      else
        if (( now - stable_since >= STABLE_SECONDS )); then
          if ! git -C "${ROOT_DIR}" diff --cached --quiet || ! git -C "${ROOT_DIR}" diff --quiet; then
            echo "[auto-sync] Stable change-set detected. Creating commit..."
            "${SCRIPT_DIR}/auto-commit.sh" || true
          fi
          last_sig=""
          stable_since=0
        fi
      fi
    else
      last_sig=""
      stable_since=0
    fi

    sleep "${INTERVAL_SECONDS}"
  done
}

cmd="${1:-status}"
case "${cmd}" in
  start) start_daemon ;;
  stop) stop_daemon ;;
  restart) stop_daemon; start_daemon ;;
  status) status_daemon ;;
  _run) run_loop ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}"
    exit 1
    ;;
esac
