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

build_auto_message() {
  local -a staged_files=()
  local staged_raw
  staged_raw="$(git diff --cached --name-only)"
  mapfile -t staged_files <<<"$staged_raw"

  local file_count="${#staged_files[@]}"
  local total_add=0
  local total_del=0
  local add del path
  while IFS=$'\t' read -r add del path; do
    if [[ "${add}" =~ ^[0-9]+$ ]]; then
      total_add=$((total_add + add))
    fi
    if [[ "${del}" =~ ^[0-9]+$ ]]; then
      total_del=$((total_del + del))
    fi
  done < <(git diff --cached --numstat)

  local action="update"
  if (( total_add > 0 && total_del == 0 )); then
    action="add"
  elif (( total_add == 0 && total_del > 0 )); then
    action="remove"
  fi

  declare -A scope_counts=()
  local file scope
  for file in "${staged_files[@]}"; do
    case "$file" in
      apps/desktop/*) scope="desktop" ;;
      engine/*) scope="engine" ;;
      scripts/*) scope="tooling" ;;
      shared/*) scope="shared" ;;
      workspace/*) scope="workspace" ;;
      README.md|docs/*) scope="docs" ;;
      *) scope="misc" ;;
    esac
    scope_counts["$scope"]=$(( ${scope_counts["$scope"]:-0} + 1 ))
  done

  local scopes
  scopes="$(
    for scope in "${!scope_counts[@]}"; do
      echo "${scope_counts[$scope]}:${scope}"
    done | sort -t: -nr | head -n 2 | cut -d: -f2 | paste -sd, -
  )"
  if [[ -z "${scopes}" ]]; then
    scopes="repo"
  fi

  local previews
  previews="$(
    printf '%s\n' "${staged_files[@]}" \
      | awk -F/ '{n=NF; if (n >= 3) print $(n-2)"/"$(n-1)"/"$n; else print $0}' \
      | awk '!seen[$0]++' \
      | head -n 2 \
      | paste -sd', ' -
  )"

  local message="chore(sync): ${action} ${scopes} (${file_count} files; +${total_add}/-${total_del})"
  if [[ -n "${previews}" ]]; then
    message="${message} [${previews}]"
  fi

  # Keep subject compact for git log readability.
  if (( ${#message} > 140 )); then
    message="chore(sync): ${action} ${scopes} (${file_count} files; +${total_add}/-${total_del})"
  fi

  echo "${message}"
}

if [[ $# -gt 0 ]]; then
  message="$*"
elif [[ -n "${AUTO_SYNC_MESSAGE:-}" ]]; then
  message="${AUTO_SYNC_MESSAGE}"
else
  message="$(build_auto_message)"
fi

git commit -m "$message"
