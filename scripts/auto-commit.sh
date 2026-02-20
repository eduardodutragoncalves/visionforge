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
  local -a staged_statuses=()
  local staged_raw
  staged_raw="$(git diff --cached --name-only)"
  mapfile -t staged_files <<<"$staged_raw"
  mapfile -t staged_statuses < <(git diff --cached --name-status)

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

  local docs_only=1
  local tooling_only=1
  local new_files=0
  local status_line
  for status_line in "${staged_statuses[@]}"; do
    [[ "$status_line" =~ ^A[[:space:]] ]] && new_files=$((new_files + 1))
    local file_path="${status_line#*$'\t'}"
    if [[ ! "$file_path" =~ ^(README\.md|docs/|\.agent/) ]]; then
      docs_only=0
    fi
    if [[ ! "$file_path" =~ ^(scripts/|\.githooks/|\.github/|\.agent/) ]]; then
      tooling_only=0
    fi
  done

  local diff_text
  diff_text="$(git diff --cached --no-color)"

  local type="chore"
  if (( docs_only == 1 )); then
    type="docs"
  elif grep -Eiq '\+(.*(fix|bug|error|fallback|guard|prevent|invalid|null|none|broken))' <<<"$diff_text"; then
    type="fix"
  elif (( new_files > 0 || (total_add > total_del && total_add >= 40) )); then
    type="feat"
  elif (( tooling_only == 1 )); then
    type="chore"
  elif (( total_del > total_add && total_del >= 40 )); then
    type="refactor"
  fi

  local verb
  case "$type" in
    feat) verb="implement" ;;
    fix) verb="fix" ;;
    refactor) verb="refactor" ;;
    docs) verb="document" ;;
    *) verb="$action" ;;
  esac

  local topics
  topics="$(
    printf '%s\n' "${staged_files[@]}" \
      | awk '
          BEGIN {
            split("src app index test tests spec specs package lock json ts js tsx jsx py md txt yaml yml config settings data route routes component components page pages service services utils helper helpers model models runner runners hook hooks store schema schemas dist build electron api project projects desktop engine shared workspace script scripts auto autosync sync commit watcher watch", stop_words, " ");
            for (i in stop_words) stop[stop_words[i]] = 1;
          }
          {
            file=$0;
            gsub("\\\\", "/", file);
            n=split(file, parts, "/");
            for (i=1; i<=n; i++) {
              token=tolower(parts[i]);
              gsub(/\.[a-z0-9]+$/, "", token);
              gsub(/[^a-z0-9_-]/, "", token);
              gsub(/[-_]/, " ", token);
              m=split(token, bits, " ");
              for (j=1; j<=m; j++) {
                b=bits[j];
                if (length(b) < 4) continue;
                if (b ~ /^[0-9]+$/) continue;
                if (stop[b]) continue;
                freq[b]++;
              }
            }
          }
          END {
            for (k in freq) {
              printf "%05d %s\n", freq[k], k;
            }
          }
        ' \
      | sort -r \
      | head -n 3 \
      | awk '{print $2}' \
      | paste -sd'/' -
  )"

  local message="${type}(sync): ${verb} ${scopes}"
  if [[ -n "${topics}" ]]; then
    message="${message} (${topics})"
  fi
  message="${message} (${file_count} files; +${total_add}/-${total_del})"

  # Keep subject compact for git log readability.
  if (( ${#message} > 160 )); then
    message="${type}(sync): ${verb} ${scopes} (${file_count} files; +${total_add}/-${total_del})"
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
