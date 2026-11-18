#!/usr/bin/env bash

set -euo pipefail
job_id="${1:-latest}"
if [[ "${job_id}" != "latest" && ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo 'Usage: scripts/cluster/logs.sh [JOB_ID]' >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

jspace_ssh "${job_id}" <<'REMOTE'
set -euo pipefail
project_root=$1
venv_path=$2
scratch_root=$3
job_id=$4
log_root="${scratch_root}/logs"
if [[ "${job_id}" != "latest" ]]; then
  [[ "${job_id}" =~ ^[0-9]+$ ]] || exit 2
  log_path=$(find "${log_root}" -maxdepth 1 -type f -name "*-${job_id}.out" -print -quit)
else
  log_path=$(find "${log_root}" -maxdepth 1 -type f -name '*.out' \
    -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
fi
if [[ -z "${log_path:-}" ]]; then
  echo 'No matching Slurm log found.' >&2
  exit 4
fi
echo "== ${log_path} =="
tail -n 200 "${log_path}"
REMOTE
