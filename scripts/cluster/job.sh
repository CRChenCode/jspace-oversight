#!/usr/bin/env bash

set -euo pipefail
job_id="${1:-}"
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
  echo 'Usage: scripts/cluster/job.sh JOB_ID' >&2
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
[[ "${job_id}" =~ ^[0-9]+$ ]] || exit 2
sacct -j "${job_id}" \
  --format=JobID,JobName%28,Partition,State,ExitCode,Elapsed,MaxRSS,AllocTRES%30
echo
scontrol show job "${job_id}" 2>/dev/null \
  | grep -E 'JobState=|Reason=|StartTime=|EndTime=|NodeList=|WorkDir=' || true
REMOTE
