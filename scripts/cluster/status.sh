#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

jspace_ssh <<'REMOTE'
set -euo pipefail
project_root=$1
venv_path=$2
scratch_root=$3
cd "${project_root}"

echo '== Repository =='
git status --short --branch
git log -3 --oneline --decorate

echo
echo '== Active Slurm jobs =='
squeue -u "${USER}" -o '%.18i %.10P %.28j %.2t %.10M %.6D %R'

echo
echo '== Recent JSpace jobs (7 days) =='
start_date=$(date -d '7 days ago' +%F)
sacct -S "${start_date}" -u "${USER}" \
  --format=JobID,JobName%28,Partition,State,ExitCode,Elapsed,MaxRSS \
  | awk 'NR <= 2 || /jspace-/'

echo
echo '== Artifacts =='
artifact_root="${scratch_root}/artifacts"
if [[ -d "${artifact_root}" ]]; then
  find "${artifact_root}" -maxdepth 3 -type f \
    -printf '%TY-%Tm-%Td %TH:%TM  %10s  %p\n' | sort
else
  echo "No artifact directory at ${artifact_root}."
fi
REMOTE
