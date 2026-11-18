#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

echo "Local repository: ${JSPACE_LOCAL_ROOT}"
echo "Cluster config:   ${JSPACE_CLUSTER_ENV}"
echo "SSH target:       ${JSPACE_SSH_HOST}"
echo "Remote project:   ${JSPACE_REMOTE_PROJECT_ROOT}"
echo "Remote venv:      ${JSPACE_REMOTE_VENV}"
echo "Remote scratch:   ${JSPACE_REMOTE_SCRATCH_ROOT}"
echo
ssh -G "${JSPACE_SSH_HOST}" 2>/dev/null \
  | awk '/^(hostname|user|port|controlmaster|controlpath|controlpersist) /'

if [[ "${1:-}" == "--connect" ]]; then
  jspace_ssh <<'REMOTE'
set -euo pipefail
project_root=$1
venv_path=$2
scratch_root=$3
printf 'remote_host=%s\n' "$(hostname)"
printf 'project_root=%s\n' "${project_root}"
printf 'venv_python=%s\n' "${venv_path}/bin/python"
printf 'scratch_root=%s\n' "${scratch_root}"
test -d "${project_root}/.git"
test -x "${venv_path}/bin/python"
test -d "${scratch_root}"
REMOTE
fi
