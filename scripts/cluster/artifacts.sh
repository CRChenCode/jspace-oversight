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
artifact_root="${scratch_root}/artifacts"
if [[ ! -d "${artifact_root}" ]]; then
  echo "No artifact directory at ${artifact_root}."
  exit 0
fi
find "${artifact_root}" -maxdepth 4 -type f \
  -printf '%TY-%Tm-%Td %TH:%TM  %10s  %p\n' | sort
REMOTE
