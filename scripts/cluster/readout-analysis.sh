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
readout_path="${scratch_root}/artifacts/readouts/qwen3-4b-pilot-risk-validation.json"
test -f "${readout_path}"
cd "${project_root}"
"${venv_path}/bin/python" scripts/analysis/analyze_paired_readouts.py \
  --input "${readout_path}" --top 12
REMOTE
