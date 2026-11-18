#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

readout_name="${1:-multiposition}"
case "${readout_name}" in
  smoke) readout_file='qwen3-4b-smoke.json' ;;
  multiposition) readout_file='qwen3-4b-smoke-multiposition.json' ;;
  *)
    echo 'Usage: scripts/cluster/readout-summary.sh {smoke|multiposition}' >&2
    exit 2
    ;;
esac

jspace_ssh "${readout_file}" <<'REMOTE'
set -euo pipefail
project_root=$1
venv_path=$2
scratch_root=$3
readout_file=$4
case "${readout_file}" in
  qwen3-4b-smoke.json|qwen3-4b-smoke-multiposition.json) ;;
  *) echo 'Rejected non-allowlisted readout file.' >&2; exit 2 ;;
esac
readout_path="${scratch_root}/artifacts/readouts/${readout_file}"
test -f "${readout_path}"
"${venv_path}/bin/python" - "${readout_path}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    result = json.load(handle)[0]

print(f"path: {path}")
print(f"model_revision: {result['model_revision']}")
print(f"lens_n_prompts: {result['lens_n_prompts']}")
print(f"input_token_count: {result.get('input_token_count')}")
print()
for readout in result["readouts"]:
    rankings = " | ".join(
        f"{item['token']!r}:{item['score']:.4g}"
        for item in readout["rankings"][:10]
    )
    print(
        f"{readout['method']:14s} "
        f"position={readout['position']:>3d} "
        f"input={readout.get('input_token')!r:14s} "
        f"{rankings}"
    )
PY
REMOTE
