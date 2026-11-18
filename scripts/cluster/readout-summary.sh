#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

readout_name="${1:-multiposition}"
case "${readout_name}" in
  smoke) readout_file='qwen3-4b-smoke.json' ;;
  multiposition) readout_file='qwen3-4b-smoke-multiposition.json' ;;
  pilot-validation) readout_file='qwen3-4b-pilot-risk-validation.json' ;;
  *)
    echo 'Usage: scripts/cluster/readout-summary.sh {smoke|multiposition|pilot-validation}' >&2
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
  qwen3-4b-smoke.json|qwen3-4b-smoke-multiposition.json|qwen3-4b-pilot-risk-validation.json) ;;
  *) echo 'Rejected non-allowlisted readout file.' >&2; exit 2 ;;
esac
readout_path="${scratch_root}/artifacts/readouts/${readout_file}"
test -f "${readout_path}"
"${venv_path}/bin/python" - "${readout_path}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)

print(f"path: {path}")
if isinstance(payload, dict):
    benchmark = payload["benchmark"]
    print(f"experiment_id: {payload['experiment_id']}")
    print(
        "median_baseline_hidden_state_forward_seconds: "
        f"{benchmark['median_baseline_hidden_state_forward_seconds']:.6f}"
    )
    print(
        "median_jacobian_and_logit_readout_seconds: "
        f"{benchmark['median_jacobian_and_logit_readout_seconds']:.6f}"
    )
    print(f"peak_gpu_memory_bytes: {benchmark['peak_gpu_memory_bytes']}")
    print()
    entries = payload["results"]
else:
    entries = [{"readout": payload[0]}]

for entry in entries:
    result = entry["readout"]
    if "prompt_id" in entry:
        print(
            f"prompt_id: {entry['prompt_id']} "
            f"condition={entry['condition']} "
            f"expected_intervention={entry['expected_intervention_opportunity']}"
        )

    print(f"model_revision: {result['model_revision']}")
    print(f"lens_n_prompts: {result['lens_n_prompts']}")
    print(f"input_token_count: {result.get('input_token_count')}")
    for readout in result["readouts"]:
        rankings = " | ".join(
            f"{item['token']!r}:{item['score']:.4g}"
            for item in readout["rankings"][:5]
        )
        print(
            f"{readout['method']:14s} "
            f"layer={readout['layer']:>2d} "
            f"position={readout['position']:>3d} "
            f"input={readout.get('input_token')!r:14s} "
            f"{rankings}"
        )
    print()
PY
REMOTE
