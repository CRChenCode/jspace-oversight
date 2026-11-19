#!/usr/bin/env bash

set -euo pipefail
job_name="${1:-}"
case "${job_name}" in
  smoke_qwen) job_file='slurm/smoke_qwen.sbatch' ;;
  smoke_jlens) job_file='slurm/smoke_jlens.sbatch' ;;
  fit_smoke) job_file='slurm/fit_jlens_smoke.sbatch' ;;
  read_smoke) job_file='slurm/read_jlens_smoke.sbatch' ;;
  fit_pilot) job_file='slurm/fit_pilot_lens.sbatch' ;;
  read_pilot) job_file='slurm/read_pilot_validation.sbatch' ;;
  read_pilot_focus) job_file='slurm/read_pilot_focus.sbatch' ;;
  counterfactual_p4) job_file='slurm/counterfactual_gpu_demo.sbatch' ;;
  *)
    echo 'Usage: scripts/cluster/submit.sh {smoke_qwen|smoke_jlens|fit_smoke|read_smoke|fit_pilot|read_pilot|read_pilot_focus|counterfactual_p4}' >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

jspace_ssh "${job_file}" <<'REMOTE'
set -euo pipefail
project_root=$1
venv_path=$2
scratch_root=$3
job_file=$4
case "${job_file}" in
  slurm/smoke_qwen.sbatch|slurm/smoke_jlens.sbatch|slurm/fit_jlens_smoke.sbatch|slurm/read_jlens_smoke.sbatch|slurm/fit_pilot_lens.sbatch|slurm/read_pilot_validation.sbatch|slurm/read_pilot_focus.sbatch|slurm/counterfactual_gpu_demo.sbatch) ;;
  *) echo 'Rejected non-allowlisted Slurm file.' >&2; exit 2 ;;
esac
cd "${project_root}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'Refusing submission: the remote worktree is dirty.' >&2
  git status --short >&2
  exit 3
fi
export JSPACE_PROJECT_ROOT="${project_root}"
export JSPACE_VENV="${venv_path}"
export JSPACE_SCRATCH_ROOT="${scratch_root}"
export JSPACE_ARTIFACT_ROOT="${scratch_root}/artifacts"
export HF_HOME="${scratch_root}/huggingface-cache"
export HF_TOKEN_PATH="${HOME}/.config/huggingface/token"
mkdir -p "${scratch_root}/logs" "${JSPACE_ARTIFACT_ROOT}" "${HF_HOME}"
echo "Submitting ${job_file} at commit $(git rev-parse HEAD)"
sbatch "${job_file}"
REMOTE
