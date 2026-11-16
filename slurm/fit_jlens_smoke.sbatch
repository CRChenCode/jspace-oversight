#!/usr/bin/env bash
#SBATCH --job-name=jspace-fit-smoke
#SBATCH --account=yyao69_super
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/jhu/yyao69/jspace-oversight/logs/%x-%j.out

set -euo pipefail
module load "${JSPACE_PYTHON_MODULE:-python/3.11.9}"
PROJECT_ROOT="${JSPACE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
VENV_PATH="${JSPACE_VENV:-/home/jhu/cchen343/research/jspace-oversight/.venv}"
SCRATCH_ROOT="${JSPACE_SCRATCH_ROOT:-/scratch/jhu/yyao69/jspace-oversight}"
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/huggingface-cache}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HOME}/.config/huggingface/token}"
export JSPACE_ARTIFACT_ROOT="${JSPACE_ARTIFACT_ROOT:-${SCRATCH_ROOT}/artifacts}"
mkdir -p "${HF_HOME}" "${JSPACE_ARTIFACT_ROOT}"
cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"
echo "git_commit=$(git rev-parse HEAD) slurm_job_id=${SLURM_JOB_ID:-unknown}"
python scripts/smoke/fit_jlens_smoke.py --config configs/fitting/smoke.yaml
