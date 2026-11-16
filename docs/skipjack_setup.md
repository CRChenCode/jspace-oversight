# Skipjack setup

## Safe initial deployment

The existing path `/home/jhu/cchen343/research/jspace-oversight/.venv` is an
environment associated with earlier manual work. Do not delete or overwrite the
existing directory around it. Clone the Git repository to a new sibling path,
for example `/home/jhu/cchen343/research/jspace-oversight-repo`, then point
`JSPACE_PROJECT_ROOT` at that clone.

```bash
mkdir -p /scratch/jhu/yyao69/jspace-oversight/{logs,artifacts,huggingface-cache}
git clone https://github.com/CRChenCode/jspace-oversight.git \
  /home/jhu/cchen343/research/jspace-oversight-repo
cd /home/jhu/cchen343/research/jspace-oversight-repo
module load python/3.11.9
source /home/jhu/cchen343/research/jspace-oversight/.venv/bin/activate
python -m pip install -e ".[interpretability]"
```

Run installation only after checking that it will not disrupt packages needed
by the historical environment; a fresh environment is safer if compatibility is
uncertain.

## Runtime variables

```bash
export JSPACE_PROJECT_ROOT=/home/jhu/cchen343/research/jspace-oversight-repo
export JSPACE_VENV=/home/jhu/cchen343/research/jspace-oversight/.venv
export JSPACE_SCRATCH_ROOT=/scratch/jhu/yyao69/jspace-oversight
export JSPACE_ARTIFACT_ROOT="$JSPACE_SCRATCH_ROOT/artifacts"
export HF_HOME="$JSPACE_SCRATCH_ROOT/huggingface-cache"
export HF_TOKEN_PATH="$HOME/.config/huggingface/token"
```

Protect the token file (`chmod 600`) and never put a token in the repository,
Slurm script, logs, or Scratch model cache. The Hugging Face cache and all lens
artifacts belong under Scratch.

## Submission

The templates use the verified defaults: account `yyao69_super`, partition
`a100`, one GPU/node/task, four CPUs, and 32 GB RAM. Create the log directory
before `sbatch`, because Slurm opens the output file before the job starts.

```bash
cd "$JSPACE_PROJECT_ROOT"
sbatch slurm/smoke_qwen.sbatch
sbatch slurm/smoke_jlens.sbatch
sbatch slurm/fit_jlens_smoke.sbatch
```

Command-line `sbatch` options override embedded defaults, for example:

```bash
sbatch --account="$JSPACE_SLURM_ACCOUNT" \
  --output="$JSPACE_SCRATCH_ROOT/logs/%x-%j.out" slurm/smoke_qwen.sbatch
```

Do not execute these Python entry points directly on a login node. Inspect the
printed commit/job ID and metadata after completion before starting a pilot run.
