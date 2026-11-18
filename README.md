# JSpace-Oversight

**Working title:** Latent-State-Informed Human Oversight of Autonomous AI Agents

JSpace-Oversight is a reproducible research codebase for asking whether internal
language-model representations can improve when human oversight is requested and
what evidence is presented to a supervisor. The first mechanism studied is the
[Jacobian Lens](https://github.com/anthropics/jacobian-lens). Lens readouts are
hypotheses about model representations—not probabilities, calibrated risk scores,
or verified statements about intent or future behavior.

## Research questions and pipeline

The central question is whether latent readouts add useful information beyond
instructions, actions, tool calls and responses, external risk indicators, and
legitimately accessible plans or reasoning traces. The planned pipeline is:

1. Fit and validate Jacobian Lenses on open-weight language models.
2. Extract latent readouts at agent decision points.
3. Relate readouts to observable behavior and carefully separated risk labels.
4. Compare oversight-triggering policies under equal intervention budgets.
5. Test whether timing and presentation improve human decisions.

## Status

This repository currently provides typed YAML configuration, lazy Hugging Face
model loading, thin wrappers over the public `jlens` fitting/readout APIs,
experiment provenance, future evaluation schemas, Skipjack Slurm templates, and
CPU-only tests/CI.

Separate manual feasibility work on Skipjack successfully loaded Qwen3-4B,
wrapped it with `jlens.from_hf`, fitted a one-prompt lens at layer 10, reloaded
it, and produced a readout. That run used 12.6 seconds for fitting and peaked at
8.35 GB of GPU memory. It was performed in the original Skipjack working
directory, not through this repository. It is exploratory—not evidence of
reliable prompt-injection detection or oversight benefit. Exact historical
metrics are in [docs/experimental_progress.md](docs/experimental_progress.md).

Planned work includes approximately 100–200 AgentDojo episodes (subject to
resources and compatibility), tau2-bench exploration, external-only,
PlowPilot-style, J-space-only, and hybrid triggering baselines, followed by
human oversight evaluation. These experiments have not been run here.

## Local installation

Python 3.11 or newer is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

GPU/model functionality is deliberately optional:

```bash
python -m pip install -e ".[dev,interpretability]"
```

This installs the official Jacobian Lens package directly from its GitHub
repository. Installation does not download Qwen weights; weights are fetched
only when a model-loading command is explicitly run.

## Configuration and smoke commands

All experimental settings live in YAML. The historical smoke shape is captured
in `configs/fitting/smoke.yaml`:

```yaml
source_layers: [10]
max_seq_len: 64
dim_batch: 8
checkpoint_every: 1
resume: true
```

The smoke configuration uses the bundled neutral example corpus. Run the
following only on a GPU compute node:

```bash
python scripts/smoke/test_qwen.py
python scripts/smoke/test_jlens.py
python scripts/smoke/fit_jlens_smoke.py --config configs/fitting/smoke.yaml
python scripts/smoke/read_jlens_smoke.py \
  --lens /path/to/lens.pt --output /path/to/readout.json \
  --prompt "Example prompt" --layer 10 --position -2
```

Scripts are import-safe: importing them does not load a model, download files,
or start computation.

## Skipjack workflow

The intended flow is Mac development → GitHub → Skipjack GPU execution → small
metadata/results via GitHub → Mac analysis. On Skipjack, create a fresh clone at
a new path rather than overwriting the existing manual-experiment directory.
Create the Scratch log directory before submission, activate the existing
environment, install this project, and submit from the clone:

```bash
mkdir -p /scratch/jhu/yyao69/jspace-oversight/logs
git clone https://github.com/CRChenCode/jspace-oversight.git \
  /home/jhu/cchen343/research/jspace-oversight-repo
cd /home/jhu/cchen343/research/jspace-oversight-repo
source /home/jhu/cchen343/research/jspace-oversight/.venv/bin/activate
python -m pip install -e ".[interpretability]"
export JSPACE_PROJECT_ROOT="$PWD"
export JSPACE_ARTIFACT_ROOT=/scratch/jhu/yyao69/jspace-oversight/artifacts
sbatch slurm/smoke_qwen.sbatch
```

Use `sbatch --account`, `--partition`, and `--output` to override `#SBATCH`
defaults. Never run model fitting on a login node. See
[docs/skipjack_setup.md](docs/skipjack_setup.md) for credentials, caches, and
safe deployment details.

## Reproducibility and data hygiene

Each experiment metadata record includes its configuration snapshot, Git state,
dependency versions, random seed, model revision, Slurm/GPU details when
available, runtime, memory, and artifact paths. Unknown values remain JSON
`null`; they are never guessed. Source/trust information and the labels for
malicious-content exposure, model recognition, unsafe action, intervention
opportunity, and intervention benefit remain distinct.

Large artifacts, `.pt` files, caches, raw trajectories, logs, tokens, and private
data are ignored. Before committing, stage intended files and run:

```bash
python scripts/check_sensitive_files.py
git diff --cached --check
```

The checker is a guardrail, not a complete secret scanner. Review the staged
diff manually. Do not enable Git LFS or commit nested clones by default.

## Documentation

- [Research plan](docs/research_plan.md)
- [Experiment protocol](docs/experiment_protocol.md)
- [Development workflow](docs/development_workflow.md)
- [Skipjack setup](docs/skipjack_setup.md)
- [Mac agent to Skipjack workflow](docs/remote_agent_workflow.md)
- [Historical experimental progress](docs/experimental_progress.md)
