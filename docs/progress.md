# Counterfactual MVP progress

Last updated: 2026-11-18 (America/New_York)

## Implemented in this change

- P0: audited existing Qwen, Jacobian Lens, provenance, test, and Skipjack paths.
- P1: added 16 versioned synthetic scenarios across eight clean/risky families,
  a deterministic in-memory workspace, structured tool errors, permission and
  terminal-state evaluation, and truth/observation separation.
- P2: added immutable T1 checkpoints and paired branch execution. Each branch
  restores a deep copy, verifies the initial snapshot hash, and owns its history,
  tool events, terminal state, and random-seed record.
- P3: added an evaluator-aware oracle reference and a seeded fallible simulator.
  Approve, reject, and revise have explicit continuation behavior; revised calls
  pass through the same workspace validation and outcome evaluator.
- P4 local implementation: added strict Qwen JSON proposal generation without
  silent repair, action-token boundary metadata, a J-lens T1 extraction check,
  one-action isolated real-model branches, a dedicated GPU scenario/config, and
  the allowlisted `counterfactual_p4` Slurm job. Remote execution is still
  pending, so no successful real-model result is claimed yet.
- P5/P6 infrastructure: added scenario coverage, paired utility calculation,
  sample counts, seeded percentile-bootstrap intervals when at least two paired
  seeds exist, conservative quadrant assignment, JSONL records, and a checksummed
  manifest.
- Public-export guard: synthetic fixture runs are rejected by the public export
  script. A real-model bundle would receive conservative content redaction and
  fresh checksums.

## Commands run and results

All commands used the repository virtual environment.

```text
.venv/bin/ruff format --check .
75 files already formatted

.venv/bin/ruff check .
All checks passed!

.venv/bin/pytest -q
52 passed in 0.36s

.venv/bin/python scripts/demo/run_mock_demo.py --config configs/demo/mock.yaml
results/counterfactual-mvp-synthetic-smoke

.venv/bin/python scripts/demo/analyze_run.py \
  --run-dir results/counterfactual-mvp-synthetic-smoke \
  --risk-penalty 5 --review-cost-weight 0.1
Integrity validation passed and three checkpoint analyses were printed.

.venv/bin/python scripts/export_public_demo.py \
  --run-dir results/counterfactual-mvp-synthetic-smoke --out /tmp/public-demo
Refused as intended because the source is a synthetic fixture.
```

The generated run is ignored runtime output and is labeled
`synthetic_fixture`. Its repeated scripted suffixes test the pipeline; they are
not independent empirical model samples and their interval estimates must not be
used as scientific evidence.

## Remaining work and blockers

- P4 requires a successful authorized Skipjack Slurm run to verify real Qwen
  structured proposal generation, exact action-token T1 extraction, and J-lens
  layer/position attachment. The first remote status attempt was blocked by
  interactive authentication, and the new code is not yet available remotely.
- P5 real-model repeated rollouts cannot begin until P4 passes. No real result or
  quadrant has been manufactured in its place.
- P6 needs cluster-by-family resampling once enough real checkpoints exist.
- P7 Streamlit was not added because current repository instructions exclude web
  apps and the dependencies are absent. Reconcile that repository-level scope
  before implementing a browser.
- P8 grant packaging depends on a validated, sanitized real-model evidence bundle.
  The exporter intentionally refuses the current synthetic fixture.
