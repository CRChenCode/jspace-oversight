# Counterfactual MVP data dictionary

The strict Pydantic definitions live in
`src/jspace_oversight/evaluation/counterfactual_schemas.py`. This document is a
navigation aid; the schema code is authoritative.

- `ScenarioRecord`: versioned visible task, permissions, sandbox inputs, static
  risk setup, evaluator-only truth, and a deterministic fixture script. Truth is
  never returned by `ToyWorkspace.observe`.
- `DecisionCheckpoint`: full pre-execution sandbox snapshot, visible history,
  exact candidate action, external observation and snapshot hashes, model and
  sampling metadata, T1 timing, and optional feature reference.
- `RepresentationRecord`: extractor identity, model revision, action-aware timing,
  layers, token positions, uncalibrated readouts, optional raw-artifact hash, and
  an interpretation warning.
- `ReviewDecision`: approve/reject/revise, optional validated replacement,
  feedback, confidence, numeric cost, and explicit cost unit.
- `BranchRecord`: treatment, reviewer, replicate seed, initial-state hash, complete
  events, terminal snapshot, objective outcome, review and compute costs, and
  evaluator version.
- `CheckpointAnalysis`: branch counts, autonomous safe-task success, paired mean
  utility difference, uncertainty when estimable, conservative quadrant label,
  exact weights, and estimator name.
- `RunManifest`: evidence kind, Git/model/reviewer/scenario/environment versions,
  generation and representation settings, seeds, record counts, sanitization
  revision, and SHA-256 checksums.

The public exporter refuses `synthetic_fixture` runs. Raw weights, hidden states,
credentials, host file contents, and private reviewer prompts do not belong in a
public bundle.
