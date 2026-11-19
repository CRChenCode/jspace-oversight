# Repository audit for the counterfactual MVP

Audit date: 2026-11-18 (America/New_York)

This audit preceded the counterfactual simulator changes. The repository began
with a working `src/` package, strict YAML configuration, lazy Hugging Face model
loading, thin wrappers around the public Jacobian Lens API, provenance capture,
paired readout analysis, CPU tests, and versioned Skipjack wrappers. The current
implementation extends those paths rather than replacing them.

## Reused code

- `src/jspace_oversight/models/loading.py` remains the model-loading boundary.
  It records requested and resolved revisions and performs no import-time load.
- `src/jspace_oversight/lenses/` remains the sole Jacobian Lens fitting/readout
  implementation. Counterfactual representation records adapt its structured
  `PromptReadout`; they do not fork or vendor the upstream library.
- `src/jspace_oversight/provenance.py` supplies Git state and dependency version
  capture. The demo manifest also records scenario, reviewer, environment,
  generation, representation, seed, count, and checksum fields.
- Existing `scripts/cluster/` and `slurm/` launchers were preserved. The P4
  implementation later added one named, allowlisted Slurm job instead of exposing
  arbitrary remote commands. No package install or remote environment change was
  performed during the audit.
- Existing historical evaluation schemas and paired readout metrics remain
  available and unchanged.

## Dependency and scope findings

- The base CPU environment already has all dependencies needed by the simulator:
  Python 3.11+, Pydantic, PyYAML, and NumPy.
- Qwen and Jacobian Lens remain optional GPU dependencies. The existing pinned
  model configuration names `Qwen/Qwen3-4B`; no local GPU run was attempted.
- Streamlit, pandas, and Plotly are not project dependencies. More importantly,
  the repository instructions explicitly exclude web applications. Therefore
  this change provides integrity-checked static artifacts and script commands,
  but does not add the proposed Streamlit browser.
- AgentDojo and tau2-bench are not installed or vendored. The MVP simulator is an
  isolated in-memory environment, not a benchmark-wide agent implementation.

## Compatibility assessment

No incompatible version requirement was found for the CPU implementation. The
new modules are import-side-effect free and do not load models or make network
requests. Existing lens, Qwen smoke, analysis, and cluster entry points were not
rewritten. A future GPU checkpoint runner should call the existing loaders and
readout functions, record exact action-token boundaries, and be executed only in
an authorized Slurm allocation.
