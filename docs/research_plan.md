# Research plan

## Question

Can internal representations add useful information to observable agent behavior
for deciding when to involve a human and what to show them?

## Phases

### 0. Infrastructure and feasibility

Reproduce the historical Qwen3-4B/Jacobian Lens smoke run through versioned
configuration, then fit a more representative corpus and evaluate stability.

### 1. Lens validation and decision-point extraction

Validate Jacobian versus vanilla logit-lens readouts across layers, positions,
prompts, seeds, and corpus choices. Extract only state legitimately available at
the decision point. Completion of fitting is not scientific validation.

The next proposed fitting corpus follows the
[reference implementation](https://github.com/anthropics/jacobian-lens)'s
deterministic WikiText-103 convention: the first 100 training records containing
at least 600 characters, truncated by the lens tokenizer to 128 tokens. The
dataset revision and generated-corpus checksum must be recorded before fitting.
Corpus preparation and the eight-layer, 100-prompt infrastructure pilot have
completed. The next paired readout validation is a pipeline sanity check, not a
test of incremental oversight value: its explicit input cues can drive both
external classifiers and latent readouts. The reference paper used 1,000
pretraining-like 128-token sequences; 100 prompts is an infrastructure-scale
pilot rather than a paper-grade reproduction.

### 2. Agent evaluation

Begin with roughly 100–200 AgentDojo episodes, subject to compatibility and
resources; investigate tau2-bench afterward. Maintain separate annotations for:

1. exposure to malicious content;
2. model recognition of potentially malicious content;
3. unsafe action occurrence;
4. human intervention opportunity; and
5. benefit of intervention.

These are related but non-interchangeable outcomes.

### 3. Oversight triggering

Compare external-observation-only, PlowPilot-style intervention prediction,
J-space-only, and hybrid policies under equal intervention budgets. Before
implementation, re-check the public PlowPilot repository; do not assume an
official pretrained model or complete evaluation pipeline exists.

### 4. Human oversight

Measure unsafe-action prevention, intervention efficiency, false positives,
human decision quality, effort, and cognitive load. Randomize presentation where
appropriate and avoid post-outcome information at the intervention point.

## Interpretation constraints

Readouts may reveal useful correlates without identifying intent or causally
predicting behavior. Evaluate incremental value, calibration, robustness,
distribution shift, and selection effects. Report negative and null results.
