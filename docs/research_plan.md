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
