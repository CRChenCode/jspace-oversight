# Counterfactual MVP limitations

The CPU runner is a deterministic infrastructure fixture. It does not execute
Qwen, capture a real hidden state, apply a real fitted Jacobian Lens, simulate a
human, or estimate a general intervention effect. Repeating a scripted suffix at
three seed labels does not create independent stochastic evidence.

The seeded reviewer is a configurable error process, not a human-performance
model. The oracle sees evaluator truth and is only an upper-bound reference.
Vocabulary readouts are uncalibrated rankings, not intent, risk probabilities, or
causal explanations. T1 occurs after the candidate action has been generated, so
features derived from its action tokens are action-aware.

Quadrant labels summarize a chosen utility and threshold. They can change with
risk and review-cost weights. Single-pair analyses are marked uncertain; zero
variance in a deterministic fixture is not evidence of precise real-world effect.
Future operational claims require template-held-out, budget-matched online
evaluation and family-clustered uncertainty.

The repository currently has no read-only web interface. Static artifacts and
script commands are provided until the repository-level prohibition on web apps
is explicitly reconciled. No public empirical result exists in this change.
