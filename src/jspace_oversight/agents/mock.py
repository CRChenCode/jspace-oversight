"""Deterministic scripted executor used only for CPU tests and fixtures."""

from __future__ import annotations

from jspace_oversight.evaluation.counterfactual_schemas import (
    ProposedAction,
    ScenarioRecord,
    TraceEvent,
)


class MockExecutorAgent:
    """Replay declared proposals without silently repairing malformed actions."""

    model_id = "synthetic/mock-executor"
    model_revision = "mock-v1"

    def __init__(self, scenario: ScenarioRecord):
        self._scenario = scenario

    def generate_proposal(self, script_offset: int) -> ProposedAction | None:
        if script_offset >= len(self._scenario.executor_script):
            return None
        return self._scenario.executor_script[script_offset].model_copy(deep=True)

    def remaining_proposals(
        self, script_offset: int, history: list[TraceEvent], max_steps: int
    ) -> list[ProposedAction]:
        """Return a bounded suffix; history is accepted to mirror real adapters."""
        del history
        end = min(len(self._scenario.executor_script), script_offset + max_steps)
        return [
            action.model_copy(deep=True)
            for action in self._scenario.executor_script[script_offset:end]
        ]
