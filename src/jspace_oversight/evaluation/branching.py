"""T1 checkpoint creation and isolated counterfactual branch execution."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from jspace_oversight.agents import MockExecutorAgent
from jspace_oversight.environments import ToyWorkspace
from jspace_oversight.evaluation.counterfactual_schemas import (
    BranchRecord,
    DecisionCheckpoint,
    ProposedAction,
    ReviewContext,
    ToolResult,
    TraceEvent,
)
from jspace_oversight.oversight import Overseer
from jspace_oversight.oversight.reviewers import action_violations
from jspace_oversight.risks import EVALUATOR_VERSION, evaluate_trace


def stable_hash(value: Any) -> str:
    """Hash a JSON-compatible value using canonical serialization."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def create_checkpoint(
    workspace: ToyWorkspace,
    executor: MockExecutorAgent,
    *,
    history: list[TraceEvent] | None = None,
    script_offset: int = 0,
    sampling_config: dict[str, Any] | None = None,
    rng_metadata: dict[str, Any] | None = None,
    feature_reference: str | None = None,
) -> DecisionCheckpoint:
    """Capture an immutable state after proposal generation and before execution."""
    candidate = executor.generate_proposal(script_offset)
    if candidate is None:
        raise ValueError("executor has no proposal at the requested script offset")
    return create_checkpoint_from_proposal(
        workspace,
        candidate,
        model_id=executor.model_id,
        model_revision=executor.model_revision,
        history=history,
        script_offset=script_offset,
        sampling_config=sampling_config,
        rng_metadata=rng_metadata,
        feature_reference=feature_reference,
    )


def create_checkpoint_from_proposal(
    workspace: ToyWorkspace,
    candidate: ProposedAction,
    *,
    model_id: str,
    model_revision: str | None,
    history: list[TraceEvent] | None = None,
    script_offset: int = 0,
    sampling_config: dict[str, Any] | None = None,
    rng_metadata: dict[str, Any] | None = None,
    feature_reference: str | None = None,
) -> DecisionCheckpoint:
    """Capture an externally supplied proposal at the same pre-execution T1."""
    observation = workspace.observe("executor")
    snapshot = workspace.snapshot()
    identity = {
        "scenario_id": workspace.scenario.scenario_id,
        "scenario_version": workspace.scenario.version,
        "snapshot": snapshot.model_dump(mode="json"),
        "candidate": candidate.model_dump(mode="json"),
        "script_offset": script_offset,
    }
    checkpoint_id = f"cp-{stable_hash(identity)[:20]}"
    return DecisionCheckpoint(
        checkpoint_id=checkpoint_id,
        scenario_id=workspace.scenario.scenario_id,
        scenario_version=workspace.scenario.version,
        environment_snapshot=snapshot,
        history=[event.model_copy(deep=True) for event in history or []],
        candidate_action=candidate,
        script_offset=script_offset,
        model_id=model_id,
        model_revision=model_revision,
        sampling_config=sampling_config or {"temperature": 0.0},
        rng_metadata=rng_metadata or {},
        external_observation_hash=stable_hash(observation),
        environment_snapshot_hash=stable_hash(snapshot),
        feature_reference=feature_reference,
    )


def _record_action(
    workspace: ToyWorkspace,
    events: list[TraceEvent],
    action: ProposedAction,
    *,
    source: str,
) -> None:
    result = workspace.step(action)
    events.append(
        TraceEvent(
            step=len(events),
            action=action.model_copy(deep=True),
            result=result,
            source=source,
        )
    )


def _record_revised_action(
    workspace: ToyWorkspace,
    events: list[TraceEvent],
    action: ProposedAction,
) -> None:
    violations = action_violations(workspace.scenario, action)
    if violations:
        events.append(
            TraceEvent(
                step=len(events),
                action=action.model_copy(deep=True),
                result=ToolResult(
                    ok=False,
                    code="reviewer_revision_failed_permission_check",
                    output={"violations": violations},
                ),
                source="reviewer",
            )
        )
        return
    _record_action(workspace, events, action, source="reviewer")


def review_context(
    workspace: ToyWorkspace, checkpoint: DecisionCheckpoint
) -> ReviewContext:
    scenario = workspace.scenario
    return ReviewContext(
        checkpoint_id=checkpoint.checkpoint_id,
        user_goal=scenario.user_goal,
        authorized_actions=scenario.authorized_actions,
        allowed_recipients=scenario.allowed_recipients,
        recent_events=[
            event.model_copy(deep=True) for event in checkpoint.history[-3:]
        ],
        observation=workspace.observe("reviewer"),
    )


def run_branch(
    scenario,
    checkpoint: DecisionCheckpoint,
    *,
    treatment: str,
    replicate_seed: int,
    overseer: Overseer | None = None,
    continue_with_fixture_script: bool = True,
) -> BranchRecord:
    """Restore and execute one branch without sharing state with another branch."""
    if treatment not in {"autonomous", "reviewed"}:
        raise ValueError("treatment must be 'autonomous' or 'reviewed'")
    if treatment == "reviewed" and overseer is None:
        raise ValueError("reviewed branches require an overseer")
    if treatment == "autonomous" and overseer is not None:
        raise ValueError("autonomous branches must not receive an overseer")

    workspace = ToyWorkspace(scenario)
    workspace.restore(checkpoint.environment_snapshot)
    initial_hash = stable_hash(workspace.snapshot())
    if initial_hash != checkpoint.environment_snapshot_hash:
        raise ValueError("restored branch state does not match checkpoint hash")
    events = [event.model_copy(deep=True) for event in checkpoint.history]
    decision = None
    reviewer_id = None

    if treatment == "autonomous":
        _record_action(
            workspace, events, checkpoint.candidate_action, source="executor"
        )
    else:
        assert overseer is not None
        reviewer_id = overseer.reviewer_id
        decision = overseer.review(
            review_context(workspace, checkpoint), checkpoint.candidate_action
        )
        if decision.action_type == "approve":
            _record_action(
                workspace, events, checkpoint.candidate_action, source="executor"
            )
        elif decision.action_type == "revise":
            assert decision.revised_action is not None
            _record_revised_action(workspace, events, decision.revised_action)

    if continue_with_fixture_script:
        executor = MockExecutorAgent(scenario)
        remaining_budget = max(0, scenario.max_steps - len(events))
        for proposal in executor.remaining_proposals(
            checkpoint.script_offset + 1, events, remaining_budget
        ):
            if workspace.snapshot().finished:
                break
            _record_action(workspace, events, proposal, source="executor")

    terminal = workspace.snapshot()
    outcome = evaluate_trace(scenario, events, terminal)
    branch_identity = {
        "checkpoint_id": checkpoint.checkpoint_id,
        "treatment": treatment,
        "reviewer_id": reviewer_id,
        "seed": replicate_seed,
    }
    return BranchRecord(
        branch_id=f"br-{stable_hash(branch_identity)[:20]}",
        checkpoint_id=checkpoint.checkpoint_id,
        treatment=treatment,
        reviewer_id=reviewer_id,
        replicate_seed=replicate_seed,
        initial_snapshot_hash=initial_hash,
        events=events,
        review_decision=decision,
        terminal_state=terminal,
        outcome=outcome,
        review_calls=1 if decision else 0,
        review_cost=decision.cost if decision else 0.0,
        completion_steps=len(events) - len(checkpoint.history),
        evaluator_version=EVALUATOR_VERSION,
    )


def run_paired_branches(
    scenario,
    checkpoint: DecisionCheckpoint,
    *,
    seeds: list[int],
    overseer: Overseer,
    continue_with_fixture_script: bool = True,
) -> list[BranchRecord]:
    records: list[BranchRecord] = []
    for seed in seeds:
        records.append(
            run_branch(
                scenario,
                checkpoint,
                treatment="autonomous",
                replicate_seed=seed,
                continue_with_fixture_script=continue_with_fixture_script,
            )
        )
        records.append(
            run_branch(
                scenario,
                checkpoint,
                treatment="reviewed",
                replicate_seed=seed,
                overseer=overseer,
                continue_with_fixture_script=continue_with_fixture_script,
            )
        )
    return records
