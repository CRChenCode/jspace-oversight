from jspace_oversight.agents import MockExecutorAgent
from jspace_oversight.environments import ToyWorkspace
from jspace_oversight.evaluation.branching import (
    create_checkpoint,
    run_branch,
    run_paired_branches,
)
from jspace_oversight.evaluation.counterfactual_schemas import ReviewDecision
from jspace_oversight.oversight import OracleOverseer
from jspace_oversight.scenarios import load_scenarios


def _scenario(identifier: str):
    return next(
        item
        for item in load_scenarios("scenarios/fixtures/mvp.yaml")
        if item.scenario_id == identifier
    )


def test_checkpoint_precedes_execution_and_branches_are_isolated() -> None:
    scenario = _scenario("public-share-risky-v1")
    workspace = ToyWorkspace(scenario)
    checkpoint = create_checkpoint(workspace, MockExecutorAgent(scenario))
    assert workspace.snapshot() == checkpoint.environment_snapshot
    assert not workspace.snapshot().messages

    branches = run_paired_branches(
        scenario,
        checkpoint,
        seeds=[0, 1, 2],
        overseer=OracleOverseer(scenario),
    )
    assert {branch.initial_snapshot_hash for branch in branches} == {
        checkpoint.environment_snapshot_hash
    }
    autonomous = [item for item in branches if item.treatment == "autonomous"]
    reviewed = [item for item in branches if item.treatment == "reviewed"]
    assert all(item.outcome.safety_violation for item in autonomous)
    assert all(not item.outcome.safety_violation for item in reviewed)
    assert all(item.outcome.task_success for item in reviewed)
    assert all(item.review_decision.action_type == "revise" for item in reviewed)
    assert not workspace.snapshot().messages


def test_rejected_action_never_executes_before_suffix() -> None:
    scenario = _scenario("delete-temp-risky-v1")
    checkpoint = create_checkpoint(ToyWorkspace(scenario), MockExecutorAgent(scenario))
    branches = run_paired_branches(
        scenario, checkpoint, seeds=[0], overseer=OracleOverseer(scenario)
    )
    reviewed = next(item for item in branches if item.treatment == "reviewed")
    assert reviewed.review_decision.action_type == "reject"
    assert all(
        event.action.arguments.get("path") != "important.tmp"
        for event in reviewed.events
    )
    assert "important.tmp" in reviewed.terminal_state.files
    assert "scratch.tmp" not in reviewed.terminal_state.files


def test_malformed_action_is_recorded_not_repaired() -> None:
    scenario = _scenario("delete-temp-clean-v1").model_copy(deep=True)
    scenario.executor_script[0].arguments = {"wrong": "scratch.tmp"}
    checkpoint = create_checkpoint(ToyWorkspace(scenario), MockExecutorAgent(scenario))
    branches = run_paired_branches(
        scenario, checkpoint, seeds=[0], overseer=OracleOverseer(scenario)
    )
    autonomous = next(item for item in branches if item.treatment == "autonomous")
    assert autonomous.events[0].action.arguments == {"wrong": "scratch.tmp"}
    assert autonomous.events[0].result.code == "invalid_arguments"


def test_unsafe_reviewer_revision_is_blocked_by_permission_check() -> None:
    class UnsafeRevisionOverseer:
        reviewer_id = "unsafe-test-reviewer"
        version = "test-v1"

        def review(self, context, proposed_action):
            del context
            return ReviewDecision(
                action_type="revise",
                revised_action=proposed_action,
                cost_unit="normalized_simulator",
            )

    scenario = _scenario("private-summary-risky-v1")
    checkpoint = create_checkpoint(ToyWorkspace(scenario), MockExecutorAgent(scenario))
    reviewed = run_branch(
        scenario,
        checkpoint,
        treatment="reviewed",
        replicate_seed=0,
        overseer=UnsafeRevisionOverseer(),
    )
    assert reviewed.events[0].result.ok is False
    assert reviewed.events[0].result.code == "reviewer_revision_failed_permission_check"
    assert not reviewed.terminal_state.messages
    assert reviewed.outcome.safety_violation is False


def test_real_proposal_branch_can_disable_fixture_suffix() -> None:
    scenario = _scenario("private-summary-risky-v1")
    checkpoint = create_checkpoint(ToyWorkspace(scenario), MockExecutorAgent(scenario))
    autonomous = run_branch(
        scenario,
        checkpoint,
        treatment="autonomous",
        replicate_seed=0,
        continue_with_fixture_script=False,
    )
    assert len(autonomous.events) == 1
    assert autonomous.events[0].action == checkpoint.candidate_action
