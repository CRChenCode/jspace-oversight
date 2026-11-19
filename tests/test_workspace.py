from jspace_oversight.environments import ToyWorkspace
from jspace_oversight.evaluation import ProposedAction
from jspace_oversight.scenarios import load_scenarios


def _scenario(identifier: str):
    return next(
        item
        for item in load_scenarios("scenarios/fixtures/mvp.yaml")
        if item.scenario_id == identifier
    )


def test_snapshot_restore_is_deterministic_and_clone_isolated() -> None:
    scenario = _scenario("delete-temp-clean-v1")
    workspace = ToyWorkspace(scenario)
    snapshot = workspace.snapshot()
    action = ProposedAction(tool="delete_file", arguments={"path": "scratch.tmp"})
    first = workspace.step(action)
    workspace.restore(snapshot)
    second = workspace.step(action)
    assert first == second

    clone = workspace.clone()
    clone.step(ProposedAction(tool="finish_task"))
    assert clone.snapshot().finished is True
    assert workspace.snapshot().finished is False


def test_invalid_call_is_logged_without_mutation() -> None:
    workspace = ToyWorkspace(_scenario("delete-temp-clean-v1"))
    before = workspace.snapshot()
    result = workspace.step(
        ProposedAction(
            tool="delete_file", arguments={"path": "scratch.tmp", "force": True}
        )
    )
    assert result.ok is False
    assert result.code == "invalid_arguments"
    assert workspace.snapshot() == before


def test_evaluator_truth_is_not_in_actor_observations() -> None:
    workspace = ToyWorkspace(_scenario("private-summary-risky-v1"))
    executor = workspace.observe("executor")
    reviewer = workspace.observe("reviewer")
    assert "truth" not in executor
    assert "truth" not in reviewer
    assert "sensitive_files" not in executor
    assert "sensitive_files" not in reviewer
