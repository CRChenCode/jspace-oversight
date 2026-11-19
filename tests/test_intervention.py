import pytest

from jspace_oversight.agents import MockExecutorAgent
from jspace_oversight.environments import ToyWorkspace
from jspace_oversight.evaluation.branching import (
    create_checkpoint,
    run_paired_branches,
)
from jspace_oversight.evaluation.intervention import (
    analyze_checkpoint,
    branch_utility,
)
from jspace_oversight.oversight import OracleOverseer
from jspace_oversight.scenarios import load_scenarios


def _branches():
    scenario = next(
        item
        for item in load_scenarios("scenarios/fixtures/mvp.yaml")
        if item.scenario_id == "private-summary-risky-v1"
    )
    checkpoint = create_checkpoint(ToyWorkspace(scenario), MockExecutorAgent(scenario))
    return run_paired_branches(
        scenario,
        checkpoint,
        seeds=[0, 1, 2],
        overseer=OracleOverseer(scenario),
    )


def test_utility_and_paired_advantage() -> None:
    branches = _branches()
    analysis = analyze_checkpoint(branches, risk_penalty=5.0, review_cost_weight=0.1)
    assert analysis.autonomous_count == 3
    assert analysis.reviewed_count == 3
    assert analysis.p0_safe_task_success == 0.0
    assert analysis.tau_hat == pytest.approx(5.9)
    assert analysis.standard_error == 0.0
    assert analysis.quadrant == "rescue"


def test_utility_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        branch_utility(_branches()[0], risk_penalty=-1, review_cost_weight=0)


def test_single_pair_is_uncertain() -> None:
    analysis = analyze_checkpoint(
        _branches()[:2], risk_penalty=5.0, review_cost_weight=0.1
    )
    assert analysis.standard_error is None
    assert analysis.interval_low is None
    assert analysis.quadrant == "uncertain"


def test_bootstrap_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="bootstrap_samples"):
        analyze_checkpoint(
            _branches(),
            risk_penalty=5.0,
            review_cost_weight=0.1,
            bootstrap_samples=0,
        )
