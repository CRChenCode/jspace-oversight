import json
from pathlib import Path

import pytest

from jspace_oversight.agents import MockExecutorAgent
from jspace_oversight.environments import ToyWorkspace
from jspace_oversight.evaluation.branching import (
    create_checkpoint,
    run_paired_branches,
)
from jspace_oversight.evaluation.intervention import analyze_checkpoint
from jspace_oversight.oversight import OracleOverseer
from jspace_oversight.reporting import validate_run_artifacts, write_run_artifacts
from jspace_oversight.representations import mock_representation
from jspace_oversight.scenarios import load_scenarios


def test_manifest_checksums_detect_mutation(tmp_path: Path) -> None:
    scenario = next(
        item
        for item in load_scenarios("scenarios/fixtures/mvp.yaml")
        if item.scenario_id == "public-share-risky-v1"
    )
    checkpoint = create_checkpoint(ToyWorkspace(scenario), MockExecutorAgent(scenario))
    feature = mock_representation(checkpoint)
    checkpoint = checkpoint.model_copy(update={"feature_reference": feature.feature_id})
    branches = run_paired_branches(
        scenario,
        checkpoint,
        seeds=[0, 1],
        overseer=OracleOverseer(scenario),
    )
    analysis = analyze_checkpoint(branches, risk_penalty=5.0, review_cost_weight=0.1)
    run_dir = tmp_path / "run"
    write_run_artifacts(
        run_dir,
        run_id="unit-test",
        evidence_kind="synthetic_fixture",
        checkpoints=[checkpoint],
        branches=branches,
        features=[feature],
        analyses=[analysis],
        code_commit=None,
        git_dirty=None,
        model_id=MockExecutorAgent.model_id,
        model_revision=MockExecutorAgent.model_revision,
        reviewer_versions={"oracle": "oracle-v1"},
        scenario_versions={scenario.scenario_id: scenario.version},
        generation_config={"temperature": 0.0},
        representation_config={"extractor": "synthetic"},
        seed_list=[0, 1],
    )
    manifest = validate_run_artifacts(run_dir)
    assert manifest.counts["branches"] == 4
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["evidence_kind"] == "synthetic_fixture"
    assert "not empirical" in summary["warning"]

    with (run_dir / "branches.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_run_artifacts(run_dir)
