"""CPU-only counterfactual fixture runner."""

from __future__ import annotations

import time
from pathlib import Path

from jspace_oversight.agents import MockExecutorAgent
from jspace_oversight.config import DemoConfig, load_config
from jspace_oversight.environments import ToyWorkspace
from jspace_oversight.evaluation.branching import (
    create_checkpoint,
    run_paired_branches,
)
from jspace_oversight.evaluation.intervention import analyze_checkpoint
from jspace_oversight.oversight import (
    OracleOverseer,
    SimulatedOverseer,
    SimulatedOverseerConfig,
)
from jspace_oversight.paths import find_project_root
from jspace_oversight.provenance import git_state
from jspace_oversight.reporting import write_run_artifacts
from jspace_oversight.representations import mock_representation
from jspace_oversight.scenarios import load_scenarios, select_scenarios


def run_mock_demo(config_path: str | Path) -> Path:
    """Run labeled synthetic fixtures; this never invokes a model or network."""
    started = time.perf_counter()
    root = find_project_root(config_path)
    config = load_config(config_path, DemoConfig)
    scenarios_path = (
        config.scenarios_path
        if config.scenarios_path.is_absolute()
        else root / config.scenarios_path
    )
    records = select_scenarios(
        load_scenarios(scenarios_path), config.scenario_ids or None
    )
    checkpoints = []
    branches = []
    features = []
    analyses = []
    reviewer_versions: dict[str, str] = {}
    for scenario in records:
        workspace = ToyWorkspace(scenario)
        executor = MockExecutorAgent(scenario)
        checkpoint = create_checkpoint(
            workspace,
            executor,
            rng_metadata={"seed_list": config.seeds},
        )
        feature = mock_representation(checkpoint)
        checkpoint = checkpoint.model_copy(
            update={"feature_reference": feature.feature_id}
        )
        if config.reviewer_type == "oracle":
            overseer = OracleOverseer(scenario)
        else:
            reviewer_settings = config.reviewer
            overseer = SimulatedOverseer(
                scenario,
                SimulatedOverseerConfig(
                    reviewer_id=reviewer_settings.reviewer_id,
                    version=reviewer_settings.version,
                    seed=config.seeds[0],
                    false_approval_rate=reviewer_settings.false_approval_rate,
                    false_rejection_rate=reviewer_settings.false_rejection_rate,
                    revision_rate=reviewer_settings.revision_rate,
                    normalized_cost=reviewer_settings.normalized_cost,
                ),
            )
        paired = run_paired_branches(
            scenario, checkpoint, seeds=config.seeds, overseer=overseer
        )
        analysis = analyze_checkpoint(
            paired,
            risk_penalty=config.risk_penalty,
            review_cost_weight=config.review_cost_weight,
        )
        checkpoints.append(checkpoint)
        features.append(feature)
        branches.extend(paired)
        analyses.append(analysis)
        reviewer_versions[overseer.reviewer_id] = overseer.version

    output_root = (
        config.output_root
        if config.output_root.is_absolute()
        else root / config.output_root
    )
    run_dir = output_root / config.run_id
    sha, dirty = git_state(root)
    write_run_artifacts(
        run_dir,
        run_id=config.run_id,
        evidence_kind="synthetic_fixture",
        checkpoints=checkpoints,
        branches=branches,
        features=features,
        analyses=analyses,
        code_commit=sha,
        git_dirty=dirty,
        model_id=MockExecutorAgent.model_id,
        model_revision=MockExecutorAgent.model_revision,
        reviewer_versions=reviewer_versions,
        scenario_versions={item.scenario_id: item.version for item in records},
        generation_config={"mode": "deterministic_script", "temperature": 0.0},
        representation_config={
            "extractor": "synthetic_fixture_no_model_state",
            "timing": "T1_after_proposal_before_execution",
            "action_aware": True,
        },
        seed_list=config.seeds,
        runtime_seconds=time.perf_counter() - started,
    )
    return run_dir
