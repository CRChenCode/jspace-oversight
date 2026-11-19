"""Run one real Qwen T1 checkpoint with J-lens and isolated paired branches."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from jspace_oversight.agents import (
    ProposalValidationError,
    QwenExecutorAgent,
    validate_action_readout_positions,
)
from jspace_oversight.config import GpuDemoConfig, ModelConfig, load_config
from jspace_oversight.environments import ToyWorkspace
from jspace_oversight.evaluation.branching import (
    create_checkpoint_from_proposal,
    run_paired_branches,
    stable_hash,
)
from jspace_oversight.evaluation.intervention import analyze_checkpoint
from jspace_oversight.lenses.fitting import resolve_runtime_path
from jspace_oversight.lenses.readout import apply_loaded_readouts
from jspace_oversight.models import load_causal_lm
from jspace_oversight.oversight import OracleOverseer
from jspace_oversight.paths import find_project_root
from jspace_oversight.provenance import git_state
from jspace_oversight.reporting import write_run_artifacts
from jspace_oversight.representations import representation_from_readout
from jspace_oversight.scenarios import load_scenarios


def _selected_scenario(config: GpuDemoConfig, root: Path):
    path = resolve_runtime_path(config.scenarios_path, project_root=root)
    matches = [
        scenario
        for scenario in load_scenarios(path)
        if scenario.scenario_id == config.scenario_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one scenario {config.scenario_id!r}, found {len(matches)}"
        )
    return matches[0]


def _write_generation_failure(
    output_root: Path,
    *,
    experiment_id: str,
    error: ProposalValidationError,
) -> Path:
    failure_root = output_root / "failures"
    failure_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = failure_root / f"{experiment_id}-{timestamp}.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "failure_type": "proposal_validation",
                "errors": error.errors,
                "raw_text": error.raw_text,
                "raw_text_sha256": hashlib.sha256(error.raw_text.encode()).hexdigest(),
                "timestamp": datetime.now(UTC).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def run(config_path: str | Path) -> Path:
    try:
        import jlens
        import torch
    except ImportError as error:
        raise RuntimeError(
            "GPU demo requires the interpretability dependency group"
        ) from error

    started = time.perf_counter()
    root = find_project_root(config_path)
    config = load_config(config_path, GpuDemoConfig)
    model_config_path = resolve_runtime_path(
        config.model_config_path, project_root=root
    )
    lens_path = resolve_runtime_path(
        config.lens_path,
        project_root=root,
        external_root="JSPACE_ARTIFACT_ROOT",
        strip_prefix="artifacts",
    )
    output_root = resolve_runtime_path(
        config.output_root,
        project_root=root,
        external_root="JSPACE_ARTIFACT_ROOT",
        strip_prefix="artifacts",
    )
    scenario = _selected_scenario(config, root)
    model_config = load_config(model_config_path, ModelConfig)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    loaded = load_causal_lm(model_config)
    model_revision = loaded.resolved_revision or loaded.requested_revision
    executor = QwenExecutorAgent(
        loaded.model,
        loaded.tokenizer,
        model_id=model_config.model_id,
        model_revision=model_revision,
    )
    workspace = ToyWorkspace(scenario)
    try:
        generation = executor.generate_proposal(
            workspace.observe("executor"), [], config.generation
        )
    except ProposalValidationError as error:
        failure_path = _write_generation_failure(
            output_root, experiment_id=config.experiment_id, error=error
        )
        raise RuntimeError(
            f"proposal validation failed; recorded at {failure_path}"
        ) from error

    validate_action_readout_positions(
        config.positions,
        decision_count=generation.decision_token_count,
        action_start=generation.action_start_token_index,
        action_end=generation.action_end_token_index_exclusive,
    )
    if generation.decision_token_count > config.max_seq_len:
        raise ValueError(
            "decision prompt would be truncated before J-lens extraction: "
            f"{generation.decision_token_count} > {config.max_seq_len}"
        )
    chat_template = getattr(loaded.tokenizer, "chat_template", None)
    sampling_config = {
        **config.generation.model_dump(mode="json"),
        "tokenizer_class": type(loaded.tokenizer).__name__,
        "chat_template_sha256": (
            hashlib.sha256(chat_template.encode()).hexdigest()
            if isinstance(chat_template, str)
            else None
        ),
        "prompt_sha256": generation.prompt_sha256,
        "raw_output_sha256": generation.raw_output_sha256,
        "prompt_token_count": generation.prompt_token_count,
        "generated_token_count": generation.generated_token_count,
        "decision_token_count": generation.decision_token_count,
        "action_start_token_index": generation.action_start_token_index,
        "action_end_token_index_exclusive": (
            generation.action_end_token_index_exclusive
        ),
    }
    checkpoint = create_checkpoint_from_proposal(
        workspace,
        generation.action,
        model_id=model_config.model_id,
        model_revision=model_revision,
        sampling_config=sampling_config,
        rng_metadata={"proposal_seed": config.generation.seed},
    )

    lens = jlens.JacobianLens.load(str(lens_path))
    wrapped = jlens.from_hf(loaded.model, loaded.tokenizer)
    decision_text = generation.formatted_prompt + generation.raw_text
    readout = apply_loaded_readouts(
        wrapped,
        loaded.tokenizer,
        decision_text,
        lens=lens,
        lens_path=lens_path,
        model_id=model_config.model_id,
        model_revision=model_revision,
        layers=config.layers,
        positions=config.positions,
        top_k=config.top_k,
        max_seq_len=config.max_seq_len,
        include_logit_lens=False,
    )
    if readout.input_token_count != generation.decision_token_count:
        raise RuntimeError(
            "J-lens tokenization does not match the recorded decision boundary: "
            f"{readout.input_token_count} != {generation.decision_token_count}"
        )
    feature = representation_from_readout(
        checkpoint,
        readout,
        raw_artifact_hash=stable_hash(readout),
        decision_token_count=generation.decision_token_count,
        action_start_token_index=generation.action_start_token_index,
        action_end_token_index_exclusive=(generation.action_end_token_index_exclusive),
    )
    checkpoint = checkpoint.model_copy(update={"feature_reference": feature.feature_id})
    overseer = OracleOverseer(scenario)
    branches = run_paired_branches(
        scenario,
        checkpoint,
        seeds=[config.generation.seed],
        overseer=overseer,
        continue_with_fixture_script=False,
    )
    analysis = analyze_checkpoint(
        branches,
        risk_penalty=config.risk_penalty,
        review_cost_weight=config.review_cost_weight,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{config.experiment_id}-seed{config.generation.seed}-{timestamp}"
    run_dir = output_root / run_id
    sha, dirty = git_state(root)
    peak_memory = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    )
    write_run_artifacts(
        run_dir,
        run_id=run_id,
        evidence_kind="real_model_run",
        checkpoints=[checkpoint],
        branches=branches,
        features=[feature],
        analyses=[analysis],
        code_commit=sha,
        git_dirty=dirty,
        model_id=model_config.model_id,
        model_revision=model_revision,
        reviewer_versions={overseer.reviewer_id: overseer.version},
        scenario_versions={scenario.scenario_id: scenario.version},
        generation_config=sampling_config,
        representation_config={
            "extractor": "existing_jacobian_lens_adapter",
            "timing": "T1_after_proposal_before_execution",
            "action_aware": True,
            "lens_path": str(lens_path),
            "layers": config.layers,
            "positions": config.positions,
            "top_k": config.top_k,
            "max_seq_len": config.max_seq_len,
        },
        seed_list=[config.generation.seed],
        runtime_seconds=time.perf_counter() - started,
        peak_memory_bytes=peak_memory,
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo/qwen.yaml")
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
