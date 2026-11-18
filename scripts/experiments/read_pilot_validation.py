"""Run paired pilot-lens readouts and record latency/provenance."""

from __future__ import annotations

import argparse
import statistics
import time

from jspace_oversight.config import EvaluationConfig, ModelConfig, load_config
from jspace_oversight.evaluation.validation import (
    load_validation_prompts,
    write_validation_output,
)
from jspace_oversight.lenses.fitting import resolve_runtime_path
from jspace_oversight.lenses.readout import apply_loaded_readouts
from jspace_oversight.models import format_prompt, load_causal_lm
from jspace_oversight.paths import find_project_root
from jspace_oversight.provenance import collect_provenance


def synchronize_cuda(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def baseline_forward_seconds(model, tokenizer, prompt: str, max_seq_len: int) -> float:
    """Time a forward pass that exposes hidden states at the decision point."""
    import torch

    device = next(model.parameters()).device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
    ).to(device)
    synchronize_cuda(torch)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=True, use_cache=False)
    synchronize_cuda(torch)
    elapsed = time.perf_counter() - started
    del output, inputs
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation/risk_validation.yaml")
    args = parser.parse_args()

    try:
        import jlens
        import torch
    except ImportError as error:
        raise SystemExit(
            "Install the interpretability dependency group first"
        ) from error

    run_started = time.perf_counter()
    root = find_project_root(args.config)
    config = load_config(args.config, EvaluationConfig)
    model_config_path = resolve_runtime_path(
        config.model_config_path, project_root=root
    )
    prompts_path = resolve_runtime_path(config.prompts_path, project_root=root)
    lens_path = resolve_runtime_path(
        config.lens_path,
        project_root=root,
        external_root="JSPACE_ARTIFACT_ROOT",
        strip_prefix="artifacts",
    )
    output_path = resolve_runtime_path(
        config.output_path,
        project_root=root,
        external_root="JSPACE_ARTIFACT_ROOT",
        strip_prefix="artifacts",
    )
    metadata_path = output_path.with_suffix(".metadata.json")
    prompts = load_validation_prompts(prompts_path)
    model_config = load_config(model_config_path, ModelConfig)

    started = time.perf_counter()
    loaded = load_causal_lm(model_config)
    model_load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    lens = jlens.JacobianLens.load(str(lens_path))
    wrapped = jlens.from_hf(loaded.model, loaded.tokenizer)
    lens_load_seconds = time.perf_counter() - started

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    results = []
    baseline_timings: list[float] = []
    readout_timings: list[float] = []
    for record in prompts:
        formatted = format_prompt(
            loaded.tokenizer,
            record.prompt,
            use_chat_template=model_config.use_chat_template,
        )
        baseline_seconds = baseline_forward_seconds(
            loaded.model, loaded.tokenizer, formatted, config.max_seq_len
        )
        synchronize_cuda(torch)
        started = time.perf_counter()
        readout = apply_loaded_readouts(
            wrapped,
            loaded.tokenizer,
            formatted,
            lens=lens,
            lens_path=lens_path,
            model_id=model_config.model_id,
            model_revision=(loaded.resolved_revision or loaded.requested_revision),
            layers=config.layers,
            positions=config.positions,
            top_k=config.top_k,
            max_seq_len=config.max_seq_len,
            include_logit_lens=config.include_logit_lens,
        )
        synchronize_cuda(torch)
        readout_seconds = time.perf_counter() - started
        baseline_timings.append(baseline_seconds)
        readout_timings.append(readout_seconds)
        results.append(
            {
                "prompt_id": record.prompt_id,
                "pair_id": record.pair_id,
                "condition": record.condition,
                "expected_intervention_opportunity": (
                    record.expected_intervention_opportunity
                ),
                "baseline_hidden_state_forward_seconds": baseline_seconds,
                "jacobian_and_logit_readout_seconds": readout_seconds,
                "readout": readout.model_dump(mode="json"),
            }
        )

    runtime_seconds = time.perf_counter() - run_started
    peak_memory = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    )
    benchmark = {
        "model_load_seconds": model_load_seconds,
        "lens_load_and_wrap_seconds": lens_load_seconds,
        "median_baseline_hidden_state_forward_seconds": statistics.median(
            baseline_timings
        ),
        "median_jacobian_and_logit_readout_seconds": statistics.median(readout_timings),
        "total_runtime_seconds": runtime_seconds,
        "peak_gpu_memory_bytes": peak_memory,
    }
    payload = {
        "experiment_id": config.experiment_id,
        "configuration": config.model_dump(mode="json", by_alias=True),
        "benchmark": benchmark,
        "results": results,
        "warning": (
            "This paired prompt set is a pipeline sanity check with explicit input "
            "cues. Raw rankings are uncalibrated and do not establish incremental "
            "oversight value, model intent, or causal prediction of unsafe action."
        ),
    }
    write_validation_output(payload, output_path)
    provenance = collect_provenance(
        config.experiment_id,
        root=root,
        model_id=model_config.model_id,
        model_revision=loaded.resolved_revision or loaded.requested_revision,
        configuration=config.model_dump(mode="json", by_alias=True),
        number_of_prompts=len(prompts),
        source_layers=config.layers,
        runtime_seconds=runtime_seconds,
        peak_gpu_memory_bytes=peak_memory,
        gpu_model=(torch.cuda.get_device_name() if torch.cuda.is_available() else None),
        output_artifacts=[str(output_path), str(metadata_path)],
    )
    provenance.save(metadata_path)
    print(output_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
