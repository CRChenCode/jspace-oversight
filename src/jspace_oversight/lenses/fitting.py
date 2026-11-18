"""Thin wrapper around the public ``jlens.fit`` API."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jspace_oversight.config import FittingConfig


def resolve_runtime_path(
    value: Path,
    *,
    project_root: Path,
    external_root: str | None = None,
    strip_prefix: str | None = None,
) -> Path:
    """Resolve a configured path against the project or an external root."""
    if value.is_absolute():
        return value
    override = os.environ.get(external_root) if external_root else None
    if override:
        parts = value.parts
        if strip_prefix and parts[:1] == (strip_prefix,):
            parts = parts[1:]
        return Path(override).expanduser().joinpath(*parts)
    return project_root / value


def load_prompt_corpus(path: str | Path) -> list[str]:
    """Read non-empty JSONL ``text`` fields or one prompt per text line."""
    import json

    corpus_path = Path(path)
    prompts: list[str] = []
    for line_number, raw in enumerate(
        corpus_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line:
            continue
        if corpus_path.suffix == ".jsonl":
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"{corpus_path}:{line_number} needs a string 'text'")
            prompts.append(record["text"])
        else:
            prompts.append(line)
    if not prompts:
        raise ValueError(f"prompt corpus is empty: {corpus_path}")
    return prompts


def validate_source_layers(source_layers: Sequence[int], n_layers: int) -> None:
    if not source_layers:
        raise ValueError("source_layers must not be empty")
    invalid = [layer for layer in source_layers if not -n_layers <= layer < n_layers]
    if invalid:
        raise ValueError(f"source layers out of range for {n_layers} layers: {invalid}")


def validate_prompt_lengths(
    model: Any,
    prompts: Sequence[str],
    *,
    max_seq_len: int,
    skip_first: int,
) -> None:
    """Ensure every prompt has a position usable by the upstream estimator."""
    too_short: list[int] = []
    for index, prompt in enumerate(prompts):
        sequence_length = int(model.encode(prompt, max_length=max_seq_len).shape[1])
        if sequence_length <= skip_first + 1:
            too_short.append(index)
    if too_short:
        raise ValueError(
            "prompts too short after tokenization for skip_first="
            f"{skip_first}: indices {too_short}"
        )


def fit_lens(model: Any, prompts: Sequence[str], config: FittingConfig) -> Any:
    """Fit and save a lens using only documented upstream parameters."""
    try:
        import jlens
    except ImportError as error:
        raise RuntimeError(
            "fitting requires the interpretability dependencies"
        ) from error

    validate_source_layers(config.source_layers, model.n_layers)
    validate_prompt_lengths(
        model,
        prompts,
        max_seq_len=config.max_seq_len,
        skip_first=config.skip_first,
    )
    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_lens_path.parent.mkdir(parents=True, exist_ok=True)
    lens = jlens.fit(
        model,
        list(prompts),
        source_layers=config.source_layers,
        target_layer=config.target_layer,
        dim_batch=config.dim_batch,
        max_seq_len=config.max_seq_len,
        skip_first=config.skip_first,
        checkpoint_path=str(config.checkpoint_path),
        checkpoint_every=config.checkpoint_every,
        resume=config.resume,
    )
    lens.save(str(config.output_lens_path))
    return lens


def run_fitting_experiment(config_path: str | Path) -> Path:
    """Load configuration, fit a lens, and write provenance beside it."""
    import random
    import time

    import numpy as np

    from jspace_oversight.config import ModelConfig, load_config
    from jspace_oversight.models import load_causal_lm
    from jspace_oversight.paths import find_project_root
    from jspace_oversight.provenance import collect_provenance

    root = find_project_root(config_path)
    fitting = load_config(config_path, FittingConfig)

    fitting = fitting.model_copy(
        update={
            "model_config_path": resolve_runtime_path(
                fitting.model_config_path, project_root=root
            ),
            "prompt_corpus": resolve_runtime_path(
                fitting.prompt_corpus,
                project_root=root,
                external_root="JSPACE_DATA_ROOT",
                strip_prefix="data",
            ),
            "checkpoint_path": resolve_runtime_path(
                fitting.checkpoint_path,
                project_root=root,
                external_root="JSPACE_ARTIFACT_ROOT",
                strip_prefix="artifacts",
            ),
            "output_lens_path": resolve_runtime_path(
                fitting.output_lens_path,
                project_root=root,
                external_root="JSPACE_ARTIFACT_ROOT",
                strip_prefix="artifacts",
            ),
        }
    )
    model_config = load_config(fitting.model_config_path, ModelConfig)
    random.seed(fitting.seed)
    np.random.seed(fitting.seed)
    try:
        import torch

        torch.manual_seed(fitting.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fitting.seed)
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None

    loaded = load_causal_lm(model_config)
    try:
        import jlens
    except ImportError as error:
        raise RuntimeError(
            "fitting requires the interpretability dependencies"
        ) from error
    lens_model = jlens.from_hf(loaded.model, loaded.tokenizer)
    prompts = load_prompt_corpus(fitting.prompt_corpus)
    started = time.perf_counter()
    fit_lens(lens_model, prompts, fitting)
    runtime = time.perf_counter() - started
    peak_memory = (
        torch.cuda.max_memory_allocated()
        if torch is not None and torch.cuda.is_available()
        else None
    )
    metadata_path = fitting.output_lens_path.with_suffix(".metadata.json")
    provenance = collect_provenance(
        fitting.experiment_id,
        root=root,
        model_id=model_config.model_id,
        model_revision=loaded.resolved_revision or loaded.requested_revision,
        configuration={
            "fitting": fitting.model_dump(mode="json"),
            "model": model_config.model_dump(mode="json"),
        },
        random_seed=fitting.seed,
        gpu_model=(
            torch.cuda.get_device_name(0)
            if torch is not None and torch.cuda.is_available()
            else None
        ),
        number_of_prompts=len(prompts),
        source_layers=fitting.source_layers,
        runtime_seconds=runtime,
        peak_gpu_memory_bytes=peak_memory,
        output_artifacts=[
            str(fitting.checkpoint_path),
            str(fitting.output_lens_path),
            str(metadata_path),
        ],
    )
    provenance.save(metadata_path)
    return fitting.output_lens_path
