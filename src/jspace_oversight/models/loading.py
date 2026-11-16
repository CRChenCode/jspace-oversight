"""Lazy Hugging Face causal-language-model loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jspace_oversight.config import ModelConfig


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    requested_revision: str | None
    resolved_revision: str | None


def _torch_dtype(name: str, torch: Any) -> Any:
    return "auto" if name == "auto" else getattr(torch, name)


def _revision_from_cache_path(path: str | os.PathLike[str] | None) -> str | None:
    """Extract a Hugging Face commit SHA from a snapshots cache path."""
    if path is None:
        return None
    parts = Path(path).parts
    try:
        snapshot_index = parts.index("snapshots")
        revision = parts[snapshot_index + 1]
    except (ValueError, IndexError):
        return None
    return revision or None


def _resolve_model_revision(
    model_id: str,
    requested_revision: str | None,
    model: Any,
    tokenizer: Any,
) -> str | None:
    """Resolve the immutable Hub revision without making a network request."""
    candidates = [
        getattr(model.config, "_commit_hash", None),
        getattr(model.config, "commit_hash", None),
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        getattr(tokenizer, "init_kwargs", {}).get("commit_hash"),
    ]
    if resolved := next((value for value in candidates if value), None):
        return str(resolved)

    try:
        from huggingface_hub import try_to_load_from_cache

        cached_config = try_to_load_from_cache(
            model_id,
            "config.json",
            revision=requested_revision or "main",
        )
    except (ImportError, OSError, ValueError):
        return None
    if not isinstance(cached_config, (str, os.PathLike)):
        return None
    return _revision_from_cache_path(cached_config)


def load_causal_lm(config: ModelConfig) -> LoadedModel:
    """Load a tokenizer/model only when explicitly called."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "model loading requires the 'interpretability' optional dependencies"
        ) from error

    device = config.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dtype_name = config.dtype
    if device == "cpu" and dtype_name in {"float16", "bfloat16"}:
        dtype_name = "float32"

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
        dtype=_torch_dtype(dtype_name, torch),
    ).to(device)
    model.eval()
    resolved = _resolve_model_revision(
        config.model_id,
        config.revision,
        model,
        tokenizer,
    )
    return LoadedModel(model, tokenizer, config.revision, resolved)


def format_prompt(tokenizer: Any, text: str, *, use_chat_template: bool) -> str:
    """Apply a tokenizer chat template while preserving a plain-text fallback."""
    if not use_chat_template or not getattr(tokenizer, "chat_template", None):
        return text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
