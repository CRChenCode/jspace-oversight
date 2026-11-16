"""Lazy Hugging Face causal-language-model loading."""

from __future__ import annotations

from dataclasses import dataclass
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
    resolved = getattr(model.config, "_commit_hash", None)
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
