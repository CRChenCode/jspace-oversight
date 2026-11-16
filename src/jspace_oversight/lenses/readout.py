"""Structured Jacobian and vanilla logit-lens readouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .metrics import top_k_entries


class TokenScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token_id: int
    token: str
    score: float


class LayerReadout(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["jacobian_lens", "logit_lens"]
    layer: int
    position: int
    rankings: list[TokenScore]


class PromptReadout(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_text: str
    model_id: str
    model_revision: str | None
    lens_path: str
    lens_n_prompts: int | None
    readouts: list[LayerReadout]
    warning: str = (
        "Raw vocabulary rankings are not probabilities, calibrated risk scores, "
        "or verified descriptions of model intent."
    )


def _rank(
    logits_by_layer: dict[int, Any],
    *,
    positions: list[int],
    tokenizer: Any,
    top_k: int,
    method: Literal["jacobian_lens", "logit_lens"],
) -> list[LayerReadout]:
    results: list[LayerReadout] = []
    for layer, logits in sorted(logits_by_layer.items()):
        for row_index, position in enumerate(positions):
            token_ids, scores = top_k_entries(logits[row_index], top_k)
            rankings = [
                TokenScore(
                    token_id=token_id,
                    token=tokenizer.decode([token_id]),
                    score=score,
                )
                for token_id, score in zip(token_ids, scores, strict=True)
            ]
            results.append(
                LayerReadout(
                    method=method,
                    layer=layer,
                    position=position,
                    rankings=rankings,
                )
            )
    return results


def apply_readouts(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    lens_path: str | Path,
    model_id: str,
    model_revision: str | None,
    layers: list[int],
    positions: list[int],
    top_k: int,
    max_seq_len: int = 512,
    include_logit_lens: bool = True,
) -> PromptReadout:
    try:
        import jlens
    except ImportError as error:
        raise RuntimeError(
            "readout requires the interpretability dependencies"
        ) from error

    path = Path(lens_path)
    lens = jlens.JacobianLens.load(str(path))
    jacobian_logits, _, _ = lens.apply(
        model,
        prompt,
        layers=layers,
        positions=positions,
        max_seq_len=max_seq_len,
        use_jacobian=True,
    )
    readouts = _rank(
        jacobian_logits,
        positions=positions,
        tokenizer=tokenizer,
        top_k=top_k,
        method="jacobian_lens",
    )
    if include_logit_lens:
        baseline_logits, _, _ = lens.apply(
            model,
            prompt,
            layers=layers,
            positions=positions,
            max_seq_len=max_seq_len,
            use_jacobian=False,
        )
        readouts.extend(
            _rank(
                baseline_logits,
                positions=positions,
                tokenizer=tokenizer,
                top_k=top_k,
                method="logit_lens",
            )
        )
    return PromptReadout(
        input_text=prompt,
        model_id=model_id,
        model_revision=model_revision,
        lens_path=str(path),
        lens_n_prompts=getattr(lens, "n_prompts", None),
        readouts=readouts,
    )


def save_readouts(results: list[PromptReadout], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.model_dump(mode="json") for result in results]
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
