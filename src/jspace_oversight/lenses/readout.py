"""Structured Jacobian and vanilla logit-lens readouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .metrics import as_numpy, top_k_entries


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
    input_token_id: int | None = None
    input_token: str | None = None
    rankings: list[TokenScore]


class PromptReadout(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_text: str
    model_id: str
    model_revision: str | None
    lens_path: str
    lens_n_prompts: int | None
    input_token_count: int | None = None
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
    input_ids: Any | None = None,
) -> list[LayerReadout]:
    results: list[LayerReadout] = []
    token_ids = None
    if input_ids is not None:
        token_ids = as_numpy(input_ids)
        if token_ids.ndim == 2 and token_ids.shape[0] == 1:
            token_ids = token_ids[0]
        if token_ids.ndim != 1:
            raise ValueError("input_ids must have shape [seq_len] or [1, seq_len]")
    for layer, logits in sorted(logits_by_layer.items()):
        for row_index, position in enumerate(positions):
            ranked_ids, scores = top_k_entries(logits[row_index], top_k)
            rankings = [
                TokenScore(
                    token_id=token_id,
                    token=tokenizer.decode([token_id]),
                    score=score,
                )
                for token_id, score in zip(ranked_ids, scores, strict=True)
            ]
            input_token_id = int(token_ids[position]) if token_ids is not None else None
            results.append(
                LayerReadout(
                    method=method,
                    layer=layer,
                    position=position,
                    input_token_id=input_token_id,
                    input_token=(
                        tokenizer.decode([input_token_id])
                        if input_token_id is not None
                        else None
                    ),
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
    jacobian_logits, _, input_ids = lens.apply(
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
        input_ids=input_ids,
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
                input_ids=input_ids,
            )
        )
    return PromptReadout(
        input_text=prompt,
        model_id=model_id,
        model_revision=model_revision,
        lens_path=str(path),
        lens_n_prompts=getattr(lens, "n_prompts", None),
        input_token_count=int(as_numpy(input_ids).shape[-1]),
        readouts=readouts,
    )


def save_readouts(results: list[PromptReadout], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.model_dump(mode="json") for result in results]
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
