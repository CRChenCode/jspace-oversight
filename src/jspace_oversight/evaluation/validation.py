"""Manifest records and serialization for paired readout validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ValidationPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    pair_id: str
    condition: Literal["control", "risk"]
    prompt: str
    expected_intervention_opportunity: bool


def load_validation_prompts(path: str | Path) -> list[ValidationPrompt]:
    """Load a non-empty JSONL manifest and reject duplicate identifiers."""
    manifest_path = Path(path)
    records = [
        ValidationPrompt.model_validate_json(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"validation prompt manifest is empty: {manifest_path}")
    prompt_ids = [record.prompt_id for record in records]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("validation prompt IDs must be unique")
    pair_conditions: dict[str, set[str]] = {}
    for record in records:
        pair_conditions.setdefault(record.pair_id, set()).add(record.condition)
    incomplete = sorted(
        pair_id
        for pair_id, conditions in pair_conditions.items()
        if conditions != {"control", "risk"}
    )
    if incomplete:
        raise ValueError(
            f"validation pairs need control and risk records: {incomplete}"
        )
    return records


def write_validation_output(payload: dict, path: str | Path) -> None:
    """Write a structured validation artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
