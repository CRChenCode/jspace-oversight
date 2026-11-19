"""Scenario fixture loading with strict schema validation."""

from __future__ import annotations

from pathlib import Path

from jspace_oversight.config import load_yaml
from jspace_oversight.evaluation.counterfactual_schemas import ScenarioRecord


def load_scenarios(path: str | Path) -> list[ScenarioRecord]:
    payload = load_yaml(path)
    raw_scenarios = payload.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("scenario fixture must contain a non-empty 'scenarios' list")
    records = [ScenarioRecord.model_validate(item) for item in raw_scenarios]
    identifiers = [record.scenario_id for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("scenario IDs must be unique")
    return records


def select_scenarios(
    records: list[ScenarioRecord], identifiers: list[str] | None
) -> list[ScenarioRecord]:
    if not identifiers:
        return records
    by_id = {record.scenario_id: record for record in records}
    missing = sorted(set(identifiers) - set(by_id))
    if missing:
        raise ValueError(f"unknown scenario IDs: {missing}")
    return [by_id[identifier] for identifier in identifiers]
