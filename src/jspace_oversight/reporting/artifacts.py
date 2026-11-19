"""JSONL run artifacts and checksummed manifests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from jspace_oversight.environments import ENVIRONMENT_VERSION
from jspace_oversight.evaluation.counterfactual_schemas import (
    BranchRecord,
    CheckpointAnalysis,
    DecisionCheckpoint,
    RepresentationRecord,
    RunManifest,
)
from jspace_oversight.provenance import dependency_versions


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_line(value: BaseModel | dict[str, Any]) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def write_jsonl(path: str | Path, records: list[BaseModel]) -> None:
    output = Path(path)
    output.write_text(
        "".join(f"{_json_line(record)}\n" for record in records), encoding="utf-8"
    )


def write_run_artifacts(
    run_dir: str | Path,
    *,
    run_id: str,
    evidence_kind: str,
    checkpoints: list[DecisionCheckpoint],
    branches: list[BranchRecord],
    features: list[RepresentationRecord],
    analyses: list[CheckpointAnalysis],
    code_commit: str | None,
    git_dirty: bool | None,
    model_id: str,
    model_revision: str | None,
    reviewer_versions: dict[str, str],
    scenario_versions: dict[str, str],
    generation_config: dict[str, Any],
    representation_config: dict[str, Any],
    seed_list: list[int],
    runtime_seconds: float | None = None,
    peak_memory_bytes: int | None = None,
) -> RunManifest:
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "checkpoints.jsonl": output / "checkpoints.jsonl",
        "branches.jsonl": output / "branches.jsonl",
        "features.jsonl": output / "features.jsonl",
        "summary.json": output / "summary.json",
    }
    write_jsonl(paths["checkpoints.jsonl"], checkpoints)
    write_jsonl(paths["branches.jsonl"], branches)
    write_jsonl(paths["features.jsonl"], features)
    paths["summary.json"].write_text(
        json.dumps(
            {
                "evidence_kind": evidence_kind,
                "analyses": [item.model_dump(mode="json") for item in analyses],
                "warning": (
                    "Synthetic fixtures test infrastructure only and are not empirical "
                    "model results or evidence of oversight effectiveness."
                    if evidence_kind == "synthetic_fixture"
                    else "Early technical demonstration; no generalizable claim."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checksums = {name: file_sha256(path) for name, path in paths.items()}
    manifest = RunManifest(
        schema_version="counterfactual-mvp-v1",
        run_id=run_id,
        evidence_kind=evidence_kind,
        created_at=datetime.now(UTC),
        code_commit=code_commit,
        git_dirty=git_dirty,
        dependencies=dependency_versions(
            ["jspace-oversight", "pydantic", "PyYAML", "numpy"]
        ),
        hostname=platform.node(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        model_id=model_id,
        model_revision=model_revision,
        reviewer_versions=reviewer_versions,
        scenario_versions=scenario_versions,
        environment_version=ENVIRONMENT_VERSION,
        generation_config=generation_config,
        representation_config=representation_config,
        seed_list=seed_list,
        number_of_prompts=len(checkpoints),
        source_layers=(
            representation_config.get("layers")
            if isinstance(representation_config.get("layers"), list)
            else None
        ),
        runtime_seconds=runtime_seconds,
        peak_memory_bytes=peak_memory_bytes,
        output_artifacts=list(paths),
        privacy_sanitization_revision="not_public_export",
        counts={
            "checkpoints": len(checkpoints),
            "branches": len(branches),
            "features": len(features),
            "analyses": len(analyses),
        },
        checksums=checksums,
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_run_artifacts(run_dir: str | Path) -> RunManifest:
    root = Path(run_dir)
    manifest = RunManifest.model_validate_json(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    for name, expected in manifest.checksums.items():
        path = root / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"artifact checksum mismatch: {name}")
    return manifest
