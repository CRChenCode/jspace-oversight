"""Experiment provenance capture and serialization."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    git_commit_sha: str | None = None
    git_dirty: bool | None = None
    model_id: str | None = None
    model_revision: str | None = None
    dependencies: dict[str, str | None] = Field(default_factory=dict)
    configuration: dict[str, Any] = Field(default_factory=dict)
    random_seed: int | None = None
    gpu_model: str | None = None
    slurm_job_id: str | None = None
    number_of_prompts: int | None = None
    source_layers: list[int] | None = None
    runtime_seconds: float | None = None
    peak_gpu_memory_bytes: int | None = None
    output_artifacts: list[str] = Field(default_factory=list)
    hostname: str = Field(default_factory=platform.node)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _git(args: list[str], root: Path) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError):
        return None


def git_state(root: str | Path) -> tuple[str | None, bool | None]:
    """Return commit SHA and dirty state, preserving unknowns as ``None``."""
    repo = Path(root)
    sha_result = _git(["rev-parse", "HEAD"], repo)
    sha = (
        sha_result.stdout.strip()
        if sha_result is not None and sha_result.returncode == 0
        else None
    )
    status_result = _git(["status", "--porcelain"], repo)
    dirty = (
        bool(status_result.stdout.strip())
        if status_result is not None and status_result.returncode == 0
        else None
    )
    return sha, dirty


def dependency_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_provenance(
    experiment_id: str,
    *,
    root: str | Path,
    dependencies: list[str] | None = None,
    **values: Any,
) -> Provenance:
    sha, dirty = git_state(root)
    return Provenance(
        experiment_id=experiment_id,
        git_commit_sha=sha,
        git_dirty=dirty,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        dependencies=dependency_versions(
            dependencies or ["jspace-oversight", "torch", "transformers", "jlens"]
        ),
        **values,
    )
