from pathlib import Path

import pytest

from jspace_oversight.lenses.fitting import (
    resolve_runtime_path,
    validate_prompt_lengths,
    validate_source_layers,
)


class Shape:
    def __init__(self, length: int) -> None:
        self.shape = (1, length)


class FakeModel:
    def encode(self, prompt: str, *, max_length: int) -> Shape:
        return Shape(min(len(prompt), max_length))


def test_validate_source_layers() -> None:
    validate_source_layers([0, 3, -1], 4)
    with pytest.raises(ValueError, match="out of range"):
        validate_source_layers([4], 4)


def test_validate_prompt_lengths_reports_corpus_indices() -> None:
    with pytest.raises(ValueError, match=r"indices \[0\]"):
        validate_prompt_lengths(
            FakeModel(),
            ["short", "a sufficiently long prompt"],
            max_seq_len=64,
            skip_first=4,
        )


def test_resolve_runtime_path_strips_logical_root(monkeypatch, tmp_path: Path) -> None:
    external = tmp_path / "scratch-data"
    monkeypatch.setenv("JSPACE_DATA_ROOT", str(external))
    resolved = resolve_runtime_path(
        Path("data/prompts/pilot.jsonl"),
        project_root=tmp_path / "project",
        external_root="JSPACE_DATA_ROOT",
        strip_prefix="data",
    )
    assert resolved == external / "prompts/pilot.jsonl"
