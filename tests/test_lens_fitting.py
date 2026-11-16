import pytest

from jspace_oversight.lenses.fitting import (
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
