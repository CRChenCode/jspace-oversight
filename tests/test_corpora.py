import hashlib
import json

import pytest

from jspace_oversight.corpora import select_long_texts, write_prompt_corpus


def test_select_long_texts_preserves_stream_order_and_source_indices() -> None:
    records = [
        {"text": "short"},
        {"text": "  long enough  "},
        {"text": "also long enough"},
        {"text": "unused long record"},
    ]
    selected = select_long_texts(
        records,
        text_field="text",
        number_of_prompts=2,
        minimum_characters=10,
    )
    assert selected == [(1, "  long enough  "), (2, "also long enough")]


def test_select_long_texts_rejects_insufficient_records() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        select_long_texts(
            [{"text": "long enough"}],
            text_field="text",
            number_of_prompts=2,
            minimum_characters=5,
        )


def test_write_prompt_corpus_is_deterministic_and_refuses_overwrite(tmp_path) -> None:
    output = tmp_path / "pilot.jsonl"
    digest = write_prompt_corpus([(4, "alpha"), (9, "beta")], output)
    content = output.read_bytes()
    assert digest == hashlib.sha256(content).hexdigest()
    records = [json.loads(line) for line in content.splitlines()]
    assert records[0] == {
        "selection_index": 0,
        "source_record_index": 4,
        "text": "alpha",
    }
    with pytest.raises(FileExistsError):
        write_prompt_corpus([(4, "alpha")], output)
