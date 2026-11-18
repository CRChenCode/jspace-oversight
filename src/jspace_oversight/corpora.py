"""Deterministic fitting-corpus selection and serialization."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def select_long_texts(
    records: Iterable[Mapping[str, Any]],
    *,
    text_field: str,
    number_of_prompts: int,
    minimum_characters: int,
) -> list[tuple[int, str]]:
    """Select the first sufficiently long text records in stream order."""
    selected: list[tuple[int, str]] = []
    for record_index, record in enumerate(records):
        text = record.get(text_field)
        if not isinstance(text, str):
            raise ValueError(f"record {record_index} lacks string field {text_field!r}")
        if len(text.strip()) >= minimum_characters:
            selected.append((record_index, text))
        if len(selected) == number_of_prompts:
            break
    if len(selected) != number_of_prompts:
        raise ValueError(
            f"found {len(selected)} qualifying records, expected {number_of_prompts}"
        )
    return selected


def write_prompt_corpus(
    selected: list[tuple[int, str]],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> str:
    """Atomically write JSONL prompts and return the file's SHA-256 digest."""
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing corpus: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as handle:
            for selection_index, (source_index, text) in enumerate(selected):
                payload = {
                    "selection_index": selection_index,
                    "source_record_index": source_index,
                    "text": text,
                }
                line = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
                handle.write(line)
                digest.update(line)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()
