"""Prepare the pinned WikiText-103 fitting corpus without loading a model."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from jspace_oversight.config import CorpusConfig, load_config
from jspace_oversight.corpora import select_long_texts, write_prompt_corpus
from jspace_oversight.paths import find_project_root
from jspace_oversight.provenance import dependency_versions, git_state


def resolve_output(config: CorpusConfig, root: Path) -> Path:
    if config.output_path.is_absolute():
        return config.output_path
    data_root = os.environ.get("JSPACE_DATA_ROOT")
    if data_root:
        suffix = (
            config.output_path.parts[1:]
            if config.output_path.parts[:1] == ("data",)
            else config.output_path.parts
        )
        return Path(data_root).expanduser().joinpath(*suffix)
    return root / config.output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/corpora/wikitext_103_pilot.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit(
            "Install the 'data' optional dependency group first"
        ) from error

    root = find_project_root(args.config)
    config = load_config(args.config, CorpusConfig)
    output = resolve_output(config, root)
    metadata_path = output.with_suffix(".metadata.json")
    existing = [path for path in (output, metadata_path) if path.exists()]
    if existing and not args.overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise SystemExit(f"refusing to overwrite existing output: {paths}")
    stream = load_dataset(
        config.dataset_id,
        config.dataset_config,
        split=config.split,
        revision=config.dataset_revision,
        streaming=True,
    )
    selected = select_long_texts(
        stream,
        text_field=config.text_field,
        number_of_prompts=config.number_of_prompts,
        minimum_characters=config.minimum_characters,
    )
    corpus_sha256 = write_prompt_corpus(selected, output, overwrite=args.overwrite)
    git_sha, git_dirty = git_state(root)
    metadata = {
        "experiment_id": config.experiment_id,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit_sha": git_sha,
        "git_dirty": git_dirty,
        "configuration": config.model_dump(mode="json"),
        "dependencies": dependency_versions(["datasets", "huggingface-hub"]),
        "selection_rule": (
            "first records in stream order whose stripped text has at least "
            f"{config.minimum_characters} characters"
        ),
        "source_record_indices": [index for index, _ in selected],
        "number_of_prompts": len(selected),
        "corpus_path": str(output),
        "corpus_sha256": corpus_sha256,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    print(metadata_path)
    print(f"sha256={corpus_sha256}")


if __name__ == "__main__":
    main()
