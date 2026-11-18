from pathlib import Path

import pytest

from jspace_oversight.evaluation.validation import load_validation_prompts


def test_load_validation_prompts_requires_complete_pairs(tmp_path: Path) -> None:
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_text(
        '{"prompt_id":"a","pair_id":"p","condition":"control",'
        '"prompt":"hello","expected_intervention_opportunity":false}\n'
    )
    with pytest.raises(ValueError, match="control and risk"):
        load_validation_prompts(manifest)


def test_repository_validation_manifest_is_paired() -> None:
    prompts = load_validation_prompts("examples/prompts/risk_validation.jsonl")
    assert len(prompts) == 8
    assert {prompt.pair_id for prompt in prompts} == {
        "calendar",
        "email",
        "file",
        "purchase",
    }
