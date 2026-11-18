from pathlib import Path

import pytest
from pydantic import ValidationError

from jspace_oversight.config import (
    EvaluationConfig,
    FittingConfig,
    ModelConfig,
    load_config,
)


def test_load_model_config(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text("model_id: org/model\ndevice: cpu\ndtype: float32\n")
    config = load_config(path, ModelConfig)
    assert config.model_id == "org/model"
    assert config.device == "cpu"


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text("model_id: org/model\ntyop: true\n")
    with pytest.raises(ValidationError):
        load_config(path, ModelConfig)


def test_fitting_config_rejects_impossible_sequence_length() -> None:
    values = {
        "experiment_id": "x",
        "model_config": "model.yaml",
        "prompt_corpus": "prompts.txt",
        "source_layers": [1],
        "max_seq_len": 17,
        "skip_first": 16,
        "checkpoint_path": "checkpoint.pt",
        "output_lens_path": "lens.pt",
    }
    with pytest.raises(ValidationError, match="max_seq_len"):
        FittingConfig.model_validate(values)


def test_repository_evaluation_config() -> None:
    config = load_config("configs/evaluation/risk_validation.yaml", EvaluationConfig)
    assert config.layers == [4, 8, 12, 16, 20, 24, 28, 32]
    assert config.positions == [-2, -1]
