"""Typed YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ConfigT = TypeVar("ConfigT", bound=BaseModel)


class StrictModel(BaseModel):
    """Base class that rejects misspelled or unknown settings."""

    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictModel):
    model_id: str
    revision: str | None = None
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "bfloat16"
    device: str = "cuda"
    trust_remote_code: bool = False
    use_chat_template: bool = True


class FittingConfig(StrictModel):
    experiment_id: str
    model_config_path: Path = Field(alias="model_config")
    prompt_corpus: Path
    source_layers: list[int] = Field(min_length=1)
    target_layer: int | None = None
    max_seq_len: int = Field(default=128, ge=2)
    dim_batch: int = Field(default=8, ge=1)
    skip_first: int = Field(default=16, ge=0)
    checkpoint_path: Path
    checkpoint_every: int | None = Field(default=1, ge=1)
    resume: bool = True
    output_lens_path: Path
    seed: int = 0

    @model_validator(mode="after")
    def validate_positions(self) -> FittingConfig:
        if self.max_seq_len <= self.skip_first + 1:
            raise ValueError("max_seq_len must leave a position after skip_first")
        if len(set(self.source_layers)) != len(self.source_layers):
            raise ValueError("source_layers must not contain duplicates")
        return self


class EvaluationConfig(StrictModel):
    experiment_id: str
    prompts_path: Path
    lens_path: Path
    layers: list[int] = Field(min_length=1)
    positions: list[int] = Field(default_factory=lambda: [-2])
    top_k: int = Field(default=10, ge=1)
    include_logit_lens: bool = True
    output_path: Path


class ClusterConfig(StrictModel):
    account: str
    partition: str
    gres: str = "gpu:1"
    cpus_per_task: int = Field(default=4, ge=1)
    memory: str = "32G"
    python_module: str = "python/3.11.9"
    virtualenv: Path
    scratch_root: Path


class CorpusConfig(StrictModel):
    experiment_id: str
    dataset_id: str
    dataset_config: str
    dataset_revision: str
    split: str = "train"
    text_field: str = "text"
    number_of_prompts: int = Field(default=100, ge=1)
    minimum_characters: int = Field(default=600, ge=1)
    output_path: Path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping without applying a schema."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a YAML mapping: {config_path}")
    return data


def load_config(path: str | Path, schema: type[ConfigT]) -> ConfigT:
    """Load and validate a YAML file using ``schema``."""
    return schema.model_validate(load_yaml(path))
