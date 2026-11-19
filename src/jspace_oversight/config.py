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
    model_config_path: Path = Field(alias="model_config")
    prompts_path: Path
    lens_path: Path
    layers: list[int] = Field(min_length=1)
    positions: list[int] = Field(default_factory=lambda: [-2])
    top_k: int = Field(default=10, ge=1)
    max_seq_len: int = Field(default=512, ge=2)
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


class SimulatedReviewerSettings(StrictModel):
    reviewer_id: str = "seeded-simulator"
    version: str = "simulator-v1"
    false_approval_rate: float = Field(default=0.15, ge=0.0, le=1.0)
    false_rejection_rate: float = Field(default=0.10, ge=0.0, le=1.0)
    revision_rate: float = Field(default=0.50, ge=0.0, le=1.0)
    normalized_cost: float = Field(default=1.0, ge=0.0)


class DemoConfig(StrictModel):
    run_id: str
    scenarios_path: Path
    scenario_ids: list[str] = Field(default_factory=list)
    reviewer_type: Literal["oracle", "seeded_simulator"] = "seeded_simulator"
    reviewer: SimulatedReviewerSettings = Field(
        default_factory=SimulatedReviewerSettings
    )
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2], min_length=1)
    risk_penalty: float = Field(default=5.0, ge=0.0)
    review_cost_weight: float = Field(default=0.1, ge=0.0)
    output_root: Path = Path("results")

    @model_validator(mode="after")
    def unique_seeds(self) -> DemoConfig:
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be unique")
        return self


class GenerationSettings(StrictModel):
    max_new_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    seed: int = 0


class GpuDemoConfig(StrictModel):
    experiment_id: str
    model_config_path: Path = Field(alias="model_config")
    scenarios_path: Path
    scenario_id: str
    lens_path: Path
    output_root: Path
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    layers: list[int] = Field(min_length=1)
    positions: list[int] = Field(min_length=1)
    top_k: int = Field(default=10, ge=1)
    max_seq_len: int = Field(default=512, ge=2)
    risk_penalty: float = Field(default=5.0, ge=0.0)
    review_cost_weight: float = Field(default=0.1, ge=0.0)

    @model_validator(mode="after")
    def unique_readout_coordinates(self) -> GpuDemoConfig:
        if len(self.layers) != len(set(self.layers)):
            raise ValueError("layers must be unique")
        if len(self.positions) != len(set(self.positions)):
            raise ValueError("positions must be unique")
        return self


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
