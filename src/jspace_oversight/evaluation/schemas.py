"""Typed records for agent decision-point evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RiskAnnotations(StrictRecord):
    """Separate labels for distinct stages of oversight relevance."""

    malicious_content_exposure: bool | None = None
    model_recognized_malicious_content: bool | None = None
    unsafe_action_occurred: bool | None = None
    human_intervention_opportunity: bool | None = None
    intervention_benefited_outcome: bool | None = None


class OversightDecision(StrictRecord):
    requested: bool
    decision: Literal["allow", "block", "modify", "defer"] | None = None
    rationale: str | None = None


class DecisionPointRecord(StrictRecord):
    episode_id: str
    decision_point_id: str
    user_task: str
    agent_action: dict[str, Any] | str | None = None
    tool_observation: dict[str, Any] | str | None = None
    source: str | None = None
    trust_boundary: str | None = None
    observable_risk_indicators: list[str] = Field(default_factory=list)
    latent_readout_reference: str | None = None
    ground_truth: RiskAnnotations = Field(default_factory=RiskAnnotations)
    oversight_decision: OversightDecision | None = None
    actual_outcome: str | None = None
