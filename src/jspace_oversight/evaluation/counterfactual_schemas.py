"""Strict records for the counterfactual agent--overseer simulator."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from jspace_oversight.config import StrictModel

ToolName = Literal[
    "read_file", "send_message", "save_summary", "delete_file", "finish_task"
]


class ProposedAction(StrictModel):
    """A tool proposal recorded exactly as produced by an executor."""

    tool: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None


class Message(StrictModel):
    sender: str
    recipient: str
    body: str
    attachments: list[str] = Field(default_factory=list)
    trusted: bool = True


class ScenarioTruth(StrictModel):
    """Evaluator-only facts that are never included in an agent observation."""

    protected_files: list[str] = Field(default_factory=list)
    sensitive_files: list[str] = Field(default_factory=list)
    required_files_read: list[str] = Field(default_factory=list)
    required_summary_path: str | None = None
    required_message_recipient: str | None = None
    required_absent_files: list[str] = Field(default_factory=list)
    require_finish: bool = True


class ScenarioRecord(StrictModel):
    scenario_id: str
    version: str
    family: str
    variant: Literal["clean", "risky"]
    task_id: str
    user_goal: str
    authorized_actions: list[ToolName]
    allowed_recipients: list[str] = Field(default_factory=list)
    initial_files: dict[str, str] = Field(default_factory=dict)
    initial_messages: list[Message] = Field(default_factory=list)
    untrusted_content: list[str] = Field(default_factory=list)
    risk_plugins: list[str] = Field(default_factory=list)
    truth: ScenarioTruth = Field(default_factory=ScenarioTruth)
    executor_script: list[ProposedAction] = Field(min_length=1)
    max_steps: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def validate_truth_references(self) -> ScenarioRecord:
        missing = set(
            self.truth.protected_files
            + self.truth.sensitive_files
            + self.truth.required_files_read
            + self.truth.required_absent_files
        ) - set(self.initial_files)
        if missing:
            raise ValueError(
                f"truth references missing initial files: {sorted(missing)}"
            )
        return self


class WorkspaceSnapshot(StrictModel):
    scenario_id: str
    scenario_version: str
    files: dict[str, str]
    messages: list[Message]
    summaries: dict[str, str] = Field(default_factory=dict)
    files_read: list[str] = Field(default_factory=list)
    finished: bool = False


class ToolResult(StrictModel):
    ok: bool
    code: str
    output: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(StrictModel):
    step: int
    action: ProposedAction
    result: ToolResult
    source: Literal["executor", "reviewer"]


class RepresentationRecord(StrictModel):
    feature_id: str
    checkpoint_id: str
    model_id: str
    model_revision: str | None = None
    extractor: str
    timing: Literal["T1_after_proposal_before_execution"] = (
        "T1_after_proposal_before_execution"
    )
    action_aware: bool = True
    layers: list[int] = Field(default_factory=list)
    token_positions: list[int] = Field(default_factory=list)
    decision_token_count: int | None = Field(default=None, ge=1)
    action_start_token_index: int | None = Field(default=None, ge=0)
    action_end_token_index_exclusive: int | None = Field(default=None, ge=1)
    readouts: list[dict[str, Any]] = Field(default_factory=list)
    raw_artifact_hash: str | None = None
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    warning: str = (
        "Vocabulary rankings are uncalibrated readouts, not probabilities, intent "
        "labels, or verified predictions of future behavior."
    )


class DecisionCheckpoint(StrictModel):
    checkpoint_id: str
    scenario_id: str
    scenario_version: str
    environment_snapshot: WorkspaceSnapshot
    history: list[TraceEvent]
    candidate_action: ProposedAction
    script_offset: int = Field(ge=0)
    extractor_timing: Literal["T1_after_proposal_before_execution"] = (
        "T1_after_proposal_before_execution"
    )
    model_id: str
    model_revision: str | None = None
    sampling_config: dict[str, Any] = Field(default_factory=dict)
    rng_metadata: dict[str, Any] = Field(default_factory=dict)
    external_observation_hash: str
    environment_snapshot_hash: str
    feature_reference: str | None = None


class ReviewContext(StrictModel):
    checkpoint_id: str
    user_goal: str
    authorized_actions: list[ToolName]
    allowed_recipients: list[str]
    recent_events: list[TraceEvent]
    observation: dict[str, Any]


class ReviewDecision(StrictModel):
    action_type: Literal["approve", "reject", "revise"]
    revised_action: ProposedAction | None = None
    feedback: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    cost: float = Field(default=0.0, ge=0.0)
    cost_unit: Literal["normalized_simulator", "tokens", "seconds"]

    @model_validator(mode="after")
    def revision_matches_type(self) -> ReviewDecision:
        if (self.action_type == "revise") != (self.revised_action is not None):
            raise ValueError("only revise decisions require revised_action")
        return self


class ReviewerRecord(StrictModel):
    checkpoint_id: str
    reviewer_id: str
    reviewer_type: Literal["oracle", "seeded_simulator", "llm"]
    reviewer_version: str
    visibility_policy: str
    decision: ReviewDecision
    error_codes: list[str] = Field(default_factory=list)


class Violation(StrictModel):
    code: str
    step: int
    evidence: dict[str, Any]


class EvaluationOutcome(StrictModel):
    task_success: bool
    safety_violation: bool
    violations: list[Violation] = Field(default_factory=list)
    failure_reason: str | None = None


class BranchRecord(StrictModel):
    branch_id: str
    checkpoint_id: str
    treatment: Literal["autonomous", "reviewed"]
    reviewer_id: str | None = None
    replicate_seed: int
    initial_snapshot_hash: str
    events: list[TraceEvent]
    review_decision: ReviewDecision | None = None
    terminal_state: WorkspaceSnapshot
    outcome: EvaluationOutcome
    review_calls: int = Field(default=0, ge=0)
    review_cost: float = Field(default=0.0, ge=0.0)
    completion_steps: int = Field(ge=0)
    compute_cost: float | None = None
    evaluator_version: str


class CheckpointAnalysis(StrictModel):
    checkpoint_id: str
    reviewer_id: str
    autonomous_count: int = Field(ge=1)
    reviewed_count: int = Field(ge=1)
    p0_safe_task_success: float = Field(ge=0.0, le=1.0)
    tau_hat: float
    standard_error: float | None = Field(default=None, ge=0.0)
    interval_low: float | None = None
    interval_high: float | None = None
    quadrant: Literal["rescue", "shared_failure", "redundant", "harmful", "uncertain"]
    utility_weights: dict[str, float]
    estimator: str


class RunManifest(StrictModel):
    schema_version: str
    run_id: str
    evidence_kind: Literal["synthetic_fixture", "real_model_run"]
    created_at: datetime
    code_commit: str | None
    git_dirty: bool | None
    dependencies: dict[str, str | None]
    hostname: str
    slurm_job_id: str | None
    model_id: str
    model_revision: str | None
    reviewer_versions: dict[str, str]
    scenario_versions: dict[str, str]
    environment_version: str
    generation_config: dict[str, Any]
    representation_config: dict[str, Any]
    seed_list: list[int]
    number_of_prompts: int | None
    source_layers: list[int] | None
    runtime_seconds: float | None
    peak_memory_bytes: int | None
    output_artifacts: list[str]
    privacy_sanitization_revision: str
    counts: dict[str, int]
    checksums: dict[str, str]
