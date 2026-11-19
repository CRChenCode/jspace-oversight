"""Decision-time representation record adapters."""

from __future__ import annotations

from jspace_oversight.evaluation.branching import stable_hash
from jspace_oversight.evaluation.counterfactual_schemas import (
    DecisionCheckpoint,
    RepresentationRecord,
)
from jspace_oversight.lenses.readout import PromptReadout


def mock_representation(checkpoint: DecisionCheckpoint) -> RepresentationRecord:
    """Create an explicitly synthetic feature record for deterministic CPU tests."""
    identity = {
        "checkpoint_id": checkpoint.checkpoint_id,
        "candidate_action": checkpoint.candidate_action.model_dump(mode="json"),
    }
    feature_id = f"feat-{stable_hash(identity)[:20]}"
    return RepresentationRecord(
        feature_id=feature_id,
        checkpoint_id=checkpoint.checkpoint_id,
        model_id=checkpoint.model_id,
        model_revision=checkpoint.model_revision,
        extractor="synthetic_fixture_no_model_state",
        action_aware=True,
    )


def representation_from_readout(
    checkpoint: DecisionCheckpoint,
    readout: PromptReadout,
    *,
    raw_artifact_hash: str | None = None,
    decision_token_count: int | None = None,
    action_start_token_index: int | None = None,
    action_end_token_index_exclusive: int | None = None,
) -> RepresentationRecord:
    """Attach existing J-lens results without changing their interpretation."""
    layers = sorted({item.layer for item in readout.readouts})
    positions = sorted({item.position for item in readout.readouts})
    payload = readout.model_dump(mode="json")
    return RepresentationRecord(
        feature_id=f"feat-{stable_hash(payload)[:20]}",
        checkpoint_id=checkpoint.checkpoint_id,
        model_id=readout.model_id,
        model_revision=readout.model_revision,
        extractor="existing_jacobian_lens_adapter",
        action_aware=True,
        layers=layers,
        token_positions=positions,
        decision_token_count=decision_token_count,
        action_start_token_index=action_start_token_index,
        action_end_token_index_exclusive=action_end_token_index_exclusive,
        readouts=payload["readouts"],
        raw_artifact_hash=raw_artifact_hash,
    )
