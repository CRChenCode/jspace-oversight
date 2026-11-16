import json

from jspace_oversight.evaluation.schemas import DecisionPointRecord, RiskAnnotations


def test_schema_json_round_trip_keeps_labels_distinct() -> None:
    record = DecisionPointRecord(
        episode_id="ep-1",
        decision_point_id="dp-1",
        user_task="Read a page",
        ground_truth=RiskAnnotations(
            malicious_content_exposure=True,
            model_recognized_malicious_content=None,
            unsafe_action_occurred=False,
            human_intervention_opportunity=True,
            intervention_benefited_outcome=None,
        ),
    )
    restored = DecisionPointRecord.model_validate_json(record.model_dump_json())
    payload = json.loads(restored.model_dump_json())
    assert payload["ground_truth"]["malicious_content_exposure"] is True
    assert payload["ground_truth"]["unsafe_action_occurred"] is False
    assert payload["ground_truth"]["model_recognized_malicious_content"] is None
