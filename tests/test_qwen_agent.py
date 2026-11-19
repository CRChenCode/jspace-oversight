import pytest

from jspace_oversight.agents.qwen import (
    ProposalValidationError,
    parse_proposed_action,
    validate_action_readout_positions,
)


def test_parse_exact_structured_action() -> None:
    action = parse_proposed_action(
        '{"tool":"send_message","arguments":{"recipient":"team@example.test",'
        '"body":"Done","attachments":["public.txt"]}}'
    )
    assert action.tool == "send_message"
    assert action.arguments["recipient"] == "team@example.test"


def test_parse_action_inside_reasoning_without_repairing_fields() -> None:
    action = parse_proposed_action(
        'analysis text\n{"tool":"finish_task","arguments":{}}\n'
    )
    assert action.tool == "finish_task"


def test_invalid_action_is_not_silently_corrected() -> None:
    raw = '{"tool":"send_email","arguments":{"to":"team@example.test"}}'
    with pytest.raises(ProposalValidationError) as captured:
        parse_proposed_action(raw)
    assert captured.value.raw_text == raw
    assert captured.value.errors


def test_unknown_action_field_is_rejected() -> None:
    with pytest.raises(ProposalValidationError):
        parse_proposed_action(
            '{"tool":"finish_task","arguments":{},"silently_fixed":true}'
        )


def test_readout_position_must_fall_inside_generated_action() -> None:
    validate_action_readout_positions(
        [-1], decision_count=100, action_start=80, action_end=100
    )
    with pytest.raises(ValueError, match="outside generated action"):
        validate_action_readout_positions(
            [-30], decision_count=100, action_start=80, action_end=100
        )
