import pytest

from jspace_oversight.evaluation.paired_metrics import paired_readout_metrics


def _entry(condition: str, rankings: list[tuple[int, str]]) -> dict:
    return {
        "pair_id": "email",
        "condition": condition,
        "readout": {
            "readouts": [
                {
                    "method": "jacobian_lens",
                    "layer": 20,
                    "position": -1,
                    "rankings": [
                        {"token_id": token_id, "token": token, "score": 1.0}
                        for token_id, token in rankings
                    ],
                }
            ]
        },
    }


def test_paired_metrics_report_overlap_and_exploratory_hits() -> None:
    payload = {
        "results": [
            _entry("control", [(1, "answer"), (2, "response")]),
            _entry("risk", [(2, "response"), (3, " safety")]),
        ]
    }
    [summary] = paired_readout_metrics(payload)
    assert summary["mean_top_k_jaccard"] == pytest.approx(1 / 3)
    assert summary["top_1_agreement_rate"] == 0.0
    assert summary["control_risk_term_hit_rate"] == 0.0
    assert summary["risk_risk_term_hit_rate"] == 1.0
    assert summary["risk_minus_control_hit_rate"] == 1.0


def test_paired_metrics_reject_incomplete_pairs() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        paired_readout_metrics({"results": [_entry("control", [(1, "answer")])]})
