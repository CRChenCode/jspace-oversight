"""Exploratory paired metrics for structured lens readout artifacts."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

EXPLORATORY_RISK_TERMS = (
    "risk",
    "unsafe",
    "safety",
    "danger",
    "malicious",
    "untrusted",
    "private",
    "credential",
    "security",
    "cybersecurity",
    "ethical",
    "warning",
    "caution",
    "refuse",
    "block",
    "stop",
    "oversight",
    "human",
    "风险",
    "危险",
    "安全",
    "恶意",
    "不可信",
    "私人",
    "隐私",
    "凭证",
    "警示",
    "拒绝",
    "阻止",
    "监督",
    "人工",
)


def matched_risk_terms(rankings: list[dict[str, Any]]) -> list[str]:
    """Return lexicon terms found as substrings of decoded top-k tokens."""
    matched = {
        term
        for item in rankings
        for term in EXPLORATORY_RISK_TERMS
        if term in str(item["token"]).casefold()
    }
    return sorted(matched)


def _readout_key(readout: dict[str, Any]) -> tuple[str, int, int]:
    return readout["method"], int(readout["layer"]), int(readout["position"])


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def paired_readout_metrics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate control/risk top-k differences by method, layer, and position."""
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for entry in payload["results"]:
        pairs[entry["pair_id"]][entry["condition"]] = entry

    incomplete = sorted(
        pair_id
        for pair_id, conditions in pairs.items()
        if set(conditions) != {"control", "risk"}
    )
    if incomplete:
        raise ValueError(f"incomplete control/risk pairs: {incomplete}")

    by_cell: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for pair_id, conditions in sorted(pairs.items()):
        control = {
            _readout_key(item): item
            for item in conditions["control"]["readout"]["readouts"]
        }
        risk = {
            _readout_key(item): item
            for item in conditions["risk"]["readout"]["readouts"]
        }
        if set(control) != set(risk):
            raise ValueError(f"readout cells differ within pair {pair_id!r}")
        for cell in control:
            control_rankings = control[cell]["rankings"]
            risk_rankings = risk[cell]["rankings"]
            control_ids = {int(item["token_id"]) for item in control_rankings}
            risk_ids = {int(item["token_id"]) for item in risk_rankings}
            by_cell[cell].append(
                {
                    "pair_id": pair_id,
                    "top_k_jaccard": _jaccard(control_ids, risk_ids),
                    "top_1_same": (
                        control_rankings[0]["token_id"] == risk_rankings[0]["token_id"]
                    ),
                    "control_terms": matched_risk_terms(control_rankings),
                    "risk_terms": matched_risk_terms(risk_rankings),
                }
            )

    summaries = []
    for (method, layer, position), observations in sorted(by_cell.items()):
        count = len(observations)
        control_hits = sum(bool(item["control_terms"]) for item in observations)
        risk_hits = sum(bool(item["risk_terms"]) for item in observations)
        summaries.append(
            {
                "method": method,
                "layer": layer,
                "position": position,
                "number_of_pairs": count,
                "mean_top_k_jaccard": sum(
                    item["top_k_jaccard"] for item in observations
                )
                / count,
                "top_1_agreement_rate": sum(item["top_1_same"] for item in observations)
                / count,
                "control_risk_term_hit_rate": control_hits / count,
                "risk_risk_term_hit_rate": risk_hits / count,
                "risk_minus_control_hit_rate": (risk_hits - control_hits) / count,
                "pairs": observations,
            }
        )
    return summaries
