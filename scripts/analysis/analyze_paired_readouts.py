"""Summarize exploratory control/risk differences in a readout artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jspace_oversight.evaluation.paired_metrics import paired_readout_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    path = Path(args.input)
    payload = json.loads(path.read_text(encoding="utf-8"))
    summaries = paired_readout_metrics(payload)
    ranked = sorted(
        summaries,
        key=lambda item: (
            -item["risk_minus_control_hit_rate"],
            -item["risk_risk_term_hit_rate"],
            item["mean_top_k_jaccard"],
            item["method"],
            item["layer"],
            item["position"],
        ),
    )

    print(f"path: {path}")
    print(f"pairs: {ranked[0]['number_of_pairs'] if ranked else 0}")
    print(
        "warning: exploratory substring lexicon defined after initial readout "
        "inspection; these are descriptive, not confirmatory metrics"
    )
    print()
    print("method          layer pos  jaccard top1same control_hit risk_hit delta")
    for item in ranked[: args.top]:
        print(
            f"{item['method']:15s} {item['layer']:>5d} {item['position']:>3d}  "
            f"{item['mean_top_k_jaccard']:.3f}   "
            f"{item['top_1_agreement_rate']:.3f}      "
            f"{item['control_risk_term_hit_rate']:.3f}     "
            f"{item['risk_risk_term_hit_rate']:.3f}  "
            f"{item['risk_minus_control_hit_rate']:+.3f}"
        )
        hits = [
            f"{pair['pair_id']}:{','.join(pair['risk_terms'])}"
            for pair in item["pairs"]
            if pair["risk_terms"]
        ]
        if hits:
            print(f"  risk matches: {' | '.join(hits)}")


if __name__ == "__main__":
    main()
