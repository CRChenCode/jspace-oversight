"""Recompute intervention utility from an integrity-checked run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jspace_oversight.evaluation.counterfactual_schemas import BranchRecord
from jspace_oversight.evaluation.intervention import analyze_checkpoint
from jspace_oversight.reporting import validate_run_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--risk-penalty", type=float, default=5.0)
    parser.add_argument("--review-cost-weight", type=float, default=0.1)
    args = parser.parse_args()
    root = Path(args.run_dir)
    validate_run_artifacts(root)
    branches = [
        BranchRecord.model_validate_json(line)
        for line in (root / "branches.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[str, list[BranchRecord]] = {}
    for branch in branches:
        grouped.setdefault(branch.checkpoint_id, []).append(branch)
    analyses = [
        analyze_checkpoint(
            records,
            risk_penalty=args.risk_penalty,
            review_cost_weight=args.review_cost_weight,
        ).model_dump(mode="json")
        for _, records in sorted(grouped.items())
    ]
    print(json.dumps(analyses, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
