"""Run the deterministic, CPU-only counterfactual fixture pipeline."""

from __future__ import annotations

import argparse

from jspace_oversight.demo import run_mock_demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo/mock.yaml")
    args = parser.parse_args()
    print(run_mock_demo(args.config))


if __name__ == "__main__":
    main()
