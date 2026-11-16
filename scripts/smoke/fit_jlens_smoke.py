"""Fit the YAML-defined smoke lens on a GPU compute node."""

from __future__ import annotations

import argparse

from jspace_oversight.lenses.fitting import run_fitting_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fitting/smoke.yaml")
    args = parser.parse_args()
    print(run_fitting_experiment(args.config))


if __name__ == "__main__":
    main()
