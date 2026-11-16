"""Explicit smoke test for wrapping Qwen with Jacobian Lens."""

from __future__ import annotations

import argparse

from jspace_oversight.config import ModelConfig, load_config
from jspace_oversight.models import load_causal_lm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models/qwen3_4b.yaml")
    args = parser.parse_args()
    try:
        import jlens
    except ImportError as error:
        raise SystemExit(
            "Install the interpretability dependency group first"
        ) from error
    loaded = load_causal_lm(load_config(args.config, ModelConfig))
    wrapped = jlens.from_hf(loaded.model, loaded.tokenizer)
    print(type(wrapped).__name__)
    print(f"n_layers={wrapped.n_layers} d_model={wrapped.d_model}")


if __name__ == "__main__":
    main()
