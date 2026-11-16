"""Apply a saved smoke lens and save structured JSON rankings."""

from __future__ import annotations

import argparse

from jspace_oversight.config import ModelConfig, load_config
from jspace_oversight.lenses.readout import apply_readouts, save_readouts
from jspace_oversight.models import format_prompt, load_causal_lm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/models/qwen3_4b.yaml")
    parser.add_argument("--lens", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--position", type=int, default=-2)
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        help="Read multiple positions in one model load; overrides --position.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.model_config, ModelConfig)
    loaded = load_causal_lm(config)
    prompt = format_prompt(
        loaded.tokenizer, args.prompt, use_chat_template=config.use_chat_template
    )
    try:
        import jlens
    except ImportError as error:
        raise SystemExit(
            "Install the interpretability dependency group first"
        ) from error
    wrapped = jlens.from_hf(loaded.model, loaded.tokenizer)
    positions = args.positions if args.positions is not None else [args.position]
    result = apply_readouts(
        wrapped,
        loaded.tokenizer,
        prompt,
        lens_path=args.lens,
        model_id=config.model_id,
        model_revision=loaded.resolved_revision or loaded.requested_revision,
        layers=[args.layer],
        positions=positions,
        top_k=args.top_k,
    )
    save_readouts([result], args.output)
    print(args.output)


if __name__ == "__main__":
    main()
