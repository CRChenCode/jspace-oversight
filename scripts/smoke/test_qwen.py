"""Explicit GPU smoke test for Qwen loading and generation."""

from __future__ import annotations

import argparse

from jspace_oversight.config import ModelConfig, load_config
from jspace_oversight.models import format_prompt, load_causal_lm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models/qwen3_4b.yaml")
    parser.add_argument("--prompt", default="Briefly explain why provenance matters.")
    args = parser.parse_args()
    config = load_config(args.config, ModelConfig)
    loaded = load_causal_lm(config)
    prompt = format_prompt(
        loaded.tokenizer, args.prompt, use_chat_template=config.use_chat_template
    )
    batch = loaded.tokenizer(prompt, return_tensors="pt")
    model_device = next(loaded.model.parameters()).device
    batch = batch.to(model_device)
    generated = loaded.model.generate(**batch, max_new_tokens=32, do_sample=False)
    print(loaded.tokenizer.decode(generated[0], skip_special_tokens=True))
    print(f"resolved_revision={loaded.resolved_revision}")


if __name__ == "__main__":
    main()
