# Experiment protocol

1. Create or review the prompt/episode manifest and record its immutable version.
2. Select a committed YAML configuration; never edit settings after launch.
3. Confirm the Git commit and whether the tree is dirty.
4. Launch GPU work through Slurm and retain its job ID and log outside Git.
5. Record resolved model revision, dependency versions, seed, hardware, prompt
   count, layers, runtime, peak memory, and every output artifact path.
6. Preserve checkpoints for recovery, but treat a resumed run as the same
   experiment only when the upstream checkpoint compatibility checks pass.
7. Save raw logits/readout scores as scores—not probabilities—and retain input,
   layer, position, tokenizer/model revision, and lens provenance.
8. Validate artifacts before analysis. Document exclusions and failed prompts.
9. Compare policies at equal intervention budgets and avoid unavailable future
   information.
10. Commit only small non-sensitive summaries and metadata after manual review.

Run IDs should be unique and meaningful (for example,
`qwen3-4b-jlens-pilot-seed0-YYYYMMDD`). Any deviation from a preregistered or
planned protocol must be explicit in metadata and reporting.
