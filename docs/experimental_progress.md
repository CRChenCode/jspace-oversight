# Experimental progress

## Historical manual Skipjack feasibility run

The following results were supplied from manual experiments in the original
Johns Hopkins ARCH Skipjack working directory. They have **not** been rerun by
this newly bootstrapped repository and are not automated-test outcomes.

Environment:

- GPU: NVIDIA A100-SXM4-80GB
- Python: 3.11.9
- PyTorch: 2.9.1+cu128
- Model: `Qwen/Qwen3-4B`
- Jacobian Lens: <https://github.com/anthropics/jacobian-lens>

| Experiment | Historical result |
|---|---|
| Qwen3-4B model loading | Successful |
| Qwen3-4B inference | Successful |
| Model parameters | 4,022,468,096 |
| Model memory allocation | 7.49 GB |
| `jlens.from_hf()` integration | Successful |
| Wrapper type | `HFLensModel` |
| Single-layer Jacobian fitting | Successful |
| Fitting configuration | 1 prompt, source layer 10, `dim_batch=8`, `max_seq_len=64` |
| Fitting time | 12.6 seconds |
| Fitting peak GPU memory | 8.35 GB |
| Saved lens reload and readout | Successful |
| Lens dimension | 2560 |
| Readout source layer | 10 |
| Readout position | -2 |

The first readout ranked tokens including `scams`, `威胁`, `deceptive`, `诈骗`,
and `blackmail` for a prompt describing malicious instructions on a webpage.
Because the lens was fitted on one prompt, this observation is exploratory. It
does not show that the model reliably predicts prompt injection, unsafe action,
agent intent, or need for human oversight.

## Repository verification status

CPU tests validate configuration, paths, provenance, schemas, and synthetic
readout ranking. They do not validate Qwen inference, Jacobian fitting, GPU
memory, scientific validity, or oversight effectiveness.
