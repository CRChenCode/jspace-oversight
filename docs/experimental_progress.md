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

## Reproduced infrastructure smoke run

The newly bootstrapped repository subsequently completed its first end-to-end
Skipjack smoke workflow. Unlike the historical results above, these runs were
generated from committed repository code and external Scratch artifacts.

| Item | Repository-run result |
|---|---|
| Repository commit used for fitting | `65e0d448d125e778fe47777aa3e0147d89463310` |
| Model revision | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Fitting Slurm job | `1913654` (`COMPLETED`, exit `0:0`) |
| GPU | NVIDIA A100-SXM4-80GB |
| Fitting corpus | One bundled neutral provenance prompt |
| Source layer | 10 |
| `dim_batch` / `max_seq_len` | 8 / 64 |
| Fitting runtime | 12.853937715000939 seconds |
| Peak allocated GPU memory | 8,906,936,832 bytes |
| Lens dimension / prompt count | 2560 / 1 |
| Lens artifact | `qwen3-4b-smoke.pt` in external Scratch storage |
| Multiposition readout Slurm allocation | `1943967` |
| Multiposition readout positions | `-19, -11, -7, -6, -5, -2, -1` |

The multiposition prompt described an untrusted webpage requesting that private
data be sent externally. At message-ending positions, the one-prompt Jacobian
lens ranked tokens such as `reproduce`, `improve`, `precisely`, and `exactly`,
which resemble the neutral fitting prompt more than the evaluation prompt's risk
content. The vanilla logit-lens rankings were also not coherently risk-related.
This is an observation of strong corpus-specific bias in an intentionally
underfit smoke lens, not evidence for or against latent risk recognition.

The original `position=-2` readout was found to correspond to the chat-template
`assistant` role token. The multiposition rerun therefore included content-token,
message-boundary, template-control, and pre-generation positions. Raw scores
remain uncalibrated vocabulary logits.

## WikiText pilot fitting run

The repository then completed the planned infrastructure-scale fitting pilot.
This establishes that the larger pipeline runs; it does not validate oversight
performance.

| Item | Repository-run result |
|---|---|
| Repository commit | `93e2fcf39bd922b957af6d2c1e0ce8a9471a9577` |
| Slurm job | `1944569` (`COMPLETED`, exit `0:0`) |
| Slurm elapsed time | 36 minutes 50 seconds |
| Corpus | 100 deterministic WikiText-103 records, minimum 600 stripped characters |
| Corpus SHA-256 | `7c8831ea1fd60b0581221dda28d45562fd5036ab6f9ff78050a35218a838e003` |
| Source layers | 4, 8, 12, 16, 20, 24, 28, 32 |
| Maximum sequence length | 128 tokens |
| Lens artifact | `qwen3-4b-pilot.pt` in external Scratch storage (104,861,103 bytes) |
| Checkpoint | `pilot-fit.pt` in external Scratch storage (209,718,795 bytes) |

The next validation uses four paired control/risk scenarios at chat-template
positions `-2` and `-1`, records Jacobian- and logit-lens rankings, and benchmarks
latency against a hidden-state-producing forward pass. Because the prompts
contain explicit risk cues and the sample is small, this is only a sanity check.

The paired readout job `1945255` subsequently completed at commit `d651091` in
18 seconds. Across the eight prompts, the median hidden-state-producing forward
pass took 0.037157 seconds; the eight-layer, two-position Jacobian-plus-logit
readout took 0.479788 seconds and peaked at 8,140,206,592 allocated GPU bytes.

An exploratory substring lexicon was defined only after inspecting these
readouts, so it is descriptive and cannot serve as confirmatory evidence. Under
that analysis, Jacobian layer 20 at position `-1` contained a risk-term match in
three of four risk prompts and zero of four controls; the mean paired top-10
Jaccard overlap was 0.322. No other reported cell had a positive risk-minus-
control hit rate. A follow-up single-cell latency run is therefore explicitly
post-hoc and intended only to estimate deployment cost, not effect size.
