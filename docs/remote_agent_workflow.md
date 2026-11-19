# Mac agent to Skipjack workflow

This workflow lets a coding agent running on the Mac inspect and operate a narrow
part of Skipjack through versioned SSH wrappers. GitHub remains the source-code
bridge: edit and test locally, commit and push, pull remotely, then submit a
reviewed Slurm template. The wrappers do not expose an arbitrary remote shell.

## One-time Mac setup

The Mac SSH configuration must provide a `skipjack` host alias. Authenticate once
in a normal terminal when JHU login requires interaction:

```bash
ssh skipjack
exit
```

A multiplexed SSH connection with `ControlPersist` allows non-interactive wrapper
calls to reuse that authenticated connection. Never copy an SSH private key into
the repository or share it with a model.

Create the untracked configuration:

```bash
cp .cluster.env.example .cluster.env
scripts/cluster/doctor.sh
scripts/cluster/doctor.sh --connect
```

`.cluster.env` contains host and path names only and is ignored by Git. Override
its location with `JSPACE_CLUSTER_ENV` if needed.

## Commands available to the agent

Read-only inspection:

```bash
scripts/cluster/status.sh
scripts/cluster/job.sh 1913654
scripts/cluster/logs.sh
scripts/cluster/logs.sh 1913654
scripts/cluster/artifacts.sh
scripts/cluster/readout-summary.sh multiposition
scripts/cluster/readout-summary.sh pilot-validation
scripts/cluster/readout-analysis.sh
```

Remote state changes requiring explicit approval:

```bash
scripts/cluster/pull.sh
scripts/cluster/submit.sh smoke_qwen
scripts/cluster/submit.sh smoke_jlens
scripts/cluster/submit.sh fit_smoke
scripts/cluster/submit.sh read_smoke
scripts/cluster/submit.sh fit_pilot
scripts/cluster/submit.sh read_pilot
scripts/cluster/submit.sh read_pilot_focus
scripts/cluster/submit.sh counterfactual_p4
scripts/cluster/upload-corpus.sh
```

`submit.sh` validates an allowlist on both the Mac and Skipjack, refuses a dirty
remote worktree, exports the established runtime paths, and invokes a committed
Slurm file. Each `fit_pilot`, `read_pilot`, or `read_pilot_focus` submission
requires separate, explicit approval.

`counterfactual_p4` runs the single-checkpoint Qwen/J-lens counterfactual smoke
defined by `configs/demo/qwen.yaml`. It also requires separate, one-run approval.

## Prepare the pilot fitting corpus

Prepare the pinned corpus on the Mac, where dataset download and validation do
not consume a cluster login node or modify the remote Python environment:

```bash
python -m pip install -e ".[dev,data]"
python scripts/data/prepare_pilot_corpus.py
wc -l data/prompts/pilot.jsonl
```

The configuration in `configs/corpora/wikitext_103_pilot.yaml` selects the
first 100 WikiText-103 training records whose stripped text contains at least
600 characters. The dataset revision, source indices, Git state, and corpus
SHA-256 are recorded beside the generated JSONL. Both generated files remain
outside Git. With explicit approval, `upload-corpus.sh` copies only those two
files to Scratch and checks their remote SHA-256 values.

## Recommended Codex prompt

```text
Inspect Skipjack using only scripts/cluster/status.sh, job.sh, logs.sh, and
artifacts.sh. Do not use ssh directly. Do not pull or submit anything unless I
explicitly approve that exact action. Never run model loading or GPU computation
on a login node; use an allowlisted Slurm wrapper.
```

Keep Codex approvals interactive (`on-request`) and its filesystem sandbox at
`workspace-write`. Do not grant blanket SSH or danger-full-access permissions.

## Recovery and troubleshooting

- `BatchMode=yes` means wrappers fail instead of requesting a password or MFA.
  Open `ssh skipjack` manually to establish or refresh the multiplexed session.
- If `pull.sh` reports a dirty worktree, inspect it manually; the wrapper will not
  overwrite remote changes.
- If a wrapper cannot find a log, use `status.sh` and `job.sh` to confirm that the
  job started and note the exact job ID.
- Scratch artifacts and logs remain outside Git. Download or commit only compact,
  reviewed metadata when appropriate.
