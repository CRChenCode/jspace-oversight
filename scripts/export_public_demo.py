"""Create a conservative public export from validated real-model artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jspace_oversight.reporting import validate_run_artifacts
from jspace_oversight.reporting.artifacts import file_sha256


def _sanitize(value):
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if not isinstance(value, dict):
        return value
    clean = {key: _sanitize(item) for key, item in value.items()}
    if "files" in clean and isinstance(clean["files"], dict):
        clean["files"] = {path: "[REDACTED]" for path in clean["files"]}
    if "messages" in clean and isinstance(clean["messages"], list):
        for message in clean["messages"]:
            if isinstance(message, dict) and "body" in message:
                message["body"] = "[REDACTED]"
    if "content" in clean:
        clean["content"] = "[REDACTED]"
    return clean


def _sanitize_jsonl(source: Path, destination: Path) -> None:
    records = [
        _sanitize(json.loads(line))
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    destination.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = Path(args.run_dir)
    destination = Path(args.out)
    manifest = validate_run_artifacts(source)
    if manifest.evidence_kind != "real_model_run":
        raise SystemExit("refusing public export: source is not a real_model_run")
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints.jsonl", "branches.jsonl", "features.jsonl"):
        _sanitize_jsonl(source / name, destination / name)
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    (destination / "summary.json").write_text(
        json.dumps(_sanitize(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = manifest.model_dump(mode="json")
    payload["privacy_sanitization_revision"] = "conservative-redaction-v1"
    payload["checksums"] = {
        name: file_sha256(destination / name)
        for name in (
            "checkpoints.jsonl",
            "branches.jsonl",
            "features.jsonl",
            "summary.json",
        )
    }
    (destination / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
