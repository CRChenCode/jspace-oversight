"""Fail if suspicious or forbidden files are tracked by Git."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".pem", ".key"}
FORBIDDEN_PARTS = {".env", ".venv", "huggingface", "artifacts", "trajectories"}
SECRET_PATTERN = re.compile(r"(?:hf|github_pat)_[A-Za-z0-9_\-]{20,}")


def main() -> None:
    result = subprocess.run(
        ["git", "ls-files", "-z"], check=True, capture_output=True, text=True
    )
    failures: list[str] = []
    for name in filter(None, result.stdout.split("\0")):
        path = Path(name)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or any(
            part in FORBIDDEN_PARTS for part in path.parts
        ):
            failures.append(f"forbidden tracked path: {name}")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if SECRET_PATTERN.search(content):
            failures.append(f"possible token in tracked file: {name}")
    if failures:
        raise SystemExit("\n".join(failures))
    print("No forbidden tracked paths or obvious token patterns found.")


if __name__ == "__main__":
    main()
