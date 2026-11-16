"""Project and runtime path resolution."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_MARKERS = ("pyproject.toml", ".git")


def find_project_root(start: str | Path | None = None) -> Path:
    """Walk upward to find the repository root."""
    current = Path(start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    raise FileNotFoundError(f"no project root found above {current}")


def resolve_path(
    value: str | Path,
    *,
    root: str | Path | None = None,
    env_var: str | None = None,
) -> Path:
    """Resolve an env-overridable path relative to the project root."""
    selected = os.environ.get(env_var) if env_var else None
    path = Path(selected or value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(root) if root else find_project_root()) / path


def ensure_parent(path: str | Path) -> Path:
    """Create a file path's parent directory and return the path."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
