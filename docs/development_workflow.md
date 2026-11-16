# Development workflow

The Mac is the primary editing, CPU testing, review, and visualization
environment. Use a Python 3.11+ virtual environment and an editable install.

```bash
python -m pip install -e ".[dev]"
ruff format .
ruff check .
pytest
python scripts/check_sensitive_files.py
```

CI repeats lint, format-check, and CPU tests with Python 3.11. It must never
authenticate with Hugging Face, download Qwen, request a GPU, fit a lens, or
submit a Slurm job.

Use feature branches and focused commits. Review `git diff` and `git status`
before staging. A human decides whether and when to commit or push. Small result
metadata may flow back through GitHub; large artifacts remain in Scratch.
