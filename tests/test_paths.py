from pathlib import Path

from jspace_oversight.paths import find_project_root, resolve_path


def test_find_project_root_from_nested_directory(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").touch()
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_project_root(nested) == tmp_path


def test_resolve_relative_and_env_override(tmp_path: Path, monkeypatch) -> None:
    assert resolve_path("results/x.json", root=tmp_path) == tmp_path / "results/x.json"
    override = tmp_path / "elsewhere"
    monkeypatch.setenv("JSPACE_TEST_PATH", str(override))
    assert (
        resolve_path("ignored", root=tmp_path, env_var="JSPACE_TEST_PATH") == override
    )
