import json
from pathlib import Path

from jspace_oversight.provenance import Provenance, dependency_versions, git_state


def test_provenance_preserves_nullable_fields(tmp_path: Path) -> None:
    record = Provenance(experiment_id="unit-test", model_revision=None)
    path = tmp_path / "metadata.json"
    record.save(path)
    data = json.loads(path.read_text())
    assert data["experiment_id"] == "unit-test"
    assert data["model_revision"] is None


def test_git_state_outside_repository(tmp_path: Path) -> None:
    assert git_state(tmp_path) == (None, None)


def test_missing_dependency_is_null() -> None:
    versions = dependency_versions(["definitely-not-a-real-package-jspace"])
    assert versions["definitely-not-a-real-package-jspace"] is None
