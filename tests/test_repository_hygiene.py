from pathlib import Path

from cluster_model_runner.models import JobRequest


def test_remote_manifest_does_not_expose_local_paths_or_environment_values(tmp_path):
    project = tmp_path / "private-project"
    project.mkdir()
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"weights")
    image = tmp_path / "image.tif"
    image.write_bytes(b"pixels")
    request = JobRequest(
        name="safe",
        command=["python", "inference.py"],
        project_dir=project,
        named_inputs={"image": image},
        checkpoint=checkpoint,
        environment={"PRIVATE_VALUE": "must-not-leak"},
    )

    manifest = request.to_manifest()
    rendered = str(manifest)
    assert str(tmp_path) not in rendered
    assert "must-not-leak" not in rendered
    assert manifest["environment"] == ["PRIVATE_VALUE"]


def test_core_has_no_cluster_specific_identifiers():
    source_root = Path(__file__).parents[1] / "src"
    forbidden = ("apu" + "ana", "cluster-" + "node", "short-" + "complex")
    for path in source_root.rglob("*.py"):
        content = path.read_text(encoding="utf-8").lower()
        assert not any(value in content for value in forbidden), path
