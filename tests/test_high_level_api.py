import json

import pytest

from cluster_model_runner import Checkpoint, Project, ResourceRequest, RuntimeRequest, Venv
from cluster_model_runner.client import ClusterClient
from cluster_model_runner.model import Model
from cluster_model_runner.models import ResolvedResources


class CapturingClient:
    def __init__(self):
        self.request = None

    def submit(self, request):
        self.request = request
        return "job-handle"


def test_model_translates_elegant_api_to_existing_job_request(tmp_path):
    entrypoint = tmp_path / "inference.py"
    entrypoint.write_text("print('ok')\n")
    requirements = tmp_path / "requirements.lock"
    requirements.write_text("torch==2.12.0\n")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    image = tmp_path / "image.tif"
    image.write_bytes(b"pixels")
    client = CapturingClient()
    model = Model(
        client=client,
        name="sam-large",
        project=Project(tmp_path, "inference.py"),
        environment=Venv(requirements=requirements, python="Python/3.10.8"),
        checkpoint=Checkpoint(checkpoint),
        resources=ResourceRequest(min_vram_gb=20),
    )

    result = model.submit(
        inputs={"image": image},
        parameters={"prompt": "buildings"},
    )

    assert result == "job-handle"
    assert client.request.command == ["python", "inference.py"]
    assert client.request.named_inputs == {"image": image}
    assert client.request.checkpoint == checkpoint
    assert client.request.requirements == requirements
    assert client.request.parameters == {"prompt": "buildings"}


def test_client_model_supports_convenience_paths(tmp_path):
    (tmp_path / "main.py").write_text("print('ok')\n")
    checkpoint = tmp_path / "model.bin"
    checkpoint.write_bytes(b"weights")
    client = object.__new__(ClusterClient)

    model = client.model(
        name="demo",
        project=tmp_path,
        entrypoint="main.py",
        checkpoint=checkpoint,
        resources=ResourceRequest(gpu_count=0),
    )

    assert model.project.root == tmp_path
    assert model.project.entrypoint == "main.py"
    assert model.checkpoint.path == checkpoint


def test_model_rejects_unsafe_input_name(tmp_path):
    (tmp_path / "main.py").write_text("print('ok')\n")
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"payload")
    model = Model(CapturingClient(), "demo", Project(tmp_path, "main.py"))

    with pytest.raises(ValueError, match="Invalid input name"):
        model.submit(inputs={"../escape": payload})


def test_runtime_request_reads_named_contract_and_writes_result(tmp_path, monkeypatch):
    input_path = tmp_path / "remote" / "inputs" / "image" / "scene.tif"
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"pixels")
    checkpoint = tmp_path / "remote" / "checkpoint" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    output = tmp_path / "output"
    artifact = tmp_path / "mask.png"
    artifact.write_bytes(b"mask")
    manifest = tmp_path / "request.json"
    manifest.write_text(
        json.dumps(
            {
                "id": "local-1",
                "parameters": {"prompt": "buildings"},
                "input_bindings": {"image": str(input_path)},
                "checkpoints": {"default": {"remote_path": str(checkpoint)}},
                "output_dir": str(output),
            }
        )
    )
    monkeypatch.setenv("CLUSTER_RUNNER_REQUEST", str(manifest))

    request = RuntimeRequest.from_cli()
    result_path = request.write_result(data={"mask_count": 1}, artifacts=[artifact])

    assert request.input("image") == input_path
    assert request.checkpoint() == checkpoint
    assert request.parameters["prompt"] == "buildings"
    result = json.loads(result_path.read_text())
    assert result["data"] == {"mask_count": 1}
    assert result["artifacts"] == ["artifacts/mask.png"]
    assert (output / "artifacts" / "mask.png").read_bytes() == b"mask"


class FakeArtifacts:
    def upload_project(self, project):
        return "/remote/projects/demo/hash", "project-hash", False

    def ensure_checkpoint(self, checkpoint):
        return "/remote/checkpoints/model.pt", "checkpoint-hash", False


class FakeStore:
    def __init__(self):
        self.manifest = None

    def update_manifest(self, job_id, manifest):
        self.manifest = manifest


class FakeSlurm:
    def render_script(self, **kwargs):
        return "#!/bin/bash\ntrue\n"

    def submit(self, script):
        return "42"


class FakeTransport:
    def __init__(self):
        self.rsync_calls = []
        self.uploads = {}

    def checked(self, argv, timeout=0):
        return ""

    def copy_to(self, source, destination):
        self.rsync_calls.append((source, destination))

    def upload_bytes(self, content, destination):
        self.uploads[destination] = content


def test_staging_writes_named_input_and_checkpoint_bindings(tmp_path):
    from cluster_model_runner.models import JobRequest

    entrypoint = tmp_path / "main.py"
    entrypoint.write_text("print('ok')\n")
    image = tmp_path / "scene.tif"
    image.write_bytes(b"pixels")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    client = object.__new__(ClusterClient)
    client.artifacts = FakeArtifacts()
    client.store = FakeStore()
    client.scheduler = FakeSlurm()
    client.transport = FakeTransport()
    request = JobRequest(
        name="demo",
        command=["python", "main.py"],
        project_dir=tmp_path,
        named_inputs={"image": image},
        checkpoint=checkpoint,
        python_module="",
    )
    manifest = request.to_manifest()
    manifest["id"] = "a" * 32
    manifest["output_dir"] = "/remote/jobs/job/output"
    resources = ResolvedResources(
        partition="debug",
        qos="",
        gpu_count=0,
        gpu_type="",
        cpus=1,
        ram_gb=1,
        time_limit="00:01:00",
    )

    slurm_id = client._stage_and_submit(
        request=request,
        resources=resources,
        job_id="a" * 32,
        remote_dir="/remote/jobs/job",
        manifest=manifest,
    )

    assert slurm_id == "42"
    assert manifest["input_bindings"] == {"image": "/remote/jobs/job/inputs/image/scene.tif"}
    assert manifest["checkpoints"]["default"]["remote_path"] == ("/remote/checkpoints/model.pt")
    uploaded = json.loads(client.transport.uploads["/remote/jobs/job/request.json"])
    assert uploaded["input_bindings"] == manifest["input_bindings"]
    assert "/remote/jobs/job/runtime/cluster_model_runner/runtime.py" in client.transport.uploads
    assert "/remote/jobs/job/runtime/nodus/runtime.py" in client.transport.uploads
