from cluster_model_runner import Checkpoint, ClusterClient, ResourceRequest, Venv
from cluster_model_runner.models import JobState, JobStatus, NodeInfo
from cluster_model_runner.resources import ResourceSelector
from cluster_model_runner.state import JobStore


class FakeConfig:
    poll_interval = 0


class FakeDiscovery:
    def __init__(self):
        self.partition_rules = {
            "gpu-batch": {
                "qos": "standard",
                "max_cpus": 48,
                "max_ram_gb": 500,
                "max_gpus": 4,
            }
        }

    def nodes(self):
        return [
            NodeInfo(
                "node-a",
                ("gpu-batch",),
                "idle",
                "gpu:1",
                (),
                512000,
                48,
                "gpu-large",
                1,
                80,
            ),
            NodeInfo(
                "node-r",
                ("gpu-batch",),
                "idle",
                "gpu:1",
                (),
                128000,
                32,
                "gpu-small",
                2,
                24,
            ),
        ]

    def python_modules(self):
        return []


class FakeArtifactCache:
    def __init__(self):
        self.project_cached = False
        self.checkpoint_cached = False
        self.project_results = []
        self.checkpoint_results = []

    def upload_project(self, project):
        uploaded = not self.project_cached
        self.project_cached = True
        self.project_results.append(uploaded)
        return "/remote/projects/demo/project-hash", "project-hash", uploaded

    def ensure_checkpoint(self, checkpoint):
        uploaded = not self.checkpoint_cached
        self.checkpoint_cached = True
        self.checkpoint_results.append(uploaded)
        return "/remote/checkpoints/checkpoint-hash/model.pt", "checkpoint-hash", uploaded


class FakeTransport:
    def __init__(self):
        self.uploads = {}

    def checked(self, argv, timeout=0):
        return ""

    def copy_to(self, source, destination, excludes=()):
        return None

    def copy_from(self, remote, local):
        local.mkdir(parents=True, exist_ok=True)
        (local / "result.json").write_text('{"ok": true}\n')

    def upload_bytes(self, content, destination):
        self.uploads[destination] = content

    def run(self, argv, timeout=0):
        raise AssertionError(f"Unexpected transport command: {argv}")


class FakeSlurm:
    def __init__(self):
        self.next_id = 100

    def submit(self, script):
        value = str(self.next_id)
        self.next_id += 1
        return value

    def render_script(self, **kwargs):
        return "#!/bin/bash\ntrue\n"

    def cancel(self, slurm_id):
        return None

    def status_info(self, slurm_id):
        return JobStatus(JobState.SUCCEEDED, slurm_id=slurm_id, elapsed="00:00:01")

    def find_by_correlation_id(self, correlation_id):
        return ""


def test_complete_high_level_batch_cycle_and_second_run_cache_hit(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "inference.py").write_text("print('remote model')\n")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    image = tmp_path / "image.tif"
    image.write_bytes(b"pixels")

    client = object.__new__(ClusterClient)
    client.config = FakeConfig()
    client.remote_root = "/remote"
    client.discovery = FakeDiscovery()
    client.selector = ResourceSelector()
    client.store = JobStore(tmp_path / "state")
    client.artifacts = FakeArtifactCache()
    client.transport = FakeTransport()
    client.scheduler = FakeSlurm()

    model = client.model(
        name="vision",
        project=project,
        entrypoint="inference.py",
        environment=Venv(path="/remote/envs/vision"),
        checkpoint=Checkpoint(checkpoint),
        resources=ResourceRequest(min_vram_gb=20),
    )

    first = model.submit(inputs={"image": image}, parameters={"prompt": "buildings"})
    assert first.record["manifest"]["resolved_resources"]["gpu_type"] == "gpu-small"
    assert first.wait(progress=False).status() == JobState.SUCCEEDED
    output = first.download(tmp_path / "results")
    assert (output / "result.json").is_file()

    second = model.submit(inputs={"image": image}, parameters={"prompt": "roads"})
    assert second.wait(progress=False).status() == JobState.SUCCEEDED
    assert client.artifacts.project_results == [True, False]
    assert client.artifacts.checkpoint_results == [True, False]
    assert [job.id for job in client.list_jobs()] == [second.id, first.id]
