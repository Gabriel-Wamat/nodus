import pytest

from cluster_model_runner.client import ClusterClient
from cluster_model_runner.models import JobRequest, JobState, JobStatus, ResolvedResources
from cluster_model_runner.state import JobStore
from cluster_model_runner.transport import CommandResult


class FakeSlurm:
    def find_by_correlation_id(self, correlation_id):
        assert correlation_id == "lost-update"
        return "9001"

    def status_info(self, slurm_id):
        assert slurm_id == "9001"
        return JobStatus(JobState.SUCCEEDED, slurm_id=slurm_id)


class FakeTransport:
    def run(self, argv, timeout=0):
        return CommandResult(1, "", "not needed")


def test_client_recovers_slurm_id_after_local_restart(tmp_path):
    store = JobStore(tmp_path)
    store.create("lost-update", "/remote/job", {"name": "demo"})
    store.update("lost-update", state=JobState.UPLOADING)

    client = object.__new__(ClusterClient)
    client.store = JobStore(tmp_path)
    client.scheduler = FakeSlurm()
    client.transport = FakeTransport()

    assert client.status("lost-update") == JobState.SUCCEEDED
    assert client.store.get("lost-update")["slurm_id"] == "9001"


class FakeDiscovery:
    def __init__(self):
        self.partition_rules = {}

    def nodes(self):
        return []


class FakeSelector:
    def resolve(self, request, nodes, rules):
        return ResolvedResources(
            partition="debug",
            qos="",
            gpu_count=0,
            gpu_type="",
            cpus=1,
            ram_gb=1,
            time_limit="00:01:00",
        )


def test_staging_failure_is_persisted_as_failed_after_restart(tmp_path):
    client = object.__new__(ClusterClient)
    client.store = JobStore(tmp_path / "state")
    client.remote_root = "/remote/nodus"
    client.discovery = FakeDiscovery()
    client.selector = FakeSelector()

    def fail_staging(**kwargs):
        raise OSError("rsync failed before sbatch")

    client._stage_and_submit = fail_staging
    request = JobRequest(name="broken", command=["python", "main.py"], project_dir=tmp_path)

    with pytest.raises(OSError, match="rsync failed"):
        client.submit(request)

    persisted = JobStore(tmp_path / "state").list()[0]
    assert persisted["state"] == "FAILED"
    assert persisted["slurm_id"] == ""
