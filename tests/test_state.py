from cluster_model_runner.client import ClusterClient
from cluster_model_runner.models import JobState
from cluster_model_runner.state import JobStore


def test_store_survives_reinstantiation_and_updates_manifest(tmp_path):
    first = JobStore(tmp_path)
    first.create("local-1", "/remote/job", {"name": "before"})
    first.update("local-1", state=JobState.SUBMITTED, slurm_id="42")
    first.update_manifest("local-1", {"name": "after", "cache": {"uploaded": False}})

    restarted = JobStore(tmp_path)
    record = restarted.get("local-1")
    assert record["slurm_id"] == "42"
    assert record["state"] == "SUBMITTED"
    assert record["manifest"]["name"] == "after"
    assert restarted.list()[0]["id"] == "local-1"


def test_client_job_and_list_jobs_use_persisted_store(tmp_path):
    store = JobStore(tmp_path)
    store.create("local-1", "/remote/job", {"name": "demo"})
    client = object.__new__(ClusterClient)
    client.store = store

    assert client.job("local-1").id == "local-1"
    assert [job.id for job in client.list_jobs()] == ["local-1"]
