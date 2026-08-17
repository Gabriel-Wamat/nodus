import json

import pytest

import nodus
from cluster_model_runner import cli
from cluster_model_runner.client import JobHandle
from cluster_model_runner.models import JobState, JobStatus


def test_public_nodus_namespace_exports_typed_sdk():
    assert nodus.ClusterClient.__name__ == "ClusterClient"
    assert nodus.JobExecutionError.__name__ == "JobExecutionError"
    assert nodus.ResourceRequest().gpu_count == 1
    assert nodus.Model.__name__ == "Model"
    assert nodus.RuntimeRequest.__name__ == "RuntimeRequest"


class FakeJob:
    id = "local-123"
    slurm_id = "456"

    def wait(self):
        return self

    def download(self, destination):
        return destination / self.id


class FakeClient:
    submitted = None

    @classmethod
    def from_env(cls):
        return cls()

    def submit(self, request):
        type(self).submitted = request
        return FakeJob()


def test_cli_submit_uses_job_request_and_same_client_core(monkeypatch, tmp_path, capsys):
    manifest = tmp_path / "job.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "cli-demo",
                "command": ["python", "main.py"],
                "project_dir": str(tmp_path),
            }
        )
    )
    monkeypatch.setattr(cli, "ClusterClient", FakeClient)

    assert cli.main(["submit", str(manifest), "--wait"]) == 0
    assert FakeClient.submitted.name == "cli-demo"
    assert '"slurm_id": "456"' in capsys.readouterr().out


class TerminalClient:
    config = type("Config", (), {"poll_interval": 0})()

    def status(self, job_id):
        return JobState.FAILED

    def status_info(self, job_id):
        return JobStatus(JobState.FAILED, slurm_id="9", reason="test")

    def logs(self, job_id, *, lines=200):
        return f"{job_id}:{lines}"


def test_job_handle_wait_raises_public_execution_error():
    handle = JobHandle(TerminalClient(), "failed-job")
    with pytest.raises(nodus.JobExecutionError, match="ended as FAILED"):
        handle.wait(progress=False)


def test_job_wait_reports_terminal_status(capsys):
    handle = JobHandle(TerminalClient(), "failed-job")

    with pytest.raises(nodus.JobExecutionError):
        handle.wait(progress=True)

    feedback = capsys.readouterr().err
    assert "[nodus] failed-job" in feedback
    assert "SLURM 9" in feedback
    assert "FAILED" in feedback


def test_job_handle_exposes_refresh_and_logs():
    handle = JobHandle(TerminalClient(), "failed-job")

    assert handle.refresh() == JobState.FAILED
    assert handle.logs(lines=12) == "failed-job:12"
