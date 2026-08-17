from cluster_model_runner.environments import EnvironmentManager
from cluster_model_runner.transport import CommandResult


class FakeEnvironmentTransport:
    def __init__(self, ready=False):
        self.ready = ready
        self.scripts = []

    def run(self, argv, timeout=0):
        if argv[:2] == ["test", "-f"]:
            return CommandResult(0 if self.ready else 1, "", "")
        return CommandResult(0, "", "")

    def checked(self, argv, timeout=0):
        if argv and argv[0] == "sbatch":
            return "77"
        return ""

    def copy_to(self, local, remote):
        return None

    def upload_bytes(self, content, remote):
        if remote.endswith(".sbatch"):
            self.scripts.append(content.decode())

    def shell(self, script, data=None, timeout=0):
        return CommandResult(0, "", "")


def test_environment_is_content_addressed_and_not_moved(tmp_path):
    requirements = tmp_path / "requirements.lock"
    requirements.write_text("torch==2.12.0\n")
    transport = FakeEnvironmentTransport()
    handle = EnvironmentManager(transport, "/remote").prepare(
        requirements, python_module="Python/3.10.8", name="vision"
    )
    assert len(handle.environment_id.rsplit("-", 1)[-1]) == 64
    assert handle.slurm_id == "77"
    script = transport.scripts[0]
    assert f"python3 -m venv {handle.path}" in script
    assert "mv " not in script
    assert "_READY" in script


def test_ready_environment_is_reused_without_submission(tmp_path):
    requirements = tmp_path / "requirements.lock"
    requirements.write_text("transformers==4.51.3\n")
    transport = FakeEnvironmentTransport(ready=True)
    handle = EnvironmentManager(transport, "/remote").prepare(requirements, name="llm")
    assert handle.reused is True
    assert handle.status().value == "SUCCEEDED"
    assert transport.scripts == []
