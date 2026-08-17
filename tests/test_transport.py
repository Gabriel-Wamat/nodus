import subprocess

import pytest

from cluster_model_runner.config import ClusterConfig
from cluster_model_runner.exceptions import ConfigurationError
from cluster_model_runner.transport import OpenSSHTransport


def test_scp_is_used_when_rsync_is_unavailable(monkeypatch, tmp_path):
    calls = []

    def which(executable):
        return None if executable == "rsync" else f"/usr/bin/{executable}"

    def run(args, **kwargs):
        calls.append(args)
        if args[0] == "ssh":
            return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr("cluster_model_runner.transport.shutil.which", which)
    monkeypatch.setattr("cluster_model_runner.transport.subprocess.run", run)
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    transport = OpenSSHTransport(ClusterConfig(host="login.cluster.example.org"))

    transport.copy_to(source, "/remote/payload.bin")

    assert any(command[0] == "scp" for command in calls)
    assert not any(command[0] == "rsync" for command in calls)


def test_missing_rsync_and_scp_has_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "cluster_model_runner.transport.shutil.which",
        lambda executable: "/usr/bin/ssh" if executable == "ssh" else None,
    )
    source = tmp_path / "payload.bin"
    source.write_bytes(b"payload")
    transport = OpenSSHTransport(ClusterConfig(host="login.cluster.example.org"))

    with pytest.raises(ConfigurationError, match="Neither a working rsync nor the scp"):
        transport.copy_to(source, "/remote/payload.bin")
