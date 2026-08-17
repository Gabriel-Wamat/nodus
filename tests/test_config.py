import pytest

from cluster_model_runner.config import ClusterConfig
from cluster_model_runner.exceptions import ConfigurationError


def test_config_from_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("CLUSTER_SSH_ALIAS", "research-cluster")
    monkeypatch.setenv("CLUSTER_STATE_DIR", str(tmp_path))
    config = ClusterConfig.from_env()
    assert config.target == "research-cluster"
    assert config.transfer_target == "research-cluster"
    assert config.state_dir == tmp_path
    assert config.discovery_ttl == 60


def test_explicit_transfer_host_overrides_ssh_alias():
    config = ClusterConfig(
        host="login.example.org",
        user="researcher",
        ssh_alias="cluster",
        transfer_host="data.example.org",
    )

    assert config.target == "cluster"
    assert config.transfer_target == "researcher@data.example.org"


@pytest.mark.parametrize("remote_root", ["../outside", "/", "unsafe path"])
def test_remote_root_rejects_escape_or_shell_unsafe_paths(remote_root):
    config = ClusterConfig(host="login.example.org", remote_root=remote_root)
    with pytest.raises(ConfigurationError, match="REMOTE_ROOT"):
        config.validate()
