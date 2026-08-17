from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .exceptions import ConfigurationError


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class ClusterConfig:
    """Connection configuration populated without embedding cluster credentials."""

    host: str
    user: str = ""
    ssh_alias: str = ""
    ssh_key: str = ""
    transfer_host: str = ""
    remote_root: str = ".cluster-model-runner"
    ssh_port: int = 22
    connect_timeout: int = 15
    poll_interval: int = 15
    discovery_ttl: int = 60
    state_dir: Path = Path(".nodus")
    inventory_file: Path | None = None
    installation_partition: str = ""
    auto_probe: str = "when-needed"
    probe_max_parallel: int = 4
    probe_wait_timeout: int = 900
    gpu_vram_reserve_gb: int = 2

    @classmethod
    def from_env(cls, prefix: str = "CLUSTER_") -> ClusterConfig:
        def value(key: str, default: str = "") -> str:
            return _env(prefix + key, default)

        try:
            port = int(value("SSH_PORT", "22"))
            timeout = int(value("CONNECT_TIMEOUT", "15"))
            poll = int(value("POLL_INTERVAL", "15"))
            discovery_ttl = int(value("DISCOVERY_TTL", "60"))
            probe_max_parallel = int(value("PROBE_MAX_PARALLEL", "4"))
            probe_wait_timeout = int(value("PROBE_WAIT_TIMEOUT", "900"))
            gpu_vram_reserve_gb = int(value("GPU_VRAM_RESERVE_GB", "2"))
        except ValueError as exc:
            raise ConfigurationError("Numeric CLUSTER settings must be integers") from exc

        config = cls(
            host=value("HOST"),
            user=value("USER"),
            ssh_alias=value("SSH_ALIAS"),
            ssh_key=value("SSH_KEY"),
            transfer_host=value("TRANSFER_HOST"),
            remote_root=value("REMOTE_ROOT", ".cluster-model-runner"),
            ssh_port=port,
            connect_timeout=timeout,
            poll_interval=poll,
            discovery_ttl=discovery_ttl,
            state_dir=Path(value("STATE_DIR", ".nodus")).expanduser(),
            inventory_file=(
                Path(value("INVENTORY_FILE")).expanduser() if value("INVENTORY_FILE") else None
            ),
            installation_partition=value("INSTALL_PARTITION"),
            auto_probe=value("AUTO_PROBE", "when-needed"),
            probe_max_parallel=probe_max_parallel,
            probe_wait_timeout=probe_wait_timeout,
            gpu_vram_reserve_gb=gpu_vram_reserve_gb,
        )
        config.validate()
        return config

    @property
    def target(self) -> str:
        if self.ssh_alias:
            return self.ssh_alias
        return f"{self.user}@{self.host}" if self.user else self.host

    @property
    def transfer_target(self) -> str:
        # When connection details live entirely in ~/.ssh/config, rsync must use
        # the same alias as ssh. An explicit transfer host still takes precedence.
        host = self.transfer_host or self.ssh_alias or self.host
        return f"{self.user}@{host}" if self.user else host

    def validate(self) -> None:
        if not self.ssh_alias and not self.host:
            raise ConfigurationError("Set CLUSTER_HOST or CLUSTER_SSH_ALIAS")
        if self.ssh_port < 1 or self.ssh_port > 65535:
            raise ConfigurationError("CLUSTER_SSH_PORT must be between 1 and 65535")
        if self.discovery_ttl < 0:
            raise ConfigurationError("CLUSTER_DISCOVERY_TTL must be zero or greater")
        connection_values = {
            "CLUSTER_HOST": self.host,
            "CLUSTER_USER": self.user,
            "CLUSTER_SSH_ALIAS": self.ssh_alias,
            "CLUSTER_TRANSFER_HOST": self.transfer_host,
        }
        for name, value in connection_values.items():
            if value and (value.startswith("-") or re.search(r"[\s\0]", value)):
                raise ConfigurationError(f"{name} contains unsafe characters")
        if (
            not self.remote_root
            or not re.fullmatch(r"[A-Za-z0-9_./~-]+", self.remote_root)
            or ".." in PurePosixPath(self.remote_root).parts
            or self.remote_root == "/"
        ):
            raise ConfigurationError("CLUSTER_REMOTE_ROOT is invalid")
        if self.installation_partition and not self.installation_partition.replace(
            "-", ""
        ).replace("_", "").isalnum():
            raise ConfigurationError("CLUSTER_INSTALL_PARTITION is invalid")
        if self.auto_probe not in {"never", "when-needed", "representative", "all-nodes"}:
            raise ConfigurationError("CLUSTER_AUTO_PROBE has an unsupported value")
        if self.probe_max_parallel < 1 or self.probe_wait_timeout < 1:
            raise ConfigurationError("Probe parallelism and timeout must be positive")
        if self.gpu_vram_reserve_gb < 0:
            raise ConfigurationError("CLUSTER_GPU_VRAM_RESERVE_GB cannot be negative")
