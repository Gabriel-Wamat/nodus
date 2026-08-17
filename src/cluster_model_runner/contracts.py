from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .models import ClusterSnapshot, JobStatus, NodeInfo, ResolvedResources


class CommandResultLike(Protocol):
    returncode: int
    stdout: str
    stderr: str


class RemoteTransport(Protocol):
    """Scheduler-independent remote command and transfer contract."""

    def run(
        self, argv: list[str], *, input_bytes: bytes | None = None, timeout: int = 30
    ) -> CommandResultLike: ...

    def checked(
        self, argv: list[str], *, input_bytes: bytes | None = None, timeout: int = 30
    ) -> str: ...

    def shell(
        self, script: str, data: bytes | None = None, timeout: int = 30
    ) -> CommandResultLike: ...

    def upload_bytes(self, content: bytes, remote_path: str) -> None: ...

    def copy_to(
        self, local: Path, remote_path: str, *, excludes: tuple[str, ...] = ()
    ) -> None: ...

    def copy_from(self, remote_path: str, local: Path) -> None: ...


class ClusterDiscovery(Protocol):
    """Information required by the orchestration core."""

    partition_rules: dict[str, dict[str, Any]]

    def nodes(self) -> list[NodeInfo]: ...

    def python_modules(self) -> list[str]: ...

    def installation_partition(
        self, nodes: list[NodeInfo] | None = None, configured: str = ""
    ) -> str: ...

    def snapshot(self) -> ClusterSnapshot: ...


class SchedulerBackend(Protocol):
    """Minimal scheduler contract consumed by ClusterClient."""

    def render_script(
        self,
        *,
        job_name: str,
        resources: ResolvedResources,
        remote_dir: str,
        project_dir: str,
        command: list[str],
        venv: str,
        python_module: str,
        environment: dict[str, str],
        correlation_id: str = "",
    ) -> str: ...

    def submit(self, script_path: str) -> str: ...

    def status_info(self, scheduler_id: str) -> JobStatus: ...

    def cancel(self, scheduler_id: str) -> None: ...

    def find_by_correlation_id(self, correlation_id: str) -> str: ...
