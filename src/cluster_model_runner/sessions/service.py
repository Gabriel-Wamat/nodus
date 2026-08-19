from __future__ import annotations

import json
import secrets
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from ..models import JobState
from .channels import RequestChannel, SharedFilesystemChannel, SshTunnelChannel
from .models import SessionChannel, SessionResult, SessionState, SessionStatus
from .store import SessionStore

if TYPE_CHECKING:
    from ..client import ClusterClient
    from ..model import Model


class SessionHandle:
    def __init__(self, service: SessionService, session_id: str):
        self._service = service
        self.id = session_id

    @property
    def scheduler_id(self) -> str:
        return self._service.store.get(self.id)["scheduler_id"]

    @property
    def slurm_id(self) -> str:
        return self.scheduler_id

    def status(self) -> SessionStatus:
        return self._service.status(self.id)

    def wait_ready(
        self,
        *,
        timeout: float | None = None,
        progress: bool = True,
        heartbeat: float = 30,
        on_update: Callable[[SessionStatus], None] | None = None,
    ) -> SessionHandle:
        started = time.monotonic()
        last: tuple[SessionState, str] | None = None
        last_report = 0.0
        while True:
            status = self.status()
            now = time.monotonic()
            signature = (status.state, status.reason)
            if signature != last or now - last_report >= heartbeat:
                if progress:
                    print(_format_status(self.id, status), file=sys.stderr, flush=True)
                if on_update:
                    on_update(status)
                last = signature
                last_report = now
            if status.state in {SessionState.READY, SessionState.DEGRADED}:
                return self
            if status.state in {SessionState.FAILED, SessionState.STOPPED}:
                raise RuntimeError(
                    f"Session {self.id} became {status.state.value}: {status.reason}"
                )
            if timeout is not None and now - started >= timeout:
                raise TimeoutError(f"Timed out waiting for session {self.id}")
            time.sleep(self._service.client.config.poll_interval)

    def infer(
        self,
        *,
        inputs: Mapping[str, str | Path] | None = None,
        parameters: Mapping[str, Any] | None = None,
        timeout: float = 300,
    ) -> SessionResult:
        return self._service.infer(
            self.id, inputs=inputs or {}, parameters=parameters or {}, timeout=timeout
        )

    def close(self) -> None:
        self._service.close(self.id)

    def logs(self, *, lines: int = 200) -> str:
        return self._service.logs(self.id, lines=lines)


class SessionService:
    """Application service orchestrating long-lived model workers."""

    def __init__(self, client: ClusterClient):
        self.client = client
        self.store = SessionStore(client.config.state_dir)
        self._channels: dict[str, RequestChannel] = {}

    def start(
        self,
        model: Model,
        *,
        entrypoint: str | None = None,
        channel: SessionChannel | str = SessionChannel.AUTO,
    ) -> SessionHandle:
        selected_channel = SessionChannel(channel)
        relative_entrypoint = PurePosixPath(entrypoint or model.project.entrypoint)
        if relative_entrypoint.is_absolute() or ".." in relative_entrypoint.parts:
            raise ValueError("Session entrypoint must be relative to the project root")
        local_entrypoint = Path(model.project.root) / relative_entrypoint
        if not local_entrypoint.is_file():
            raise ValueError(f"Session entrypoint does not exist: {local_entrypoint}")

        environment = model.environment
        venv = environment.path if environment else ""
        python_module = environment.python if environment else "auto"
        if environment and environment.requirements is not None and not venv:
            prepared = self.client.prepare_environment(
                environment.requirements,
                name=model.name,
                python_module=python_module,
            ).wait()
            venv = prepared.path
        if python_module == "auto":
            modules = self.client.discovery.python_modules()
            python_module = modules[-1] if modules else ""

        nodes = self.client.discovery.nodes()
        requires_vram = bool(
            model.resources.gpu_count
            and (
                model.resources.min_vram_gb
                or model.resources.policy in {"smallest-compatible", "safe", "exact"}
            )
        )
        if requires_vram and any(node.gpu_count > 0 and node.vram_gb <= 0 for node in nodes):
            self.client.bootstrap(require_vram=True)
            nodes = self.client.discovery.nodes()
        resources = self.client.selector.resolve(
            model.resources, nodes, self.client.discovery.partition_rules
        )
        session_id = uuid.uuid4().hex
        remote_dir = f"{self.client.remote_root}/sessions/{session_id}"
        token = secrets.token_urlsafe(32)
        manifest: dict[str, Any] = {
            "id": session_id,
            "name": model.name,
            "entrypoint": relative_entrypoint.as_posix(),
            "requested_channel": selected_channel.value,
            "resolved_resources": resources.__dict__,
        }
        self.store.create(session_id, remote_dir, token, selected_channel.value, manifest)
        self.store.update(session_id, state=SessionState.UPLOADING)
        try:
            remote_project, project_hash, project_uploaded = self.client.artifacts.upload_project(
                Path(model.project.root)
            )
            manifest["project"] = {
                "remote_path": remote_project,
                "sha256": project_hash,
                "uploaded": project_uploaded,
            }
            remote_checkpoint = ""
            if model.checkpoint:
                remote_checkpoint, checkpoint_hash, checkpoint_uploaded = (
                    self.client.artifacts.ensure_checkpoint(Path(model.checkpoint.path))
                )
                manifest["checkpoint"] = {
                    "remote_path": remote_checkpoint,
                    "sha256": checkpoint_hash,
                    "uploaded": checkpoint_uploaded,
                }
            self.client.transport.checked(
                [
                    "mkdir",
                    "-p",
                    f"{remote_dir}/control",
                    f"{remote_dir}/requests",
                    f"{remote_dir}/responses",
                    f"{remote_dir}/inputs",
                    f"{remote_dir}/outputs",
                    f"{remote_dir}/logs",
                    f"{remote_dir}/runtime",
                ]
            )
            runtime_source = Path(__file__).parents[1] / "session_runtime.py"
            for package in ("nodus", "cluster_model_runner"):
                package_dir = f"{remote_dir}/runtime/{package}"
                self.client.transport.upload_bytes(b"", f"{package_dir}/__init__.py")
                self.client.transport.upload_bytes(
                    runtime_source.read_bytes(), f"{package_dir}/session_runtime.py"
                )
            environment_values = {
                "NODUS_SESSION_ID": session_id,
                "NODUS_SESSION_TOKEN": token,
                "PYTHONPATH": f"{remote_dir}/runtime",
            }
            if remote_checkpoint:
                environment_values["NODUS_SESSION_CHECKPOINT"] = remote_checkpoint
            command = [
                "python",
                f"{remote_dir}/runtime/cluster_model_runner/session_runtime.py",
                "worker",
                "--session-dir",
                remote_dir,
                "--entrypoint",
                f"{remote_project}/{relative_entrypoint.as_posix()}",
            ]
            script = self.client.scheduler.render_script(
                job_name=f"{model.name}-session",
                resources=resources,
                remote_dir=remote_dir,
                project_dir=remote_project,
                command=command,
                venv=venv,
                python_module=python_module,
                environment=environment_values,
                correlation_id=session_id,
            )
            manifest["runtime"] = {"python_module": python_module, "venv": venv}
            self.store.update(session_id, manifest=manifest)
            self.client.transport.upload_bytes(
                json.dumps(manifest, sort_keys=True, indent=2).encode(),
                f"{remote_dir}/session.json",
            )
            self.client.transport.upload_bytes(script.encode(), f"{remote_dir}/run.sbatch")
            scheduler_id = self.client.scheduler.submit(f"{remote_dir}/run.sbatch")
        except Exception:
            self.store.update(session_id, state=SessionState.FAILED)
            raise
        self.store.update(session_id, state=SessionState.SUBMITTED, scheduler_id=scheduler_id)
        return SessionHandle(self, session_id)

    def attach(self, session_id: str) -> SessionHandle:
        self.store.get(session_id)
        return SessionHandle(self, session_id)

    def list(self) -> list[SessionHandle]:
        return [SessionHandle(self, item["id"]) for item in self.store.list()]

    def status(self, session_id: str) -> SessionStatus:
        record = self.store.get(session_id)
        scheduler_id = record["scheduler_id"]
        if not scheduler_id:
            recovered = self.client.scheduler.find_by_correlation_id(session_id)
            if recovered:
                self.store.update(
                    session_id, state=SessionState.SUBMITTED, scheduler_id=recovered
                )
                record = self.store.get(session_id)
                scheduler_id = recovered
            else:
                return SessionStatus(SessionState(record["state"]))
        job = self.client.scheduler.status_info(scheduler_id)
        if job.state == JobState.PENDING:
            state = SessionState.PENDING
        elif job.state == JobState.RUNNING:
            ready = self.client.transport.run(
                ["cat", f"{record['remote_dir']}/control/ready.json"], timeout=10
            )
            if ready.returncode == 0:
                state = (
                    SessionState.DEGRADED
                    if record["state"] == SessionState.DEGRADED.value
                    else SessionState.READY
                )
            else:
                state = SessionState.STARTING
        elif job.state == JobState.CANCELLED:
            state = SessionState.STOPPED
        elif job.state in {JobState.FAILED, JobState.SUCCEEDED}:
            state = SessionState.FAILED
        else:
            state = SessionState.UNKNOWN
        self.store.update(session_id, state=state)
        active = self._channels.get(session_id)
        channel = SessionChannel(active.name) if active else None
        return SessionStatus(
            state,
            scheduler_id=scheduler_id,
            reason=job.reason,
            elapsed=job.elapsed,
            node=job.node,
            channel=channel,
        )

    def infer(
        self,
        session_id: str,
        *,
        inputs: Mapping[str, str | Path],
        parameters: Mapping[str, Any],
        timeout: float,
    ) -> SessionResult:
        status = self.status(session_id)
        if status.state not in {SessionState.READY, SessionState.DEGRADED}:
            raise RuntimeError(f"Session {session_id} is not ready: {status.state.value}")
        record = self.store.get(session_id)
        request_id = uuid.uuid4().hex
        bindings: dict[str, str] = {}
        for name, value in sorted(inputs.items()):
            if not name or not name.replace("_", "").replace("-", "").isalnum():
                raise ValueError(f"Invalid input name: {name!r}")
            source = Path(value).expanduser().resolve()
            if not source.exists():
                raise ValueError(f"Input does not exist: {source}")
            target_dir = f"{record['remote_dir']}/inputs/{request_id}/{name}"
            self.client.transport.checked(["mkdir", "-p", target_dir])
            target = f"{target_dir}/{source.name}"
            self.client.transport.copy_to(source, target)
            bindings[name] = target
        payload = {"id": request_id, "inputs": bindings, "parameters": dict(parameters)}
        channel = self._channel(record)
        try:
            response = channel.request(payload, timeout=timeout)
        except Exception:
            requested = SessionChannel(record["channel"])
            if requested != SessionChannel.AUTO or channel.name == "filesystem":
                raise
            channel.close()
            channel = SharedFilesystemChannel(
                self.client.transport,
                record["remote_dir"],
                max(0.05, self.client.config.poll_interval),
            )
            self._channels[session_id] = channel
            self.store.update(session_id, state=SessionState.DEGRADED)
            response = channel.request(payload, timeout=timeout)
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "Remote inference failed"))
        result = response.get("result", {})
        if not isinstance(result, dict):
            raise TypeError("Invalid session response")
        data = result.get("data", {})
        artifacts = result.get("artifacts", [])
        return SessionResult(
            session_id,
            request_id,
            dict(data) if isinstance(data, dict) else {},
            tuple(str(item) for item in artifacts) if isinstance(artifacts, list) else (),
            f"{record['remote_dir']}/outputs/{request_id}",
            self.client.transport,
        )

    def _channel(self, record: Any) -> RequestChannel:
        session_id = str(record["id"])
        existing = self._channels.get(session_id)
        if existing:
            return existing
        requested = SessionChannel(record["channel"])
        if requested in {SessionChannel.AUTO, SessionChannel.SSH}:
            ready = self.client.transport.checked(
                ["cat", f"{record['remote_dir']}/control/ready.json"], timeout=10
            )
            endpoint = json.loads(ready)
            try:
                compute_user = self.client.transport.checked(["id", "-un"], timeout=10)
                ssh = SshTunnelChannel(
                    self.client.config,
                    node=str(endpoint["host"]),
                    remote_port=int(endpoint["port"]),
                    token=str(record["token"]),
                    compute_user=compute_user,
                )
                self._channels[session_id] = ssh
                return ssh
            except Exception:
                if requested == SessionChannel.SSH:
                    raise
        filesystem = SharedFilesystemChannel(
            self.client.transport,
            str(record["remote_dir"]),
            max(0.05, self.client.config.poll_interval),
        )
        self._channels[session_id] = filesystem
        if requested == SessionChannel.AUTO:
            self.store.update(session_id, state=SessionState.DEGRADED)
        return filesystem

    def close(self, session_id: str) -> None:
        record = self.store.get(session_id)
        channel = self._channels.pop(session_id, None)
        if channel:
            channel.close()
        if record["scheduler_id"]:
            self.client.scheduler.cancel(record["scheduler_id"])
        self.store.update(session_id, state=SessionState.STOPPED)

    def logs(self, session_id: str, *, lines: int = 200) -> str:
        record = self.store.get(session_id)
        limit = max(1, min(lines, 5000))
        return self.client.transport.shell(
            "find "
            + _quote(record["remote_dir"] + "/logs")
            + f" -maxdepth 1 -type f -name 'slurm_*' -exec tail -n {limit} {{}} + 2>/dev/null || true",
            timeout=20,
        ).stdout


def _quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def _format_status(session_id: str, status: SessionStatus) -> str:
    parts = [f"[nodus] session {session_id}", status.state.value]
    if status.scheduler_id:
        parts.append(f"SLURM {status.scheduler_id}")
    if status.state == SessionState.PENDING and status.reason:
        parts.append(f"reason={status.reason}")
    if status.elapsed:
        parts.append(f"elapsed={status.elapsed}")
    if status.channel:
        parts.append(f"channel={status.channel.value}")
    return " | ".join(parts)
