from __future__ import annotations

import json
import shlex
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .artifacts import ArtifactCache
from .bootstrap import ClusterBootstrapper, ProbePolicy
from .config import ClusterConfig
from .contracts import ClusterDiscovery, RemoteTransport, SchedulerBackend
from .discovery import SlurmDiscovery
from .environments import EnvironmentHandle, EnvironmentManager
from .exceptions import JobExecutionError
from .model import Checkpoint, Model, Project, Venv
from .models import (
    ClusterSnapshot,
    JobRecord,
    JobRequest,
    JobState,
    JobStatus,
    ResolvedResources,
    ResourceRequest,
)
from .resources import ResourceSelector
from .slurm import SlurmBackend
from .state import JobStore
from .transport import OpenSSHTransport


@dataclass
class JobHandle:
    client: ClusterClient
    id: str

    @property
    def record(self) -> JobRecord:
        return self.client.store.get(self.id)

    @property
    def slurm_id(self) -> str:
        return self.record["slurm_id"]

    def status(self) -> JobState:
        return self.client.status(self.id)

    def status_info(self) -> JobStatus:
        return self.client.status_info(self.id)

    def refresh(self) -> JobState:
        return self.status()

    def wait(
        self,
        *,
        timeout: float | None = None,
        progress: bool = True,
        heartbeat: float = 30.0,
        on_update: Callable[[JobStatus], None] | None = None,
    ) -> JobHandle:
        started = time.monotonic()
        last_signature: tuple[JobState, str] | None = None
        last_report = 0.0
        while True:
            info = self.status_info()
            state = info.state
            now = time.monotonic()
            signature = (state, info.reason)
            should_report = signature != last_signature or now - last_report >= heartbeat
            if should_report:
                if progress:
                    print(_format_status(self.id, info), file=sys.stderr, flush=True)
                if on_update is not None:
                    on_update(info)
                last_signature = signature
                last_report = now
            if state == JobState.SUCCEEDED:
                return self
            if state in {JobState.FAILED, JobState.CANCELLED}:
                raise JobExecutionError(self.id, state.value)
            if timeout is not None and time.monotonic() - started >= timeout:
                raise TimeoutError(f"Timed out waiting for job {self.id}")
            time.sleep(self.client.config.poll_interval)

    def download(self, destination: str | Path) -> Path:
        return self.client.download(self.id, destination)

    def cancel(self) -> None:
        self.client.cancel(self.id)

    def logs(self, *, lines: int = 200) -> str:
        return self.client.logs(self.id, lines=lines)


class ClusterClient:
    """Importable SDK facade for remote model execution on a SLURM cluster."""

    def __init__(
        self,
        config: ClusterConfig,
        *,
        transport: RemoteTransport | None = None,
        discovery: ClusterDiscovery | None = None,
        scheduler: SchedulerBackend | None = None,
        store: JobStore | None = None,
    ):
        self.config = config
        self.transport = cast(RemoteTransport, transport or OpenSSHTransport(config))
        self.remote_root = self._resolve_remote_root(config.remote_root)
        self.discovery: ClusterDiscovery = discovery or SlurmDiscovery(
            self.transport, config.inventory_file
        )
        self.selector = ResourceSelector()
        self.scheduler: SchedulerBackend = scheduler or SlurmBackend(self.transport)
        self.store = store or JobStore(config.state_dir)
        self.artifacts = ArtifactCache(self.transport, self.remote_root)
        self.environments = EnvironmentManager(self.transport, self.remote_root)
        self.bootstrapper = ClusterBootstrapper(
            transport=self.transport,
            scheduler=self.scheduler,
            discovery=self.discovery,
            state_dir=config.state_dir,
            remote_root=self.remote_root,
            poll_interval=config.poll_interval,
            ttl_seconds=config.discovery_ttl,
        )
        self._cluster_snapshot: ClusterSnapshot | None = None
        self._cluster_snapshot_at = 0.0

    def _resolve_remote_root(self, configured: str) -> str:
        if configured == "~" or configured.startswith("~/"):
            home = self.transport.checked(["printenv", "HOME"], timeout=15)
            configured = home + configured[1:]
        if not configured.startswith("/"):
            home = self.transport.checked(["printenv", "HOME"], timeout=15)
            configured = f"{home.rstrip('/')}/{configured}"
        return configured.rstrip("/")

    @classmethod
    def from_env(cls, prefix: str = "CLUSTER_") -> ClusterClient:
        return cls(ClusterConfig.from_env(prefix))

    def inspect_cluster(self) -> ClusterSnapshot:
        return self.discover()

    def discover(self) -> ClusterSnapshot:
        now = time.monotonic()
        cached = self._cluster_snapshot
        cached_at = self._cluster_snapshot_at
        if cached is not None and now - cached_at < self.config.discovery_ttl:
            return cached
        snapshot = self.discovery.snapshot()
        self._cluster_snapshot = snapshot
        self._cluster_snapshot_at = now
        return snapshot

    def refresh_cluster(self) -> ClusterSnapshot:
        self._cluster_snapshot = None
        return self.discover()

    def bootstrap(
        self,
        *,
        probe_policy: ProbePolicy | str | None = None,
        refresh: bool = False,
        require_vram: bool = False,
        progress: bool = True,
    ) -> dict[str, object]:
        policy = probe_policy or ProbePolicy(
            mode=self.config.auto_probe,
            max_parallel=self.config.probe_max_parallel,
            max_wait_seconds=self.config.probe_wait_timeout,
            safe_vram_reserve_gb=self.config.gpu_vram_reserve_gb,
        )
        inventory = self.bootstrapper.bootstrap(
            policy=policy,
            refresh=refresh,
            require_vram=require_vram,
            on_update=(
                lambda message: print(f"[nodus] bootstrap | {message}", file=sys.stderr, flush=True)
                if progress
                else None
            ),
        )
        self._cluster_snapshot = None
        return inventory

    def model(
        self,
        *,
        name: str,
        project: Project | str | Path,
        entrypoint: str | None = None,
        environment: Venv | None = None,
        requirements: str | Path | None = None,
        checkpoint: Checkpoint | str | Path | None = None,
        resources: ResourceRequest | None = None,
    ) -> Model:
        if isinstance(project, Project):
            if entrypoint is not None:
                raise ValueError("entrypoint is already defined by Project")
            project_contract = project
        else:
            if not entrypoint:
                raise ValueError("entrypoint is required when project is a path")
            project_contract = Project(root=project, entrypoint=entrypoint)
        if environment is not None and requirements is not None:
            raise ValueError("Use environment or requirements, not both")
        environment_contract = environment
        if environment_contract is None and requirements is not None:
            environment_contract = Venv(requirements=requirements)
        checkpoint_contract = (
            checkpoint
            if isinstance(checkpoint, Checkpoint) or checkpoint is None
            else Checkpoint(path=checkpoint)
        )
        return Model(
            client=self,
            name=name,
            project=project_contract,
            environment=environment_contract,
            checkpoint=checkpoint_contract,
            resources=resources or ResourceRequest(),
        )

    def prepare_environment(
        self,
        requirements: str | Path,
        *,
        name: str = "runtime",
        python_module: str = "auto",
    ) -> EnvironmentHandle:
        if python_module == "auto":
            modules = self.discovery.python_modules()
            python_module = modules[-1] if modules else ""
        nodes = self.discovery.nodes()
        partition = self.discovery.installation_partition(
            nodes, configured=self.config.installation_partition
        )
        return self.environments.prepare(
            Path(requirements), name=name, python_module=python_module, partition=partition
        )

    def submit(self, request: JobRequest) -> JobHandle:
        request.validate()
        if request.requirements and not request.venv:
            prepared = self.prepare_environment(
                request.requirements,
                name=request.name,
                python_module=request.python_module,
            ).wait()
            request.venv = prepared.path
            if request.python_module == "auto":
                modules = self.discovery.python_modules()
                request.python_module = modules[-1] if modules else ""
        nodes = self.discovery.nodes()
        requires_vram = bool(
            request.resources.gpu_count
            and (
                request.resources.min_vram_gb
                or request.resources.policy in {"smallest-compatible", "safe", "exact"}
            )
        )
        if requires_vram and any(
            node.gpu_count > 0 and node.vram_gb <= 0 for node in nodes
        ):
            self.bootstrap(require_vram=True)
            nodes = self.discovery.nodes()
        resources = self.selector.resolve(request.resources, nodes, self.discovery.partition_rules)
        job_id = uuid.uuid4().hex
        remote_dir = f"{self.remote_root}/jobs/{job_id}"
        manifest = request.to_manifest()
        manifest["id"] = job_id
        manifest["resolved_resources"] = resources.__dict__
        manifest["output_dir"] = f"{remote_dir}/output"
        self.store.create(job_id, remote_dir, manifest)
        self.store.update(job_id, state=JobState.UPLOADING)

        try:
            slurm_id = self._stage_and_submit(
                request=request,
                resources=resources,
                job_id=job_id,
                remote_dir=remote_dir,
                manifest=manifest,
            )
        except Exception:
            # A failure before sbatch returns a job ID is terminal and must not
            # leave a permanently misleading UPLOADING record after restart.
            self.store.update(job_id, state=JobState.FAILED)
            raise
        self.store.update(job_id, state=JobState.SUBMITTED, slurm_id=slurm_id)
        return JobHandle(self, job_id)

    def _stage_and_submit(
        self,
        *,
        request: JobRequest,
        resources: ResolvedResources,
        job_id: str,
        remote_dir: str,
        manifest: dict[str, object],
    ) -> str:
        """Stage immutable inputs and return only after sbatch reports a job ID."""
        remote_project, project_hash, project_uploaded = self.artifacts.upload_project(
            request.project_dir
        )
        manifest["project"] = {
            "remote_path": remote_project,
            "sha256": project_hash,
            "uploaded": project_uploaded,
        }
        remote_checkpoint = ""
        if request.checkpoint:
            remote_checkpoint, checkpoint_hash, checkpoint_uploaded = (
                self.artifacts.ensure_checkpoint(request.checkpoint)
            )
            checkpoint_manifest: dict[str, object] = {
                "remote_path": remote_checkpoint,
                "sha256": checkpoint_hash,
                "uploaded": checkpoint_uploaded,
            }
            manifest["checkpoint"] = checkpoint_manifest
            manifest["checkpoints"] = {request.checkpoint_name: checkpoint_manifest}

        self.transport.checked(
            ["mkdir", "-p", f"{remote_dir}/inputs", f"{remote_dir}/logs", f"{remote_dir}/output"]
        )
        for source in request.inputs:
            self.transport.copy_to(source, f"{remote_dir}/inputs/{source.name}")
        input_bindings: dict[str, str] = {}
        for name, source in sorted(request.named_inputs.items()):
            remote_input_dir = f"{remote_dir}/inputs/{name}"
            self.transport.checked(["mkdir", "-p", remote_input_dir])
            remote_input = f"{remote_input_dir}/{source.name}"
            self.transport.copy_to(source, remote_input)
            input_bindings[name] = remote_input
        manifest["input_bindings"] = input_bindings

        runtime_root = f"{remote_dir}/runtime"
        runtime_source = Path(__file__).with_name("runtime.py").read_bytes()
        for package_name in ("nodus", "cluster_model_runner"):
            runtime_package = f"{runtime_root}/{package_name}"
            self.transport.upload_bytes(b"", f"{runtime_package}/__init__.py")
            self.transport.upload_bytes(runtime_source, f"{runtime_package}/runtime.py")

        environment = dict(request.environment)
        environment.update(
            {
                "CLUSTER_RUNNER_JOB_ID": job_id,
                "CLUSTER_RUNNER_INPUT_DIR": f"{remote_dir}/inputs",
                "CLUSTER_RUNNER_OUTPUT_DIR": f"{remote_dir}/output",
                "CLUSTER_RUNNER_REQUEST": f"{remote_dir}/request.json",
                "PYTHONPATH": runtime_root,
            }
        )
        if remote_checkpoint:
            environment["CLUSTER_RUNNER_CHECKPOINT"] = remote_checkpoint

        python_module = request.python_module
        if python_module == "auto":
            modules = self.discovery.python_modules()
            python_module = modules[-1] if modules else ""
        script = self.scheduler.render_script(
            job_name=request.name,
            resources=resources,
            remote_dir=remote_dir,
            project_dir=remote_project,
            command=request.command,
            venv=request.venv,
            python_module=python_module,
            environment=environment,
            correlation_id=job_id,
        )
        manifest["runtime"] = {"python_module": python_module, "venv": request.venv}
        self.store.update_manifest(job_id, manifest)
        self.transport.upload_bytes(
            json.dumps(manifest, indent=2, sort_keys=True).encode(), f"{remote_dir}/request.json"
        )
        self.transport.upload_bytes(script.encode(), f"{remote_dir}/run.sbatch")
        return self.scheduler.submit(f"{remote_dir}/run.sbatch")

    def attach(self, job_id: str) -> JobHandle:
        self.store.get(job_id)
        return JobHandle(self, job_id)

    def job(self, job_id: str) -> JobHandle:
        return self.attach(job_id)

    def list_jobs(self) -> list[JobHandle]:
        return [JobHandle(self, record["id"]) for record in self.store.list()]

    def status(self, job_id: str) -> JobState:
        return self.status_info(job_id).state

    def status_info(self, job_id: str) -> JobStatus:
        record = self.store.get(job_id)
        if not record["slurm_id"]:
            recovered = self.scheduler.find_by_correlation_id(job_id)
            if recovered:
                self.store.update(job_id, state=JobState.SUBMITTED, slurm_id=recovered)
                record = self.store.get(job_id)
            else:
                return JobStatus(JobState(record["state"]))
        info = self.scheduler.status_info(record["slurm_id"])
        if info.state == JobState.UNKNOWN:
            remote_status = self.transport.run(
                ["cat", f"{record['remote_dir']}/status"], timeout=10
            )
            if remote_status.returncode == 0:
                state = {
                    "RUNNING": JobState.RUNNING,
                    "SUCCEEDED": JobState.SUCCEEDED,
                    "FAILED": JobState.FAILED,
                }.get(remote_status.stdout.strip().upper(), JobState.UNKNOWN)
                info = JobStatus(state, slurm_id=record["slurm_id"], reason="remote marker")
        self.store.update(job_id, state=info.state)
        return info

    def cancel(self, job_id: str) -> None:
        record = self.store.get(job_id)
        if record["slurm_id"]:
            self.scheduler.cancel(record["slurm_id"])
        self.store.update(job_id, state=JobState.CANCELLED)

    def download(self, job_id: str, destination: str | Path) -> Path:
        record = self.store.get(job_id)
        destination = Path(destination).expanduser().resolve()
        target = destination / job_id
        self.transport.copy_from(f"{record['remote_dir']}/output", target)
        return target

    def logs(self, job_id: str, *, lines: int = 200) -> str:
        record = self.store.get(job_id)
        log_dir = shlex.quote(record["remote_dir"] + "/logs")
        script = f"find {log_dir} -maxdepth 1 -type f -name 'slurm_*' -exec tail -n {max(1, min(lines, 5000))} {{}} + 2>/dev/null || true"
        return self.transport.shell(script, timeout=20).stdout


def _format_status(job_id: str, info: JobStatus) -> str:
    parts = [f"[nodus] {job_id}"]
    if info.slurm_id:
        parts.append(f"SLURM {info.slurm_id}")
    parts.append(info.state.value)
    if info.state == JobState.PENDING and info.reason:
        parts.append(f"reason={info.reason}")
    if info.elapsed:
        parts.append(f"elapsed={info.elapsed}")
    if info.node and info.node not in {"(null)", "None assigned"}:
        parts.append(f"node={info.node}")
    if info.exit_code:
        parts.append(f"exit={info.exit_code}")
    return " | ".join(parts)
