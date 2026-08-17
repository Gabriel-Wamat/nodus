from __future__ import annotations

import hashlib
import re
import shlex
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .contracts import RemoteTransport
from .exceptions import ConfigurationError
from .models import JobState
from .slurm import STATE_MAP


@dataclass
class EnvironmentHandle:
    manager: EnvironmentManager
    path: str
    environment_id: str
    slurm_id: str = ""
    reused: bool = False

    def status(self) -> JobState:
        if self.reused:
            return JobState.SUCCEEDED
        return self.manager.status(self)

    def wait(self, *, timeout: float | None = None, poll_interval: int = 10) -> EnvironmentHandle:
        started = time.monotonic()
        while True:
            state = self.status()
            if state == JobState.SUCCEEDED:
                env_root = self.path.rsplit("/venv", 1)[0]
                if self.manager.transport.run(["test", "-f", f"{env_root}/_READY"]).returncode != 0:
                    raise RuntimeError("Environment job completed without _READY marker")
                return self
            if state in {JobState.FAILED, JobState.CANCELLED}:
                raise RuntimeError(f"Environment installation ended as {state.value}")
            if timeout is not None and time.monotonic() - started >= timeout:
                raise TimeoutError(f"Timed out preparing environment {self.environment_id}")
            time.sleep(poll_interval)


class EnvironmentManager:
    def __init__(self, transport: RemoteTransport, remote_root: str):
        self.transport = transport
        self.remote_root = remote_root.rstrip("/")

    def prepare(
        self,
        requirements: Path,
        *,
        python_module: str = "",
        name: str = "runtime",
        partition: str = "install",
        time_limit: str = "00:30:00",
        cpus: int = 8,
        ram_gb: int = 32,
    ) -> EnvironmentHandle:
        requirements = requirements.expanduser().resolve()
        if not requirements.is_file():
            raise ConfigurationError(f"Requirements file not found: {requirements}")
        python_identity = python_module or "system:python3"
        digest = hashlib.sha256(
            requirements.read_bytes() + b"\0" + python_identity.encode()
        ).hexdigest()
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "-", name)[:48]
        env_id = f"{safe_name}-{digest}"
        env_dir = f"{self.remote_root}/environments/{env_id}"
        if self.transport.run(["test", "-f", f"{env_dir}/_READY"]).returncode == 0:
            return EnvironmentHandle(self, f"{env_dir}/venv", env_id, reused=True)

        lock_dir = f"{self.remote_root}/locks/environment-{digest}"
        specs_dir = f"{self.remote_root}/environment_specs/{digest}"
        acquire = (
            f"mkdir -p {shlex.quote(self.remote_root + '/locks')} "
            f"{shlex.quote(specs_dir)} {shlex.quote(self.remote_root + '/logs')}; "
            f"if [ -d {shlex.quote(lock_dir)} ] && "
            f"find {shlex.quote(lock_dir)} -maxdepth 0 -mmin +120 | grep -q .; then "
            f"rm -rf {shlex.quote(lock_dir)}; fi; "
            f"mkdir {shlex.quote(lock_dir)} 2>/dev/null"
        )
        acquired = self.transport.shell(acquire, timeout=20).returncode == 0
        if not acquired:
            return EnvironmentHandle(self, f"{env_dir}/venv", env_id)

        remote_requirements = f"{specs_dir}/requirements.lock"
        self.transport.copy_to(requirements, remote_requirements)
        script_path = f"{specs_dir}/install-{uuid.uuid4().hex}.sbatch"
        setup = []
        if python_module:
            setup = ["module purge", f"module load {shlex.quote(python_module)}"]
        script = "\n".join(
            [
                "#!/bin/bash",
                f"#SBATCH --job-name=env-{safe_name}",
                f"#SBATCH --partition={partition}",
                f"#SBATCH --cpus-per-task={cpus}",
                f"#SBATCH --mem={ram_gb}G",
                f"#SBATCH --time={time_limit}",
                f"#SBATCH --output={self.remote_root}/logs/{env_id}_%j.out",
                f"#SBATCH --error={self.remote_root}/logs/{env_id}_%j.err",
                "set -Eeuo pipefail",
                f"trap 'rm -rf {shlex.quote(lock_dir)}' EXIT",
                *setup,
                f"rm -rf {shlex.quote(env_dir)}",
                f"mkdir -p {shlex.quote(env_dir)}",
                f"cp {shlex.quote(remote_requirements)} {shlex.quote(env_dir + '/requirements.lock')}",
                f"python3 -m venv {shlex.quote(env_dir + '/venv')}",
                f"source {shlex.quote(env_dir + '/venv/bin/activate')}",
                f"python -m pip install --disable-pip-version-check --requirement {shlex.quote(remote_requirements)}",
                "python -m pip check",
                f"python -m pip freeze > {shlex.quote(env_dir + '/installed.txt')}",
                f"touch {shlex.quote(env_dir + '/_READY')}",
                "",
            ]
        )
        try:
            self.transport.upload_bytes(script.encode(), script_path)
            output = self.transport.checked(["sbatch", "--parsable", script_path])
            slurm_id = output.split(";", 1)[0].strip()
            self.transport.upload_bytes(slurm_id.encode(), f"{lock_dir}/slurm_id")
        except Exception:
            self.transport.run(["rm", "-rf", lock_dir], timeout=15)
            raise
        return EnvironmentHandle(self, f"{env_dir}/venv", env_id, slurm_id=slurm_id)

    def status(self, handle: EnvironmentHandle) -> JobState:
        env_root = handle.path.rsplit("/venv", 1)[0]
        if self.transport.run(["test", "-f", f"{env_root}/_READY"]).returncode == 0:
            return JobState.SUCCEEDED
        if not handle.slurm_id:
            lock_dir = (
                f"{self.remote_root}/locks/environment-{handle.environment_id.rsplit('-', 1)[-1]}"
            )
            job = self.transport.run(["cat", f"{lock_dir}/slurm_id"], timeout=10)
            if job.returncode != 0 or not job.stdout.strip():
                return JobState.PENDING
            handle.slurm_id = job.stdout.strip()
        queue = self.transport.run(["squeue", "-h", "-j", handle.slurm_id, "-o", "%T"])
        if queue.returncode == 0 and queue.stdout:
            return STATE_MAP.get(queue.stdout.splitlines()[0].upper(), JobState.UNKNOWN)
        account = self.transport.run(
            ["sacct", "-n", "-X", "-j", handle.slurm_id, "--format=State", "--parsable2"]
        )
        for line in account.stdout.splitlines():
            state = line.split("|", 1)[0].split("+", 1)[0].strip().upper()
            if state:
                return STATE_MAP.get(state, JobState.UNKNOWN)
        return JobState.UNKNOWN
