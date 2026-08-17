from __future__ import annotations

import re
import shlex

from .contracts import RemoteTransport
from .exceptions import RemoteCommandError
from .models import JobState, JobStatus, ResolvedResources

STATE_MAP = {
    "PENDING": JobState.PENDING,
    "CONFIGURING": JobState.PENDING,
    "RUNNING": JobState.RUNNING,
    "COMPLETING": JobState.RUNNING,
    "COMPLETED": JobState.SUCCEEDED,
    "FAILED": JobState.FAILED,
    "TIMEOUT": JobState.FAILED,
    "OUT_OF_MEMORY": JobState.FAILED,
    "NODE_FAIL": JobState.FAILED,
    "PREEMPTED": JobState.FAILED,
    "CANCELLED": JobState.CANCELLED,
}


def render_sbatch(
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
) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "-", job_name)[:64] or "model-job"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={safe_name}",
        f"#SBATCH --partition={resources.partition}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={resources.cpus}",
        f"#SBATCH --mem={resources.ram_gb}G",
        f"#SBATCH --time={resources.time_limit}",
        f"#SBATCH --output={remote_dir}/logs/slurm_%j.out",
        f"#SBATCH --error={remote_dir}/logs/slurm_%j.err",
    ]
    if resources.qos:
        lines.append(f"#SBATCH --qos={resources.qos}")
    if correlation_id:
        if not re.fullmatch(r"[a-f0-9]{32}", correlation_id):
            raise ValueError("Invalid correlation ID")
        lines.append(f"#SBATCH --comment=nodus:{correlation_id}")
    if resources.gpu_count:
        gres = (
            f"gpu:{resources.gpu_type}:{resources.gpu_count}"
            if resources.gpu_type and resources.typed_gres
            else f"gpu:{resources.gpu_count}"
        )
        lines.append(f"#SBATCH --gres={gres}")
    if resources.constraint:
        lines.append(f"#SBATCH --constraint={resources.constraint}")
    # Untyped GRES cannot request a GPU model. Excluding incompatible nodes preserves
    # one-node scheduling while guaranteeing the selected hardware class.
    if resources.gpu_count and not resources.typed_gres and resources.excluded_nodes:
        lines.append(f"#SBATCH --exclude={','.join(resources.excluded_nodes)}")
    lines += [
        "",
        "set -Eeuo pipefail",
        f"RUN_DIR={shlex.quote(remote_dir)}",
        f"PROJECT_DIR={shlex.quote(project_dir)}",
        'mkdir -p "$RUN_DIR/logs" "$RUN_DIR/output"',
        'status() { printf \'%s\' "$1" > "$RUN_DIR/status"; }',
        "trap 'code=$?; status FAILED; exit $code' ERR",
        "status RUNNING",
    ]
    if python_module and python_module != "auto":
        lines += ["module purge", f"module load {shlex.quote(python_module)}"]
    if venv:
        lines.append(f"source {shlex.quote(venv)}/bin/activate")
    for key, value in sorted(environment.items()):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid environment variable name: {key}")
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines += [
        f"cd {shlex.quote(project_dir)}",
        f"{shlex.join(command)}",
        "status SUCCEEDED",
        'touch "$RUN_DIR/output/_SUCCESS"',
    ]
    return "\n".join(lines) + "\n"


class SlurmBackend:
    def __init__(self, transport: RemoteTransport):
        self.transport = transport

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
    ) -> str:
        return render_sbatch(
            job_name=job_name,
            resources=resources,
            remote_dir=remote_dir,
            project_dir=project_dir,
            command=command,
            venv=venv,
            python_module=python_module,
            environment=environment,
            correlation_id=correlation_id,
        )

    def submit(self, script_path: str) -> str:
        output = self.transport.checked(["sbatch", "--parsable", script_path], timeout=30)
        job_id = output.split(";", 1)[0].strip()
        if not re.fullmatch(r"\d+(?:_\d+)?", job_id):
            raise RemoteCommandError(f"Unexpected sbatch response: {output}")
        return job_id

    def status(self, slurm_id: str) -> JobState:
        return self.status_info(slurm_id).state

    def status_info(self, slurm_id: str) -> JobStatus:
        queue = self.transport.run(
            ["squeue", "-h", "-j", slurm_id, "-o", "%T|%R|%M|%N"], timeout=15
        )
        if queue.returncode == 0 and queue.stdout:
            state, reason, elapsed, node = (queue.stdout.splitlines()[0].split("|", 3) + [""] * 4)[
                :4
            ]
            return JobStatus(
                STATE_MAP.get(state.strip().upper(), JobState.UNKNOWN),
                slurm_id=slurm_id,
                reason=reason.strip(),
                elapsed=elapsed.strip(),
                node=node.strip(),
            )
        account = self.transport.run(
            [
                "sacct",
                "-n",
                "-X",
                "-j",
                slurm_id,
                "--format=State,ExitCode,Elapsed,NodeList",
                "--parsable2",
            ],
            timeout=15,
        )
        if account.returncode == 0:
            for line in account.stdout.splitlines():
                fields = line.split("|")
                state = fields[0].split("+", 1)[0].strip().upper()
                if state:
                    return JobStatus(
                        STATE_MAP.get(state, JobState.UNKNOWN),
                        slurm_id=slurm_id,
                        exit_code=fields[1].strip() if len(fields) > 1 else "",
                        elapsed=fields[2].strip() if len(fields) > 2 else "",
                        node=fields[3].strip() if len(fields) > 3 else "",
                    )
        control = self.transport.run(["scontrol", "show", "job", "-o", slurm_id], timeout=15)
        if control.returncode == 0:
            match = re.search(r"(?:^|\s)JobState=([^\s]+)", control.stdout)
            if match:
                reason_match = re.search(r"(?:^|\s)Reason=([^\s]+)", control.stdout)
                elapsed_match = re.search(r"(?:^|\s)RunTime=([^\s]+)", control.stdout)
                node_match = re.search(r"(?:^|\s)NodeList=([^\s]+)", control.stdout)
                exit_code_match = re.search(r"(?:^|\s)ExitCode=([^\s]+)", control.stdout)
                return JobStatus(
                    STATE_MAP.get(match.group(1).split("+", 1)[0].upper(), JobState.UNKNOWN),
                    slurm_id=slurm_id,
                    reason=reason_match.group(1) if reason_match else "",
                    elapsed=elapsed_match.group(1) if elapsed_match else "",
                    node=node_match.group(1) if node_match else "",
                    exit_code=exit_code_match.group(1) if exit_code_match else "",
                )
        return JobStatus(JobState.UNKNOWN, slurm_id=slurm_id)

    def cancel(self, slurm_id: str) -> None:
        self.transport.checked(["scancel", slurm_id], timeout=15)

    def find_by_correlation_id(self, correlation_id: str) -> str:
        marker = f"nodus:{correlation_id}"
        queue = self.transport.run(["squeue", "-h", "-o", "%i|%k"], timeout=15)
        if queue.returncode == 0:
            for line in queue.stdout.splitlines():
                job_id, _, comment = line.partition("|")
                if comment.strip() == marker and re.fullmatch(r"\d+(?:_\d+)?", job_id.strip()):
                    return job_id.strip()
        history = self.transport.run(
            [
                "sacct",
                "-n",
                "-X",
                "--starttime",
                "now-14days",
                "--format=JobIDRaw,Comment",
                "--parsable2",
            ],
            timeout=20,
        )
        if history.returncode == 0:
            for line in history.stdout.splitlines():
                job_id, _, comment = line.partition("|")
                if comment.strip().rstrip("|") == marker and re.fullmatch(
                    r"\d+(?:_\d+)?", job_id.strip()
                ):
                    return job_id.strip()
        return ""
