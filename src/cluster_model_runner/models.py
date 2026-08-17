from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict


class JobRecord(TypedDict):
    id: str
    slurm_id: str
    state: str
    remote_dir: str
    manifest: dict[str, Any]
    created_at: str
    updated_at: str


class JobState(str, Enum):
    CREATED = "CREATED"
    UPLOADING = "UPLOADING"
    SUBMITTED = "SUBMITTED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class JobStatus:
    state: JobState
    slurm_id: str = ""
    reason: str = ""
    elapsed: str = ""
    node: str = ""
    exit_code: str = ""


@dataclass
class ResourceRequest:
    policy: str = "smallest-compatible"
    min_vram_gb: int = 0
    gpu_count: int = 1
    gpu_type: str = ""
    cpus: int = 4
    ram_gb: int = 32
    time_limit: str = "01:00:00"
    partition: str = ""
    qos: str = ""
    constraint: str = ""


@dataclass
class JobRequest:
    name: str
    command: list[str]
    project_dir: Path
    inputs: list[Path] = field(default_factory=list)
    named_inputs: dict[str, Path] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    venv: str = ""
    python_module: str = "auto"
    requirements: Path | None = None
    checkpoint: Path | None = None
    checkpoint_name: str = "default"
    environment: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name or not self.command:
            raise ValueError("Job name and command are required")
        if not self.project_dir.expanduser().exists():
            raise ValueError(f"Project directory does not exist: {self.project_dir}")
        for path in self.inputs:
            if not path.expanduser().exists():
                raise ValueError(f"Input does not exist: {path}")
        for name, path in self.named_inputs.items():
            if not name or not name.replace("_", "").replace("-", "").isalnum():
                raise ValueError(f"Invalid input name: {name!r}")
            if not path.expanduser().exists():
                raise ValueError(f"Input does not exist: {path}")
        if self.resources.gpu_count < 0 or self.resources.cpus < 1 or self.resources.ram_gb < 1:
            raise ValueError("Invalid resource request")
        if (
            not self.checkpoint_name
            or not self.checkpoint_name.replace("_", "").replace("-", "").isalnum()
        ):
            raise ValueError(f"Invalid checkpoint name: {self.checkpoint_name!r}")

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        # Remote manifests describe staged objects, never absolute source paths
        # or environment values from the user's workstation.
        payload["project_dir"] = self.project_dir.name
        payload["inputs"] = [path.name for path in self.inputs]
        payload["named_inputs"] = {
            name: path.name for name, path in sorted(self.named_inputs.items())
        }
        payload["requirements"] = self.requirements.name if self.requirements else None
        payload["checkpoint"] = self.checkpoint.name if self.checkpoint else None
        payload["environment"] = sorted(self.environment)
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> JobRequest:
        resources_payload = payload.get("resources", {})
        if not isinstance(resources_payload, Mapping):
            raise TypeError("resources must be an object")
        raw_inputs = payload.get("inputs", [])
        raw_named_inputs = payload.get("named_inputs", {})
        if isinstance(raw_inputs, Mapping):
            raw_named_inputs = raw_inputs
            raw_inputs = []
        raw_command = payload.get("command")
        raw_parameters = payload.get("parameters", {})
        raw_environment = payload.get("environment", {})
        if not isinstance(raw_inputs, list):
            raise TypeError("inputs must be a list or object")
        if not isinstance(raw_named_inputs, Mapping):
            raise TypeError("named_inputs must be an object")
        if not isinstance(raw_command, list) or not all(
            isinstance(item, str) for item in raw_command
        ):
            raise ValueError("command must be a list of strings")
        if not isinstance(raw_parameters, Mapping) or not isinstance(raw_environment, Mapping):
            raise TypeError("parameters and environment must be objects")
        return cls(
            name=str(payload.get("name") or ""),
            command=list(raw_command),
            project_dir=Path(str(payload.get("project_dir") or ".")),
            inputs=[Path(str(item)) for item in raw_inputs],
            named_inputs={str(key): Path(str(value)) for key, value in raw_named_inputs.items()},
            parameters=dict(raw_parameters),
            resources=ResourceRequest(**dict(resources_payload)),
            venv=str(payload.get("venv") or ""),
            python_module=str(payload.get("python_module") or "auto"),
            requirements=(
                Path(str(payload["requirements"])) if payload.get("requirements") else None
            ),
            checkpoint=(Path(str(payload["checkpoint"])) if payload.get("checkpoint") else None),
            checkpoint_name=str(payload.get("checkpoint_name") or "default"),
            environment={str(key): str(value) for key, value in raw_environment.items()},
        )


@dataclass(frozen=True)
class NodeInfo:
    name: str
    partitions: tuple[str, ...]
    state: str
    gres: str
    features: tuple[str, ...]
    memory_mb: int
    cpus: int
    gpu_type: str = ""
    gpu_count: int = 0
    vram_gb: int = 0


@dataclass(frozen=True)
class ResolvedResources:
    partition: str
    qos: str
    gpu_count: int
    gpu_type: str
    cpus: int
    ram_gb: int
    time_limit: str
    constraint: str = ""
    eligible_nodes: tuple[str, ...] = ()
    excluded_nodes: tuple[str, ...] = ()
    typed_gres: bool = False


class ClusterSnapshot(TypedDict):
    nodes: list[dict[str, Any]]
    python_modules: list[str]
    scontrol_nodes: dict[str, dict[str, str]]
