from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models import JobRequest, ResourceRequest

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .client import ClusterClient, JobHandle
    from .sessions import SessionChannel, SessionHandle


@dataclass(frozen=True)
class Project:
    """Immutable local project definition used to create remote snapshots."""

    root: Path | str
    entrypoint: str

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        entrypoint = Path(self.entrypoint)
        if entrypoint.is_absolute() or ".." in entrypoint.parts:
            raise ValueError("Project entrypoint must be relative to the project root")
        if not root.is_dir():
            raise ValueError(f"Project root does not exist: {root}")
        if not (root / entrypoint).is_file():
            raise ValueError(f"Project entrypoint does not exist: {root / entrypoint}")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "entrypoint", entrypoint.as_posix())


@dataclass(frozen=True)
class Checkpoint:
    """A local checkpoint addressed remotely by its content hash."""

    path: Path | str
    name: str = "default"

    def __post_init__(self) -> None:
        path = Path(self.path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Checkpoint does not exist: {path}")
        if not _valid_name(self.name):
            raise ValueError(f"Invalid checkpoint name: {self.name!r}")
        object.__setattr__(self, "path", path)


@dataclass(frozen=True)
class Venv:
    """Content-addressed environment request or an existing remote venv path."""

    requirements: Path | str | None = None
    python: str = "auto"
    path: str = ""

    def __post_init__(self) -> None:
        requirements = self.requirements
        if requirements is not None:
            resolved = Path(requirements).expanduser().resolve()
            if not resolved.is_file():
                raise ValueError(f"Requirements file does not exist: {resolved}")
            object.__setattr__(self, "requirements", resolved)
        if not requirements and not self.path:
            raise ValueError("Venv requires a requirements file or an existing remote path")
        if self.path and ("\n" in self.path or "\0" in self.path):
            raise ValueError("Invalid remote venv path")


@dataclass(frozen=True)
class Model:
    """Reusable model definition backed by the existing batch submission core."""

    client: ClusterClient = field(repr=False, compare=False)
    name: str
    project: Project
    environment: Venv | None = None
    checkpoint: Checkpoint | None = None
    resources: ResourceRequest = field(default_factory=ResourceRequest)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Model name is required")

    def submit(
        self,
        *,
        inputs: Mapping[str, str | Path] | None = None,
        parameters: Mapping[str, Any] | None = None,
        resources: ResourceRequest | None = None,
    ) -> JobHandle:
        named_inputs: dict[str, Path] = {}
        for name, value in (inputs or {}).items():
            if not _valid_name(name):
                raise ValueError(f"Invalid input name: {name!r}")
            named_inputs[name] = Path(value).expanduser().resolve()

        environment = self.environment
        request = JobRequest(
            name=self.name,
            project_dir=Path(self.project.root),
            command=["python", self.project.entrypoint],
            named_inputs=named_inputs,
            parameters=dict(parameters or {}),
            resources=resources or self.resources,
            venv=environment.path if environment else "",
            python_module=environment.python if environment else "auto",
            requirements=(
                Path(environment.requirements)
                if environment and environment.requirements is not None
                else None
            ),
            checkpoint=Path(self.checkpoint.path) if self.checkpoint else None,
            checkpoint_name=self.checkpoint.name if self.checkpoint else "default",
        )
        return self.client.submit(request)

    def session(
        self,
        *,
        entrypoint: str | None = None,
        channel: SessionChannel | str = "auto",
    ) -> SessionHandle:
        """Submit a long-lived worker that loads this model exactly once."""
        return self.client.session_service.start(self, entrypoint=entrypoint, channel=channel)


def _valid_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", value))
