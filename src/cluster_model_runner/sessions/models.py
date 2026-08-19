from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class SessionChannel(str, Enum):
    AUTO = "auto"
    SSH = "ssh"
    FILESYSTEM = "filesystem"


class SessionState(str, Enum):
    CREATED = "CREATED"
    UPLOADING = "UPLOADING"
    SUBMITTED = "SUBMITTED"
    PENDING = "PENDING"
    STARTING = "STARTING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SessionStatus:
    state: SessionState
    scheduler_id: str = ""
    reason: str = ""
    elapsed: str = ""
    node: str = ""
    channel: SessionChannel | None = None


@dataclass(frozen=True)
class SessionResult:
    session_id: str
    request_id: str
    data: dict[str, Any]
    artifacts: tuple[str, ...]
    remote_output_dir: str
    _transport: Any

    def download(self, destination: str | Path) -> Path:
        target = Path(destination).expanduser().resolve() / self.request_id
        self._transport.copy_from(self.remote_output_dir, target)
        return target
