from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from ipaddress import IPv4Address
from typing import Any, Protocol

from ..config import ClusterConfig
from ..contracts import RemoteTransport


class RequestChannel(Protocol):
    name: str

    def request(self, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]: ...

    def close(self) -> None: ...


@dataclass
class SharedFilesystemChannel:
    transport: RemoteTransport
    session_dir: str
    poll_interval: float
    name: str = "filesystem"

    def request(self, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        request_id = str(payload["id"])
        temporary = f"{self.session_dir}/requests/.{request_id}.tmp"
        published = f"{self.session_dir}/requests/{request_id}.json"
        response = f"{self.session_dir}/responses/{request_id}.json"
        self.transport.upload_bytes(json.dumps(payload, sort_keys=True).encode(), temporary)
        self.transport.checked(["mv", temporary, published], timeout=15)
        started = time.monotonic()
        while True:
            result = self.transport.run(["cat", response], timeout=15)
            if result.returncode == 0 and result.stdout:
                self.transport.run(["rm", "-f", response], timeout=10)
                return dict(json.loads(result.stdout))
            if time.monotonic() - started >= timeout:
                raise TimeoutError(f"Timed out waiting for session request {request_id}")
            time.sleep(self.poll_interval)

    def close(self) -> None:
        return


class SshTunnelChannel:
    """Authenticated HTTP over an OpenSSH local-forward process."""

    name = "ssh"

    def __init__(
        self,
        config: ClusterConfig,
        *,
        node: str,
        remote_port: int,
        token: str,
        compute_user: str = "",
        startup_timeout: float = 5,
    ):
        self.token = token
        self.local_host = str(IPv4Address(0x7F000001))
        self.local_port = _free_port()
        self.process: subprocess.Popen[bytes] | None = None
        self.strategy = ""
        errors: list[str] = []
        for strategy, command in self._commands(config, node, remote_port, compute_user):
            self.process = subprocess.Popen(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            try:
                self._wait_healthy(startup_timeout)
                self.strategy = strategy
                return
            except Exception as exc:  # noqa: BLE001 - try the next OpenSSH topology
                errors.append(f"{strategy}: {exc}")
                self.close()
        raise ConnectionError("; ".join(errors) or "No SSH tunnel strategy succeeded")

    def _commands(
        self, config: ClusterConfig, node: str, remote_port: int, compute_user: str
    ) -> list[tuple[str, list[str]]]:
        base = [
            "ssh",
            "-N",
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            f"ConnectTimeout={config.connect_timeout}",
        ]
        if config.ssh_port != 22:
            base += ["-p", str(config.ssh_port)]
        if config.ssh_key:
            base += ["-i", config.ssh_key]
        login_forward = base + [
            "-L",
            f"{self.local_host}:{self.local_port}:{node}:{remote_port}",
            config.target,
        ]
        jump = config.target
        if not config.ssh_alias and config.ssh_port != 22:
            jump = f"{config.target}:{config.ssh_port}"
        compute_target = f"{compute_user}@{node}" if compute_user else node
        direct_forward = [
            "ssh",
            "-N",
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            f"ConnectTimeout={config.connect_timeout}",
            "-J",
            jump,
        ]
        if config.ssh_key:
            direct_forward += ["-i", config.ssh_key]
        direct_forward += [
            "-L",
            f"{self.local_host}:{self.local_port}:{self.local_host}:{remote_port}",
            compute_target,
        ]
        return [("login-forward", login_forward), ("proxy-jump", direct_forward)]

    def _wait_healthy(self, timeout: float) -> None:
        if self.process is None:
            raise ConnectionError("SSH tunnel process was not started")
        started = time.monotonic()
        while time.monotonic() - started < timeout:
            if self.process.poll() is not None:
                detail = (self.process.stderr.read() if self.process.stderr else b"").decode(
                    "utf-8", errors="replace"
                )
                raise ConnectionError(detail.strip() or "SSH tunnel exited")
            try:
                self._http("GET", "/health", None, timeout=0.5)
                return
            except (OSError, urllib.error.URLError, TimeoutError):
                time.sleep(0.1)
        raise TimeoutError("SSH tunnel health check timed out")

    def _http(
        self, method: str, path: str, payload: dict[str, Any] | None, *, timeout: float
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f"http://{self.local_host}:{self.local_port}{path}",
            data=body,
            method=method,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return dict(json.loads(response.read()))
        except urllib.error.HTTPError as exc:
            payload_value = json.loads(exc.read())
            if isinstance(payload_value, dict):
                return dict(payload_value)
            raise

    def request(self, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        return self._http("POST", "/infer", payload, timeout=timeout)

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        self.process = None


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind((str(IPv4Address(0x7F000001)), 0))
        return int(listener.getsockname()[1])
