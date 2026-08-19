from __future__ import annotations

import json
from types import SimpleNamespace
from typing import ClassVar

import pytest

from cluster_model_runner.config import ClusterConfig
from cluster_model_runner.model import Model, Project, Venv
from cluster_model_runner.models import JobState, JobStatus, NodeInfo, ResourceRequest
from cluster_model_runner.resources import ResourceSelector
from cluster_model_runner.session_runtime import _Worker
from cluster_model_runner.sessions.channels import SharedFilesystemChannel, SshTunnelChannel
from cluster_model_runner.sessions.models import SessionChannel, SessionState
from cluster_model_runner.sessions.service import SessionService
from cluster_model_runner.sessions.store import SessionStore
from cluster_model_runner.transport import CommandResult


class QueueTransport:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.moves: list[tuple[str, str]] = []

    def upload_bytes(self, content: bytes, remote_path: str) -> None:
        self.files[remote_path] = content

    def checked(
        self, argv: list[str], *, input_bytes: bytes | None = None, timeout: int = 30
    ) -> str:
        assert argv[0] == "mv"
        source, destination = argv[1:]
        self.moves.append((source, destination))
        request = json.loads(self.files.pop(source))
        request_id = request["id"]
        self.files[f"/remote/session/responses/{request_id}.json"] = json.dumps(
            {"ok": True, "result": {"data": {"value": 7}, "artifacts": []}}
        ).encode()
        return ""

    def run(
        self, argv: list[str], *, input_bytes: bytes | None = None, timeout: int = 30
    ) -> CommandResult:
        if argv[0] == "cat" and argv[1] in self.files:
            return CommandResult(0, self.files[argv[1]].decode(), "")
        if argv[0] == "rm":
            self.files.pop(argv[-1], None)
            return CommandResult(0, "", "")
        return CommandResult(1, "", "missing")


def test_remote_worker_loads_model_once_for_multiple_requests(tmp_path, monkeypatch):
    entrypoint = tmp_path / "persistent_model.py"
    load_counter = tmp_path / "loads.txt"
    entrypoint.write_text(
        """
def load_model(context):
    counter = context.session_dir / "loads.txt"
    count = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(count))
    return {"loads": count, "calls": 0}

def infer(model, request):
    model["calls"] += 1
    return {"loads": model["loads"], "calls": model["calls"], "value": request.parameters["value"]}
"""
    )
    monkeypatch.setenv("NODUS_SESSION_ID", "session-1")
    worker = _Worker(tmp_path, entrypoint, "secret")

    first = worker.process({"id": "request-1", "inputs": {}, "parameters": {"value": 3}})
    second = worker.process({"id": "request-2", "inputs": {}, "parameters": {"value": 5}})
    retried = worker.process({"id": "request-2", "inputs": {}, "parameters": {"value": 99}})

    assert load_counter.read_text() == "1"
    assert first["result"]["data"] == {"loads": 1, "calls": 1, "value": 3}
    assert second["result"]["data"] == {"loads": 1, "calls": 2, "value": 5}
    assert retried["result"]["data"] == second["result"]["data"]


def test_filesystem_channel_publishes_request_atomically():
    transport = QueueTransport()
    channel = SharedFilesystemChannel(transport, "/remote/session", poll_interval=0.01)

    response = channel.request(
        {"id": "request-1", "inputs": {}, "parameters": {"prompt": "hello"}}, timeout=1
    )

    assert response["result"]["data"] == {"value": 7}
    assert transport.moves == [
        (
            "/remote/session/requests/.request-1.tmp",
            "/remote/session/requests/request-1.json",
        )
    ]


def test_session_registry_survives_client_restart(tmp_path):
    first = SessionStore(tmp_path)
    first.create(
        "session-1",
        "/remote/sessions/session-1",
        "ephemeral-token",
        "auto",
        {"name": "vision"},
    )
    first.update("session-1", state=SessionState.READY, scheduler_id="42")

    restored = SessionStore(tmp_path).get("session-1")

    assert restored["state"] == SessionState.READY.value
    assert restored["scheduler_id"] == "42"
    assert restored["manifest"] == {"name": "vision"}


def test_ssh_channel_supports_login_forward_and_proxy_jump():
    channel = object.__new__(SshTunnelChannel)
    channel.local_host = "loopback-address"
    channel.local_port = 41000
    config = ClusterConfig(host="login.cluster.example.org", user="researcher")

    commands = channel._commands(config, "compute-node", 42000, "researcher")

    assert [name for name, _ in commands] == ["login-forward", "proxy-jump"]
    assert "loopback-address:41000:compute-node:42000" in commands[0][1]
    assert "researcher@compute-node" in commands[1][1]
    assert "loopback-address:41000:loopback-address:42000" in commands[1][1]


def test_auto_channel_falls_back_to_shared_filesystem(monkeypatch, tmp_path):
    class EndpointTransport:
        def checked(self, argv, timeout=0):
            if argv[:1] == ["cat"]:
                return json.dumps({"host": "compute-node", "port": 42000})
            if argv == ["id", "-un"]:
                return "researcher"
            raise AssertionError(argv)

    class TrackingStore:
        def __init__(self):
            self.state = None

        def update(self, session_id, **values):
            self.state = values.get("state")

    def reject_tunnel(*args, **kwargs):
        raise ConnectionError("forwarding disabled")

    service = object.__new__(SessionService)
    service.client = SimpleNamespace(
        config=ClusterConfig(
            host="login.cluster.example.org", state_dir=tmp_path, poll_interval=1
        ),
        transport=EndpointTransport(),
    )
    service.store = TrackingStore()
    service._channels = {}
    monkeypatch.setattr("cluster_model_runner.sessions.service.SshTunnelChannel", reject_tunnel)

    channel = service._channel(
        {
            "id": "session-1",
            "remote_dir": "/remote/session",
            "channel": SessionChannel.AUTO.value,
            "token": "token",
        }
    )

    assert isinstance(channel, SharedFilesystemChannel)
    assert service.store.state == SessionState.DEGRADED


def test_explicit_ssh_channel_does_not_hide_tunnel_failure(monkeypatch, tmp_path):
    class EndpointTransport:
        def checked(self, argv, timeout=0):
            if argv[:1] == ["cat"]:
                return json.dumps({"host": "compute-node", "port": 42000})
            return "researcher"

    service = object.__new__(SessionService)
    service.client = SimpleNamespace(
        config=ClusterConfig(host="login.cluster.example.org", state_dir=tmp_path),
        transport=EndpointTransport(),
    )
    service.store = SimpleNamespace(update=lambda *args, **kwargs: None)
    service._channels = {}
    monkeypatch.setattr(
        "cluster_model_runner.sessions.service.SshTunnelChannel",
        lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionError("forwarding disabled")),
    )

    with pytest.raises(ConnectionError, match="forwarding disabled"):
        service._channel(
            {
                "id": "session-1",
                "remote_dir": "/remote/session",
                "channel": SessionChannel.SSH.value,
                "token": "token",
            }
        )


def test_session_start_stages_worker_and_persists_scheduler_id(tmp_path):
    class Artifacts:
        def upload_project(self, project):
            return "/remote/projects/vision/hash", "project-hash", True

        def ensure_checkpoint(self, checkpoint):
            raise AssertionError("No checkpoint expected")

    class Transport:
        def __init__(self):
            self.uploads = {}

        def checked(self, argv, timeout=30):
            return ""

        def upload_bytes(self, content, remote_path):
            self.uploads[remote_path] = content

    class Discovery:
        partition_rules: ClassVar = {"gpu-batch": {}}

        def nodes(self):
            return [
                NodeInfo(
                    "gpu-node",
                    ("gpu-batch",),
                    "idle",
                    "gpu:generic:1",
                    (),
                    128000,
                    32,
                    "generic",
                    1,
                    24,
                )
            ]

        def python_modules(self):
            return []

    class Scheduler:
        def __init__(self):
            self.rendered = None

        def render_script(self, **kwargs):
            self.rendered = kwargs
            return "#!/bin/bash\ntrue\n"

        def submit(self, script_path):
            return "42"

        def status_info(self, scheduler_id):
            return JobStatus(JobState.PENDING, scheduler_id, reason="Resources")

        def find_by_correlation_id(self, correlation_id):
            return ""

    (tmp_path / "persistent.py").write_text(
        "def load_model(context): return object()\n"
        "def infer(model, request): return {'ok': True}\n"
    )
    transport = Transport()
    scheduler = Scheduler()
    client = SimpleNamespace(
        config=ClusterConfig(host="login.cluster.example.org", state_dir=tmp_path),
        remote_root="/remote",
        artifacts=Artifacts(),
        transport=transport,
        discovery=Discovery(),
        selector=ResourceSelector(),
        scheduler=scheduler,
    )
    service = SessionService(client)
    model = Model(
        client,
        "vision",
        Project(tmp_path, "persistent.py"),
        Venv(path="/remote/environments/vision"),
        resources=ResourceRequest(min_vram_gb=8),
    )

    session = service.start(model, channel="filesystem")

    record = service.store.get(session.id)
    assert session.scheduler_id == "42"
    assert record["state"] == SessionState.SUBMITTED.value
    assert scheduler.rendered["command"][2] == "worker"
    assert any(path.endswith("/runtime/nodus/session_runtime.py") for path in transport.uploads)
    assert service.status(session.id).reason == "Resources"
