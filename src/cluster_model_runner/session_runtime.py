"""Remote runtime for persistent Nodus model sessions.

This module intentionally depends only on the Python standard library.  The SDK
uploads it beside every session so the project venv does not need Nodus installed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import socket
import threading
import time
import traceback
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SessionContext:
    """Stable values passed once to the project's ``load_model`` function."""

    session_id: str
    checkpoint_path: Path | None
    session_dir: Path


@dataclass(frozen=True)
class SessionRequest:
    """One inference request delivered to the project's ``infer`` function."""

    id: str
    inputs: Mapping[str, Path]
    parameters: Mapping[str, Any]
    output_dir: Path

    def input(self, name: str) -> Path:
        try:
            return self.inputs[name]
        except KeyError as exc:
            raise KeyError(f"Input {name!r} is not present") from exc

    def write_result(
        self,
        *,
        data: Mapping[str, Any] | None = None,
        artifacts: list[str | Path] | None = None,
    ) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir = self.output_dir / "artifacts"
        names: list[str] = []
        for source_value in artifacts or []:
            source = Path(source_value).resolve()
            artifact_dir.mkdir(parents=True, exist_ok=True)
            target = artifact_dir / source.name
            if source != target.resolve():
                shutil.copy2(source, target)
            names.append(f"artifacts/{target.name}")
        payload = {"data": dict(data or {}), "artifacts": names}
        _atomic_json(self.output_dir / "result.json", payload)
        (self.output_dir / "_SUCCESS").touch()
        return payload


class _Worker:
    def __init__(self, session_dir: Path, entrypoint: Path, token: str):
        self.session_dir = session_dir
        self.token = token
        self.stop_event = threading.Event()
        self.inference_lock = threading.Lock()
        module = _load_module(entrypoint)
        load_model = getattr(module, "load_model", None)
        infer = getattr(module, "infer", None)
        if not callable(load_model) or not callable(infer):
            raise TypeError(
                "Session entrypoint must define callable load_model(context) and infer(model, request)"
            )
        checkpoint = os.environ.get("NODUS_SESSION_CHECKPOINT", "")
        context = SessionContext(
            session_id=os.environ["NODUS_SESSION_ID"],
            checkpoint_path=Path(checkpoint) if checkpoint else None,
            session_dir=session_dir,
        )
        self.model = load_model(context)
        self.infer = infer

    def process(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request_id = str(payload.get("id", ""))
        if not request_id or not request_id.replace("-", "").isalnum():
            raise ValueError("Invalid request id")
        output_dir = (self.session_dir / "outputs" / request_id).resolve()
        if self.session_dir.resolve() not in output_dir.parents:
            raise ValueError("Output path escaped the session directory")
        raw_inputs = payload.get("inputs", {})
        raw_parameters = payload.get("parameters", {})
        if not isinstance(raw_inputs, Mapping) or not isinstance(raw_parameters, Mapping):
            raise TypeError("inputs and parameters must be objects")
        input_root = (self.session_dir / "inputs" / request_id).resolve()
        inputs: dict[str, Path] = {}
        for name, path_value in raw_inputs.items():
            path = Path(str(path_value)).resolve()
            if input_root != path and input_root not in path.parents:
                raise ValueError(f"Input {name!r} escaped its request directory")
            inputs[str(name)] = path
        request = SessionRequest(request_id, inputs, dict(raw_parameters), output_dir)
        with self.inference_lock:
            result_path = output_dir / "result.json"
            if result_path.is_file() and (output_dir / "_SUCCESS").is_file():
                response = json.loads(result_path.read_text())
            else:
                result = self.infer(self.model, request)
                if result_path.is_file():
                    response = json.loads(result_path.read_text())
                elif result is None:
                    response = request.write_result()
                elif isinstance(result, Mapping):
                    response = request.write_result(data=result)
                else:
                    raise TypeError(
                        "infer() must return a mapping, None, or call request.write_result()"
                    )
        return {"id": request_id, "ok": True, "result": response}


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def do_GET(self) -> None:
        if self.path != "/health" or not self._authorized():
            self.send_error(404)
            return
        self._json(200, {"ready": True})

    def do_POST(self) -> None:
        if self.path != "/infer" or not self._authorized():
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 8 * 1024 * 1024:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length))
            self._json(200, self.server.worker.process(payload))
        except Exception as exc:  # noqa: BLE001 - return project errors to the caller
            self._json(500, {"ok": False, "error": str(exc)})

    def log_message(self, _format: str, *args: object) -> None:
        return

    def _authorized(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {self.server.worker.token}"

    def _json(self, status: int, payload: Mapping[str, Any]) -> None:
        content = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


class _Server(ThreadingHTTPServer):
    def __init__(self, worker: _Worker):
        super().__init__((os.environ.get("NODUS_SESSION_WORKER_BIND", ""), 0), _Handler)
        self.worker = worker


def _load_module(entrypoint: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"nodus_session_{uuid.uuid4().hex}", entrypoint)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load session entrypoint: {entrypoint}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True))
    os.replace(temporary, path)


def _queue_loop(worker: _Worker) -> None:
    requests = worker.session_dir / "requests"
    responses = worker.session_dir / "responses"
    requests.mkdir(parents=True, exist_ok=True)
    responses.mkdir(parents=True, exist_ok=True)
    while not worker.stop_event.wait(0.05):
        for request_path in sorted(requests.glob("*.json")):
            claimed = request_path.with_suffix(".processing")
            try:
                os.replace(request_path, claimed)
            except FileNotFoundError:
                continue
            request_id = claimed.stem
            try:
                payload = json.loads(claimed.read_text())
                response = worker.process(payload)
            except Exception as exc:  # noqa: BLE001 - isolate failures per request
                response = {"id": request_id, "ok": False, "error": str(exc)}
                traceback.print_exc()
            _atomic_json(responses / f"{request_id}.json", response)
            claimed.unlink(missing_ok=True)


def run_worker(session_dir: Path, entrypoint: Path, token: str) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    worker = _Worker(session_dir, entrypoint, token)
    server = _Server(worker)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    queue = threading.Thread(target=_queue_loop, args=(worker,), daemon=True)
    queue.start()
    _atomic_json(
        session_dir / "control" / "ready.json",
        {
            "ready": True,
            "host": socket.gethostname(),
            "port": server.server_port,
            "pid": os.getpid(),
            "started_at": time.time(),
        },
    )
    try:
        while not worker.stop_event.wait(1):
            _atomic_json(session_dir / "control" / "heartbeat.json", {"at": time.time()})
    finally:
        server.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["worker"])
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--entrypoint", required=True, type=Path)
    args = parser.parse_args(argv)
    token = os.environ.get("NODUS_SESSION_TOKEN", "")
    if not token:
        parser.error("NODUS_SESSION_TOKEN is required")
    run_worker(args.session_dir, args.entrypoint, token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
