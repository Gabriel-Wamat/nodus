from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import shlex
import uuid
from pathlib import Path

from .contracts import RemoteTransport


def content_hash(path: Path, *, excludes: tuple[str, ...] = ()) -> str:
    path = path.expanduser().resolve()
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path)
        if any(
            fnmatch.fnmatch(part, pattern) or fnmatch.fnmatch(str(relative), pattern)
            for pattern in excludes
            for part in relative.parts
        ):
            continue
        digest.update(str(relative).encode("utf-8"))
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", value)[:120] or "artifact"


class ArtifactCache:
    """Content-addressed remote cache for checkpoints and immutable project snapshots."""

    def __init__(self, transport: RemoteTransport, remote_root: str):
        self.transport = transport
        self.remote_root = remote_root.rstrip("/")

    def _stored_checkpoint_path(self, final_dir: str, fallback_name: str) -> str:
        metadata = self.transport.run(["cat", f"{final_dir}/metadata.json"], timeout=10)
        if metadata.returncode == 0 and metadata.stdout:
            try:
                filename = str(json.loads(metadata.stdout).get("filename") or "")
            except (json.JSONDecodeError, AttributeError):
                filename = ""
            if filename and _safe_name(filename) == filename:
                return f"{final_dir}/{filename}"
        return f"{final_dir}/{fallback_name}"

    def ensure_checkpoint(self, local_path: Path) -> tuple[str, str, bool]:
        local_path = local_path.expanduser().resolve()
        digest = content_hash(local_path)
        name = _safe_name(local_path.name)
        final_dir = f"{self.remote_root}/model_store/sha256/{digest}"
        final_path = f"{final_dir}/{name}"
        ready = f"{final_dir}/_READY"
        check = self.transport.run(["test", "-f", ready], timeout=15)
        if check.returncode == 0:
            return self._stored_checkpoint_path(final_dir, name), digest, False

        cache_root = f"{self.remote_root}/model_store/sha256"
        lock = f"{self.remote_root}/locks/checkpoint-{digest}"
        acquire = (
            f"mkdir -p {shlex.quote(cache_root)} {shlex.quote(self.remote_root + '/locks')}; "
            f"if [ -d {shlex.quote(lock)} ] && "
            f"find {shlex.quote(lock)} -maxdepth 0 -mmin +120 | grep -q .; then "
            f"rm -rf {shlex.quote(lock)}; fi; "
            f"mkdir {shlex.quote(lock)} 2>/dev/null"
        )
        if self.transport.shell(acquire, timeout=20).returncode != 0:
            wait = (
                f"for i in $(seq 1 600); do [ -f {shlex.quote(ready)} ] && exit 0; "
                "sleep 1; done; exit 75"
            )
            waited = self.transport.shell(wait, timeout=620)
            if waited.returncode != 0:
                raise RuntimeError("Timed out waiting for concurrent checkpoint upload")
            return self._stored_checkpoint_path(final_dir, name), digest, False

        upload_id = uuid.uuid4().hex
        temp_dir = f"{self.remote_root}/uploads/{digest}.{upload_id}"
        try:
            self.transport.checked(["mkdir", "-p", temp_dir], timeout=15)
            remote_temp = f"{temp_dir}/{name}"
            self.transport.copy_to(local_path, remote_temp)
            metadata = json.dumps({"sha256": digest, "filename": name}, sort_keys=True).encode()
            self.transport.upload_bytes(metadata, f"{temp_dir}/metadata.json")

            publish = (
                f"trap 'rm -rf {shlex.quote(lock)}' EXIT; "
                f"if [ -f {shlex.quote(ready)} ]; then rm -rf {shlex.quote(temp_dir)}; "
                f"else mv {shlex.quote(temp_dir)} {shlex.quote(final_dir)} && "
                f"touch {shlex.quote(ready)}; fi"
            )
            result = self.transport.shell(publish, timeout=30)
            if (
                result.returncode != 0
                and self.transport.run(["test", "-f", ready], timeout=10).returncode != 0
            ):
                raise RuntimeError(result.stderr or "Could not publish checkpoint cache entry")
        except Exception:
            self.transport.run(["rm", "-rf", lock, temp_dir], timeout=20)
            raise
        return final_path, digest, True

    def upload_project(self, project_dir: Path) -> tuple[str, str, bool]:
        project_dir = project_dir.expanduser().resolve()
        excludes = (
            ".git",
            ".venv",
            "__pycache__",
            ".cluster-runner",
            ".demo-cache",
            "runs",
            "results",
            "*.pt",
            "*.pth",
            "*.ckpt",
            "*.safetensors",
        )
        digest = content_hash(project_dir, excludes=excludes)
        final_dir = f"{self.remote_root}/projects/{_safe_name(project_dir.name)}/{digest}"
        ready = f"{final_dir}/_READY"
        if self.transport.run(["test", "-f", ready], timeout=15).returncode == 0:
            return final_dir, digest, False
        lock = f"{self.remote_root}/locks/project-{digest}"
        acquire = (
            f"mkdir -p {shlex.quote(self.remote_root + '/locks')}; "
            f"if [ -d {shlex.quote(lock)} ] && "
            f"find {shlex.quote(lock)} -maxdepth 0 -mmin +120 | grep -q .; then "
            f"rm -rf {shlex.quote(lock)}; fi; "
            f"mkdir {shlex.quote(lock)} 2>/dev/null"
        )
        if self.transport.shell(acquire, timeout=20).returncode != 0:
            wait = (
                f"for i in $(seq 1 300); do [ -f {shlex.quote(ready)} ] && exit 0; "
                "sleep 1; done; exit 75"
            )
            waited = self.transport.shell(wait, timeout=320)
            if waited.returncode != 0:
                raise RuntimeError("Timed out waiting for concurrent project upload")
            return final_dir, digest, False
        upload_id = uuid.uuid4().hex
        temp_dir = f"{self.remote_root}/uploads/project-{digest}.{upload_id}"
        try:
            self.transport.checked(["mkdir", "-p", temp_dir], timeout=15)
            self.transport.copy_to(
                project_dir,
                temp_dir,
                excludes=excludes,
            )
            project_root = final_dir.rsplit("/", 1)[0]
            publish = (
                f"mkdir -p {shlex.quote(project_root)}; "
                f"trap 'rm -rf {shlex.quote(lock)}' EXIT; "
                f"if [ -f {shlex.quote(ready)} ]; then rm -rf {shlex.quote(temp_dir)}; "
                f"else mv {shlex.quote(temp_dir)} {shlex.quote(final_dir)} && "
                f"touch {shlex.quote(ready)}; fi"
            )
            result = self.transport.shell(publish, timeout=75)
            if result.returncode != 0:
                raise RuntimeError(result.stderr or "Could not publish project snapshot")
        except Exception:
            self.transport.run(["rm", "-rf", lock, temp_dir], timeout=20)
            raise
        return final_dir, digest, True
