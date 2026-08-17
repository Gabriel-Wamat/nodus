from __future__ import annotations

import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import ClusterConfig
from .exceptions import ConfigurationError, RemoteCommandError


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class OpenSSHTransport:
    def __init__(self, config: ClusterConfig):
        self.config = config
        if not shutil.which("ssh"):
            raise ConfigurationError("OpenSSH executable not found")

    def _ssh_base(self) -> list[str]:
        args = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.config.connect_timeout}",
            "-o",
            "ServerAliveInterval=30",
        ]
        if self.config.ssh_port != 22:
            args += ["-p", str(self.config.ssh_port)]
        if self.config.ssh_key:
            args += ["-i", str(Path(self.config.ssh_key).expanduser())]
        return args

    def run(
        self, argv: list[str], *, input_bytes: bytes | None = None, timeout: int = 30
    ) -> CommandResult:
        command = self._ssh_base() + [self.config.target, shlex.join(argv)]
        completed = subprocess.run(
            command,
            input=input_bytes,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        result = CommandResult(
            completed.returncode,
            completed.stdout.decode("utf-8", errors="replace").strip(),
            completed.stderr.decode("utf-8", errors="replace").strip(),
        )
        return result

    def checked(
        self, argv: list[str], *, input_bytes: bytes | None = None, timeout: int = 30
    ) -> str:
        result = self.run(argv, input_bytes=input_bytes, timeout=timeout)
        if result.returncode != 0:
            raise RemoteCommandError(
                f"Remote command failed ({result.returncode}): {result.stderr or result.stdout}"
            )
        return result.stdout

    def upload_bytes(self, content: bytes, remote_path: str) -> None:
        parent = remote_path.rsplit("/", 1)[0]
        script = f"mkdir -p {shlex.quote(parent)} && umask 077 && cat > {shlex.quote(remote_path)}"
        result = self.shell(script, content, timeout=60)
        if result.returncode != 0:
            raise RemoteCommandError(result.stderr or "Unable to upload remote file")

    def shell(
        self, script: str, data: bytes | None = None, timeout: int = 30
    ) -> CommandResult:
        command = self._ssh_base() + [self.config.target, script]
        completed = subprocess.run(
            command, input=data, capture_output=True, timeout=timeout, check=False
        )
        return CommandResult(
            completed.returncode,
            completed.stdout.decode("utf-8", errors="replace").strip(),
            completed.stderr.decode("utf-8", errors="replace").strip(),
        )

    def copy_to(self, local: Path, remote_path: str, *, excludes: tuple[str, ...] = ()) -> None:
        rsync_error = ""
        if shutil.which("rsync"):
            rsync_error = self._rsync_to(local, remote_path, excludes=excludes)
            if not rsync_error:
                return
        self._scp_to(local, remote_path, excludes=excludes, prior_error=rsync_error)

    def _rsync_to(self, local: Path, remote_path: str, *, excludes: tuple[str, ...]) -> str:
        ssh_parts = ["ssh"]
        if self.config.ssh_port != 22:
            ssh_parts += ["-p", str(self.config.ssh_port)]
        if self.config.ssh_key:
            ssh_parts += ["-i", str(Path(self.config.ssh_key).expanduser())]
        args = ["rsync", "-az", "--partial", "-e", shlex.join(ssh_parts)]
        for pattern in excludes:
            args += ["--exclude", pattern]
        source = str(local.expanduser())
        if local.is_dir() and not source.endswith("/"):
            source += "/"
        args += [source, f"{self.config.transfer_target}:{remote_path}"]
        completed = subprocess.run(args, capture_output=True, text=True, check=False)
        return completed.stderr.strip() if completed.returncode != 0 else ""

    def copy_from(self, remote_path: str, local: Path) -> None:
        local.expanduser().mkdir(parents=True, exist_ok=True)
        rsync_error = ""
        if shutil.which("rsync"):
            rsync_error = self._rsync_from(remote_path, local)
            if not rsync_error:
                return
        self._scp_from(remote_path, local, prior_error=rsync_error)

    def _rsync_from(self, remote_path: str, local: Path) -> str:
        ssh_parts = ["ssh"]
        if self.config.ssh_port != 22:
            ssh_parts += ["-p", str(self.config.ssh_port)]
        if self.config.ssh_key:
            ssh_parts += ["-i", str(Path(self.config.ssh_key).expanduser())]
        args = [
            "rsync",
            "-az",
            "--partial",
            "-e",
            shlex.join(ssh_parts),
            f"{self.config.transfer_target}:{remote_path.rstrip('/')}/",
            str(local.expanduser()) + "/",
        ]
        completed = subprocess.run(args, capture_output=True, text=True, check=False)
        return completed.stderr.strip() if completed.returncode != 0 else ""

    def _scp_base(self) -> list[str]:
        if not shutil.which("scp"):
            raise ConfigurationError("Neither a working rsync nor the scp executable is available")
        args = ["scp", "-q"]
        if self.config.ssh_port != 22:
            args += ["-P", str(self.config.ssh_port)]
        if self.config.ssh_key:
            args += ["-i", str(Path(self.config.ssh_key).expanduser())]
        return args

    def _scp_to(
        self,
        local: Path,
        remote_path: str,
        *,
        excludes: tuple[str, ...],
        prior_error: str,
    ) -> None:
        source = local.expanduser()
        scp_base = self._scp_base()
        target_dir = remote_path if source.is_dir() else remote_path.rsplit("/", 1)[0]
        self.checked(["mkdir", "-p", target_dir])
        with tempfile.TemporaryDirectory(prefix="nodus-transfer-") as temporary:
            if source.is_dir() and excludes:
                staged = Path(temporary) / "payload"
                shutil.copytree(source, staged, ignore=shutil.ignore_patterns(*excludes))
                source = staged
            args = list(scp_base)
            if source.is_dir():
                args.append("-r")
                source_value = str(source) + "/."
            else:
                source_value = str(source)
            args += [source_value, f"{self.config.transfer_target}:{remote_path}"]
            completed = subprocess.run(args, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or prior_error or "file upload failed"
            raise RemoteCommandError(detail)

    def _scp_from(self, remote_path: str, local: Path, *, prior_error: str) -> None:
        args = self._scp_base() + [
            "-r",
            f"{self.config.transfer_target}:{remote_path.rstrip('/')}/.",
            str(local.expanduser()) + "/",
        ]
        completed = subprocess.run(args, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or prior_error or "file download failed"
            raise RemoteCommandError(detail)
