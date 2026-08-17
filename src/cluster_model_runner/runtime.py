from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeRequest:
    """Typed view of the manifest exposed to code running inside a SLURM job."""

    manifest: dict[str, Any]

    @classmethod
    def from_cli(cls) -> RuntimeRequest:
        manifest_path = os.environ.get("CLUSTER_RUNNER_REQUEST", "")
        if not manifest_path:
            raise RuntimeError("CLUSTER_RUNNER_REQUEST is not set")
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Runtime manifest must be a JSON object")
        return cls(payload)

    @property
    def parameters(self) -> dict[str, Any]:
        value = self.manifest.get("parameters", {})
        if not isinstance(value, dict):
            raise TypeError("Manifest parameters must be an object")
        return value

    @property
    def output_dir(self) -> Path:
        configured = self.manifest.get("output_dir") or os.environ.get(
            "CLUSTER_RUNNER_OUTPUT_DIR", ""
        )
        if not configured:
            raise RuntimeError("Runtime output directory is not configured")
        return Path(str(configured))

    def input(self, name: str) -> Path:
        bindings = self.manifest.get("input_bindings", {})
        if not isinstance(bindings, dict) or name not in bindings:
            raise KeyError(f"Unknown runtime input: {name}")
        return Path(str(bindings[name]))

    def checkpoint(self, name: str = "default") -> Path:
        checkpoints = self.manifest.get("checkpoints", {})
        if isinstance(checkpoints, dict) and name in checkpoints:
            entry = checkpoints[name]
            if isinstance(entry, dict) and entry.get("remote_path"):
                return Path(str(entry["remote_path"]))
        legacy = self.manifest.get("checkpoint")
        if name == "default" and isinstance(legacy, dict) and legacy.get("remote_path"):
            return Path(str(legacy["remote_path"]))
        raise KeyError(f"Unknown runtime checkpoint: {name}")

    def write_result(
        self,
        *,
        data: dict[str, Any],
        artifacts: list[str | Path] | None = None,
    ) -> Path:
        output_dir = self.output_dir
        artifact_dir = output_dir / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for source_value in artifacts or []:
            source = Path(source_value).expanduser().resolve()
            if not source.exists():
                raise FileNotFoundError(source)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            destination = artifact_dir / source.name
            if destination.exists():
                raise FileExistsError(destination)
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
            copied.append(f"artifacts/{source.name}")
        result = {
            "job_id": self.manifest.get("id", ""),
            "data": data,
            "artifacts": copied,
        }
        target = output_dir / "result.json"
        target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return target
