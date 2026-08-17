from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .contracts import RemoteTransport
from .exceptions import DiscoveryError
from .models import ClusterSnapshot, NodeInfo


def _gpu_from_gres(gres: str, features: tuple[str, ...]) -> tuple[str, int, int]:
    text = gres.lower()
    gpu_type = ""
    count = 0
    for match in re.finditer(r"gpu(?::([^:,()]+))?:(\d+)", text):
        candidate, raw_count = match.groups()
        count += int(raw_count)
        if candidate and not gpu_type:
            gpu_type = candidate
    joined = " ".join(features).lower()
    if not gpu_type:
        feature_type = next(
            (item for item in features if item.lower().startswith(("gpu_", "gpu-"))), ""
        )
        gpu_type = re.sub(r"^gpu[_-]", "", feature_type, flags=re.IGNORECASE)
    memory_match = re.search(
        r"(?:^|[_-])(\d+)(?:gb|g)(?=$|[_\s-])", f"{gpu_type} {joined}"
    )
    vram = int(memory_match.group(1)) if memory_match else 0
    return gpu_type, count, vram


class SlurmDiscovery:
    def __init__(self, transport: RemoteTransport, inventory_file: Path | None = None):
        self.transport = transport
        self.inventory, self.partition_rules = self._load_inventory(inventory_file)

    @staticmethod
    def _load_inventory(
        path: Path | None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        if path is None:
            return {}, {}
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
        raw_nodes = payload.get("nodes", [])
        if isinstance(raw_nodes, dict):
            nodes = {str(name): dict(item) for name, item in raw_nodes.items()}
        else:
            nodes = {str(item["name"]): item for item in raw_nodes}
        return nodes, dict(payload.get("partitions", {}))

    def apply_inventory(self, payload: dict[str, Any]) -> None:
        raw_nodes = payload.get("nodes", {})
        if isinstance(raw_nodes, dict):
            self.inventory.update({str(name): dict(item) for name, item in raw_nodes.items()})
        raw_partitions = payload.get("partitions", {})
        if isinstance(raw_partitions, dict):
            self.partition_rules.update(
                {str(name): dict(rule) for name, rule in raw_partitions.items()}
            )

    def nodes(self) -> list[NodeInfo]:
        result = self.transport.run(["sinfo", "-N", "-h", "-o", "%N|%P|%t|%G|%f|%m|%c"], timeout=15)
        if result.returncode != 0:
            raise DiscoveryError(result.stderr or "sinfo node discovery failed")
        nodes_by_name: dict[str, NodeInfo] = {}
        seen: set[tuple[str, str]] = set()
        for line in result.stdout.splitlines():
            parts = line.split("|", 6)
            if len(parts) != 7:
                continue
            name, partition, state, gres, raw_features, memory, cpus = parts
            key = (name, partition.rstrip("*"))
            if key in seen:
                continue
            seen.add(key)
            features = tuple(item for item in raw_features.split(",") if item and item != "(null)")
            gpu_type, gpu_count, vram = _gpu_from_gres(gres, features)
            override = self.inventory.get(name, {})
            gpu_type = str(override.get("gpu_type") or gpu_type)
            gpu_count = int(override.get("gpu_count") or gpu_count)
            vram = int(override.get("vram_gb") or vram)
            clean_partition = partition.rstrip("*")
            existing = nodes_by_name.get(name)
            partitions = (
                tuple(sorted(set(existing.partitions) | {clean_partition}))
                if existing
                else (clean_partition,)
            )
            nodes_by_name[name] = NodeInfo(
                name=name,
                partitions=partitions,
                state=state.rstrip("*~#!+"),
                gres=gres or (existing.gres if existing else ""),
                features=features or (existing.features if existing else ()),
                memory_mb=int(memory) if memory.isdigit() else (existing.memory_mb if existing else 0),
                cpus=int(cpus) if cpus.isdigit() else (existing.cpus if existing else 0),
                gpu_type=gpu_type or (existing.gpu_type if existing else ""),
                gpu_count=gpu_count or (existing.gpu_count if existing else 0),
                vram_gb=vram or (existing.vram_gb if existing else 0),
            )
        nodes = [nodes_by_name[name] for name in sorted(nodes_by_name)]
        if not nodes:
            raise DiscoveryError("sinfo returned no parseable nodes")
        return nodes

    def python_modules(self) -> list[str]:
        script = (
            "command -v module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1 || true; "
            "module -t avail Python 2>&1 || true"
        )
        result = self.transport.shell(script, timeout=20)
        modules = []
        for line in result.stdout.splitlines() + result.stderr.splitlines():
            value = line.strip()
            if "Python/" in value and not value.startswith("-"):
                modules.append(value.split()[0])

        def version_key(value: str) -> tuple[int, ...]:
            match = re.search(r"Python/(\d+(?:\.\d+)+)", value)
            return tuple(int(part) for part in match.group(1).split(".")) if match else ()

        return sorted(set(modules), key=version_key)

    def scontrol_nodes(self) -> dict[str, dict[str, str]]:
        """Return authoritative SLURM node fields without assuming JSON support."""
        result = self.transport.run(["scontrol", "show", "node", "-o"], timeout=20)
        if result.returncode != 0:
            raise DiscoveryError(result.stderr or "scontrol node discovery failed")
        parsed: dict[str, dict[str, str]] = {}
        field_re = re.compile(r"(?<!\S)([A-Za-z][A-Za-z0-9_/:]*)=")
        for line in result.stdout.splitlines():
            matches = list(field_re.finditer(line))
            fields: dict[str, str] = {}
            for index, match in enumerate(matches):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
                fields[match.group(1)] = line[match.end() : end].strip()
            name = fields.get("NodeName")
            if name:
                parsed[name] = fields
        if not parsed:
            raise DiscoveryError("scontrol returned no parseable nodes")
        return parsed

    def installation_partition(
        self, nodes: list[NodeInfo] | None = None, configured: str = ""
    ) -> str:
        """Choose a visible and account-authorized short partition for building a venv."""
        nodes = nodes or self.nodes()
        available = {
            node.partitions[0]
            for node in nodes
            if node.state.lower() not in {"down", "drain", "drained", "fail", "unknown"}
        }
        inventory_candidates = sorted(
            name for name, rule in self.partition_rules.items() if rule.get("installation")
        )
        cpu_only = sorted(
            {
                node.partitions[0]
                for node in nodes
                if node.gpu_count == 0 and node.partitions[0] in available
            }
        )
        candidates = [configured] if configured else []
        candidates += [name for name in inventory_candidates if name not in candidates]
        candidates += [name for name in cpu_only if name not in candidates]
        candidates += [name for name in sorted(available) if name not in candidates]
        for partition in candidates:
            probe = self.transport.run(
                [
                    "sbatch",
                    "--test-only",
                    "--partition",
                    partition,
                    "--time",
                    "00:05:00",
                    "--mem",
                    "1G",
                    "--cpus-per-task",
                    "1",
                    "--wrap=true",
                ],
                timeout=20,
            )
            if probe.returncode == 0:
                return partition
        raise DiscoveryError("No visible and authorized partition for environment creation was found")

    def snapshot(self) -> ClusterSnapshot:
        nodes = self.nodes()
        return {
            "nodes": [node.__dict__ for node in nodes],
            "python_modules": self.python_modules(),
            "scontrol_nodes": self.scontrol_nodes(),
        }
