from __future__ import annotations

import hashlib
import json
import re
import shlex
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import ClusterDiscovery, RemoteTransport, SchedulerBackend
from .models import JobState, NodeInfo
from .resources import node_is_schedulable


@dataclass(frozen=True)
class ProbePolicy:
    """Controls the optional SLURM jobs used to characterize GPUs."""

    mode: str = "when-needed"
    max_parallel: int = 4
    time_limit: str = "00:02:00"
    max_wait_seconds: int = 900
    safe_vram_reserve_gb: int = 2

    def __post_init__(self) -> None:
        if self.mode not in {"never", "when-needed", "representative", "all-nodes"}:
            raise ValueError(f"Unsupported probe policy: {self.mode}")
        if self.max_parallel < 1:
            raise ValueError("Probe max_parallel must be at least one")
        if self.max_wait_seconds < 1 or self.safe_vram_reserve_gb < 0:
            raise ValueError("Probe wait and VRAM reserve must be non-negative")
        if not re.fullmatch(r"\d{2,3}:\d{2}:\d{2}", self.time_limit):
            raise ValueError("Probe time_limit must use HH:MM:SS")


def parse_nvidia_smi_csv(output: str, *, reserve_gb: int = 2) -> list[dict[str, Any]]:
    """Parse the stable CSV query emitted by a Nodus GPU probe."""

    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) < 5 or not fields[2].isdigit():
            continue
        index, name, raw_memory, driver, compute_capability = fields[:5]
        total_mb = int(raw_memory)
        reserve_mb = reserve_gb * 1024
        safe_mb = max(0, total_mb - reserve_mb)
        gpus.append(
            {
                "index": int(index) if index.isdigit() else index,
                "name": name,
                "canonical_name": _canonical_gpu_name(name),
                "total_vram_mb": total_mb,
                "safe_vram_mb": safe_mb,
                "driver_version": driver,
                "compute_capability": compute_capability,
            }
        )
    return gpus


def cluster_fingerprint(cluster_name: str, nodes: list[NodeInfo]) -> str:
    topology = {
        "cluster": cluster_name,
        "nodes": [
            {
                "name": node.name,
                "partitions": sorted(node.partitions),
                "gres": node.gres,
                "features": sorted(node.features),
                "cpus": node.cpus,
                "memory_mb": node.memory_mb,
            }
            for node in sorted(nodes, key=lambda item: item.name)
        ],
    }
    encoded = json.dumps(topology, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class ClusterBootstrapper:
    """Build and persist a cluster inventory without cluster-specific tables."""

    def __init__(
        self,
        *,
        transport: RemoteTransport,
        scheduler: SchedulerBackend,
        discovery: ClusterDiscovery,
        state_dir: Path,
        remote_root: str,
        poll_interval: int = 5,
        ttl_seconds: int = 86400,
    ) -> None:
        self.transport = transport
        self.scheduler = scheduler
        self.discovery = discovery
        self.state_dir = state_dir
        self.remote_root = remote_root.rstrip("/")
        self.poll_interval = max(1, poll_interval)
        self.ttl_seconds = max(0, ttl_seconds)

    def bootstrap(
        self,
        *,
        policy: ProbePolicy | str = "when-needed",
        refresh: bool = False,
        require_vram: bool = False,
        on_update: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        selected_policy = policy if isinstance(policy, ProbePolicy) else ProbePolicy(mode=policy)
        nodes = self.discovery.nodes()
        cluster_name = self._cluster_name()
        fingerprint = cluster_fingerprint(cluster_name, nodes)
        inventory_path = self.state_dir / "clusters" / fingerprint / "inventory.json"
        cache_fresh = (
            inventory_path.is_file()
            and self.ttl_seconds > 0
            and time.time() - inventory_path.stat().st_mtime < self.ttl_seconds
        )
        if cache_fresh and not refresh:
            raw_cached = json.loads(inventory_path.read_text(encoding="utf-8"))
            if not isinstance(raw_cached, dict):
                raise TypeError("Cached cluster inventory must be a JSON object")
            cached: dict[str, Any] = raw_cached
            cached_nodes = cached.get("nodes", {})
            if isinstance(cached_nodes, dict):
                for node in nodes:
                    entry = cached_nodes.get(node.name)
                    if not isinstance(entry, dict):
                        continue
                    entry["state"] = node.state
                    entry["schedulable"] = node_is_schedulable(node.state)
                    if not entry["schedulable"] and entry.get("probe_status") == "succeeded":
                        entry["probe_status"] = "stale"
            self._apply_inventory(cached)
            missing_required_vram = require_vram and isinstance(cached_nodes, dict) and any(
                int(entry.get("gpu_count", 0)) > 0 and int(entry.get("vram_gb", 0)) <= 0
                for entry in cached_nodes.values()
                if isinstance(entry, dict)
            )
            if not missing_required_vram or selected_policy.mode == "never":
                return cached
            nodes = self.discovery.nodes()

        entries = {node.name: self._node_entry(node) for node in nodes}
        candidates = self._probe_candidates(nodes, selected_policy.mode, require_vram)
        probe_results = self._probe_nodes(
            candidates, fingerprint, selected_policy, on_update=on_update
        )
        representative_results = {
            self._group_key(node): probe_results[node.name]
            for node in candidates
            if node.name in probe_results
        }
        for node_name, result in probe_results.items():
            entries[node_name].update(result)
            gpus = result.get("gpus", [])
            if gpus:
                entries[node_name]["gpu_type"] = gpus[0]["canonical_name"]
                entries[node_name]["vram_gb"] = min(
                    int(gpu["safe_vram_mb"]) // 1024 for gpu in gpus
                )
        if selected_policy.mode in {"when-needed", "representative"}:
            for node in nodes:
                if node.name in probe_results or node.vram_gb > 0:
                    continue
                representative = representative_results.get(self._group_key(node))
                if representative and representative.get("gpus"):
                    inherited = dict(representative)
                    inherited["probe_status"] = "inferred-from-representative"
                    inherited["source"] = {
                        "slurm": True,
                        "gpu_probe": False,
                        "representative_probe": True,
                    }
                    entries[node.name].update(inherited)
                    gpus = inherited["gpus"]
                    entries[node.name]["gpu_type"] = gpus[0]["canonical_name"]
                    entries[node.name]["vram_gb"] = min(
                        int(gpu["safe_vram_mb"]) // 1024 for gpu in gpus
                    )

        inventory: dict[str, Any] = {
            "schema_version": 1,
            "cluster": {
                "name": cluster_name,
                "fingerprint": fingerprint,
                "discovered_at": datetime.now(timezone.utc).isoformat(),
            },
            "nodes": entries,
            "probe_policy": asdict(selected_policy),
        }
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = inventory_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(inventory_path)
        remote_path = f"{self.remote_root}/discovery/{fingerprint}/inventory.json"
        self.transport.upload_bytes(
            json.dumps(inventory, indent=2, sort_keys=True).encode(), remote_path
        )
        self._apply_inventory(inventory)
        return inventory

    def _apply_inventory(self, payload: dict[str, Any]) -> None:
        apply_inventory = getattr(self.discovery, "apply_inventory", None)
        if callable(apply_inventory):
            apply_inventory(payload)

    def _cluster_name(self) -> str:
        result = self.transport.run(["scontrol", "show", "config"], timeout=20)
        if result.returncode == 0:
            match = re.search(r"(?m)^\s*ClusterName\s*=\s*(\S+)", result.stdout)
            if match:
                return match.group(1)
        return "slurm-cluster"

    @staticmethod
    def _node_entry(node: NodeInfo) -> dict[str, Any]:
        return {
            "name": node.name,
            "state": node.state,
            "partitions": list(node.partitions),
            "cpus": node.cpus,
            "ram_mb": node.memory_mb,
            "gres": node.gres,
            "features": list(node.features),
            "gpu_type": node.gpu_type,
            "gpu_count": node.gpu_count,
            "vram_gb": node.vram_gb,
            "probe_status": "not-required" if node.vram_gb else "not-run",
            "schedulable": node_is_schedulable(node.state),
            "source": {"slurm": True, "gpu_probe": False},
        }

    @staticmethod
    def _probe_candidates(
        nodes: list[NodeInfo], mode: str, require_vram: bool
    ) -> list[NodeInfo]:
        unknown = [node for node in nodes if node.gpu_count > 0 and node.vram_gb <= 0]
        if mode == "never" or (mode == "when-needed" and not require_vram):
            return []
        if mode == "all-nodes":
            return [node for node in nodes if node.gpu_count > 0]
        groups: dict[tuple[object, ...], NodeInfo] = {}
        for node in unknown:
            key = (
                node.gres,
                tuple(sorted(node.features)),
                tuple(sorted(node.partitions)),
                node.cpus,
                node.memory_mb,
            )
            groups.setdefault(key, node)
        return sorted(groups.values(), key=lambda item: item.name)

    @staticmethod
    def _group_key(node: NodeInfo) -> tuple[object, ...]:
        return (
            node.gres,
            tuple(sorted(node.features)),
            tuple(sorted(node.partitions)),
            node.cpus,
            node.memory_mb,
        )

    def _probe_nodes(
        self,
        nodes: list[NodeInfo],
        fingerprint: str,
        policy: ProbePolicy,
        *,
        on_update: Callable[[str], None] | None = None,
    ) -> dict[str, dict[str, Any]]:
        pending = list(nodes)
        active: dict[str, tuple[NodeInfo, str]] = {}
        results: dict[str, dict[str, Any]] = {}
        last_status: dict[str, tuple[JobState, str]] = {}
        started = time.monotonic()
        while pending or active:
            while pending and len(active) < policy.max_parallel:
                node = pending.pop(0)
                result_path, scheduler_id = self._submit_probe(node, fingerprint, policy)
                active[scheduler_id] = (node, result_path)
                if on_update:
                    on_update(f"probe {node.name} submitted as scheduler job {scheduler_id}")
            for scheduler_id, (node, result_path) in list(active.items()):
                status = self.scheduler.status_info(scheduler_id)
                signature = (status.state, status.reason)
                if on_update and last_status.get(scheduler_id) != signature:
                    reason = f" reason={status.reason}" if status.reason else ""
                    on_update(
                        f"probe {node.name} job {scheduler_id} {status.state.value}{reason}"
                    )
                    last_status[scheduler_id] = signature
                if status.state == JobState.SUCCEEDED:
                    output = self.transport.run(["cat", result_path], timeout=20)
                    gpus = (
                        parse_nvidia_smi_csv(
                            output.stdout, reserve_gb=policy.safe_vram_reserve_gb
                        )
                        if output.returncode == 0
                        else []
                    )
                    results[node.name] = {
                        "gpus": gpus,
                        "probe_status": "succeeded" if gpus else "incomplete",
                        "source": {"slurm": True, "gpu_probe": bool(gpus)},
                    }
                    del active[scheduler_id]
                elif status.state in {JobState.FAILED, JobState.CANCELLED}:
                    results[node.name] = {
                        "probe_status": "failed",
                        "probe_error": status.reason or status.state.value,
                        "source": {"slurm": True, "gpu_probe": False},
                    }
                    del active[scheduler_id]
            if active and time.monotonic() - started >= policy.max_wait_seconds:
                for scheduler_id, (node, _) in active.items():
                    self.scheduler.cancel(scheduler_id)
                    results[node.name] = {
                        "probe_status": "timed-out",
                        "probe_error": "probe wait limit exceeded",
                        "source": {"slurm": True, "gpu_probe": False},
                    }
                    if on_update:
                        on_update(f"probe {node.name} job {scheduler_id} timed out and was cancelled")
                active.clear()
            if active:
                time.sleep(self.poll_interval)
        return results

    def _submit_probe(
        self, node: NodeInfo, fingerprint: str, policy: ProbePolicy
    ) -> tuple[str, str]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", node.name):
            raise ValueError(f"Unsafe node name returned by SLURM: {node.name!r}")
        probe_dir = f"{self.remote_root}/discovery/{fingerprint}/nodes/{node.name}"
        result_path = f"{probe_dir}/gpu.csv"
        script_path = f"{probe_dir}/probe.sbatch"
        partition = next(
            (
                name
                for name in node.partitions
                if self.discovery.partition_rules.get(name, {}).get("batch") is not False
            ),
            node.partitions[0],
        )
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", partition):
            raise ValueError(f"Unsafe partition name returned by SLURM: {partition!r}")
        script = "\n".join(
            [
                "#!/bin/bash",
                "#SBATCH --job-name=nodus-probe",
                f"#SBATCH --partition={partition}",
                "#SBATCH --nodes=1",
                "#SBATCH --ntasks=1",
                "#SBATCH --cpus-per-task=1",
                "#SBATCH --mem=512M",
                "#SBATCH --gres=gpu:1",
                f"#SBATCH --time={policy.time_limit}",
                f"#SBATCH --nodelist={node.name}",
                f"#SBATCH --output={probe_dir}/slurm_%j.out",
                f"#SBATCH --error={probe_dir}/slurm_%j.err",
                "set -Eeuo pipefail",
                (
                    "nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap "
                    f"--format=csv,noheader,nounits > {shlex.quote(result_path)}"
                ),
                "",
            ]
        )
        self.transport.upload_bytes(script.encode(), script_path)
        return result_path, self.scheduler.submit(script_path)


def _canonical_gpu_name(name: str) -> str:
    value = name.lower()
    for word in ("nvidia", "corporation", "geforce", "tesla"):
        value = re.sub(rf"\b{word}\b", "", value)
    return re.sub(r"[^a-z0-9]+", "", value) or "gpu"
