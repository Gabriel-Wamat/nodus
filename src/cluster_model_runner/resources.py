from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any

from .exceptions import DiscoveryError
from .models import NodeInfo, ResolvedResources, ResourceRequest

UNAVAILABLE_STATES = {"down", "drain", "drained", "fail", "failing", "maint", "unknown"}


def node_is_schedulable(state: str) -> bool:
    return state.lower() not in UNAVAILABLE_STATES


class ResourceSelector:
    """Resolve requirements to a node class while leaving final placement to SLURM."""

    def resolve(
        self,
        request: ResourceRequest,
        nodes: list[NodeInfo],
        partition_rules: dict[str, dict[str, Any]] | None = None,
    ) -> ResolvedResources:
        partition_rules = partition_rules or {}
        if request.policy not in {"smallest-compatible", "fastest-queue", "safe", "exact"}:
            raise DiscoveryError(f"Unsupported resource policy: {request.policy}")
        if request.policy == "exact" and not request.gpu_type:
            raise DiscoveryError("The exact policy requires gpu_type")
        candidates = []
        for node in nodes:
            if not node_is_schedulable(node.state):
                continue
            compatible_partitions = tuple(
                partition
                for partition in node.partitions
                if self._partition_accepts(partition, request, partition_rules)
            )
            if not compatible_partitions:
                continue
            if node.cpus and node.cpus < request.cpus:
                continue
            if node.memory_mb and node.memory_mb < request.ram_gb * 1024:
                continue
            if request.gpu_count:
                if node.gpu_count < request.gpu_count:
                    continue
                if request.policy in {"smallest-compatible", "safe"} and node.vram_gb <= 0:
                    continue
                if request.min_vram_gb and (not node.vram_gb or node.vram_gb < request.min_vram_gb):
                    continue
                if request.gpu_type and request.gpu_type.lower() not in node.gpu_type.lower():
                    continue
            candidates.append(replace(node, partitions=compatible_partitions))

        if not candidates:
            raise DiscoveryError("No discovered node satisfies the requested resources")

        if request.policy == "fastest-queue":
            state_rank = {"idle": 0, "mix": 1, "alloc": 2, "allocated": 2}
            candidates.sort(key=lambda n: (state_rank.get(n.state.lower(), 3), n.vram_gb or 10_000))
        elif request.policy == "safe":
            candidates.sort(key=lambda n: (-(n.vram_gb or 0), -n.memory_mb, n.name))
        else:
            candidates.sort(key=lambda n: (n.vram_gb or 10_000, n.memory_mb, n.name))

        chosen = candidates[0]
        gpu_type = request.gpu_type or (
            chosen.gpu_type if request.policy != "fastest-queue" else ""
        )
        partition = request.partition or self._choose_partition(candidates)
        partition_candidates = [n for n in candidates if partition in n.partitions]
        if request.policy in {"smallest-compatible", "safe", "exact"} and chosen.gpu_type:
            class_candidates = [n for n in partition_candidates if n.gpu_type == chosen.gpu_type]
        else:
            class_candidates = partition_candidates
        eligible = tuple(sorted({n.name for n in class_candidates}))
        all_gpu_nodes = {n.name for n in nodes if n.gpu_count > 0}
        excluded = tuple(sorted(all_gpu_nodes - set(eligible)))
        rule = partition_rules.get(partition, {})
        return ResolvedResources(
            partition=partition,
            qos=request.qos or str(rule.get("qos") or ""),
            gpu_count=request.gpu_count,
            gpu_type=gpu_type,
            cpus=request.cpus,
            ram_gb=request.ram_gb,
            time_limit=request.time_limit,
            constraint=request.constraint,
            eligible_nodes=eligible,
            excluded_nodes=excluded,
            typed_gres=bool(gpu_type and f"gpu:{gpu_type.lower()}:" in chosen.gres.lower()),
        )

    @staticmethod
    def _partition_accepts(
        partition: str,
        request: ResourceRequest,
        partition_rules: dict[str, dict[str, Any]],
    ) -> bool:
        if request.partition and request.partition != partition:
            return False
        rule = partition_rules.get(partition, {})
        if not request.partition and rule.get("batch") is False:
            return False
        if rule.get("max_cpus") and request.cpus > int(rule["max_cpus"]):
            return False
        if rule.get("max_ram_gb") and request.ram_gb > int(rule["max_ram_gb"]):
            return False
        return not (
            rule.get("max_gpus") is not None
            and request.gpu_count > int(rule["max_gpus"])
        )

    @staticmethod
    def _choose_partition(nodes: list[NodeInfo]) -> str:
        counts: dict[str, int] = defaultdict(int)
        for node in nodes:
            for partition in node.partitions:
                counts[partition] += 1
        if not counts:
            raise DiscoveryError("Compatible nodes have no partition")
        return min(counts, key=lambda name: (-counts[name], name))
