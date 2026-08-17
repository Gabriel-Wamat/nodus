import json

from cluster_model_runner.bootstrap import (
    ClusterBootstrapper,
    ProbePolicy,
    cluster_fingerprint,
    parse_nvidia_smi_csv,
)
from cluster_model_runner.models import JobState, JobStatus, NodeInfo
from cluster_model_runner.transport import CommandResult


def gpu_node(name):
    return NodeInfo(name, ("gpu",), "idle", "gpu:1", (), 128000, 32, "", 1, 0)


class FakeDiscovery:
    def __init__(self):
        self.partition_rules = {}
        self.applied = None
        self.state = "idle"

    def nodes(self):
        return [
            NodeInfo(
                node.name,
                node.partitions,
                self.state,
                node.gres,
                node.features,
                node.memory_mb,
                node.cpus,
                node.gpu_type,
                node.gpu_count,
                node.vram_gb,
            )
            for node in (gpu_node("gpu-a"), gpu_node("gpu-b"))
        ]

    def apply_inventory(self, payload):
        self.applied = payload


class FakeTransport:
    def __init__(self):
        self.uploads = {}

    def run(self, argv, timeout=0):
        if argv[:3] == ["scontrol", "show", "config"]:
            return CommandResult(0, "ClusterName = research\n", "")
        if argv[:1] == ["cat"]:
            return CommandResult(0, "0, Example GPU, 24576, 550.1, 8.6\n", "")
        return CommandResult(0, "", "")

    def upload_bytes(self, content, remote):
        self.uploads[remote] = content


class FakeScheduler:
    def __init__(self):
        self.submissions = []

    def submit(self, script_path):
        self.submissions.append(script_path)
        return str(len(self.submissions))

    def status_info(self, scheduler_id):
        return JobStatus(JobState.SUCCEEDED, slurm_id=scheduler_id)

    def cancel(self, scheduler_id):
        raise AssertionError("completed probes must not be cancelled")


def test_gpu_probe_parser_separates_physical_and_safe_vram():
    parsed = parse_nvidia_smi_csv("0, Example GPU, 24576, 550.1, 8.6", reserve_gb=2)
    assert parsed[0]["total_vram_mb"] == 24576
    assert parsed[0]["safe_vram_mb"] == 22528
    assert parsed[0]["canonical_name"] == "examplegpu"


def test_fingerprint_is_independent_of_node_order():
    first = [gpu_node("gpu-a"), gpu_node("gpu-b")]
    assert cluster_fingerprint("research", first) == cluster_fingerprint(
        "research", list(reversed(first))
    )


def test_representative_probe_is_persisted_and_reused(tmp_path):
    transport = FakeTransport()
    scheduler = FakeScheduler()
    discovery = FakeDiscovery()
    bootstrapper = ClusterBootstrapper(
        transport=transport,
        scheduler=scheduler,
        discovery=discovery,
        state_dir=tmp_path,
        remote_root="/remote",
        poll_interval=1,
        ttl_seconds=60,
    )

    inventory = bootstrapper.bootstrap(
        policy=ProbePolicy(mode="representative"), require_vram=True
    )
    assert len(scheduler.submissions) == 1
    assert inventory["nodes"]["gpu-a"]["vram_gb"] == 22
    assert inventory["nodes"]["gpu-b"]["vram_gb"] == 22
    assert inventory["nodes"]["gpu-b"]["probe_status"] == "inferred-from-representative"
    assert discovery.applied == inventory

    discovery.state = "down"
    cached = bootstrapper.bootstrap(policy="representative", require_vram=True)
    assert cached["cluster"]["fingerprint"] == inventory["cluster"]["fingerprint"]
    assert cached["nodes"]["gpu-a"]["schedulable"] is False
    assert cached["nodes"]["gpu-a"]["probe_status"] == "stale"
    assert len(scheduler.submissions) == 1
    stored = next((tmp_path / "clusters").glob("*/inventory.json"))
    assert json.loads(stored.read_text()) == inventory
