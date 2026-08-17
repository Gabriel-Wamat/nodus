from cluster_model_runner.client import ClusterClient
from cluster_model_runner.discovery import SlurmDiscovery, _gpu_from_gres
from cluster_model_runner.transport import CommandResult


class FakeTransport:
    def run(self, argv, timeout=0):
        return CommandResult(
            0,
            "node1|gpu-batch|idle|gpu:gpu-large:1|gpu-large|512000|48\n"
            "node2|short|mix|gpu:2(S:0-1)|(null)|128000|32",
            "",
        )

    def shell(self, script, data=None, timeout=0):
        return CommandResult(0, "Python/3.10.8\nPython/3.12.2", "")


def test_gres_parser_supports_typed_and_untyped_gpu():
    assert _gpu_from_gres("gpu:gpu-large:1", ()) == ("gpu-large", 1, 0)
    assert _gpu_from_gres("gpu:gpu-large-80gb:1", ()) == ("gpu-large-80gb", 1, 80)
    assert _gpu_from_gres("gpu:2(S:0-1)", ()) == ("", 2, 0)


def test_discovery_parses_nodes_and_sorts_python_numerically():
    discovery = SlurmDiscovery(FakeTransport())
    nodes = discovery.nodes()
    assert nodes[0].gpu_type == "gpu-large"
    assert nodes[1].gpu_count == 2
    assert discovery.python_modules()[-1] == "Python/3.12.2"


def test_discovery_merges_partitions_for_the_same_node():
    class MultiPartitionTransport(FakeTransport):
        def run(self, argv, timeout=0):
            return CommandResult(
                0,
                "gpu-a|batch-a|idle|gpu:1|(null)|128000|32\n"
                "gpu-a|batch-b|idle|gpu:1|(null)|128000|32",
                "",
            )

    nodes = SlurmDiscovery(MultiPartitionTransport()).nodes()
    assert len(nodes) == 1
    assert nodes[0].partitions == ("batch-a", "batch-b")


class CountingDiscovery:
    def __init__(self):
        self.calls = 0

    def snapshot(self):
        self.calls += 1
        return {"nodes": [], "python_modules": [], "scontrol_nodes": {}}


def test_client_discovery_uses_ttl_and_explicit_refresh():
    client = object.__new__(ClusterClient)
    client.config = type("Config", (), {"discovery_ttl": 60})()
    client.discovery = CountingDiscovery()
    client._cluster_snapshot = None
    client._cluster_snapshot_at = 0.0

    assert client.discover() is client.discover()
    assert client.discovery.calls == 1
    client.refresh_cluster()
    assert client.discovery.calls == 2
