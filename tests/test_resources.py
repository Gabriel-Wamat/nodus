import pytest

from cluster_model_runner.exceptions import DiscoveryError
from cluster_model_runner.models import NodeInfo, ResourceRequest
from cluster_model_runner.resources import ResourceSelector


def node(name, gpu, vram, state="idle", gres="gpu:1"):
    return NodeInfo(name, ("gpu-batch",), state, gres, (), 512000, 48, gpu, 1, vram)


def test_smallest_compatible_prefers_3090():
    selected = ResourceSelector().resolve(
        ResourceRequest(min_vram_gb=20),
        [node("a", "gpu-large", 80), node("b", "gpu-small", 24)],
        {"gpu-batch": {"qos": "standard"}},
    )
    assert selected.gpu_type == "gpu-small"
    assert selected.eligible_nodes == ("b",)
    assert selected.excluded_nodes == ("a",)
    assert selected.qos == "standard"


def test_large_model_restricts_to_large_gpu():
    selected = ResourceSelector().resolve(
        ResourceRequest(min_vram_gb=40),
        [node("a", "gpu-large", 80), node("b", "gpu-small", 24)],
        {"gpu-batch": {"qos": "standard"}},
    )
    assert selected.gpu_type == "gpu-large"
    assert selected.eligible_nodes == ("a",)
    assert selected.excluded_nodes == ("b",)


def test_safe_policy_prefers_largest_known_gpu():
    selected = ResourceSelector().resolve(
        ResourceRequest(policy="safe"),
        [node("large", "gpu-large", 80), node("small", "gpu-small", 24)],
    )
    assert selected.gpu_type == "gpu-large"
    assert selected.eligible_nodes == ("large",)


def test_smallest_policy_rejects_unknown_vram():
    with pytest.raises(DiscoveryError, match="No discovered node"):
        ResourceSelector().resolve(
            ResourceRequest(policy="smallest-compatible"),
            [node("unknown", "", 0)],
        )


def test_exact_policy_requires_gpu_type():
    with pytest.raises(DiscoveryError, match="requires gpu_type"):
        ResourceSelector().resolve(ResourceRequest(policy="exact"), [node("a", "gpu", 24)])


def test_selector_uses_any_compatible_partition_on_a_node():
    candidate = NodeInfo(
        "gpu-a", ("maintenance", "batch"), "idle", "gpu:1", (), 128000, 32, "gpu", 1, 24
    )
    selected = ResourceSelector().resolve(
        ResourceRequest(),
        [candidate],
        {"maintenance": {"batch": False}, "batch": {"max_gpus": 1}},
    )
    assert selected.partition == "batch"
