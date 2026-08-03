"""CoreWeave cluster connection settings shared by Iris operator scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


COREWEAVE_KUBECONFIG = Path.home() / ".kube" / "coreweave-iris"
COREWEAVE_OBJECT_ENDPOINT = "https://cwobject.com"


@dataclass(frozen=True)
class ClusterConfig:
    kubeconfig: Path
    context: str
    object_endpoint: str


CLUSTERS = {
    "cw-rno2a": ClusterConfig(
        kubeconfig=COREWEAVE_KUBECONFIG,
        context="marin-rn02a_RNO2A",
        object_endpoint=COREWEAVE_OBJECT_ENDPOINT,
    ),
    "cw-us-east-02a": ClusterConfig(
        kubeconfig=COREWEAVE_KUBECONFIG,
        context="marin-gpu_US-EAST-02A",
        object_endpoint=COREWEAVE_OBJECT_ENDPOINT,
    ),
    "cw-us-east-08a": ClusterConfig(
        kubeconfig=COREWEAVE_KUBECONFIG,
        context="marin-us-east-08a_US-EAST-08A",
        object_endpoint=COREWEAVE_OBJECT_ENDPOINT,
    ),
}
