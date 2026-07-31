"""Immutable GPU-RL image registry and cluster-aware selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


IMAGE_REPOSITORY = "ghcr.io/marin-community/marinskyrl"
MEGATRON_STRATEGY = "megatron"


class ImageArchitecture(StrEnum):
    AMD64 = "amd64"
    ARM64 = "arm64"


class ImageVariant(StrEnum):
    STANDARD = "standard"
    MEGATRON = "megatron"


@dataclass(frozen=True)
class GpuRlImage:
    architecture: ImageArchitecture
    variant: ImageVariant
    digest: str
    tag: str
    source_commit: str
    harbor_commit: str

    @property
    def reference(self) -> str:
        return f"{IMAGE_REPOSITORY}@{self.digest}"


SOURCE_COMMIT = "fa640da3dc653be902395ecd15440f4fcdd80c2f"
HARBOR_COMMIT = "1ffb4003f202daadcb7e407f449bd62278b5e8e1"

GPU_RL_IMAGES = {
    (ImageArchitecture.AMD64, ImageVariant.STANDARD): GpuRlImage(
        architecture=ImageArchitecture.AMD64,
        variant=ImageVariant.STANDARD,
        digest="sha256:5e6e160e648c2ec6cd62d08aa1d06e1c0f5e02c31e90adad83cd2609898597ec",
        tag=f"gpu-rl-{SOURCE_COMMIT}",
        source_commit=SOURCE_COMMIT,
        harbor_commit=HARBOR_COMMIT,
    ),
    (ImageArchitecture.AMD64, ImageVariant.MEGATRON): GpuRlImage(
        architecture=ImageArchitecture.AMD64,
        variant=ImageVariant.MEGATRON,
        digest="sha256:c41f7d589043dc422ad4d2d3962a3d48fa68bfe8f37344384594b53223b8b7e7",
        tag=f"gpu-rl-megatron-{SOURCE_COMMIT}",
        source_commit=SOURCE_COMMIT,
        harbor_commit=HARBOR_COMMIT,
    ),
    (ImageArchitecture.ARM64, ImageVariant.STANDARD): GpuRlImage(
        architecture=ImageArchitecture.ARM64,
        variant=ImageVariant.STANDARD,
        digest="sha256:0dd75103cf56bc4735d1c0155c8591822c10a6825e8137c6943400cbef628dd3",
        tag=f"gpu-rl-{SOURCE_COMMIT}-arm64",
        source_commit=SOURCE_COMMIT,
        harbor_commit=HARBOR_COMMIT,
    ),
    (ImageArchitecture.ARM64, ImageVariant.MEGATRON): GpuRlImage(
        architecture=ImageArchitecture.ARM64,
        variant=ImageVariant.MEGATRON,
        digest="sha256:9b50a18d4729a22bb70277bfadcf6f3f7017187b33af69e360a73c1eb1b53638",
        tag=f"gpu-rl-megatron-{SOURCE_COMMIT}-arm64",
        source_commit=SOURCE_COMMIT,
        harbor_commit=HARBOR_COMMIT,
    ),
}

CLUSTER_ARCHITECTURES = {
    "cw-rno2a": ImageArchitecture.AMD64,
    "cw-us-east-02a": ImageArchitecture.AMD64,
    "cw-us-east-08a": ImageArchitecture.ARM64,
}


def image_for_cluster(cluster: str, strategy: str | None) -> GpuRlImage:
    """Return the immutable image matching an execution cluster and trainer strategy."""
    try:
        architecture = CLUSTER_ARCHITECTURES[cluster]
    except KeyError as error:
        raise ValueError(f"No GPU-RL image architecture is registered for cluster {cluster!r}") from error
    variant = ImageVariant.MEGATRON if strategy == MEGATRON_STRATEGY else ImageVariant.STANDARD
    return GPU_RL_IMAGES[(architecture, variant)]
