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


AMD64_SOURCE_COMMIT = "814abea09aea8c52e19006fd47ac10951e8b9308"
ARM64_SOURCE_COMMIT = "fa640da3dc653be902395ecd15440f4fcdd80c2f"
HARBOR_COMMIT = "1ffb4003f202daadcb7e407f449bd62278b5e8e1"

GPU_RL_IMAGES = {
    (ImageArchitecture.AMD64, ImageVariant.STANDARD): GpuRlImage(
        architecture=ImageArchitecture.AMD64,
        variant=ImageVariant.STANDARD,
        digest="sha256:ba3ddd58f8c6a3fc77b91fb4eef016115f5e7aa1d4c57f06ab693c61be1f3426",
        tag=f"gpu-rl-{AMD64_SOURCE_COMMIT}",
        source_commit=AMD64_SOURCE_COMMIT,
        harbor_commit=HARBOR_COMMIT,
    ),
    (ImageArchitecture.AMD64, ImageVariant.MEGATRON): GpuRlImage(
        architecture=ImageArchitecture.AMD64,
        variant=ImageVariant.MEGATRON,
        digest="sha256:084e368ead8e12665f541ce89701560ee1357584a1d93861c27997a25659fedb",
        tag=f"gpu-rl-megatron-{AMD64_SOURCE_COMMIT}",
        source_commit=AMD64_SOURCE_COMMIT,
        harbor_commit=HARBOR_COMMIT,
    ),
    (ImageArchitecture.ARM64, ImageVariant.STANDARD): GpuRlImage(
        architecture=ImageArchitecture.ARM64,
        variant=ImageVariant.STANDARD,
        digest="sha256:0dd75103cf56bc4735d1c0155c8591822c10a6825e8137c6943400cbef628dd3",
        tag=f"gpu-rl-{ARM64_SOURCE_COMMIT}-arm64",
        source_commit=ARM64_SOURCE_COMMIT,
        harbor_commit=HARBOR_COMMIT,
    ),
    (ImageArchitecture.ARM64, ImageVariant.MEGATRON): GpuRlImage(
        architecture=ImageArchitecture.ARM64,
        variant=ImageVariant.MEGATRON,
        digest="sha256:9b50a18d4729a22bb70277bfadcf6f3f7017187b33af69e360a73c1eb1b53638",
        tag=f"gpu-rl-megatron-{ARM64_SOURCE_COMMIT}-arm64",
        source_commit=ARM64_SOURCE_COMMIT,
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
