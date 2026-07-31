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


SOURCE_COMMIT = "dec8f94176cacf9f891aab720488782260ae3c28"
HARBOR_COMMIT = "f2dd505f609021611eb85521a097e6dff384d12d"

GPU_RL_IMAGES = {
    (ImageArchitecture.AMD64, ImageVariant.STANDARD): GpuRlImage(
        architecture=ImageArchitecture.AMD64,
        variant=ImageVariant.STANDARD,
        digest="sha256:39fcb3aff359d423fc7ac822173ffb6193d818357f5970cdf7ded889a14c295b",
        tag=f"gpu-rl-{SOURCE_COMMIT}",
        source_commit=SOURCE_COMMIT,
        harbor_commit=HARBOR_COMMIT,
    ),
    (ImageArchitecture.AMD64, ImageVariant.MEGATRON): GpuRlImage(
        architecture=ImageArchitecture.AMD64,
        variant=ImageVariant.MEGATRON,
        digest="sha256:d473ac28c59fd101ba32200b6ac056a6353a326597c6e34130081d1473bd88cc",
        tag=f"gpu-rl-megatron-{SOURCE_COMMIT}",
        source_commit=SOURCE_COMMIT,
        harbor_commit=HARBOR_COMMIT,
    ),
    (ImageArchitecture.ARM64, ImageVariant.STANDARD): GpuRlImage(
        architecture=ImageArchitecture.ARM64,
        variant=ImageVariant.STANDARD,
        digest="sha256:fa879139e5f3ed523891d06533277a3da5d03b87b374cf167d852ae2e836a215",
        tag=f"gpu-rl-{SOURCE_COMMIT}-arm64",
        source_commit=SOURCE_COMMIT,
        harbor_commit=HARBOR_COMMIT,
    ),
    (ImageArchitecture.ARM64, ImageVariant.MEGATRON): GpuRlImage(
        architecture=ImageArchitecture.ARM64,
        variant=ImageVariant.MEGATRON,
        digest="sha256:8c16dfd5266cb894dd48d11ab6a1ab72a804390c0155eccd45a054fff5c5c57c",
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
