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


SOURCE_COMMIT = "457723430c2715030032bd1c293e13c9f6cbb05a"
HARBOR_COMMIT = "1b02e70c3ec5778e0ef46a70b66156ba554501be"

GPU_RL_IMAGES = {
    (ImageArchitecture.AMD64, ImageVariant.STANDARD): GpuRlImage(
        architecture=ImageArchitecture.AMD64,
        variant=ImageVariant.STANDARD,
        digest="sha256:c18563800354a547950d840c45f9d4b1f31a08ab9dab26f5a11bf0b9350d9c05",
        tag=f"gpu-rl-{SOURCE_COMMIT}",
        source_commit=SOURCE_COMMIT,
        harbor_commit=HARBOR_COMMIT,
    ),
    (ImageArchitecture.AMD64, ImageVariant.MEGATRON): GpuRlImage(
        architecture=ImageArchitecture.AMD64,
        variant=ImageVariant.MEGATRON,
        digest="sha256:ed419ba2363565d44aacaf5552d058ac998ab189d7d95ca08d93693655697cd1",
        tag=f"gpu-rl-megatron-{SOURCE_COMMIT}",
        source_commit=SOURCE_COMMIT,
        harbor_commit=HARBOR_COMMIT,
    ),
    (ImageArchitecture.ARM64, ImageVariant.STANDARD): GpuRlImage(
        architecture=ImageArchitecture.ARM64,
        variant=ImageVariant.STANDARD,
        digest="sha256:c102c53e4d3b44616669e288d6424782922c1861e7ec2eec84516b4afeb8a3a9",
        tag=f"gpu-rl-{SOURCE_COMMIT}-arm64",
        source_commit=SOURCE_COMMIT,
        harbor_commit=HARBOR_COMMIT,
    ),
    (ImageArchitecture.ARM64, ImageVariant.MEGATRON): GpuRlImage(
        architecture=ImageArchitecture.ARM64,
        variant=ImageVariant.MEGATRON,
        digest="sha256:6d9dd15ed45d7a54468fbf2c3f8488efe88a9eec299bd8372131eab1d4bae4d8",
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
