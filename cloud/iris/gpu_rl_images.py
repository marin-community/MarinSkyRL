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

    @property
    def platform(self) -> str:
        return f"linux/{self.architecture}"


SOURCE_COMMIT = "2f281a199b700fb61d83378b209baffa46c6d71c"
HARBOR_COMMIT = "f2dd505f609021611eb85521a097e6dff384d12d"

GPU_RL_IMAGES = {
    (ImageArchitecture.AMD64, ImageVariant.STANDARD): GpuRlImage(
        architecture=ImageArchitecture.AMD64,
        variant=ImageVariant.STANDARD,
        digest="sha256:b0220be3badbc408e902b2800ce6718208935904b35b42daa1db4702ac9e9bae",
        tag=f"gpu-rl-{SOURCE_COMMIT}",
        source_commit=SOURCE_COMMIT,
        harbor_commit=HARBOR_COMMIT,
    ),
    (ImageArchitecture.AMD64, ImageVariant.MEGATRON): GpuRlImage(
        architecture=ImageArchitecture.AMD64,
        variant=ImageVariant.MEGATRON,
        digest="sha256:5f1291fb453d7439d96375eb103f535c815674ebf40a7c40324c2fcde589af4a",
        tag=f"gpu-rl-megatron-{SOURCE_COMMIT}",
        source_commit=SOURCE_COMMIT,
        harbor_commit=HARBOR_COMMIT,
    ),
    (ImageArchitecture.ARM64, ImageVariant.STANDARD): GpuRlImage(
        architecture=ImageArchitecture.ARM64,
        variant=ImageVariant.STANDARD,
        digest="sha256:36a0d327aed94d26dba411864332cc88d29bc593d769b6544719000f8174d412",
        tag=f"gpu-rl-{SOURCE_COMMIT}-arm64",
        source_commit=SOURCE_COMMIT,
        harbor_commit=HARBOR_COMMIT,
    ),
    (ImageArchitecture.ARM64, ImageVariant.MEGATRON): GpuRlImage(
        architecture=ImageArchitecture.ARM64,
        variant=ImageVariant.MEGATRON,
        digest="sha256:50a5c9584be412a0bb5657ea9c0be28d2689c5a587351ed815ef0f73fb30aac4",
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
