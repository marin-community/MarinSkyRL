# Debugging log for cluster-aware GPU-RL images

Make GPU-RL image selection follow the execution cluster, centralize immutable image metadata, and
make baked Harbor provenance inspectable from the OCI configuration.

## Initial status

The launcher stores one standard and one Megatron digest, both linux/amd64. Image selection considers
only `trainer.strategy`, so an east-08a launch must override `--task-image` to avoid scheduling an
amd64 image on an arm64 host. The Dockerfiles assert the Harbor source revision during the build but
do not preserve it in OCI metadata, leaving retained build logs as the only artifact-level evidence.

## Hypothesis 1

Selecting an image from the effective execution cluster (`target_cluster` for federation, otherwise
`cluster`) and strategy removes the manual arm64 override without weakening immutable digest pins.

## Changes to make

Add a regression test for east-08a standard and Megatron defaults, then move the deployed image
registry and selection logic into a dependency-free helper used by the launcher.

## Results

The regression failed for both strategies: east-08a received the registered amd64 digest. A
cluster-and-strategy registry now supplies all four deployed images, and the launcher resolves it
from `target_cluster` when federating or `cluster` for direct submission. Explicit image overrides
remain unchanged.

## Hypothesis 2

OCI labels populated from the Dockerfiles' existing `GITSHA` and `HARBOR_COMMIT` build arguments
make the built artifact self-describing without adding a second source of build inputs.

## Changes to make

Label both runtime images with the MarinSkyRL and Harbor revisions, add a static build-contract test,
and make image validation compare those labels with the code registry.

## Results

Both runtime Dockerfiles now expose the source and Harbor revisions as OCI labels, and a parametrized
contract test covers both architectures. The build procedure validates those labels directly from the
published image configuration and uses build logs only for build assertions and digest correlation.
Existing image digests are unchanged, so the labels become available when the next images are built.

## Future work

- [x] Record the baked Harbor revision as an OCI label in both runtime Dockerfiles.
- [x] Remove image inventory from mutable operations prose and clarify immutable tag syntax.
