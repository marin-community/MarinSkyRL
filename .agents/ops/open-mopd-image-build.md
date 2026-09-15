# Open-MOPD fidelity image build

This experiment image is a thin mutation of the authors' published verl runtime,
`docker.io/verlai/verl@sha256:3ce56ff018516b28ab9c4f4fc09d3aa67589074495ace75e2674b720aa4d0e5d`.
It preserves that image's CUDA, Torch, vLLM, FlashAttention, and FlashInfer binaries. The overlay aligns
Transformers and protobuf with the authors' setup script and adds a version-matched s3fs/fsspec pair for durable Iris
output. It does not build native wheels or install MarinSkyRL's training environment.

## Build

The `Build Open-MOPD image` GitHub Actions workflow submits a disposable, GPU-free amd64 builder to `cw-rno2a`.
The workflow authenticates to a dedicated GHCR experiment package with its short-lived repository token and publishes
`ghcr.io/marin-community/marinskyrl-opd-repro:opd-repro-<full-sha>`. It uses 8 CPUs, 32 GB memory, 100 GB disk, no
preemption, and no task retries. The repository checkout is bundled at `/app` by Iris.

The builder extracts only `/kaniko` from the Kaniko image, writes registry credentials without tracing them, and builds
`docker/Dockerfile.open-mopd`. No Hugging Face token is needed because this overlay reuses the base image's native
wheels. Native MarinSkyRL image builds separately preserve expensive wheelhouses in
`open-athena/marinskyrl-gpu-wheelhouse`.

After a successful build, resolve the tag to an immutable digest. Inspect the image labels and `linux/amd64` platform,
verify anonymous pull access, then run the Open-MOPD runtime inventory on an H100. The inventory must match
`cloud/iris/configs/open_mopd_fidelity.json`, import `s3fs`, and confirm the expected accelerator before the one-step
training gate is submitted.
