# Open-MOPD Marin image build

This experiment image runs the current MarinSkyRL source and frozen root GPU-RL environment. It deliberately
reuses `docker/Dockerfile.gpu-rl`; it does not install the authors' patched `verl`, consume their image, or create
a second dependency lock. The `opd-repro-<full-sha>` tag keeps experiment selection separate without changing
the maintained `gpu-rl-<full-sha>` image contract.

## Build

Work from a clean, committed revision and complete the standard GPU-RL image preflight. The x86_64 wrapper
`docker/build_open_mopd_kaniko.sh` fixes the Dockerfile, FSDP variant, and tag prefix, then delegates to the
maintained kaniko driver. It defaults to a source wheel build; a prebuilt wheelhouse is allowed only under the
standard manifest and digest checks.

The default destination is
`us-east1-docker.pkg.dev/hai-gcp-models/marin/marinskyrl:opd-repro-<full-sha>`. The shared driver derives the
authentication host from the selected repository, so the same path supports GHCR or GAR credentials. It does
not change IAM. A token must remain valid through the final push; a one-hour `gcloud auth print-access-token`
is unsafe for an uncached native wheel build. Use an already-authorized refreshable workload identity or a
suitably scoped build credential, or stop before submission.

Submit only with explicit build authorization. Use a disposable, GPU-free amd64 Iris task on `cw-rno2a` with
`docker.io/library/ubuntu:22.04`, `--no-sync`, `--enable-extra-resources`, `--no-preemptible`, `--max-retries 0`,
and the standard amd64 build resources. Iris still bundles the committed workspace at `/app`. Pass:

```text
GITSHA=<full committed MarinSkyRL revision>
REGISTRY_USER=<registry username; oauth2accesstoken for a GAR access token>
REGISTRY_TOKEN=<runtime-only registry credential>
```

Do not print or persist the credential. After the build, resolve the tag to an immutable digest and apply every
inspection, runtime, import, and H100 smoke gate in `.agents/ops/gpu-rl-image-build.md`. Record that digest in
the native Marin OPD experiment plan. This image is not evidence that the authors' patched-verl fidelity control
can run; that remains a separate comparison track.
