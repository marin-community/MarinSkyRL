# Debugging log for macOS CPU test setup

Make the frozen `skyrl-train` CPU-test environment resolve on macOS without relying on an ambient Python environment.

## Initial status

`skyrl-train` pins `torch` and `torchvision` to the CUDA 12.8 index through unconditional
`tool.uv.sources` entries. The universal lock therefore contains `torch==2.11.0+cu128`, whose locked wheels do not
include a macOS distribution. `uv run --frozen --extra dev pytest tests/cpu/` fails during installation on macOS.

## Hypothesis 1

Limiting the CUDA 12.8 sources to non-macOS platforms will make uv fall back to PyPI's CPU-only macOS wheels while
preserving the CUDA source for the existing Linux and Windows environments.

## Changes to make

Add macOS-excluding markers to the `torch` and `torchvision` sources, regenerate the universal lock, and verify a
frozen dry-run against an Apple Silicon target.

## Results

Confirmed. Regenerating the lock after adding the source markers produced separate branches:

- macOS resolves `torch==2.11.0` and `torchvision==0.26.0` from PyPI, including their Apple Silicon wheels.
- Non-macOS platforms retain `torch==2.11.0+cu128` and `torchvision==0.26.0+cu128` from the CUDA 12.8 index.

With `MACOSX_DEPLOYMENT_TARGET=14.0`, a frozen Apple Silicon dry-run resolves all 131 packages in the base plus `dev`
environment, including `torchdata`. The frozen Linux environment still imports `torch==2.11.0+cu128` and
`torchdata`.

The 769-test CPU CI selection started successfully in the frozen Linux environment. Its Ray workers eventually filled
`/tmp`, so the run was interrupted without a final suite result. The five tests that reported failures after disk space
reached zero passed when rerun in isolation.

## Future work

None identified.
