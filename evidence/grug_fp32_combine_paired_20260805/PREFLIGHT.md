# Grug FP32 combine paired-measurement preflight

The experimental rules were frozen before any numerical result. The exact
parent-to-candidate source transformation, Python compilation, project-pinned
Ruff, JSON parsing, and whitespace checks passed locally. The exact candidate
CPU regression
`test_bfloat16_expert_combine_uses_float32_accumulation` passed in 6.13 s.

Iris granted one interactive `h100-8x` node in `cw-rno2a`: eight NVIDIA H100
80GB HBM3 devices, 32 requested CPU cores, a 128 GiB memory limit, and 200 GiB
ephemeral storage. Before numerical work the pod had about 1.9 TiB host memory
available, no host swap, and about 15 GiB cgroup use after installing the
pinned Python environment. The local host had about 8.2 GiB memory available
and 6.0 GiB of its 8.0 GiB swap free. No measurement worker ran locally.

The first remote correctness-worker launch stopped before importing the Grug
module or constructing a tensor: the minimal environment lacked `ray`, which
the SkyRL package initializer imports. No numerical output or timing existed.
The common runtime was completed with the project versions `ray==2.51.1`,
`omegaconf==2.3.0`, and `antlr4-python3-runtime==4.9.3`; a direct import of the
pinned parent module then passed. A second worker reached the grouped path but
stopped at its lazy `torchtitan` import. It again wrote no result JSON and
performance did not start. The runtime was completed with the project's exact
Torchtitan pin
`a1fdd7e43694bbfeff5d6ad8ac738c067bb90d41`; a direct import of the grouped
kernel then passed. Only the frozen runtime inventory digest and freeze
timestamp changed after these import-only failures. The fixtures, gates,
tolerances, schedule, verdict, driver, and reader did not change.

The live command must use the hashes and runtime identity in
`FROZEN_PROTOCOL.json`. Correctness runs parent then candidate on GPU 0. The
four finite performance waves can start only after every correctness gate
passes. The driver aborts before 100 GiB cgroup use, below 256 GiB host memory
available, or below 50% free host swap when swap exists.
