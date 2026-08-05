# Later distributed diagnostic source record

This directory preserves the exact local source material used for the
route-aware pair and focused causal probe reported on Marin issue #7903.
It is an audit record, not a request to rerun either diagnostic.

The directory-local `.gitattributes` rule disables outer Git whitespace
diagnostics for the archived patch files. Unified-diff context lines require a
single leading space; those syntax bytes must remain unchanged for the recorded
patch hashes and exact reconstruction below.

Both `*_driver_worker.patch` files apply to public evidence commit
`7c3bac451a69d34fa8b8f027ceb91998a6e0ff2c`. At that revision the base blobs
are:

- driver `6e6d6dfac25f785bb32b71f5d56d60157938bad2` at
  `skyrl-train/scripts/grug_fixed_replay_benchmark.py`;
- worker `880aa28ee09b5ee2a97e0999d3bb68b6b5664955` at
  `skyrl-train/skyrl_train/workers/fsdp/fsdp_worker.py`.

Applying `route_driver_worker.patch` reconstructs the writer's two pinned
files exactly:

- driver SHA-256
  `bbdc711b3d26b5127a71b3b8e24f7f3dfb5e00ba94e1a51f28f9bf83111dd084`;
- worker SHA-256
  `d66c1c3ee148a8aef0007d1d3e17af4ef522381c0107f836d3c0817805fc0de4`.

Applying `causal_driver_worker.patch` reconstructs the focused probe's two
pinned files exactly:

- driver SHA-256
  `4f0d7a468558849a5567d96e1c05d49fcc67d55df085ce82773265cab6481373`;
- worker SHA-256
  `cced801b807d5d75ca15b7fc1e81c83121e83ecddba2cbe0807b663fb1cce0eb`.

The evidence-only candidate model source is already preserved by the exact
mail patch in `7c3bac451a69d34fa8b8f027ceb91998a6e0ff2c`; both writer launchers pin its
copied model file to SHA-256
`2dd51d842afe6f57fe00337c21945923da68664d9c23678633c578c27504ba93`.

## Preserved file manifest

| File | SHA-256 |
| --- | --- |
| `route_driver_worker.patch` | `d74c876e86cb321edd2c9bd4382d235683832800d7b172dcea4263c1d74b7790` |
| `launch_route_discriminator_headline_fbb1fc8.sh` | `1f817bf0903dea26fc74eca5c7c32e90af7a054324f662191c498ce48d6668ca` |
| `run_route_discriminator_fbb1fc8.sh` | `39db06d8462c805e6bb3c4f658354c602eccd01727ce16b2747eaf2b263a1e5b` |
| `launch_route_discriminator_headline_readback.sh` | `a2cc752476433958d6db71c2d134d9a04d64bec919557b985c5cf740b0f2d7c6` |
| `verify_route_discriminator_headline.py` | `ea32f61a06c8e75d93cacf2c93a057e9452eb9c5d181130faff8c596ddfeea6b` |
| `causal_driver_worker.patch` | `344b7e6677b6f211aa5196591dece6f56fba9cb39ad27e431e86b563cdcf43e5` |
| `launch_causal_probe_fbb1fc8.sh` | `0a623dda1d89113d85721a4c1d6d27e93e6dfc7e4322212c7f7f3c333b63f6f2` |
| `run_causal_probe_fbb1fc8.sh` | `284b9b75011b3e9259c2f94a3d1757b82e8d5aaf54326a9b70704daeb5566ae6` |
| `launch_causal_probe_readback.sh` | `0048c8858064bdc06d609459ae7d4a098e300f003c1deee61ce1318cee2e9cca` |
| `verify_causal_probe.py` | `dd7b7c0544714ab4bb3dbc5e0c839a7f20bf46cd56d678a73815b28275a32683` |

The patches were independently applied to an archive of `7c3bac4` during the
no-new-runs closeout. The reconstructed file hashes matched the execution pins
above. No experiment, test, benchmark, GPU job, or distributed workload was
run during this recovery.

The launchers contain the original absolute worktree paths and are preserved
verbatim. They are evidence of the submitted bundle construction and pins, not
portable launch instructions.
