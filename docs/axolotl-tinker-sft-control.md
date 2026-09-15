# Native Axolotl control for Tinker SFT

This harness runs a native eight-H100 approximation of the public Tinker
OpenThoughts3 SFT recipe. It exists because Levanter does not yet implement the
Qwen3.5 hybrid Gated DeltaNet architecture. It is a cross-runtime control, not
an exact reproduction of Tinker's hosted optimizer implementation.

The checked-in Axolotl configuration pins Axolotl `0.19.0`,
`Qwen/Qwen3.5-9B-Base` at commit
`68c46c4b3498877f3ef123c856ecfde50c39f404`, and
`open-thoughts/OpenThoughts3-1.2M` at commit
`61bcf9d4eb38b30295efc2021227a63cc5bb34c8`. The full stage uses rank-128
LoRA, a global batch of 128, 16,384-token sequences, 3,000 optimizer steps,
linear decay from `1e-3` with no warmup, and Adam betas `0.9/0.95` with epsilon
`1e-8`. Only assistant turns and their turn-ending token receive loss.

The worker reproduces the Cookbook's seed-0 streaming shuffle with a
384,000-row buffer, materializes the resulting 384,000 examples, and tells
Axolotl to consume them sequentially. This is closer than applying Axolotl's
ordinary whole-dataset shuffle, but distributed worker partitioning and
tokenization still differ. The manifest explicitly records this and the more
important LoRA mismatch: Tinker has separate low-rank Q/K/V parameters for the
fused Gated DeltaNet input projection, while PEFT wraps that fused projection
once. Tinker's service-side initialization and scaling are not public. The
Axolotl config uses `lora_alpha: 1`, matching Tinker's exported PEFT convention,
and includes the language attention, Gated DeltaNet, MLP, and LM-head paths.

## Staged launch

The task image is built from `docker/Dockerfile.axolotl-tinker-sft`. It pins the
amd64 manifest of Axolotl's `0.19.0`, Python 3.12, CUDA 13.0, and PyTorch 2.11
image, then adds this checkout's Iris worker under a revision label. Build it
with `docker/build_axolotl_tinker_sft_kaniko.sh` from a clean committed checkout,
resolve the resulting tag to a digest, and use only that digest in the launch.
The task revalidates Axolotl and the eight visible GPUs before downloading the
pinned model or dataset.

Every invocation is a dry run unless `--submit` is present:

```bash
uv run --frozen python -m cloud.iris.axolotl_tinker_sft \
  --stage plumbing \
  --run-id repro-20260915 \
  --cluster-config "$IRIS_CLUSTER_CONFIG" \
  --output-uri s3://marin-us-east-02a/experiments/axolotl-tinker-sft/repro-20260915/plumbing \
  --task-image '<registry>/axolotl@sha256:<digest>'
```

The stages are deliberately closed:

| Stage | Shape | Authorization gate |
| --- | --- | --- |
| `plumbing` | 1 step, 8 examples, 2,048 tokens | None |
| `fidelity_step` | 1 exact-shape step and the full shuffle buffer | `--acknowledge-cost-usd 150` |
| `full` | 3,000 exact-shape steps | `--acknowledge-cost-usd 10000` |

The dollar amount is an authorization record, not a billing limit. After
reviewing a dry-run plan, repeat it with the stage's exact acknowledgement plus
`--submit --allow-known-deviations`. Run each stage into a new, empty durable
prefix. Iris retries and preemption are disabled so a failed distributed
optimizer cannot silently restart against partial local state.

The worker uploads `native-sft-reproduction-manifest.json`, the fully resolved
Axolotl YAML, logs and checkpoints. Successful completion additionally requires
a rank-128, alpha-1 PEFT `adapter_config.json` and adapter safetensors; their
sizes and SHA-256 digests are added to the final manifest.

## Local validation

This checks the secret-free plan, stage bounds, config lowering, submission
gates, and final PEFT artifact contract without loading model weights:

```bash
uv run --frozen --extra cpu --group dev --group harbor-test pytest -q \
  cloud/iris/tests/test_axolotl_tinker_sft.py
```
