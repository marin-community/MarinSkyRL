# CatCount CPU starting policy

The CPU canary downloads the calibrated 529,792-parameter FP32 policy from
`s3://marin-us-east-02a/marin/rl-canaries/cat-count/cpu/pretrained/llama-530k/v1/`.
Its manifest SHA256 is `7a5f1047648a90262514a663168580610b9f6bbe1522b2c38b7578bbc5eb82ae`;
each downloaded file must match the manifest's byte count and SHA256.
Credentials or network failures select cached local pretraining. Actions caches
key every producer source, pretrain parameter and dependency-lock input.
CI read access is provided separately through a read-only role.

PR CI runs ten normal training steps on seed0 with the positive assertions.
Nightly checks normal/reversed pairs on seeds0/1 and asynchronous resume,
staleness and optimizer updates. At some resumed evaluation the mean train and
held-out reward must be at least0.65 and at least0.3 above its initial value.
The resume run retains100steps; a recorded devbox run first crossed at44.

Regenerate from the repository root in the frozen CPU environment:

```bash
uv sync --frozen --group dev --group harbor-test --extra cpu --extra telemetry
uv run --frozen python skyrl-train/examples/cat_count/cpu_canary.py \
  --out /tmp/cat-count-policy --steps 3000 --lr 3e-4 \
  --width 128 --layers 2 --seed 0
sha256sum /tmp/cat-count-policy/model.safetensors
```

Pretraining uses one CPU thread, counts other words and teaches `cat` only at
N=1. The retained `pretrain()` implementation produces the fixture. Byte-level
reproduction depends on the frozen dependencies and CPU numerical operations.
Review regenerated weights and rerun both CI selections before replacing them.

Expected SHA-256 hashes:

| File | SHA-256 |
| --- | --- |
| `model.safetensors` | `5f1774fdaf2faa6f7b53cb932df09ffc31ff6cd7a7358eabac0939e67a51474d` |
| `config.json` | `ce6bb019786d3256ad09c468c7338a20f446b5aee0d1123cc1d272a2145b1014` |
| `tokenizer.json` | `cd269f2e791d29a5690b3e9c3b1dbaf289eb05078bb9ee1e300f870672e94b53` |
| `tokenizer_config.json` | `46853c9c870f60e630b8b687dd6eaaeb8bc26d8645c1535047d9b7d93890c48c` |
| `generation_config.json` | `2a9736c3d3a24836d662195c61e61b2a216c14c07c527b6f562f7437a6b83184` |
| `chat_template.jinja` | `64df6474eeca98c150c72e02a30cb15620a1abf774238decb961176d735eead4` |

Producer SHA-256: `70caebfde03f1a17dd4653c8a3972a5b86bad7fe134be9c15f99e994db970fd2`.
