# Open-MOPD released-checkpoint evaluation

This harness evaluates the released Open-MOPD final checkpoint with the authors' pinned source, model revision,
published evaluation Parquets, and documented sampling protocol. It defaults to a local dry-run that prints the entire
Iris command. Submission requires both `--submit` and `--allow-known-omissions`.

The versioned inputs are:

- Open-MOPD source commit `4809a96cf85a869106ff0ff3f37d0a51e12010ae`;
- formal evaluation specification commit `4460e57ad87fef996a0c21f96bde9a7d1ba029b6`;
- final model revision `228a146a5d95f00136057347ac4810e6635061b6`, including per-file LFS SHA-256 values;
- Open-MOPD-Data revision `9e897efe3257599d4300e2d5ee865a1cc714af87`, with exact size and SHA-256 for each
  evaluation Parquet;
- a caller-supplied digest-addressed task image.

The task verifies all model and data files before starting vLLM. It records the resolved commands, runtime package
inventory, expected and observed artifact digests, benchmark coverage, and outcome in `evaluation-manifest.json` under
the durable output prefix. Outputs are synchronized while the task runs and once more during teardown.

## Coverage

| Benchmark | Full gate | Released scorer available at the pinned commit |
| --- | ---: | --- |
| AIME24 | 30 prompts × 64 samples | Yes |
| AIME25 | 30 prompts × 64 samples | Yes |
| LiveCodeBench v5 | 167 prompts × 10 samples | No; rollout only |
| LiveCodeBench v6 | 175 prompts × 10 samples | No; rollout only |
| IFEval | 541 prompts × 1 sample | Yes |
| IFBench_test | 300 prompts × 1 sample | No; rollout only |

The pinned source vendors its AIME and Google IFEval scorers. The official LiveCodeBench checkout is identifiable at
`28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`, and IFBench at
`1091c4c3de6c1f6ed12c012ed68f11ea450b0117`. However, the release does not include the `repos.lock.json`, third-party
checkouts, or LiveCodeBench patch/assets referenced by its own evaluation README. The LiveCodeBench asset revision and
hashes remain unknown, and the IFBench setup is not reproducible from the pinned tree alone. Consequently, the harness
preserves those rollouts but does not silently replace their official scorers or report invented aggregate scores.

The full gate uses the immutable formal specification: bfloat16, 32,768-token model length, 1,024 maximum sequences,
top-p 0.95, top-k disabled, stop token 128012, and thinking enabled for every domain. Math uses a 31,000-token cap,
temperature 0.6, and 64 samples; code uses a 30,000-token cap, temperature 1.0, and 10 samples; instruction following
uses a 10,000-token cap, temperature 1.0, and one sample. The specification does not set a random seed.

The formal specification names a private step-175 checkpoint, while the released Hugging Face final model card
identifies its checkpoint as step 200. The public record does not establish that these weights are identical. Results
from this harness therefore evaluate the immutable released model and must retain this checkpoint-identity caveat when
compared with the paper table.

## Gates and launch

`smoke` runs one prompt and one sample from every benchmark with a 512-token cap. It validates artifact access, model
loading, prompt rendering, rollout persistence, and the available scorers, but its scores are explicitly non-comparable.
`full` generates 8,101 completions and can request up to 230,050,000 output tokens. Review expected runtime and
GPU cost before submitting it.

```bash
uv run --frozen python -m cloud.iris.open_mopd_evaluation \
  --gate smoke \
  --cluster-config /path/to/cw-rno2a.yaml \
  --gpu-slice H100x8 \
  --task-image registry.example/open-mopd-eval@sha256:<64-hex-digest> \
  --output-uri s3://bucket/unique/open-mopd-final-eval-smoke
```

Review the JSON plan and command. Only then add:

```text
--submit --allow-known-omissions
```

The authors report A100x8. Using H100x8 is recorded as a hardware deviation. The public release does not provide a
complete locked evaluation environment or digest-addressed image, so an image must be validated against the package
contract in `open_mopd_fidelity.json` before any submission. `--no-sync` is intentional: Iris still bundles the current
workspace, while skipping dependency setup that would replace the image's pinned GPU environment.
