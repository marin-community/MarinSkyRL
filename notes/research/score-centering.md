# PPO/TIS score centering experiment

Research branch: `research/score-centering-simple-01a0bb6f`. Base: [`544d5d6f14116a06bde0209352585903133bd618`](https://github.com/marin-community/MarinSkyRL/commit/544d5d6f14116a06bde0209352585903133bd618). This branch keeps the algorithm from [the repaired version](https://github.com/marin-community/MarinSkyRL/tree/1b5dd6b6ae6c7d87bfc01097d198aa72e5a5c968), removes coverage metrics and shared distillation refactors, and computes centering beside current-policy scoring. The objective receives a differentiable per-token correction and reduces it separately from sampled-action TIS weights.

For each prefix, `q` is the sampling policy, `o` is the stored policy before training the batch, and `p` is the current policy. Capture the natural behavior top K, then score those same IDs under `o` and `p`. Center the PPO/TIS score coefficient using detached probabilities and clip decisions. Outside the head, model `q` and `o` as proportional copies of the current tail. This approximation does not recover arbitrary omitted gradients.

## Resume

Check out the research branch and install the root frozen environment. On an approved GPU allocation, install `--extra megatron --extra vllm` and use the standard training entrypoint. Start from an already working launch config with model, datasets, placement and artifact paths. Add these Hydra overrides:

```text
trainer.strategy=megatron
trainer.use_sample_packing=false
trainer.policy.sequence_parallel_size=1
trainer.policy.megatron_config.tensor_model_parallel_size=1
trainer.policy.megatron_config.context_parallel_size=1
trainer.algorithm.policy_loss_type=regular
trainer.algorithm.off_policy_correction=custom
'trainer.algorithm.off_policy_correction_rules=[{kind:token,action:truncate,high:1.05}]'
trainer.algorithm.score_centering_topk=32
generator.backend=vllm
generator.run_engines_locally=true
generator.sampling_params.logprobs=32
```

From `skyrl-train/`, launch with `uv run --project .. --frozen --no-sync -m skyrl_train.entrypoints.main_base` followed by the working config's overrides and those above. For Iris, put the same values under `skyrl` in the standard resolved launch config and submit through `marinskyrl launch --config <config.yaml>`. Set `runtime.launcher_commit` to `git rev-parse HEAD` from the research checkout; the selected launcher checkout must match it. Follow the repository's launch and resource coordination procedure before submission.

Width zero disables centering. Also set generation `logprobs=null` when returning to ordinary PPO/TIS, unless another supported consumer needs the capture. Use full-distribution, temperature-only training sampling: `top_p=1`, `min_p=0`, `top_k=-1`, without probability-changing processors. Normal startup validates these settings and configures compatible serving log probabilities.

Supported: regular PPO with exactly one token TIS truncation; local vLLM; unpacked Megatron with TP/CP/SP=1. Pipeline parallelism remains supported. Distillation, packing, remote capture and other parallel sizes fail configuration validation. DP and all existing loss reductions remain supported. Chosen and candidate scores share the same differentiable normalizer; non-final pipeline stages return no selected scores.

## Evidence and checks

The [study and complete results](https://github.com/yonromai/marin/blob/4ac4d2dfb4cb3a1734969d0b72b1bbae7b3169ab/experiments/post_training/score_centering_study.md) used the [original runtime](https://github.com/marin-community/MarinSkyRL/tree/5f53efd1300ac41fcff0811db56842dacb4c735d), preserved in `research/score-centering-record-20261006-01a0bb6f` at `f36bbf37b347fdcd4fa2537e2b67fb15413ee0b3`. Twelve paired seeds at 40 updates showed no established correct-answer-rate improvement over matched TIS: +0.06 percentage points, adjusted joint 95% interval [-1.23, +1.34]. Token ages were 0–2; PPO clipping never activated. This is not a universal negative result.

The simplified branch has CPU validation only. The historical GPU runs used forward microbatch size one and do not qualify this revision or a multi-stage pipeline. Preserve the repaired runtime at `research/score-centering-repaired-20261006-01a0bb6f` (`1b5dd6b6ae6c7d87bfc01097d198aa72e5a5c968`). Historical design: [Echo 608](https://echo.oa.dev/wiki/608).

From the repository root:

```bash
uv sync --frozen --group dev --group harbor-test --extra cpu --extra telemetry
OMP_NUM_THREADS=1 uv run --frozen --no-sync pytest -n 0 -q \
  skyrl-train/tests/cpu/objective/test_score_centering.py \
  skyrl-train/tests/cpu/test_score_centering_megatron_positions.py
uv run infra/pre-commit.py --changed-files --fix
```

These checks cover numerical values and gradients, modeled tails and tiny masses, masked sentinels, loss reductions, THINK weights, microbatch/DP partitioning, evidence serialization, token alignment, and output construction before pipeline filtering. Scoring uses a real one-rank CPU Gloo group; the external Megatron model, scheduler and pipeline ranks are simulated. The pipeline check reproduces worker output assembly with the real container and collector. It is not a GPU peak-memory measurement or hardware qualification.
