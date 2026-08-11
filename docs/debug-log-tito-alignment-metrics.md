# Debug log: TITO alignment metrics on standard generators

## Reported behavior

Standard SkyRLGym rollouts can carry position-aligned rollout logprobs without emitting the
`generate/tis/*` alignment-health metrics. TerminalBench emits those metrics from its own generator
implementation, so the trainer and W&B see different observability depending on the generator class.

## Hypothesis 1: metric finalization is owned by a specialized generator

**Prediction.** A SkyRLGym `GeneratorOutput` with aligned `response_ids`, `loss_masks`, and
`rollout_logprobs` will not contain alignment metrics, while TerminalBench adds them explicitly.

**Evidence.** `AlignmentStats.as_metrics()` has one production caller, in
`examples/terminal_bench/terminal_bench_generator.py`. Both SkyRLGym generation strategies return
aligned logprobs but call only `get_rollout_metrics()`, which derives reward and length metrics.

**Result.** Confirmed by code tracing. The shared concatenation layer can preserve explicit metrics,
but it cannot preserve metrics that SkyRLGym never creates. Direct trainer consumption also reads the
generator's metrics before any concatenation.

## Fix contract

The public `GeneratorInterface.generate()` method owns output finalization. Concrete implementations
produce raw output through `_generate()`. Finalization derives exact-alignment health metrics from the
existing `GeneratorOutput` contract whenever aligned rollout logprobs are present and preserves richer
metrics supplied by generators that perform reconstruction or fallback alignment.

Regression coverage exercises both SkyRLGym strategies through the public `generate()` API and the
base interface with masked tokens. It must prove the metric family is present before trainer logging or
concatenation.

## Result

The red tests failed because SkyRLGym outputs contained no `generate/tis/*` keys. After moving the
finalization lifecycle to `GeneratorInterface.generate()`, the batched and non-batched public APIs emit
exact-alignment metrics, and the metrics survive the fully asynchronous concatenation boundary. A
separate regression proves explicit reconstruction measurements are not overwritten.

## Hypothesis 2: TIS still selects batched generation because non-batched logprobs were once absent

**Prediction.** The broader SkyRLGym suite will fail its generation-strategy contract even though the
non-batched path now returns aligned rollout logprobs.

**Evidence.** `validate_cfg()` unconditionally changed `generator.batched` to `True` for TIS and said
only the batched path returns logprobs. The non-batched regression returns a position-aligned logprob
row, and `test_tis_config_does_not_select_a_generation_strategy` failed on the forced mutation.

**Result.** Confirmed. Removed the obsolete strategy mutation. TIS still requires logprobs and rejects
unsupported SGLang generation, but it no longer chooses between supported SkyRLGym strategies.
