"""Evaluate a Tinker sampler checkpoint on the pinned AIME 2024 corpus."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import Any, Protocol

from datasets import Dataset, load_dataset

AIME24_DATASET = "HuggingFaceH4/aime_2024"
AIME24_REVISION = "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"
AIME24_SPLIT = "train"
AIME24_SIZE = 30
MODEL_NAME = "Qwen/Qwen3.5-9B-Base"
RENDERER_NAME = "qwen3_5"
MAX_TOKENS = 64_000
CONTEXT_WINDOW = 65_536
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = -1
TIMEOUT_SECONDS = 1_800


@dataclass(frozen=True)
class AIME24Example:
    """One normalized AIME 2024 evaluation example."""

    example_id: str
    problem: str
    answer: int


@dataclass(frozen=True)
class EvaluationConfig:
    """Inputs that identify one Tinker AIME 2024 evaluation."""

    checkpoint: str
    save_dir: str
    max_examples: int | None
    num_samples: int
    concurrency: int


DatasetLoader = Callable[..., Dataset]


class SamplingParams(Protocol):
    """Sampling fields whose defaults define published-score comparability."""

    top_p: float
    top_k: int


class SamplingParamsFactory(Protocol):
    """Construct sampling parameters using the installed SDK defaults."""

    def __call__(self) -> SamplingParams: ...


def load_aime24_examples(dataset_loader: DatasetLoader = load_dataset) -> list[AIME24Example]:
    """Load and validate the immutable AIME 2024 evaluation set."""
    dataset = dataset_loader(AIME24_DATASET, split=AIME24_SPLIT, revision=AIME24_REVISION)
    if len(dataset) != AIME24_SIZE:
        raise ValueError(f"Expected {AIME24_SIZE} AIME 2024 examples, found {len(dataset)}")

    examples: list[AIME24Example] = []
    for row in dataset:
        row_id = int(row["id"])
        problem = str(row["problem"]).strip()
        if not problem:
            raise ValueError(f"AIME 2024 example {row_id} has an empty problem")

        answer = int(str(row["answer"]).strip())
        if not 0 <= answer <= 999:
            raise ValueError(f"AIME 2024 example {row_id} has an answer outside [0, 999]: {answer}")
        examples.append(AIME24Example(example_id=f"aime_2024:{row_id}", problem=problem, answer=answer))
    return examples


def validate_sampling_defaults(sampling_params_factory: SamplingParamsFactory) -> None:
    """Require the transitive Tinker SDK to preserve the published sampling contract."""
    sampling_params = sampling_params_factory()
    if sampling_params.top_p != TOP_P or sampling_params.top_k != TOP_K:
        raise RuntimeError(
            "The installed Tinker SDK sampling defaults are incompatible with this evaluation: "
            f"expected top_p={TOP_P} and top_k={TOP_K}, found "
            f"top_p={sampling_params.top_p} and top_k={sampling_params.top_k}"
        )


def validate_comparable_result(
    config: EvaluationConfig,
    *,
    num_examples: int,
    num_errors: int,
    num_truncated: int,
) -> None:
    """Reject incomplete evaluation results that cannot support a published-score comparison."""
    expected_examples = (config.max_examples or AIME24_SIZE) * config.num_samples
    problems: list[str] = []
    if num_examples != expected_examples:
        problems.append(f"expected {expected_examples} scored samples, found {num_examples}")
    if num_errors:
        problems.append(f"encountered {num_errors} sampling errors")
    if num_truncated:
        problems.append(f"encountered {num_truncated} truncated responses")
    if problems:
        detail = "; ".join(problems)
        raise RuntimeError(
            f"AIME 2024 result is not comparable to the published score: {detail}. "
            "Inspect the saved trajectories, resolve the failures, and rerun into a new output directory."
        )


def _benchmark_builder(examples: list[AIME24Example]) -> Any:
    # Tinker Cookbook is intentionally an isolated runtime dependency. Importing it
    # here keeps the root MarinSkyRL environment and lockfile independent of it.
    from tinker_cookbook.eval.benchmarks import BenchmarkBuilder
    from tinker_cookbook.eval.benchmarks.aime import AIMEMessageEnv
    from tinker_cookbook.rl.message_env import EnvFromMessageEnv

    class AIME24Benchmark(BenchmarkBuilder):
        name = "aime_2024"
        recommended_system_prompt = "Put your final answer in \\boxed{}."

        def make_envs(self, renderer: Any, config: Any) -> list[Any]:
            selected = examples[: config.max_examples]
            return [
                EnvFromMessageEnv(
                    renderer=renderer,
                    message_env=AIMEMessageEnv(
                        example.problem,
                        example.answer,
                        example_id=example.example_id,
                        system_prompt=config.system_prompt,
                    ),
                    failed_parse_reward=0.0,
                    context_overflow_reward=0.0,
                )
                for example in selected
            ]

    return AIME24Benchmark()


async def evaluate(config: EvaluationConfig) -> dict[str, Any]:
    """Run the pinned benchmark and return a JSON-serializable result."""
    import tinker
    from tinker_cookbook.eval.benchmarks import BenchmarkConfig, run_benchmark
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    validate_sampling_defaults(tinker.SamplingParams)
    examples = load_aime24_examples()
    service_client = tinker.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(model_path=config.checkpoint)
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = get_renderer(RENDERER_NAME, tokenizer=tokenizer, model_name=MODEL_NAME)
    benchmark_config = BenchmarkConfig(
        max_examples=config.max_examples,
        concurrency=config.concurrency,
        timeout_seconds=TIMEOUT_SECONDS,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        context_window=CONTEXT_WINDOW,
        save_dir=config.save_dir,
        num_samples=config.num_samples,
    )
    result = await run_benchmark(
        _benchmark_builder(examples),
        sampling_client,
        renderer,
        benchmark_config,
    )
    validate_comparable_result(
        config,
        num_examples=result.num_examples,
        num_errors=result.num_errors,
        num_truncated=result.num_truncated,
    )
    return {
        "checkpoint": config.checkpoint,
        "dataset": AIME24_DATASET,
        "dataset_revision": AIME24_REVISION,
        "model_name": MODEL_NAME,
        "renderer_name": RENDERER_NAME,
        "runtime_versions": {
            package: version(package) for package in ("datasets", "tinker", "tinker-cookbook", "transformers")
        },
        "sampling": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "max_tokens": MAX_TOKENS,
            "context_window": CONTEXT_WINDOW,
            "num_samples": config.num_samples,
        },
        "result": asdict(result),
        "score_completed": result.score_completed,
    }


def _parse_args() -> EvaluationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Tinker sampler checkpoint URI")
    parser.add_argument("--save-dir", required=True, help="Local or cloud directory for trajectories")
    parser.add_argument("--max-examples", type=int, help="Limit examples for a smoke test")
    parser.add_argument("--num-samples", type=int, default=1, help="Samples per problem")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent Tinker sampling requests")
    args = parser.parse_args()
    if args.max_examples is not None and not 1 <= args.max_examples <= AIME24_SIZE:
        parser.error(f"--max-examples must be between 1 and {AIME24_SIZE}")
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    return EvaluationConfig(
        checkpoint=args.checkpoint,
        save_dir=args.save_dir,
        max_examples=args.max_examples,
        num_samples=args.num_samples,
        concurrency=args.concurrency,
    )


def main() -> None:
    """Run the evaluator CLI."""
    result = asyncio.run(evaluate(_parse_args()))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
