"""Route TaskCompendium lowerings to native chat or Harbor execution."""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from omegaconf import DictConfig
from taskcompendium.grading import Outcome
from taskcompendium.lowering import (
    ENVIRONMENT_CONFIG_FILE,
    SPECIFICATION_FILE,
    SUBMISSION_CONVENTION_FILE,
    HarborEnvironmentConfig,
    read_environment_config,
    read_specification,
    read_submission_convention,
)
from taskcompendium.models import AnswerType, TaskSpec, VerifierKind
from taskcompendium.resources import ResourceVisibility
from taskcompendium.submission import AnswerFormat, SubmissionConvention, chat_request
from taskcompendium.verifier_registry import grade_answer
from transformers import PreTrainedTokenizerBase

from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.trajectory_runners.base import (
    TrajectoryBatch,
    TrajectoryRequestBatch,
    TrajectoryRunner,
    propagate_teacher_routes,
    run_rollout_task,
)
from skyrl_train.rollouts.buffer import RolloutTask, RolloutWriter
from skyrl_train.trajectory_runners.model_clients import ModelClient
from skyrl_train.trajectory_runners.trajectory_processing import (
    concatenate_trajectory_batches,
    get_custom_chat_template,
    get_response_ids_and_loss_mask_from_messages,
    normalize_token_ids,
)
from skyrl_train.trajectory_runners.trajectory_retention import TrajectorySink, retain_trajectories


HARBOR_ENV_CLASS = "taskcompendium_harbor"
NATIVE_CHAT_ENV_CLASS = "taskcompendium_native_chat"


class UngradedTaskCompendiumBatchError(RuntimeError):
    """A verifier or task failure prevented construction of a numeric batch."""


def _native_chat_eligible(
    specification: TaskSpec, convention: SubmissionConvention, binding: HarborEnvironmentConfig
) -> bool:
    """Return whether the lowering can run as one host-graded chat completion."""
    return (
        specification.answer_type in (AnswerType.TEXT, AnswerType.NUMBER)
        and specification.verifier.kind is VerifierKind.EXACT_ANSWER
        and convention.answer_format in (AnswerFormat.PLAIN, AnswerFormat.JSON)
        and convention.supports(specification.answer_type)
        and not specification.environment_requirements.capabilities
        and not specification.tool_providers
        and not any(resource.visibility is ResourceVisibility.AGENT for resource in specification.resources)
        and binding == HarborEnvironmentConfig()
    )


class TaskCompendiumTaskDataset:
    """Load lowerings and choose the least capable execution path that satisfies each one."""

    def __init__(self, data_files: Sequence[str | Mapping[str, Any]], *, api_base: str, model_name: str):
        self._rows = self._load_rows(data_files, api_base=api_base, model_name=model_name)

    @staticmethod
    def _task_directories(root: Path) -> list[Path]:
        if not root.is_dir():
            raise ValueError(f"TaskCompendium data root does not exist: {root}")
        if (root / SPECIFICATION_FILE).is_file():
            return [root]
        tasks = sorted(path for path in root.iterdir() if (path / SPECIFICATION_FILE).is_file())
        if not tasks:
            raise ValueError(f"TaskCompendium data root has no lowering packages: {root}")
        return tasks

    @classmethod
    def _load_rows(
        cls,
        data_files: Sequence[str | Mapping[str, Any]],
        *,
        api_base: str,
        model_name: str,
    ) -> list[dict[str, Any]]:
        rows = []
        for value in data_files:
            if not isinstance(value, str):
                value = str(value["local_path"])
            for task_dir in cls._task_directories(Path(value)):
                specification = read_specification(task_dir / SPECIFICATION_FILE)
                convention = read_submission_convention(task_dir / SUBMISSION_CONVENTION_FILE)
                binding = read_environment_config(task_dir / ENVIRONMENT_CONFIG_FILE)
                if _native_chat_eligible(specification, convention, binding):
                    rows.append(
                        {
                            "uid": task_dir.name,
                            "prompt": chat_request(specification, convention)["messages"],
                            "env_class": NATIVE_CHAT_ENV_CLASS,
                            "env_extras": {"task_dir": str(task_dir)},
                        }
                    )
                    continue

                rows.append(
                    {
                        "uid": task_dir.name,
                        "prompt": [{"role": "user", "content": str(task_dir)}],
                        "env_class": HARBOR_ENV_CLASS,
                        "env_extras": {"task_dir": str(task_dir), "api_base": api_base, "model_name": model_name},
                    }
                )
        return rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._rows[index]

    def uid(self, index: int) -> str:
        return str(self._rows[index]["uid"])

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def collate_fn(self, item_list):
        return item_list


def _select_rows(batch: TrajectoryRequestBatch, indices: list[int]) -> TrajectoryRequestBatch:
    size = len(batch["prompts"])
    selected: dict[str, Any] = {}
    for key, value in batch.items():
        selected[key] = [value[index] for index in indices] if isinstance(value, list) and len(value) == size else value
    return selected  # type: ignore[return-value]


class NativeTaskCompendiumRunner(TrajectoryRunner):
    """Generate exact policy tokens and grade simple answer-only tasks on the host."""

    def __init__(
        self,
        trajectory_runner_cfg: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
        model_client: ModelClient,
    ) -> None:
        self.trajectory_runner_cfg = trajectory_runner_cfg
        self.tokenizer = tokenizer
        self.model_client = model_client
        self.custom_chat_template = get_custom_chat_template(trajectory_runner_cfg.get("chat_template"))
        self.trajectory_sink = None
        self.global_step_fn = None

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def start_eval_session(self, **kwargs) -> None:
        del kwargs

    async def stop_eval_session(self) -> None:
        pass

    async def _run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        del disable_tqdm
        extras = input_batch.get("env_extras")
        identities = input_batch.get("trajectory_ids")
        size = len(input_batch["prompts"])
        if (
            extras is None
            or identities is None
            or not (len(extras) == len(identities) == len(input_batch["env_classes"]) == size)
        ):
            raise ValueError("Native TaskCompendium requests require aligned task metadata and trajectory IDs")
        if any(name != NATIVE_CHAT_ENV_CLASS for name in input_batch["env_classes"]):
            raise ValueError(f"This runner requires env_class={NATIVE_CHAT_ENV_CLASS}")

        prompt_token_ids = [
            normalize_token_ids(
                self.tokenizer.apply_chat_template(
                    prompt,
                    add_generation_prompt=True,
                    chat_template=self.custom_chat_template,
                    tokenize=True,
                    **dict(self.trajectory_runner_cfg.get("chat_template_kwargs", {})),
                )
            )
            for prompt in input_batch["prompts"]
        ]
        request = InferenceEngineInput(
            prompts=None,
            prompt_token_ids=prompt_token_ids,
            sampling_params=input_batch.get("sampling_params") or dict(self.trajectory_runner_cfg.sampling_params),
            session_ids=[identity.to_string() for identity in identities],
        )
        generated = await self.model_client.generate(request)
        responses = generated["responses"]
        response_ids = generated["response_ids"]
        stop_reasons = generated["stop_reasons"]
        rollout_logprobs = generated.get("response_logprobs")
        if not (len(responses) == len(response_ids) == len(stop_reasons) == size):
            raise ValueError("Inference engine output must align with the TaskCompendium request batch")
        if rollout_logprobs is not None and any(
            len(token_ids) != len(logprobs) for token_ids, logprobs in zip(response_ids, rollout_logprobs, strict=True)
        ):
            raise ValueError("Inference engine logprobs must align with response token IDs")

        rewards: list[float] = []
        exception_types: list[str | None] = []
        error_treatments: list[str | None] = []
        extraction_errors = 0
        with tempfile.TemporaryDirectory(prefix="taskcompendium-native-") as temporary:
            workspace = Path(temporary)
            for extra, response, identity in zip(extras, responses, identities, strict=True):
                task_dir = Path(extra["task_dir"])
                specification = read_specification(task_dir / SPECIFICATION_FILE)
                convention = read_submission_convention(task_dir / SUBMISSION_CONVENTION_FILE)
                result = grade_answer(specification, convention, response, workspace)
                if result.status is Outcome.GRADED and result.reward is not None:
                    rewards.append(result.reward)
                    exception_types.append(None)
                    error_treatments.append(None)
                    continue
                if result.status is Outcome.EXTRACTION_ERROR:
                    rewards.append(0.0)
                    exception_types.append(Outcome.EXTRACTION_ERROR.value)
                    error_treatments.append("zero")
                    extraction_errors += 1
                    continue
                raise UngradedTaskCompendiumBatchError(
                    f"{identity.to_string()} has semantic status {result.status.value}; "
                    "the result cannot enter a numeric training batch"
                )

        output: TrajectoryBatch = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": response_ids,
            "rewards": rewards,
            "unshaped_rewards": rewards,
            "loss_masks": [[1] * len(tokens) for tokens in response_ids],
            "stop_reasons": stop_reasons,
            "exception_types": exception_types,
            "error_treatments": error_treatments,
            "trajectory_ids": list(identities),
            "rollout_metrics": {
                "taskcompendium/native_chat_trajectories": float(size),
                "taskcompendium/extraction_errors_zeroed": float(extraction_errors),
            },
            "rollout_logprobs": rollout_logprobs,
            "is_last_step": [True] * size,
            "exclude_from_baseline": [False] * size,
        }
        selected_indices = generated.get("student_topk_indices")
        selected_logprobs = generated.get("behavior_topk_logprobs")
        if (selected_indices is None) != (selected_logprobs is None):
            raise ValueError("Inference engine must return student top-K IDs and behavior scores together")
        if selected_indices is not None:
            output["student_topk_indices"] = selected_indices
            output["behavior_topk_logprobs"] = selected_logprobs
        return output


class TaskCompendiumHarborRunner(TrajectoryRunner):
    """Run tool-bearing task packages and reconstruct learner actions from Harbor traces."""

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, output_dir: Path, *, concurrency: int, max_turns: int, timeout: float
    ):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.concurrency = concurrency
        self.max_turns = max_turns
        self.timeout = timeout
        self.trajectory_sink = None
        self.global_step_fn = None

    async def startup(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def shutdown(self) -> None:
        pass

    async def start_eval_session(self, **kwargs) -> None:
        del kwargs

    async def stop_eval_session(self) -> None:
        pass

    async def _run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        del disable_tqdm
        from taskcompendium.harbor.runner import ChatLaunch, run_trial  # noqa: PLC0415

        extras = input_batch.get("env_extras")
        identities = input_batch.get("trajectory_ids")
        size = len(input_batch["prompts"])
        if extras is None or identities is None or not (len(extras) == len(identities) == size):
            raise ValueError("Harbor requests require aligned task metadata and trajectory IDs")
        if any(name != HARBOR_ENV_CLASS for name in input_batch["env_classes"]):
            raise ValueError(f"This runner requires env_class={HARBOR_ENV_CLASS}")
        semaphore = asyncio.Semaphore(self.concurrency)

        async def trial(extra, prompt, identity):
            task_dir = Path(extra["task_dir"])
            if prompt != [{"role": "user", "content": str(task_dir)}]:
                raise ValueError("Harbor request prompts must identify the corresponding task package")
            binding = read_environment_config(task_dir / ENVIRONMENT_CONFIG_FILE)
            launch = ChatLaunch(
                model=extra["model_name"],
                api_base=extra["api_base"],
                max_turns=self.max_turns,
                request_timeout=self.timeout,
                trial_timeout=self.timeout,
            )
            async with semaphore:
                result = await run_trial(task_dir, binding, launch, self.output_dir, uuid4().hex)
            trial_path = result.resolve_trial_path(self.output_dir)
            if result.exception_info is not None or result.verifier_result is None:
                raise UngradedTaskCompendiumBatchError(
                    f"{identity.to_string()} is ungraded; Harbor trace retained at {trial_path}"
                )
            reward = result.verifier_result.rewards.get("reward")
            metadata = result.agent_result.metadata if result.agent_result is not None else None
            if reward is None or not isinstance(metadata, dict) or not isinstance(metadata.get("all_messages"), list):
                raise UngradedTaskCompendiumBatchError(
                    f"{identity.to_string()} lacks a reward or assistant trace; Harbor trace retained at {trial_path}"
                )
            messages = metadata["all_messages"]
            first_assistant = next(
                (index for index, message in enumerate(messages) if message.get("role") == "assistant"), None
            )
            if first_assistant is None:
                raise UngradedTaskCompendiumBatchError(
                    f"{identity.to_string()} lacks assistant actions; Harbor trace retained at {trial_path}"
                )
            tools = metadata.get("tool_definitions")
            template_kwargs = {"tools": tools} if tools else {}
            prompt_ids = normalize_token_ids(
                self.tokenizer.apply_chat_template(
                    messages[:first_assistant], tokenize=True, add_generation_prompt=False, **template_kwargs
                )
            )
            response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
                messages[first_assistant:],
                self.tokenizer,
                rollout_logprobs_required=False,
                tito_full=False,
                chat_template_kwargs=template_kwargs,
            )
            return prompt_ids, response_ids, loss_mask, float(reward)

        rows = await asyncio.gather(
            *(
                trial(extra, prompt, identity)
                for extra, prompt, identity in zip(extras, input_batch["prompts"], identities, strict=True)
            )
        )
        return {
            "prompt_token_ids": [row[0] for row in rows],
            "response_ids": [row[1] for row in rows],
            "loss_masks": [row[2] for row in rows],
            "rewards": [row[3] for row in rows],
            "unshaped_rewards": [row[3] for row in rows],
            "stop_reasons": ["stop"] * size,
            "exception_types": [None] * size,
            "error_treatments": [None] * size,
            "trajectory_ids": list(identities),
            "rollout_metrics": {"taskcompendium/reconstructed_trajectories": float(size)},
            "rollout_logprobs": None,
            "is_last_step": [True] * size,
            "exclude_from_baseline": [False] * size,
        }


class TaskCompendiumTrajectoryRouter:
    """Run simple answer tasks natively and capability-bearing tasks through Harbor."""

    def __init__(
        self,
        *,
        native_runner,
        harbor_runner,
        require_rollout_logprobs: bool,
        tis_lcs_alert_threshold: float,
    ) -> None:
        self.native_runner = native_runner
        self.harbor_runner = harbor_runner
        self.require_rollout_logprobs = require_rollout_logprobs
        self.tis_lcs_alert_threshold = tis_lcs_alert_threshold
        self._global_step_fn = None
        self.trajectory_sink: TrajectorySink | None = None

    @property
    def global_step_fn(self):
        return self._global_step_fn

    @global_step_fn.setter
    def global_step_fn(self, callback) -> None:
        self._global_step_fn = callback
        self.native_runner.global_step_fn = callback
        self.harbor_runner.global_step_fn = callback

    async def startup(self) -> None:
        await asyncio.gather(self.native_runner.startup(), self.harbor_runner.startup())

    async def shutdown(self) -> None:
        await asyncio.gather(self.native_runner.shutdown(), self.harbor_runner.shutdown())

    def set_trajectory_sink(self, sink: TrajectorySink) -> None:
        sink.bind_runner(type(self).__name__)
        self.trajectory_sink = sink

    async def start_eval_session(self, **kwargs) -> None:
        await asyncio.gather(
            self.native_runner.start_eval_session(**kwargs),
            self.harbor_runner.start_eval_session(**kwargs),
        )

    async def stop_eval_session(self) -> None:
        await asyncio.gather(self.native_runner.stop_eval_session(), self.harbor_runner.stop_eval_session())

    async def run_task(self, task: RolloutTask, writer: RolloutWriter) -> int:
        return await run_rollout_task(self, task, writer)

    async def run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        classes = input_batch["env_classes"]
        if len(classes) != len(input_batch["prompts"]):
            raise ValueError("TaskCompendium routing requires one environment class per request row")
        native_indices = [index for index, name in enumerate(classes) if name == NATIVE_CHAT_ENV_CLASS]
        harbor_indices = [index for index, name in enumerate(classes) if name == HARBOR_ENV_CLASS]
        if len(native_indices) + len(harbor_indices) != len(classes):
            unknown = sorted(set(classes) - {NATIVE_CHAT_ENV_CLASS, HARBOR_ENV_CLASS})
            raise ValueError(f"Unsupported TaskCompendium environment classes: {unknown}")

        jobs = []
        index_groups = []
        if native_indices:
            jobs.append(self.native_runner.run(_select_rows(input_batch, native_indices), disable_tqdm=disable_tqdm))
            index_groups.append(native_indices)
        if harbor_indices:
            jobs.append(self.harbor_runner.run(_select_rows(input_batch, harbor_indices), disable_tqdm=disable_tqdm))
            index_groups.append(harbor_indices)
        outputs = await asyncio.gather(*jobs)
        if native_indices and harbor_indices:
            # The Harbor trace is retokenized. Do not preserve partial behavior-policy
            # evidence on a batch that also contains reconstructed actions.
            outputs[0]["rollout_logprobs"] = None
            outputs[0].pop("student_topk_indices", None)
            outputs[0].pop("behavior_topk_logprobs", None)
        result = (
            outputs[0]
            if len(outputs) == 1
            else concatenate_trajectory_batches(
                outputs,
                require_rollout_logprobs=self.require_rollout_logprobs,
                tis_lcs_alert_threshold=self.tis_lcs_alert_threshold,
            )
        )

        concatenated_indices = [index for group in index_groups for index in group]
        restore = sorted(range(len(concatenated_indices)), key=concatenated_indices.__getitem__)
        for key, value in list(result.items()):
            if isinstance(value, list) and len(value) == len(concatenated_indices):
                result[key] = [value[index] for index in restore]
        rollout_metrics = result.setdefault("rollout_metrics", {})
        if rollout_metrics is None:
            rollout_metrics = result["rollout_metrics"] = {}
        rollout_metrics["taskcompendium/native_chat_trajectories"] = float(len(native_indices))
        rollout_metrics["taskcompendium/harbor_trajectories"] = float(len(harbor_indices))
        propagate_teacher_routes(input_batch, result)
        if self.trajectory_sink is not None:
            await retain_trajectories(self.trajectory_sink, input_batch, result)
        return result
