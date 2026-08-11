from collections import OrderedDict
from collections.abc import Mapping, Sequence
import asyncio
from dataclasses import dataclass
import gzip
import hashlib
import json
import posixpath
import threading
from typing import Any, Protocol

from omegaconf import DictConfig
from loguru import logger
from transformers import PreTrainedTokenizerBase

from skyrl_train.generators.generator_types import GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl_train.generators.trajectory_reward_shaping import DEFAULT_ACCEPTED_STOP_REASONS, NormalizedReward
from skyrl_train.utils.io import io


RETENTION_METRIC_PREFIX = "generate/trajectory_retention"
RETENTION_SCHEMA_VERSION = 1
_LEDGER_NAME = "_retention_ledger.json"
_SELECTION_COUNT = "count"
_SELECTION_FRACTION = "fraction"
_SELECTION_MANDATORY = "mandatory"


@dataclass(frozen=True)
class TrajectoryRetentionConfig:
    schema_version: int = RETENTION_SCHEMA_VERSION
    enabled: bool = True
    output_path: str = ""
    run_id: str = ""
    phases: tuple[str, ...] = ("train",)
    sample_count_per_step: int = 1
    sample_fraction: float = 0.0
    always_retain_failures: bool = True
    always_retain_non_terminating: bool = True
    always_retain_loops: bool = True
    accepted_stop_reasons: tuple[str, ...] = DEFAULT_ACCEPTED_STOP_REASONS
    reward_below: float | None = None
    reward_above: float | None = None
    max_bytes_per_step: int = 8 * 1024 * 1024
    max_bytes_per_run: int = 256 * 1024 * 1024
    required: bool = False
    redact_fields: tuple[str, ...] = ()
    model_path: str | None = None
    model_source_identity: str | None = None
    resume_path: str | None = None
    inference_backend: str | None = None


@dataclass(frozen=True)
class _LedgerEntry:
    path: str
    bytes: int
    step: str
    reasons: tuple[str, ...]
    sample_score: int

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "_LedgerEntry":
        return cls(
            path=str(value["path"]),
            bytes=int(value["bytes"]),
            step=str(value["step"]),
            reasons=tuple(str(reason) for reason in value.get("reasons", ())),
            sample_score=int(value.get("sample_score", 0)),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "step": self.step,
            "reasons": list(self.reasons),
            "sample_score": self.sample_score,
        }


@dataclass
class _RetentionLedger:
    total_bytes: int
    step_bytes: dict[str, int]
    records: dict[str, _LedgerEntry]
    schema_version: int = RETENTION_SCHEMA_VERSION

    @classmethod
    def from_json(cls, value: Mapping[str, Any] | None) -> "_RetentionLedger":
        if value is None:
            return cls(total_bytes=0, step_bytes={}, records={})
        schema_version = int(value.get("schema_version", 0))
        if schema_version != RETENTION_SCHEMA_VERSION:
            raise ValueError("trajectory retention ledger schema version is incompatible")
        return cls(
            schema_version=schema_version,
            total_bytes=int(value.get("total_bytes", 0)),
            step_bytes={str(step): int(byte_count) for step, byte_count in value.get("step_bytes", {}).items()},
            records={
                str(record_id): _LedgerEntry.from_json(entry)
                for record_id, entry in value.get("records", {}).items()
            },
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "total_bytes": self.total_bytes,
            "step_bytes": self.step_bytes,
            "records": {record_id: entry.to_json() for record_id, entry in self.records.items()},
        }


@dataclass(frozen=True)
class _TrajectoryRows:
    trajectory_id: TrajectoryID
    row_indices: tuple[int, ...]
    prompt_index: int


@dataclass(frozen=True)
class _Selection:
    reasons: tuple[str, ...]
    displaced_id: str | None = None


@dataclass(frozen=True)
class _TrajectoryIdentity:
    instance_id: str
    repetition_id: int
    row_indices: tuple[int, ...]
    environment_class: str
    environment_extras: Any

    def to_json(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "repetition_id": self.repetition_id,
            "row_indices": list(self.row_indices),
            "environment_class": self.environment_class,
            "environment_extras": self.environment_extras,
        }


@dataclass(frozen=True)
class _PromptTrace:
    messages: Any
    token_ids: tuple[int, ...]

    def to_json(self) -> dict[str, Any]:
        return {"messages": self.messages, "token_ids": list(self.token_ids)}


@dataclass(frozen=True)
class _ResponseTrace:
    messages: Any
    text: str | None
    token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    trainable_spans: tuple[dict[str, int], ...]
    step_boundaries: tuple[dict[str, Any], ...]
    stop_reason: str | None
    generation_limit: int | None
    loop_spans: tuple[Any, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "text": self.text,
            "token_ids": list(self.token_ids),
            "loss_mask": list(self.loss_mask),
            "trainable_spans": list(self.trainable_spans),
            "step_boundaries": list(self.step_boundaries),
            "stop_reason": self.stop_reason,
            "generation_limit": self.generation_limit,
            "loop_spans": list(self.loop_spans),
        }


@dataclass(frozen=True)
class _RewardTrace:
    outcome: float
    shaped: float
    components: Any

    def to_json(self) -> dict[str, Any]:
        return {"outcome": self.outcome, "shaped": self.shaped, "components": self.components}


@dataclass(frozen=True)
class _ProvenanceTrace:
    generator: str
    inference_backend: str | None
    model_path: str | None
    model_source_identity: str | None
    resume_path: str | None
    model_version_step: int
    sampling: dict[str, Any]
    reward_shaping_schema_version: int | None

    def to_json(self) -> dict[str, Any]:
        return {
            "generator": self.generator,
            "inference_backend": self.inference_backend,
            "model_path": self.model_path,
            "model_source_identity": self.model_source_identity,
            "resume_path": self.resume_path,
            "model_version_step": self.model_version_step,
            "sampling": self.sampling,
            "reward_shaping_schema_version": self.reward_shaping_schema_version,
        }


@dataclass(frozen=True)
class TrajectoryRecord:
    schema_version: int
    record_id: str
    run_id: str
    global_step: int
    phase: str
    trajectory: _TrajectoryIdentity
    prompt: _PromptTrace
    response: _ResponseTrace
    reward: _RewardTrace
    metrics: dict[str, Any]
    provenance: _ProvenanceTrace

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "TrajectoryRecord":
        return cls(
            schema_version=int(value["schema_version"]),
            record_id=str(value["record_id"]),
            run_id=str(value["run_id"]),
            global_step=int(value["global_step"]),
            phase=str(value["phase"]),
            trajectory=_TrajectoryIdentity(
                instance_id=str(value["trajectory"]["instance_id"]),
                repetition_id=int(value["trajectory"]["repetition_id"]),
                row_indices=tuple(int(index) for index in value["trajectory"]["row_indices"]),
                environment_class=str(value["trajectory"]["environment_class"]),
                environment_extras=value["trajectory"]["environment_extras"],
            ),
            prompt=_PromptTrace(
                messages=value["prompt"]["messages"],
                token_ids=tuple(int(token) for token in value["prompt"]["token_ids"]),
            ),
            response=_ResponseTrace(
                messages=value["response"]["messages"],
                text=value["response"]["text"],
                token_ids=tuple(int(token) for token in value["response"]["token_ids"]),
                loss_mask=tuple(int(active) for active in value["response"]["loss_mask"]),
                trainable_spans=tuple(value["response"]["trainable_spans"]),
                step_boundaries=tuple(value["response"]["step_boundaries"]),
                stop_reason=value["response"]["stop_reason"],
                generation_limit=value["response"]["generation_limit"],
                loop_spans=tuple(value["response"]["loop_spans"]),
            ),
            reward=_RewardTrace(
                outcome=float(value["reward"]["outcome"]),
                shaped=float(value["reward"]["shaped"]),
                components=value["reward"]["components"],
            ),
            metrics=dict(value["metrics"]),
            provenance=_ProvenanceTrace(
                generator=str(value["provenance"]["generator"]),
                inference_backend=value["provenance"]["inference_backend"],
                model_path=value["provenance"]["model_path"],
                model_source_identity=value["provenance"]["model_source_identity"],
                resume_path=value["provenance"]["resume_path"],
                model_version_step=int(value["provenance"]["model_version_step"]),
                sampling=dict(value["provenance"]["sampling"]),
                reward_shaping_schema_version=value["provenance"]["reward_shaping_schema_version"],
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "run_id": self.run_id,
            "global_step": self.global_step,
            "phase": self.phase,
            "trajectory": self.trajectory.to_json(),
            "prompt": self.prompt.to_json(),
            "response": self.response.to_json(),
            "reward": self.reward.to_json(),
            "metrics": self.metrics,
            "provenance": self.provenance.to_json(),
        }

    @property
    def instance_id(self) -> str:
        return self.trajectory.instance_id

    @property
    def repetition_id(self) -> int:
        return self.trajectory.repetition_id

    @property
    def outcome(self) -> float:
        return self.reward.outcome

    @property
    def stop_reason(self) -> str | None:
        return self.response.stop_reason

    @property
    def loop_spans(self) -> Sequence[Any]:
        return self.response.loop_spans


class TrajectoryWriter(Protocol):
    def exists(self, relative_path: str) -> bool: ...

    def read_json(self, relative_path: str) -> dict[str, Any] | None: ...

    def write_bytes(self, relative_path: str, payload: bytes) -> None: ...

    def write_json(self, relative_path: str, value: dict[str, Any]) -> None: ...

    def remove(self, relative_path: str) -> None: ...


class _FilesystemTrajectoryWriter:
    def __init__(self, output_path: str):
        self.output_path = output_path.rstrip("/")

    def _path(self, relative_path: str) -> str:
        return posixpath.join(self.output_path, relative_path)

    def exists(self, relative_path: str) -> bool:
        return io.exists(self._path(relative_path))

    def read_json(self, relative_path: str) -> dict[str, Any] | None:
        path = self._path(relative_path)
        if not io.exists(path):
            return None
        with io.open_file(path, "r") as source:
            value = json.load(source)
        if not isinstance(value, dict):
            raise ValueError(f"trajectory retention ledger must contain an object: {path}")
        return value

    def write_bytes(self, relative_path: str, payload: bytes) -> None:
        self._write(relative_path, payload)

    def write_json(self, relative_path: str, value: dict[str, Any]) -> None:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self._write(relative_path, payload)

    def remove(self, relative_path: str) -> None:
        io.remove(self._path(relative_path))

    def _write(self, relative_path: str, payload: bytes) -> None:
        io.write_bytes_atomic(self._path(relative_path), payload)


def parse_trajectory_retention_config(config: Mapping[str, Any] | None) -> TrajectoryRetentionConfig:
    """Parse and validate the shared trajectory-retention policy."""
    defaults = TrajectoryRetentionConfig()
    if config is None:
        return TrajectoryRetentionConfig(enabled=False)
    if not isinstance(config, Mapping):
        raise ValueError("generator.trajectory_retention must be a mapping")

    phases_value = config.get("phases", defaults.phases)
    redact_value = config.get("redact_fields", defaults.redact_fields)
    accepted_stops_value = config.get("accepted_stop_reasons", defaults.accepted_stop_reasons)
    if not isinstance(phases_value, Sequence) or isinstance(phases_value, (str, bytes)):
        raise ValueError("trajectory_retention.phases must be a sequence")
    if not isinstance(redact_value, Sequence) or isinstance(redact_value, (str, bytes)):
        raise ValueError("trajectory_retention.redact_fields must be a sequence")
    if not isinstance(accepted_stops_value, Sequence) or isinstance(accepted_stops_value, (str, bytes)):
        raise ValueError("trajectory_retention.accepted_stop_reasons must be a sequence")

    parsed = TrajectoryRetentionConfig(
        schema_version=int(config.get("schema_version", defaults.schema_version)),
        enabled=bool(config.get("enabled", defaults.enabled)),
        output_path=str(config.get("output_path", defaults.output_path) or ""),
        run_id=str(config.get("run_id", defaults.run_id) or ""),
        phases=tuple(str(phase).lower() for phase in phases_value),
        sample_count_per_step=int(config.get("sample_count_per_step", defaults.sample_count_per_step)),
        sample_fraction=float(config.get("sample_fraction", defaults.sample_fraction)),
        always_retain_failures=bool(config.get("always_retain_failures", defaults.always_retain_failures)),
        always_retain_non_terminating=bool(
            config.get("always_retain_non_terminating", defaults.always_retain_non_terminating)
        ),
        always_retain_loops=bool(config.get("always_retain_loops", defaults.always_retain_loops)),
        accepted_stop_reasons=tuple(str(reason).strip().lower() for reason in accepted_stops_value),
        reward_below=_optional_float(config.get("reward_below", defaults.reward_below)),
        reward_above=_optional_float(config.get("reward_above", defaults.reward_above)),
        max_bytes_per_step=int(config.get("max_bytes_per_step", defaults.max_bytes_per_step)),
        max_bytes_per_run=int(config.get("max_bytes_per_run", defaults.max_bytes_per_run)),
        required=bool(config.get("required", defaults.required)),
        redact_fields=tuple(str(field) for field in redact_value),
        model_path=_optional_string(config.get("model_path", defaults.model_path)),
        model_source_identity=_optional_string(
            config.get("model_source_identity", defaults.model_source_identity)
        ),
        resume_path=_optional_string(config.get("resume_path", defaults.resume_path)),
        inference_backend=_optional_string(config.get("inference_backend", defaults.inference_backend)),
    )
    _validate_config(parsed)
    return parsed


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_string(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _validate_config(config: TrajectoryRetentionConfig) -> None:
    if config.schema_version != RETENTION_SCHEMA_VERSION:
        raise ValueError(f"trajectory retention schema_version must be {RETENTION_SCHEMA_VERSION}")
    if config.enabled and (not config.output_path or not config.run_id):
        raise ValueError("enabled trajectory retention requires output_path and run_id")
    if not set(config.phases).issubset({"train", "eval"}) or not config.phases:
        raise ValueError("trajectory_retention.phases must contain train and/or eval")
    if config.sample_count_per_step < 0:
        raise ValueError("trajectory_retention.sample_count_per_step must be non-negative")
    if not 0.0 <= config.sample_fraction <= 1.0:
        raise ValueError("trajectory_retention.sample_fraction must be between 0 and 1")
    if config.max_bytes_per_step < 0 or config.max_bytes_per_run < 0:
        raise ValueError("trajectory retention byte bounds must be non-negative")
    if config.max_bytes_per_step > config.max_bytes_per_run:
        raise ValueError("trajectory retention per-step bound cannot exceed its run bound")
    if not config.accepted_stop_reasons:
        raise ValueError("trajectory_retention.accepted_stop_reasons cannot be empty")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    return str(value)


def _trajectory_key(trajectory_id: TrajectoryID) -> tuple[str, int]:
    return trajectory_id.instance_id, trajectory_id.repetition_id


def _group_rows(input_batch: GeneratorInput, output: GeneratorOutput) -> list[_TrajectoryRows]:
    batch_size = len(output["response_ids"])
    output_ids = output.get("trajectory_ids")
    input_ids = input_batch.get("trajectory_ids")
    if output_ids is None:
        if input_ids is None or len(input_ids) != batch_size:
            raise ValueError("trajectory retention requires one trajectory ID per normalized output row")
        output_ids = input_ids
    if len(output_ids) != batch_size:
        raise ValueError("trajectory IDs must align with normalized output rows")

    input_index = {_trajectory_key(trajectory_id): index for index, trajectory_id in enumerate(input_ids or [])}
    groups: OrderedDict[tuple[str, int], tuple[TrajectoryID, list[int], int]] = OrderedDict()
    for row_index, trajectory_id in enumerate(output_ids):
        key = _trajectory_key(trajectory_id)
        prompt_index = input_index.get(key, row_index)
        if key not in groups:
            groups[key] = trajectory_id, [], prompt_index
        groups[key][1].append(row_index)
    return [
        _TrajectoryRows(trajectory_id=trajectory_id, row_indices=tuple(row_indices), prompt_index=prompt_index)
        for trajectory_id, row_indices, prompt_index in groups.values()
    ]


def _active_spans(loss_mask: Sequence[int]) -> list[dict[str, int]]:
    spans = []
    start = None
    for index, active in enumerate(loss_mask):
        if active and start is None:
            start = index
        if not active and start is not None:
            spans.append({"start": start, "end": index})
            start = None
    if start is not None:
        spans.append({"start": start, "end": len(loss_mask)})
    return spans


def _decode(tokenizer: PreTrainedTokenizerBase | None, token_ids: Sequence[int]) -> str | None:
    if tokenizer is None:
        return None
    return str(tokenizer.decode(list(token_ids), skip_special_tokens=False))


def _step_boundaries(
    output: GeneratorOutput, row_indices: Sequence[int], stop_reasons: Sequence[str | None]
) -> list[dict[str, Any]]:
    boundaries = []
    token_start = 0
    final_steps = output.get("is_last_step")
    for row_index in row_indices:
        token_end = token_start + len(output["response_ids"][row_index])
        boundaries.append(
            {
                "row_index": row_index,
                "token_start": token_start,
                "token_end": token_end,
                "stop_reason": stop_reasons[row_index],
                "is_last_step": None if final_steps is None else final_steps[row_index],
            }
        )
        token_start = token_end
    return boundaries


def _redact(record: dict[str, Any], fields: Sequence[str]) -> None:
    for field in fields:
        parts = field.split(".")
        current: Any = record
        for part in parts[:-1]:
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if isinstance(current, dict) and parts[-1] in current:
            current[parts[-1]] = "[REDACTED]"


def build_trajectory_records(
    input_batch: GeneratorInput,
    output: GeneratorOutput,
    config: TrajectoryRetentionConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    generator_name: str,
) -> list[TrajectoryRecord]:
    """Build generator-neutral records from one normalized batch."""
    metadata = input_batch.get("batch_metadata")
    if metadata is None:
        raise ValueError("trajectory retention requires batch metadata")
    stop_reasons = output.get("stop_reasons") or [None] * len(output["response_ids"])
    unshaped = output.get("unshaped_rewards")
    components = output.get("reward_shaping_components")
    loop_spans = output.get("reward_shaping_loop_spans")
    shaping_versions = output.get("reward_shaping_versions")
    model_version_step = output.get("actual_global_step")
    if model_version_step is None:
        model_version_step = max(0, metadata.global_step - 1)

    records = []
    for group in _group_rows(input_batch, output):
        trajectory_id = group.trajectory_id
        row_indices = group.row_indices
        prompt_index = group.prompt_index
        final_index = row_indices[-1]
        response_ids = [token for index in row_indices for token in output["response_ids"][index]]
        loss_mask = [value for index in row_indices for value in output["loss_masks"][index]]
        prompt_ids = output["prompt_token_ids"][row_indices[0]]
        normalized_reward = NormalizedReward.from_output(output["rewards"][final_index])
        shaped_reward = normalized_reward.total
        outcome = (
            float(unshaped[final_index]) if unshaped is not None else normalized_reward.outcome
        )
        response_text = _decode(tokenizer, response_ids)
        record = {
            "schema_version": config.schema_version,
            "record_id": "",
            "run_id": config.run_id,
            "global_step": metadata.global_step,
            "phase": metadata.training_phase,
            "trajectory": {
                "instance_id": trajectory_id.instance_id,
                "repetition_id": trajectory_id.repetition_id,
                "row_indices": row_indices,
                "environment_class": input_batch["env_classes"][prompt_index],
                "environment_extras": _json_safe(input_batch["env_extras"][prompt_index]),
            },
            "prompt": {
                "messages": _json_safe(input_batch["prompts"][prompt_index]),
                "token_ids": list(prompt_ids),
            },
            "response": {
                "messages": None if response_text is None else [{"role": "assistant", "content": response_text}],
                "text": response_text,
                "token_ids": response_ids,
                "loss_mask": loss_mask,
                "trainable_spans": _active_spans(loss_mask),
                "step_boundaries": _step_boundaries(output, row_indices, stop_reasons),
                "stop_reason": stop_reasons[final_index],
                "generation_limit": (input_batch.get("sampling_params") or {}).get(
                    "max_tokens", (input_batch.get("sampling_params") or {}).get("max_generate_length")
                ),
                "loop_spans": [] if loop_spans is None else loop_spans[final_index],
            },
            "reward": {
                "outcome": outcome,
                "shaped": shaped_reward,
                "components": None if components is None else components[final_index],
            },
            "metrics": _json_safe(output.get("rollout_metrics") or {}),
            "provenance": {
                "generator": generator_name,
                "inference_backend": config.inference_backend,
                "model_path": config.model_path,
                "model_source_identity": config.model_source_identity,
                "resume_path": config.resume_path,
                "model_version_step": model_version_step,
                "sampling": _json_safe(input_batch.get("sampling_params") or {}),
                "reward_shaping_schema_version": (
                    None if shaping_versions is None else shaping_versions[final_index]
                ),
            },
        }
        _redact(record, config.redact_fields)
        digest_source = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        record["record_id"] = hashlib.sha256(digest_source).hexdigest()
        records.append(TrajectoryRecord.from_json(record))
    return records


def _empty_metrics() -> dict[str, float]:
    return {
        f"{RETENTION_METRIC_PREFIX}/candidates": 0.0,
        f"{RETENTION_METRIC_PREFIX}/selected": 0.0,
        f"{RETENTION_METRIC_PREFIX}/written": 0.0,
        f"{RETENTION_METRIC_PREFIX}/duplicates": 0.0,
        f"{RETENTION_METRIC_PREFIX}/dropped_by_bounds": 0.0,
        f"{RETENTION_METRIC_PREFIX}/write_errors": 0.0,
        f"{RETENTION_METRIC_PREFIX}/bytes_written": 0.0,
    }


class TrajectorySink:
    """Select, bound, and persist normalized trajectory records."""

    def __init__(
        self,
        config: TrajectoryRetentionConfig,
        tokenizer: PreTrainedTokenizerBase,
        writer: TrajectoryWriter | None = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.writer = writer or _FilesystemTrajectoryWriter(config.output_path)
        self._lock = threading.Lock()
        self._ledger: _RetentionLedger | None = None

    def retain(
        self,
        input_batch: GeneratorInput,
        output: GeneratorOutput,
        *,
        generator_name: str,
    ) -> dict[str, float]:
        if not self.config.enabled:
            return {}
        metadata = input_batch.get("batch_metadata")
        if metadata is None or metadata.training_phase not in self.config.phases:
            return {}

        with self._lock:
            return self._retain_locked(input_batch, output, generator_name)

    def _retain_locked(
        self,
        input_batch: GeneratorInput,
        output: GeneratorOutput,
        generator_name: str,
    ) -> dict[str, float]:
        metrics = _empty_metrics()
        records = build_trajectory_records(
            input_batch,
            output,
            self.config,
            self.tokenizer,
            generator_name=generator_name,
        )
        metrics[f"{RETENTION_METRIC_PREFIX}/candidates"] = float(len(records))
        try:
            ledger = self._load_ledger()
        except Exception as error:
            if self.config.required:
                raise
            logger.error("Failed to load trajectory retention ledger: {}", error)
            metrics[f"{RETENTION_METRIC_PREFIX}/write_errors"] += float(len(records))
            return metrics

        ledger_changed = False
        for record in sorted(records, key=self._sample_score):
            ledger_changed |= self._retain_record(record, ledger, metrics)

        if ledger_changed:
            try:
                self.writer.write_json(_LEDGER_NAME, ledger.to_json())
            except Exception as error:
                if self.config.required:
                    raise
                logger.error("Failed to update trajectory retention ledger: {}", error)
                metrics[f"{RETENTION_METRIC_PREFIX}/write_errors"] += 1.0
        return metrics

    def _retain_record(
        self,
        record: TrajectoryRecord,
        ledger: _RetentionLedger,
        metrics: dict[str, float],
    ) -> bool:
        record_id = record.record_id
        if record_id in ledger.records:
            metrics[f"{RETENTION_METRIC_PREFIX}/duplicates"] += 1.0
            return False

        selection = self._selection(record, ledger)
        if selection is None:
            return False
        metrics[f"{RETENTION_METRIC_PREFIX}/selected"] += 1.0
        payload = self._encode_record(record)
        relative_path = self._record_path(record)
        try:
            return self._write_selected_record(record, payload, relative_path, selection, ledger, metrics)
        except Exception as error:
            if self.config.required:
                raise
            logger.error("Failed to retain trajectory {}: {}", record_id, error)
            metrics[f"{RETENTION_METRIC_PREFIX}/write_errors"] += 1.0
            return False

    def _write_selected_record(
        self,
        record: TrajectoryRecord,
        payload: bytes,
        relative_path: str,
        selection: _Selection,
        ledger: _RetentionLedger,
        metrics: dict[str, float],
    ) -> bool:
        if self.writer.exists(relative_path):
            metrics[f"{RETENTION_METRIC_PREFIX}/duplicates"] += 1.0
            return False

        step_key = self._step_key(record)
        step_bytes = ledger.step_bytes.get(step_key, 0)
        displaced = None if selection.displaced_id is None else ledger.records[selection.displaced_id]
        replaced_bytes = displaced.bytes if displaced is not None and displaced.reasons == (_SELECTION_COUNT,) else 0
        retained_bytes = len(payload) - replaced_bytes
        if (
            step_bytes + retained_bytes > self.config.max_bytes_per_step
            or ledger.total_bytes + retained_bytes > self.config.max_bytes_per_run
        ):
            metrics[f"{RETENTION_METRIC_PREFIX}/dropped_by_bounds"] += 1.0
            return False

        self.writer.write_bytes(relative_path, payload)
        if displaced is not None:
            self._displace_count_sample(ledger, selection.displaced_id, displaced, relative_path)

        ledger.records[record.record_id] = _LedgerEntry(
            path=relative_path,
            bytes=len(payload),
            step=step_key,
            reasons=selection.reasons,
            sample_score=self._sample_score(record),
        )
        ledger.step_bytes[step_key] = step_bytes + retained_bytes
        ledger.total_bytes += retained_bytes
        metrics[f"{RETENTION_METRIC_PREFIX}/written"] += 1.0
        metrics[f"{RETENTION_METRIC_PREFIX}/bytes_written"] += float(len(payload))
        return True

    @staticmethod
    def _encode_record(record: TrajectoryRecord) -> bytes:
        return gzip.compress(
            json.dumps(record.to_json(), sort_keys=True, separators=(",", ":")).encode("utf-8"),
            mtime=0,
        )

    def _load_ledger(self) -> _RetentionLedger:
        if self._ledger is None:
            self._ledger = _RetentionLedger.from_json(self.writer.read_json(_LEDGER_NAME))
        return self._ledger

    def _selection(self, record: TrajectoryRecord, ledger: _RetentionLedger) -> _Selection | None:
        reasons = []
        if self._is_mandatory(record):
            return _Selection((_SELECTION_MANDATORY,))

        score = self._sample_score(record)
        if score / (1 << 256) < self.config.sample_fraction:
            reasons.append(_SELECTION_FRACTION)

        displaced_id = None
        count_samples = self._count_samples(ledger, self._step_key(record))
        if self.config.sample_count_per_step > 0:
            if len(count_samples) < self.config.sample_count_per_step:
                reasons.append(_SELECTION_COUNT)
            else:
                largest_id, largest_entry = max(count_samples, key=lambda item: item[1].sample_score)
                if score < largest_entry.sample_score:
                    reasons.append(_SELECTION_COUNT)
                    displaced_id = largest_id

        return _Selection(tuple(reasons), displaced_id) if reasons else None

    @staticmethod
    def _count_samples(ledger: _RetentionLedger, step_key: str) -> list[tuple[str, _LedgerEntry]]:
        return [
            (record_id, entry)
            for record_id, entry in ledger.records.items()
            if entry.step == step_key and _SELECTION_COUNT in entry.reasons
        ]

    def _displace_count_sample(
        self,
        ledger: _RetentionLedger,
        displaced_id: str,
        displaced: _LedgerEntry,
        replacement_path: str,
    ) -> None:
        remaining_reasons = tuple(reason for reason in displaced.reasons if reason != _SELECTION_COUNT)
        if remaining_reasons:
            ledger.records[displaced_id] = _LedgerEntry(
                path=displaced.path,
                bytes=displaced.bytes,
                step=displaced.step,
                reasons=remaining_reasons,
                sample_score=displaced.sample_score,
            )
            return

        try:
            self.writer.remove(displaced.path)
        except Exception:
            try:
                self.writer.remove(replacement_path)
            except Exception as rollback_error:
                logger.error(
                    "Failed to remove replacement {} while rolling back trajectory retention: {}",
                    replacement_path,
                    rollback_error,
                )
            raise
        del ledger.records[displaced_id]

    def _is_mandatory(self, record: TrajectoryRecord) -> bool:
        outcome = record.outcome
        stop_reason = record.stop_reason
        normalized_stop = None if stop_reason is None else str(stop_reason).strip().lower()
        return any(
            (
                self.config.always_retain_failures and outcome <= 0.0,
                self.config.always_retain_non_terminating
                and normalized_stop not in self.config.accepted_stop_reasons,
                self.config.always_retain_loops and bool(record.loop_spans),
                self.config.reward_below is not None and outcome <= self.config.reward_below,
                self.config.reward_above is not None and outcome >= self.config.reward_above,
            )
        )

    @staticmethod
    def _sample_key(record: TrajectoryRecord) -> str:
        return ":".join(
            (
                record.run_id,
                record.phase,
                str(record.global_step),
                record.instance_id,
                str(record.repetition_id),
            )
        )

    @classmethod
    def _sample_score(cls, record: TrajectoryRecord) -> int:
        return int(hashlib.sha256(cls._sample_key(record).encode("utf-8")).hexdigest(), 16)

    @staticmethod
    def _step_key(record: TrajectoryRecord) -> str:
        return f"{record.phase}:{record.global_step}"

    @staticmethod
    def _record_path(record: TrajectoryRecord) -> str:
        safe_instance = hashlib.sha256(record.instance_id.encode("utf-8")).hexdigest()[:12]
        return posixpath.join(
            f"schema_v{record.schema_version}",
            f"phase={record.phase}",
            f"step={record.global_step:08d}",
            f"{safe_instance}-r{record.repetition_id}-{record.record_id}.json.gz",
        )


async def retain_trajectories(
    sink: TrajectorySink,
    input_batch: GeneratorInput,
    output: GeneratorOutput,
    *,
    generator_name: str,
) -> None:
    """Persist normalized trajectories without blocking the trainer event loop."""
    metrics = await asyncio.to_thread(
        sink.retain,
        input_batch,
        output,
        generator_name=generator_name,
    )
    if not metrics:
        return
    rollout_metrics = output.get("rollout_metrics") or {}
    rollout_metrics.update(metrics)
    output["rollout_metrics"] = rollout_metrics


def make_trajectory_sink(config: DictConfig, tokenizer: PreTrainedTokenizerBase) -> TrajectorySink:
    """Build the shared sink from the generator configuration."""
    return TrajectorySink(parse_trajectory_retention_config(config.get("trajectory_retention")), tokenizer)
