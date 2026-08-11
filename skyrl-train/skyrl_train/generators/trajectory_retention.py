from collections import OrderedDict
from collections.abc import Mapping, Sequence
import asyncio
from dataclasses import dataclass, replace
import gzip
import hashlib
import json
import posixpath
import threading
from typing import Any, Protocol

from omegaconf import DictConfig
from loguru import logger
from transformers import PreTrainedTokenizerBase

from marinskyrl.resource_locator import join_resource_path
from skyrl_train.generators.generator_types import (
    GeneratorInput,
    GeneratorOutput,
    RewardShapingComponents,
    RewardShapingLoopSpan,
    TrajectoryID,
)
from skyrl_train.generators.trajectory_retention_config import (
    TrajectoryRetentionConfig,
    parse_trajectory_retention_config,
)
from skyrl_train.generators.trajectory_reward_shaping import NormalizedReward
from skyrl_train.json_serialization import canonical_json_bytes, to_jsonable
from skyrl_train.utils.io import io


RETENTION_METRIC_PREFIX = "generate/trajectory_retention"
RETENTION_SCHEMA_VERSION = 1
_LEDGER_NAME = "_retention_ledger.json"
_SELECTION_COUNT = "count"
_SELECTION_FRACTION = "fraction"
_SELECTION_MANDATORY = "mandatory"


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
                str(record_id): _LedgerEntry.from_json(entry) for record_id, entry in value.get("records", {}).items()
            },
        )


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


@dataclass(frozen=True)
class _PromptTrace:
    messages: Any
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class _TraceMessage:
    role: str
    content: str


@dataclass(frozen=True)
class _StepBoundary:
    row_index: int
    token_start: int
    token_end: int
    stop_reason: str | None
    is_last_step: bool | None


@dataclass(frozen=True)
class _ResponseTrace:
    messages: tuple[_TraceMessage, ...] | None
    text: str | None
    token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    trainable_spans: tuple[RewardShapingLoopSpan, ...]
    step_boundaries: tuple[_StepBoundary, ...]
    stop_reason: str | None
    generation_limit: int | None
    loop_spans: tuple[RewardShapingLoopSpan, ...]


@dataclass(frozen=True)
class _RewardTrace:
    outcome: float
    shaped: float
    components: RewardShapingComponents | None


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

    def to_json(self) -> dict[str, Any]:
        return to_jsonable(self)


class TrajectoryWriter(Protocol):
    def exists(self, relative_path: str) -> bool: ...

    def read_json(self, relative_path: str) -> dict[str, Any] | None: ...

    def write_bytes(self, relative_path: str, payload: bytes) -> None: ...

    def write_json(self, relative_path: str, value: dict[str, Any]) -> None: ...

    def remove(self, relative_path: str) -> None: ...


class _FilesystemTrajectoryWriter:
    def __init__(self, output_path: str):
        self.output_path = output_path

    def _path(self, relative_path: str) -> str:
        return join_resource_path(self.output_path, relative_path)

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
        self._write(relative_path, canonical_json_bytes(value))

    def remove(self, relative_path: str) -> None:
        io.remove(self._path(relative_path))

    def _write(self, relative_path: str, payload: bytes) -> None:
        io.write_bytes_atomic(self._path(relative_path), payload)


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


def _active_spans(loss_mask: Sequence[int]) -> list[RewardShapingLoopSpan]:
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
) -> list[_StepBoundary]:
    boundaries = []
    token_start = 0
    final_steps = output.get("is_last_step")
    for row_index in row_indices:
        token_end = token_start + len(output["response_ids"][row_index])
        boundaries.append(
            _StepBoundary(
                row_index=row_index,
                token_start=token_start,
                token_end=token_end,
                stop_reason=stop_reasons[row_index],
                is_last_step=None if final_steps is None else final_steps[row_index],
            )
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


def _serialized_record(record: TrajectoryRecord, redact_fields: Sequence[str]) -> dict[str, Any]:
    serialized = record.to_json()
    _redact(serialized, redact_fields)
    return serialized


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
        outcome = float(unshaped[final_index]) if unshaped is not None else normalized_reward.outcome
        response_text = _decode(tokenizer, response_ids)
        record = TrajectoryRecord(
            schema_version=RETENTION_SCHEMA_VERSION,
            record_id="",
            run_id=config.run_id,
            global_step=metadata.global_step,
            phase=metadata.training_phase,
            trajectory=_TrajectoryIdentity(
                instance_id=trajectory_id.instance_id,
                repetition_id=trajectory_id.repetition_id,
                row_indices=row_indices,
                environment_class=input_batch["env_classes"][prompt_index],
                environment_extras=to_jsonable(input_batch["env_extras"][prompt_index]),
            ),
            prompt=_PromptTrace(
                messages=to_jsonable(input_batch["prompts"][prompt_index]),
                token_ids=tuple(prompt_ids),
            ),
            response=_ResponseTrace(
                messages=None if response_text is None else (_TraceMessage(role="assistant", content=response_text),),
                text=response_text,
                token_ids=tuple(response_ids),
                loss_mask=tuple(loss_mask),
                trainable_spans=tuple(_active_spans(loss_mask)),
                step_boundaries=tuple(_step_boundaries(output, row_indices, stop_reasons)),
                stop_reason=stop_reasons[final_index],
                generation_limit=(input_batch.get("sampling_params") or {}).get(
                    "max_tokens", (input_batch.get("sampling_params") or {}).get("max_generate_length")
                ),
                loop_spans=tuple(() if loop_spans is None else loop_spans[final_index]),
            ),
            reward=_RewardTrace(
                outcome=outcome,
                shaped=shaped_reward,
                components=None if components is None else components[final_index],
            ),
            metrics=to_jsonable(output.get("rollout_metrics") or {}),
            provenance=_ProvenanceTrace(
                generator=generator_name,
                inference_backend=config.inference_backend,
                model_path=config.model_path,
                model_source_identity=config.model_source_identity,
                resume_path=config.resume_path,
                model_version_step=model_version_step,
                sampling=to_jsonable(input_batch.get("sampling_params") or {}),
                reward_shaping_schema_version=None if shaping_versions is None else shaping_versions[final_index],
            ),
        )
        digest_source = canonical_json_bytes(_serialized_record(record, config.redact_fields))
        records.append(replace(record, record_id=hashlib.sha256(digest_source).hexdigest()))
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
        self._generator_name: str | None = None

    def bind_generator(self, generator_name: str) -> None:
        """Bind the generator identity once when the sink is attached."""
        if self._generator_name not in (None, generator_name):
            raise ValueError(f"trajectory sink is already bound to {self._generator_name}")
        self._generator_name = generator_name

    def retain(
        self,
        input_batch: GeneratorInput,
        output: GeneratorOutput,
    ) -> dict[str, float]:
        """Retain eligible records, or return no metrics when retention does not apply."""
        if not self.config.enabled:
            return {}
        metadata = input_batch.get("batch_metadata")
        if metadata is None or metadata.training_phase not in self.config.phases:
            return {}
        if self._generator_name is None:
            raise ValueError("trajectory sink must be bound to a generator before retention")

        with self._lock:
            return self._retain_locked(input_batch, output)

    def _retain_locked(
        self,
        input_batch: GeneratorInput,
        output: GeneratorOutput,
    ) -> dict[str, float]:
        metrics = _empty_metrics()
        records = build_trajectory_records(
            input_batch,
            output,
            self.config,
            self.tokenizer,
            generator_name=self._generator_name,
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
                self.writer.write_json(_LEDGER_NAME, to_jsonable(ledger))
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

    def _encode_record(self, record: TrajectoryRecord) -> bytes:
        return gzip.compress(
            canonical_json_bytes(_serialized_record(record, self.config.redact_fields)),
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
        outcome = record.reward.outcome
        stop_reason = record.response.stop_reason
        normalized_stop = None if stop_reason is None else str(stop_reason).strip().lower()
        return any(
            (
                self.config.always_retain_failures and outcome <= 0.0,
                self.config.always_retain_non_terminating and normalized_stop not in self.config.accepted_stop_reasons,
                self.config.always_retain_loops and bool(record.response.loop_spans),
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
                record.trajectory.instance_id,
                str(record.trajectory.repetition_id),
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
        safe_instance = hashlib.sha256(record.trajectory.instance_id.encode("utf-8")).hexdigest()[:12]
        return posixpath.join(
            f"schema_v{record.schema_version}",
            f"phase={record.phase}",
            f"step={record.global_step:08d}",
            f"{safe_instance}-r{record.trajectory.repetition_id}-{record.record_id}.json.gz",
        )


async def retain_trajectories(
    sink: TrajectorySink,
    input_batch: GeneratorInput,
    output: GeneratorOutput,
) -> None:
    """Persist normalized trajectories without blocking the trainer event loop."""
    metrics = await asyncio.to_thread(
        sink.retain,
        input_batch,
        output,
    )
    if not metrics:
        return
    rollout_metrics = output.get("rollout_metrics") or {}
    rollout_metrics.update(metrics)
    output["rollout_metrics"] = rollout_metrics


def make_trajectory_sink(config: DictConfig, tokenizer: PreTrainedTokenizerBase) -> TrajectorySink:
    """Build the shared sink from the generator configuration."""
    return TrajectorySink(parse_trajectory_retention_config(config.get("trajectory_retention")), tokenizer)
