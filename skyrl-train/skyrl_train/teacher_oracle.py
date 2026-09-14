"""Transport-neutral teacher scoring and lifecycle ownership."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, Self

import torch

from marinskyrl.distillation import TeacherEvidenceKind
from skyrl_train.distillation import (
    TeacherEvidenceBatch,
    TeacherScoreRequest,
    validate_teacher_evidence,
    validate_teacher_score_request,
)


@dataclass(frozen=True)
class TeacherCapabilities:
    """Static scoring contract advertised by one logical teacher."""

    teacher_id: str
    teacher_revision: str
    tokenizer_fingerprint: str
    evidence_kinds: frozenset[TeacherEvidenceKind]
    max_sequence_length: int
    supports_prompt_token_scoring: bool
    max_concurrency: int

    def __post_init__(self) -> None:
        for field_name in ("teacher_id", "teacher_revision", "tokenizer_fingerprint"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"teacher capabilities {field_name} must be a non-empty string")
        if not self.evidence_kinds:
            raise ValueError("teacher capabilities must advertise at least one evidence kind")
        if self.max_sequence_length <= 0:
            raise ValueError("teacher capabilities max_sequence_length must be positive")
        if self.max_concurrency <= 0:
            raise ValueError("teacher capabilities max_concurrency must be positive")


class TeacherOracle(Protocol):
    """Score exact student-token sequences without prescribing deployment."""

    capabilities: TeacherCapabilities

    async def score(self, request: TeacherScoreRequest) -> TeacherEvidenceBatch: ...

    async def close(self) -> None: ...


def validate_teacher_request_against_capabilities(
    request: TeacherScoreRequest, capabilities: TeacherCapabilities
) -> None:
    """Validate a request against an oracle before performing remote work."""
    validate_teacher_score_request(request)
    if capabilities.teacher_id != request.teacher_id:
        raise ValueError("teacher capability identity does not match the score request")
    if capabilities.tokenizer_fingerprint != request.tokenizer_fingerprint:
        raise ValueError("teacher and student tokenizer fingerprints must match for vocabulary-level scoring")
    if request.evidence not in capabilities.evidence_kinds:
        raise ValueError(f"teacher {request.teacher_id!r} cannot provide {request.evidence.value} evidence")
    if not capabilities.supports_prompt_token_scoring:
        raise ValueError(f"teacher {request.teacher_id!r} cannot score prompt tokens")

    sequence_lengths = request.prompt_mask.sum(dim=1) + request.response_mask.sum(dim=1)
    if torch.any(sequence_lengths > capabilities.max_sequence_length):
        raise ValueError(
            f"teacher {request.teacher_id!r} maximum sequence length is {capabilities.max_sequence_length}"
        )


class ValidatedTeacherOracle:
    """Enforce capabilities, provenance, concurrency, and idempotent closure."""

    def __init__(self, oracle: TeacherOracle) -> None:
        self.capabilities = oracle.capabilities
        self._oracle = oracle
        self._capacity = asyncio.Semaphore(self.capabilities.max_concurrency)
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._in_flight: set[asyncio.Task] = set()

    async def score(self, request: TeacherScoreRequest) -> TeacherEvidenceBatch:
        if self._closed:
            raise RuntimeError("teacher oracle is closed")
        validate_teacher_request_against_capabilities(request, self.capabilities)
        task = asyncio.current_task()
        assert task is not None
        self._in_flight.add(task)
        try:
            async with self._capacity:
                evidence = await self._oracle.score(request)
        finally:
            self._in_flight.remove(task)
        validate_teacher_evidence(request, evidence)
        if evidence.teacher_revision != self.capabilities.teacher_revision:
            raise ValueError("teacher evidence does not match the negotiated teacher revision")
        return evidence

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            if self._in_flight:
                await asyncio.gather(*tuple(self._in_flight), return_exceptions=True)
            await self._oracle.close()


TeacherOracleFactory = Callable[[], Awaitable[TeacherOracle]]


class TeacherOracleOwner:
    """Own a set of logical oracles, including partial-startup cleanup."""

    def __init__(self, oracles: dict[str, ValidatedTeacherOracle]) -> None:
        self._oracles = oracles
        self._close_lock = asyncio.Lock()
        self._closed = False

    @classmethod
    async def create(cls, factories: Mapping[str, TeacherOracleFactory]) -> Self:
        if not factories:
            raise ValueError("teacher oracle owner requires at least one factory")
        if any(not teacher_id.strip() for teacher_id in factories):
            raise ValueError("teacher oracle factory names must be non-empty strings")

        oracles: dict[str, ValidatedTeacherOracle] = {}
        try:
            for teacher_id, factory in factories.items():
                oracle = ValidatedTeacherOracle(await factory())
                oracles[teacher_id] = oracle
                if oracle.capabilities.teacher_id != teacher_id:
                    raise ValueError(
                        f"teacher oracle factory {teacher_id!r} returned {oracle.capabilities.teacher_id!r}"
                    )
        except BaseException as startup_error:
            cleanup_errors = await _close_oracles(oracles.values())
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "teacher oracle startup and cleanup failed",
                    [startup_error, *cleanup_errors],
                )
            raise
        return cls(oracles)

    async def score(self, teacher_id: str, request: TeacherScoreRequest) -> TeacherEvidenceBatch:
        if self._closed:
            raise RuntimeError("teacher oracle owner is closed")
        try:
            oracle = self._oracles[teacher_id]
        except KeyError as error:
            raise ValueError(f"unknown teacher oracle {teacher_id!r}") from error
        return await oracle.score(request)

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            cleanup_errors = await _close_oracles(self._oracles.values())
            if cleanup_errors:
                raise BaseExceptionGroup("teacher oracle cleanup failed", cleanup_errors)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()


async def _close_oracles(oracles: Iterable[ValidatedTeacherOracle]) -> list[BaseException]:
    errors: list[BaseException] = []
    for oracle in reversed(tuple(oracles)):
        try:
            await oracle.close()
        except BaseException as error:
            errors.append(error)
    return errors
