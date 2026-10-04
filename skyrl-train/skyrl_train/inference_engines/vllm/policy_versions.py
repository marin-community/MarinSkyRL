"""Record generating weights before vLLM merges output chunks.

The pinned multiprocess vLLM client receives its pause acknowledgement after
the old outputs on the same FIFO socket. A marker in its output queue then
waits for the frontend to process those outputs. This barrier prevents delayed
old chunks from acquiring the version installed while generation is paused.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class TokenVersionTrace:
    token_ids: list[int] = field(default_factory=list)
    versions: list[int] = field(default_factory=list)


class PolicyVersionRecorder:
    """Observe accepted engine tokens; leave vLLM's response assembly intact."""

    def __init__(self, engine):
        self.engine = engine
        self.version: int | None = None
        self.traces: dict[tuple[str, int], TokenVersionTrace] = {}
        self._markers: dict[int, asyncio.Future] = {}
        core = engine.engine_core
        if not isinstance(getattr(core, "outputs_queue", None), asyncio.Queue):
            raise ValueError("token policy versions require the pinned multiprocess vLLM output queue")
        original_get = core.get_output_async
        original_process = engine.output_processor.process_outputs

        async def get_output():
            while True:
                output = await original_get()
                marker = self._markers.pop(id(output), None)
                if marker is None:
                    return output
                if not marker.done():
                    marker.set_result(None)

        def process_outputs(outputs, *args, **kwargs):
            for output in outputs:
                state = engine.output_processor.request_states.get(output.request_id)
                if state is None or not output.new_token_ids:
                    continue
                if self.version is None:
                    raise RuntimeError("generation began before its policy version was installed")
                key = (state.external_req_id, state.request_index)
                trace = self.traces.setdefault(key, TokenVersionTrace())
                trace.token_ids.extend(output.new_token_ids)
                trace.versions.extend([self.version] * len(output.new_token_ids))
            return original_process(outputs, *args, **kwargs)

        core.get_output_async = get_output
        engine.output_processor.process_outputs = process_outputs

    async def install(self, version: int) -> None:
        """Publish a completed trainer step while the engine cannot generate."""
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError("policy version must be a nonnegative completed trainer step")
        if self.version is not None and version < self.version:
            raise ValueError("policy versions cannot move backward")
        if self.engine.output_handler is not None:
            # pause_scheduler's acknowledgement precedes this marker. The output
            # handler reaches it only after processing every earlier token chunk.
            marker = object()
            done = asyncio.get_running_loop().create_future()
            self._markers[id(marker)] = done
            self.engine.engine_core.outputs_queue.put_nowait(marker)
            try:
                await asyncio.wait_for(done, timeout=30)
            finally:
                self._markers.pop(id(marker), None)
        self.version = version

    def discard(self, request_ids: list[str]) -> None:
        """Release traces for aborted requests whose responses will never return."""
        requested = set(request_ids)
        for key in list(self.traces):
            if key[0] in requested:
                del self.traces[key]

    def take(self, request_id: str, index: int, token_ids: list[int]) -> list[int]:
        """Return exact aligned versions once, allowing a trimmed stop suffix."""
        trace = self.traces.pop((request_id, index), None)
        if trace is None:
            if not token_ids:
                return []
            raise ValueError("vLLM response has no generating-policy trace")
        if trace.token_ids[: len(token_ids)] != token_ids:
            raise ValueError("vLLM policy-version trace does not match returned token IDs")
        return trace.versions[: len(token_ids)]
