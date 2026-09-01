import asyncio
import socket

import httpx
import pytest
import uvicorn

from skyrl_train.inference_engines.inference_engine_client_http_endpoint import create_app, set_global_state
from skyrl_train.inference_engines.vllm.stats import HTTPBridgeStatsAccumulator, IntervalReadMode


class _Backend:
    model_name = "test-model"

    async def chat_completion(self, _request):
        return {"choices": [{"message": {"content": "ok"}}]}

    async def completion(self, _request):
        return {"choices": [{"text": "ok"}]}

    async def chat_completion_stream(self, _request):
        yield "data: [DONE]\n\n"


async def _wait_until_started(server: uvicorn.Server) -> None:
    while not server.started:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_real_uvicorn_bridge_records_96_concurrent_requests():
    accumulator = HTTPBridgeStatsAccumulator()
    set_global_state(_Backend(), None)
    app = create_app(accumulator, event_loop_lag_interval_seconds=0.001)
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server_task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        await asyncio.wait_for(_wait_until_started(server), timeout=5)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
            responses = await asyncio.gather(
                *(
                    client.post(
                        "/v1/chat/completions",
                        json={"model": "test-model", "messages": [{"role": "user", "content": str(index)}]},
                    )
                    for index in range(96)
                )
            )
        assert all(response.status_code == 200 for response in responses)

        snapshot = accumulator.snapshot(IntervalReadMode.PEEK)
        assert snapshot.response_bytes.count == 96
        assert snapshot.json_serialization_seconds.count == 96
        assert snapshot.event_loop_lag_seconds.count > 0
    finally:
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=5)
        listener.close()
