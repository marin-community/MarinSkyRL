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


class _WeightBackend(_Backend):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.request = None

    async def update_named_weights(self, request):
        self.request = request
        self.started.set()
        await self.release.wait()


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


@pytest.mark.asyncio
async def test_weight_bridge_arms_receive_before_waiting_for_load() -> None:
    backend = _WeightBackend()
    set_global_state(backend, None)
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/weights/update",
            json={
                "names": ["model.embed_tokens.weight"],
                "dtypes": ["bfloat16"],
                "shapes": [[16, 8]],
                "extras": [],
                "packed": False,
            },
        )
        assert response.json() == {"status": "receiving"}
        await asyncio.wait_for(backend.started.wait(), timeout=1)
        assert backend.request == {
            "names": ["model.embed_tokens.weight"],
            "dtypes": ["bfloat16"],
            "shapes": [[16, 8]],
            "extras": [],
            "packed": False,
        }

        wait_task = asyncio.create_task(client.post("/weights/wait"))
        await asyncio.sleep(0)
        assert not wait_task.done()
        backend.release.set()
        assert (await wait_task).json() == {"status": "loaded"}
