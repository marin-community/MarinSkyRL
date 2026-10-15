import asyncio

import httpx
import pytest
from fastapi import FastAPI

from skyrl_train.rollouts.agent_service import serve_agent_app


@pytest.mark.asyncio
async def test_agent_http_service_uses_engine_loop_and_closes_listener():
    app = FastAPI()
    engine_loop = asyncio.get_running_loop()
    completed = asyncio.Event()

    @app.post("/complete")
    async def complete():
        assert asyncio.get_running_loop() is engine_loop
        completed.set()
        return {"tokens": [12]}

    async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
        async with serve_agent_app(app, host="127.0.0.1", startup_timeout=5, shutdown_timeout=5) as port:
            url = f"http://127.0.0.1:{port}/complete"
            response = await client.post(url)
            assert response.json() == {"tokens": [12]}
            assert completed.is_set()
        with pytest.raises(httpx.ConnectError):
            await client.post(url)
