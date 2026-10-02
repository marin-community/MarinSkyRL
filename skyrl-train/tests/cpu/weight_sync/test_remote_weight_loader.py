"""RemoteWeightLoader against a local HTTP server that records what each backend receives."""

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

from skyrl_train.inference_engines.remote_inference_engine import RemoteWeightLoader

INIT_COMMUNICATOR_PARAMS = {
    "master_address": "127.0.0.1",
    "master_port": 29500,
    "rank_offset": 1,
    "world_size": 2,
    "group_name": "test_group",
    "backend": "nccl",
    "override_existing": False,
}


@pytest_asyncio.fixture
async def engine_server():
    """Yield the server URL and the (path, JSON body) of every POST it receives."""
    received: list[tuple[str, dict | None]] = []

    async def record(request: web.Request) -> web.Response:
        body = await request.json() if request.can_read_body else None
        received.append((request.path, body))
        return web.json_response({"success": True})

    app = web.Application()
    app.router.add_post("/{endpoint}", record)
    async with TestServer(app) as server:
        yield str(server.make_url("")).rstrip("/"), received


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "init_path", "update_path"),
    [
        ("vllm", "/init_weight_update_communicator", "/update_weights"),
        ("sglang", "/init_weights_update_group", "/update_weights_from_distributed"),
    ],
)
async def test_remote_weight_loader_uses_backend_endpoints(engine_server, backend, init_path, update_path):
    url, received = engine_server
    loader = RemoteWeightLoader(url=url, engine_backend=backend)
    request = {"names": ["model.layer.weight"], "dtypes": ["bfloat16"], "shapes": [[4096, 4096]]}

    assert await loader.init_communicator(**INIT_COMMUNICATOR_PARAMS) == {"success": True}
    assert await loader.load_weights(request) == {"results": [{"success": True}]}
    assert await loader.destroy_group() == {"success": True}

    assert received == [
        (init_path, INIT_COMMUNICATOR_PARAMS),
        (update_path, {"name": "model.layer.weight", "dtype": "bfloat16", "shape": [4096, 4096]}),
        ("/destroy_weights_update_group", None),
    ]
