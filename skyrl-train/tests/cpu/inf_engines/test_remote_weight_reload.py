import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from skyrl_train.inference_engines.remote_inference_engine import RemoteWeightLoader


@pytest.mark.asyncio
@pytest.mark.parametrize("native_publication", [False, True])
async def test_remote_bucket_keeps_publication_identity_and_broadcast_order(native_publication):
    publication = {"publication_id": "candidate-7", "model_version": 6} if native_publication else {}
    received = []

    async def begin(request):
        return web.json_response(publication or {"status": "ok"})

    async def receive(request):
        received.append(await request.json())
        return web.json_response({"status": "ok"})

    async def finish(request):
        assert await request.json() == publication
        assert [item["name"] for item in received] == ["router.weight", "experts.gate_proj.weight"]
        return web.json_response({"model_version": 7} if native_publication else {"status": "ok"})

    app = web.Application()
    app.router.add_post("/begin_weight_reload", begin)
    app.router.add_post("/update_weights", receive)
    app.router.add_post("/finish_weight_reload", finish)
    async with TestServer(app) as server:
        loader = RemoteWeightLoader(str(server.make_url("")).rstrip("/"), "vllm")
        await loader.begin_weight_reload()
        await loader.load_weights(
            {
                "names": ["router.weight", "experts.gate_proj.weight"],
                "dtypes": ["torch.float32", "torch.bfloat16"],
                "shapes": [[2, 4], [2, 8, 4]],
            }
        )
        result = await loader.finish_weight_reload()
    assert received == [
        {**publication, "name": "router.weight", "dtype": "torch.float32", "shape": [2, 4]},
        {**publication, "name": "experts.gate_proj.weight", "dtype": "torch.bfloat16", "shape": [2, 8, 4]},
    ]
    assert result == ({"model_version": 7} if native_publication else {"status": "ok"})


@pytest.mark.asyncio
async def test_remote_bucket_stops_after_rejected_tensor():
    received = []

    async def receive(request):
        tensor = await request.json()
        received.append(tensor["name"])
        if tensor["name"] == "bad-shape":
            raise web.HTTPConflict(text="tensor shape does not match")
        return web.json_response({"status": "ok"})

    app = web.Application()
    app.router.add_post("/update_weights", receive)
    async with TestServer(app) as server:
        loader = RemoteWeightLoader(str(server.make_url("")).rstrip("/"), "vllm")
        with pytest.raises(aiohttp.ClientResponseError) as error:
            await loader.load_weights(
                {
                    "names": ["first", "bad-shape", "unreceived"],
                    "dtypes": ["torch.float32"] * 3,
                    "shapes": [[2, 2]] * 3,
                }
            )
    assert error.value.status == 409
    assert received == ["first", "bad-shape"]
