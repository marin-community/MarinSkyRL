import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from skyrl_train.inference_engines.remote_inference_engine import RemoteInferenceEngine, RemoteWeightLoader


@pytest.mark.asyncio
@pytest.mark.parametrize("native_publication", [False, True])
async def test_remote_bucket_keeps_publication_identity_and_broadcast_order(native_publication):
    publication = {"publication_id": "candidate-7", "model_version": 6} if native_publication else {}
    received = []

    async def begin(_request):
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


@pytest.mark.asyncio
@pytest.mark.parametrize("native_publication", [False, True])
async def test_remote_draft_checkpoint_preserves_publication_identity_and_order(native_publication):
    publication = {"publication_id": "draft-9", "model_version": 6} if native_publication else {}
    received = []
    weights_path = "/shared/online-eagle/step-12"

    async def start(_request):
        received.append(("start", None))
        return web.json_response(publication or {"status": "ok"})

    async def update(request):
        received.append(("update", await request.json()))
        return web.json_response({"status": "staged"})

    async def finish(request):
        received.append(("finish", await request.json()))
        return web.json_response({"model_version": 6} if native_publication else {"status": "ok"})

    app = web.Application()
    app.router.add_post("/start_draft_weight_update", start)
    app.router.add_post("/update_weights", update)
    app.router.add_post("/finish_weight_update", finish)
    async with TestServer(app) as server:
        engine = RemoteInferenceEngine(
            str(server.make_url("")).removeprefix("http://").rstrip("/"), "test", "vllm", None
        )
        result = await engine.update_draft_weights(weights_path)
    assert result == {"active": True}
    assert received == [
        ("start", None),
        ("update", {**publication, "update_info": {"weights_path": weights_path}}),
        ("finish", publication),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("rejected_phase", ["start", "update", "finish"])
async def test_remote_draft_checkpoint_propagates_rejection_without_publishing(rejected_phase):
    received = []
    publication = {"publication_id": "draft-9", "model_version": 6}

    async def handle(request):
        phase = {
            "/start_draft_weight_update": "start",
            "/update_weights": "update",
            "/finish_weight_update": "finish",
        }[request.path]
        received.append(phase)
        if phase == rejected_phase:
            raise web.HTTPConflict(text="draft publication rejected")
        return web.json_response(publication if phase == "start" else {"status": "ok"})

    app = web.Application()
    for path in ("/start_draft_weight_update", "/update_weights", "/finish_weight_update"):
        app.router.add_post(path, handle)
    async with TestServer(app) as server:
        engine = RemoteInferenceEngine(
            str(server.make_url("")).removeprefix("http://").rstrip("/"), "test", "vllm", None
        )
        with pytest.raises(aiohttp.ClientResponseError) as error:
            await engine.update_draft_weights("/shared/online-eagle/rejected")
    assert error.value.status == 409
    phases = ["start", "update", "finish"]
    assert received == phases[: phases.index(rejected_phase) + 1]
