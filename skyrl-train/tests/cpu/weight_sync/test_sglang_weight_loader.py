import pytest
import torch

from skyrl_train.inference_engines.sglang.ipc_utils import (
    IPC_REQUEST_END_MARKER,
    deserialize_ipc_request,
    serialize_ipc_request,
)


@pytest.mark.parametrize(
    "request_body",
    [
        pytest.param({"names": ["test"]}, id="minimal"),
        pytest.param(
            {
                "names": ["layer1.weight", "layer2.weight", "layer3.bias"],
                "dtypes": ["bfloat16", "bfloat16", "float32"],
                "shapes": [[4096, 4096], [4096, 1024], [1024]],
                "extras": [{"ipc_handles": {"gpu-0": f"handle{index}"}} for index in range(3)],
            },
            id="multiple-weights",
        ),
    ],
)
def test_ipc_request_roundtrip_is_4_byte_aligned(request_body):
    tensor = serialize_ipc_request(request_body)

    assert len(tensor) % 4 == 0
    assert deserialize_ipc_request(tensor) == request_body


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        pytest.param(bytes([1, 2, 3, 4]), "End marker not found", id="missing-end-marker"),
        pytest.param(b"not_valid_base64!!!" + IPC_REQUEST_END_MARKER, "Failed to deserialize", id="corrupt-body"),
    ],
)
def test_ipc_request_rejects_malformed_payload(payload, error):
    with pytest.raises(ValueError, match=error):
        deserialize_ipc_request(torch.frombuffer(bytearray(payload), dtype=torch.uint8))
