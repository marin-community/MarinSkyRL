import pytest
import torch

from skyrl_train.mismatch_harness.numerics import ulp_distance

BF16_SMALLEST_SUBNORMAL = 2.0**-133
FP32_SMALLEST_SUBNORMAL = 2.0**-149


@pytest.mark.parametrize(
    ("dtype", "left", "right", "expected"),
    [
        (torch.bfloat16, 1.0, 1.0 + 2.0**-7, 1),
        (torch.bfloat16, -1.0, -(1.0 + 2.0**-7), 1),
        (torch.bfloat16, -0.0, 0.0, 0),
        (torch.bfloat16, BF16_SMALLEST_SUBNORMAL, -BF16_SMALLEST_SUBNORMAL, 2),
        (torch.bfloat16, 2.0**-126, -(2.0**-126), 2 * 2**7),
        (torch.float32, 1.0, 1.0 + 2.0**-23, 1),
        (torch.float32, FP32_SMALLEST_SUBNORMAL, -FP32_SMALLEST_SUBNORMAL, 2),
    ],
)
def test_ulp_distance_counts_representable_values_between_signs(dtype, left, right, expected):
    a = torch.tensor([left], dtype=dtype)
    b = torch.tensor([right], dtype=dtype)
    assert ulp_distance(a, b).item() == expected
    assert ulp_distance(b, a).item() == expected
