"""ShortConv values and derivatives against independent per-document convolutions."""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from skyrl_train.models.grug_shortconv import causal_short_conv


@pytest.mark.parametrize("lengths", [(9,), (1, 2, 6), (3, 3, 3)])
@pytest.mark.parametrize("width", [1, 4, 12])
def test_shortconv_packed_values_and_gradients(lengths, width):
    torch.manual_seed(5)
    x = torch.randn(sum(lengths), 2, 3, requires_grad=True)
    w = torch.randn(width, 3, requires_grad=True)
    reference = []
    start = 0
    # Independent channel-by-channel PyTorch convolution, restarted per document.
    for length in lengths:
        doc = x[start : start + length].permute(1, 2, 0)
        reference.append(
            torch.cat(
                [F.conv1d(F.pad(doc[:, c : c + 1], (width - 1, 0)), w[:, c].flip(0).view(1, 1, -1)) for c in range(3)],
                dim=1,
            ).permute(2, 0, 1)
        )
        start += length
    expected = torch.cat(reference)
    actual = causal_short_conv(x, w, lengths)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, (x, w), upstream, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (x, w), upstream)
    for a, e in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, e)


def _cp_conv_worker(rank, world_size, rendezvous, output_dir, lengths):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(11)
        full = torch.randn(sum(lengths), 1, 3)
        weight = torch.randn(4, 3, requires_grad=True)
        selected = []
        offset = 0
        for length in lengths:
            chunks = torch.arange(offset, offset + length).chunk(2 * world_size)
            selected.extend((chunks[rank], chunks[2 * world_size - rank - 1]))
            offset += length
        indices = torch.cat(selected)
        local = full[indices].detach().requires_grad_()
        actual = causal_short_conv(local, weight, lengths, dist.group.WORLD)
        actual.square().sum().backward()
        dist.all_reduce(weight.grad)
        torch.save(
            {"indices": indices, "output": actual.detach(), "dx": local.grad, "dw": weight.grad},
            output_dir / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("lengths", [(24,), (4, 8, 12)])
def test_shortconv_cp_halos_match_unsharded_gradients(tmp_path, lengths):
    torch.multiprocessing.spawn(_cp_conv_worker, args=(2, str(tmp_path / "rendezvous"), tmp_path, lengths), nprocs=2)
    torch.manual_seed(11)
    x = torch.randn(sum(lengths), 1, 3, requires_grad=True)
    weight = torch.randn(4, 3, requires_grad=True)
    expected = causal_short_conv(x, weight, lengths)
    expected.square().sum().backward()
    for rank in range(2):
        actual = torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        indices = actual["indices"]
        torch.testing.assert_close(actual["output"], expected[indices])
        torch.testing.assert_close(actual["dx"], x.grad[indices])
        torch.testing.assert_close(actual["dw"], weight.grad)
