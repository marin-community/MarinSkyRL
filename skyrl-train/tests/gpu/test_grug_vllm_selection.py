import torch
from vllm.model_executor.models.grugmoe import GrugMoeRouter

from skyrl_train.models.grug_vllm_kernels import vllm_topk_experts
from tests.gpu.grug_gpu_gates import require_hoppers

ROWS, EXPERTS, TOP_K = 512, 256, 4


def test_trainer_selects_vllms_experts_in_vllms_order_when_biased_logits_tie():
    require_hoppers(1)
    generator = torch.Generator().manual_seed(0)
    biased = torch.randn(ROWS, EXPERTS, generator=generator)
    ranked = biased.argsort(dim=-1, descending=True)
    # Half the rows tie their 4th and 5th largest biased logits exactly, the other half their two largest.
    boundary, inside = torch.arange(ROWS // 2), torch.arange(ROWS // 2, ROWS)
    biased[boundary, ranked[boundary, 4]] = biased[boundary, ranked[boundary, 3]]
    biased[inside, ranked[inside, 1]] = biased[inside, ranked[inside, 0]]
    biased = biased.cuda()
    # vLLM's router adds its bias to the logits it is given; a zero bias leaves these biased logits as they are.
    router = GrugMoeRouter(top_k=TOP_K, global_num_experts=EXPERTS, bias=torch.zeros(EXPERTS, device="cuda"))
    _, expected = router._compute_routing(None, biased, torch.int64)

    assert torch.equal(vllm_topk_experts(biased, TOP_K), expected)
    # A row's selection does not depend on the rows beside it: engine steps and trainer micro-batches hold other rows.
    for row in (0, ROWS // 2 - 1, ROWS // 2, ROWS - 1):
        assert torch.equal(vllm_topk_experts(biased[row : row + 1], TOP_K), expected[row : row + 1])
