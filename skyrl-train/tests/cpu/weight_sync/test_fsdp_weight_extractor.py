from collections import OrderedDict
from types import SimpleNamespace

import torch

from skyrl_train.workers.fsdp.fsdp_worker import _omit_tied_lm_head_weight


def test_tied_lm_head_is_omitted_from_weight_sync():
    params = OrderedDict(
        [
            ("model.embed_tokens.weight", torch.zeros(2, 2)),
            ("model.layers.0.weight", torch.ones(2, 2)),
            ("lm_head.weight", torch.zeros(2, 2)),
        ]
    )

    filtered = _omit_tied_lm_head_weight(params, SimpleNamespace(tie_word_embeddings=True))

    assert list(filtered) == ["model.embed_tokens.weight", "model.layers.0.weight"]


def test_untied_lm_head_is_preserved_for_weight_sync():
    params = OrderedDict([("lm_head.weight", torch.zeros(2, 2))])

    filtered = _omit_tied_lm_head_weight(params, SimpleNamespace(tie_word_embeddings=False))

    assert filtered is params
