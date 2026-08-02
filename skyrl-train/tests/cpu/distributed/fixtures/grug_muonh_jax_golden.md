# Grug MuonH JAX golden

`grug_muonh_jax_golden.npz` comes from Marin
`61a4a8c81dc6d9171c02ea0490bc941bdf2de238`, using
`experiments/grug/moe/{optimizer,adamh}.py` and
`lib/levanter/src/levanter/optim/{grugmuon,util}.py`.

The fixture contains embeddings, dense attention, attention gates, routers,
rank-3 routed experts, shared experts, GatedNorms, one-dimensional norms and
biases, and the output head. It uses seed `20260730`, FP32 parameters and
gradients, three steps, MuonH/AdamH LR `0.03`, Adam LR `0.004`, momentum
`0.95`, Nesterov, five BF16 Newton--Schulz steps, betas `(0.9, 0.95)`, and
epsilons `1e-8`. Final matrix axes are transposed when saved to match PyTorch
layout. Weight decay inputs are deliberately nonzero; the expected recipe
applies no decay.

The [full generator at the preserved validation commit](https://github.com/marin-community/MarinSkyRL/blob/0792437b14c712331102b39ca7e3e73d118e6d14/skyrl-train/tests/cpu/distributed/fixtures/grug_muonh_jax_golden.md)
includes the complete parameter tree, routing keys, state extraction, and
layout conversion. From the MarinSkyRL repository root, save that generator as
`/tmp/generate_grug_muonh_golden.py`, archive the pinned Marin revision to
`/tmp/marin-grug-muonh-61a4a8c`, then run:

```sh
cd skyrl-train/tests/cpu/distributed/fixtures
PYTHONPATH=/tmp/marin-grug-muonh-61a4a8c:/tmp/marin-grug-muonh-61a4a8c/lib/levanter/src \
  uv run --project /tmp/marin-grug-muonh-61a4a8c --no-sync \
  python /tmp/generate_grug_muonh_golden.py
sha256sum grug_muonh_jax_golden.npz
```

Expected SHA-256:
`57a66c2b0d36f1fbaffe1646b016457b7b92773fbc631aac318ceac45c9cb387`.
