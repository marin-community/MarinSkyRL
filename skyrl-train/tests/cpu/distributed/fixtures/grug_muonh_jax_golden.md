# Grug MuonH JAX golden

`grug_muonh_jax_golden.npz` comes from Marin
`61a4a8c81dc6d9171c02ea0490bc941bdf2de238`, using
`experiments/grug/moe/{optimizer,adamh}.py` and
`lib/levanter/src/levanter/optim/{grugmuon,util}.py`.

The fixture contains embeddings, dense attention, attention gates, routers,
rank-3 routed experts, shared experts, GatedNorms, one-dimensional norms and
biases, and the output head. The grouped-query attention query matrix has
shape `(12, 4)` and weights from `-0.1` to `0.1`; its fused test has two query
heads per key/value group. It uses seed `20260730`, FP32 parameters and
gradients, three steps, MuonH/AdamH LR `0.03`, Adam LR `0.004`, momentum
`0.95`, Nesterov, five BF16 Newton--Schulz steps, betas `(0.9, 0.95)`, and
epsilons `1e-8`. Final matrix axes are transposed when saved to match PyTorch
layout. The recipe applies no weight decay.

[`generate_grug_muonh_golden.py`](generate_grug_muonh_golden.py) contains the
complete parameter tree, routing keys, state extraction, and layout conversion.
The JAX reference receives nonzero weight decay to confirm that Marin's recipe
does not apply it. The MarinSkyRL runtime rejects nonzero weight decay so an
operator cannot mistake the ignored setting for active decay.

Archive the pinned Marin revision to `/tmp/marin-grug-muonh-61a4a8c`, then run:

```sh
cd skyrl-train/tests/cpu/distributed/fixtures
PYTHONPATH=/tmp/marin-grug-muonh-61a4a8c:/tmp/marin-grug-muonh-61a4a8c/lib/levanter/src \
  uv run --project /tmp/marin-grug-muonh-61a4a8c --no-sync \
  python generate_grug_muonh_golden.py
sha256sum grug_muonh_jax_golden.npz
```

Expected SHA-256:
`333b546ed2d878fa675d8d1976c1eccd8075ff2a4cda014a50f682af644b27f2`.
