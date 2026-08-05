🤖 Bounded closeout update. The earlier fixed-route localization placed the
first observed eager/grouped difference at the final BF16 `index_add_`
boundary. Independently, the
[paired local report](https://github.com/marin-community/MarinSkyRL/tree/40d365d661f3acc411a78b8801000a4eae964512/evidence/grug_fp32_combine_paired_20260805)
showed that both parent paths share a BF16 accumulation error relative to the
independent FP32 reduction, while `fbb1fc8` satisfies that bounded local FP32
contract.

The distributed result remains unresolved. The first candidate pair failed
`3 / 12,288` action log probabilities; the later route-aware pair failed
`1 / 12,288` at a new coordinate and did not reproduce the earlier three. Its
focused probe found no difference in the captured sparse-block cone. This is
not a stable `3 -> 1` improvement and proves neither production equivalence nor
a cause.

The candidate stays off #276 because the unchanged distributed semantic gate
failed and the local implementation has measured time and combine-memory cost.
Separate checkpoint-provenance work also traced the target step-630 recipe to
ring EP8 at [#7250 head](https://github.com/marin-community/marin/commit/9f527141435cc3606511b331326d09b1f92d696b)
and its corresponding
[merge](https://github.com/marin-community/marin/commit/4b90671a51cf3d1b0f2146203ef965402d2bfd2d).
That historical ring path does not request `fbb1fc8`'s FP32-local-sum contract,
so checkpoint parity supplies no reason to add it. The original training run's
exact recorded Git SHA remains unrecovered.

#7903 stays open. Any predeclared ring-specific acceptance contract belongs to
the separate MarinSkyRL/Levanter training-gap owner; it must remain separate
from this non-repeatable residual and cannot retroactively validate this gate.
No new run was performed for this closeout.
