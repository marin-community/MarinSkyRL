# Rollout buffer checkpoint fixture

`rollout_buffer_pre_row_facts.pkl` contains public `RolloutBuffer.snapshot()` results
for `full_batch` and `rolling`, produced by source commit
`a76960121f89965e74a34254185d1f2d489055bb`. That source's `RolloutVerdict` has no
`row_facts` member. The pickle uses protocol 4 and contains no payload objects or
external references: payload references are the strings `admitted` and `extra`.

For each policy, the generator constructed
`RolloutBuffer(RolloutBufferConfig(2, 4, 1, policy, None, None))`, published policy
step 1 and acquired one outstanding lease. It then acquired and committed groups
`admitted` and `extra`, admitting once after the first commit. Each verdict came
from `RolloutContentPolicy` with an exact physical group size of 2 and no dynamic
selection. Each payload had prompts `[[1], [1, 2]]`, responses `[[], [2, 3]]`,
loss masks `[[], [1, 1]]` and rewards `[9.0, 1.25]`.

The generator serialized the dictionary `{policy.value: buffer.snapshot()}` using
`pickle.dumps(..., protocol=4)`. The current snapshot owner test exercises this
same construction and loads this fixture through the public restore/admit APIs.
