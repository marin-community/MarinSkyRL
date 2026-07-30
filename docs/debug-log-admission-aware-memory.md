# Debugging log for admission-aware launcher memory

Make automatic Iris RL resource requests fit the requested gang on a busy cluster.

## Initial status

The launcher requests 80% of the smallest matching node's total allocatable memory. It does not
subtract resources requested by running pods or consider `--num-nodes`. An eight-node interactive
job therefore requested 1586Gi per node and remained unadmitted because memory excluded 57 of 64
nodes; the same job was admitted immediately with an explicit 700GB request.

## Hypothesis 1

Choosing the requested gang's Nth-largest live memory headroom, capped at 80% of node allocatable
memory, gives every automatic launch a request that fits at least N matching nodes at inspection
time.

## Changes to make

Add a regression test with four matching nodes whose existing pod requests leave two nodes with
700Gi free and two with 300Gi free. A two-node gang should resolve to 700Gi per node.

## Results

The regression failed before the implementation because resource resolution had no `num_nodes`
input and only inspected total node allocatable resources.

## Hypothesis 2

Memory and disk defaults must come from one feasible node set. Selecting each resource's Nth-largest
headroom independently can produce a pair that no N nodes can jointly satisfy.

## Changes to make

Select the N nodes with the most memory headroom after applying explicit constraints, then cap
automatic disk at the minimum headroom on those same nodes. When memory is explicit, select the N
eligible nodes with the most disk headroom instead.

## Results

The resolver now reads one live node-and-pod snapshot, ignores terminal pods and unscheduled nodes,
and subtracts effective pod requests from allocatable memory and ephemeral storage. Automatic memory
resolves to 700Gi in the reported two-of-four shape. Automatic disk resolves to 3000Gi on those same
two nodes, avoiding a memory/disk pair that exists only across disjoint node sets. Explicit decimal
GB constraints are preserved and participate in node selection.

## Future work

- [x] Record the failing regression and final suite results: 131 passed and 1 skipped under
  `cloud/iris/tests/`.
