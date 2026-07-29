# Debugging log for issue 195

Make Megatron context parallel sample packing handle sequences whose real tokens end before a padded shard boundary.

## Initial status

`preprocess_packed_seqs` assigns a fixed-width first context-parallel chunk from an unclamped source slice. A short sequence can therefore produce a narrower source than destination and raise a tensor shape error.

## Hypothesis 1

The first zigzag chunk needs the same real-length clamp and empty-slice guard as the second chunk.

## Changes to make

Add a CPU regression that runs `preprocess_packed_seqs` for both ranks of context parallel size 2 with one real token padded to four. Assert each rank's packed tokens, then clamp and guard the first chunk assignment.

## Results

The CPU regression passed for CP rank 0 and reproduced the reported `RuntimeError` for CP rank 1: the destination expected one token while the source slice was empty. This confirms that the first chunk must be bounded by the real sequence length.

Clamping the first chunk's end and guarding non-positive lengths makes both ranks return the expected packed tokens. The focused regression passes for both CP ranks.

The full CPU suite completed with 870 passed and 19 skipped. Two tests failed, and both failures reproduce unchanged on `main`: `test_generator_output_concatenation` expects a stale `GeneratorOutput` schema, and `test_all_defaults_is_structurally_identical_to_pre_ep` compares against a stale configuration golden file.

## Future work

None.
