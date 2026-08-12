# Debugging log for checkpoint export publication

Make checkpoint exports observable and prevent an interrupted upload from publishing an index that references absent weight shards.

## Initial status

A 67B FSDP2 export published config, tokenizer, and `model.safetensors.index.json` to object storage, then
emitted no progress for 107 minutes. No referenced safetensors shard reached the destination. The export caller
only checked whether the destination prefix existed, so a metadata-only directory could otherwise pass its
success check.

## Hypothesis 1

The model was serialized into the temporary local work directory before publication. The recursive object-store
upload then selected metadata and the index before the large weight shards and exposed no per-file progress.
Publishing safetensors first, logging each completed shard, and publishing the index last will make the long phase
observable and prevent a new interrupted upload from advertising absent shards.

## Changes to make

- Add an HF-model staging context that orders cloud publication and invalidates a stale destination index before
  a retry.
- Use the context in the FSDP, Megatron, and DeepSpeed HF serializers.
- Add behavioral tests over the visible object publication order.

## Results

The original cloud path serialized the entire model into `local_work_dir`, then called one recursive `fsspec`
upload after serialization completed. The object-store listing therefore captured upload order, not serialization
order. Targeted tests now publish referenced safetensors shards first, emit start/completion records with byte and
elapsed-time fields for every shard, and publish the index last. An interrupted retry removes any stale destination
index before work begins and does not restore it.

Megatron Bridge 0.5 writes HF weights on rank 0 by default while every other rank exhausts the conversion generator
to participate in collectives. The Megatron path now gives rank 0 the ordered publisher and gives other ranks
unpublished scratch directories, preserving Bridge's collective contract without racing multiple uploads.

## Hypothesis 2

Prefix existence cannot distinguish a complete model from the five metadata objects left by the failed export.
Validating the index weight map against the destination before publishing or recording request completion will make
exit code zero mean that every referenced shard exists.

## Changes to make

- Add one completeness validator for sharded and unsharded safetensors exports.
- Run it before ordered cloud publication, after strategy conversion, and after a successful Iris subprocess.
- Keep request-driven exports pending and skip Hub publication when validation fails.

## Results

The metadata-only reproduction now fails with the missing shard named in the error. A complete unsharded export and
a two-shard indexed export pass. The targeted checkpoint exporter, Iris lifecycle, object-store ordering,
non-publishing Megatron rank, and FSDP staging tests pass: 54 tests total.

## Future work

- [ ] Evaluate a CPU-only FSDP2 checkpoint reassembler separately; it changes checkpoint loading geometry and
  requires large-memory validation.
- [ ] Set a production watchdog threshold after shard timing is available from real exports.
