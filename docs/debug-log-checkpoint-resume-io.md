# Debugging log for checkpoint resume I/O

Reproduce the FSDP2 checkpoint read amplification and determine which proposed timeout and retry changes act at the failing transfer boundary.

## Initial status

A 64-rank FSDP2 resume failed while reading a complete 4.19 GB optimizer shard from S3. The loader recursively stages the policy checkpoint directory on every rank, even though each rank subsequently opens only its model, optimizer, and extra-state shards. This proves roughly 12,000 redundant object transfers per resume, but not that all transfers were simultaneously in flight. The S3 client has the SDK's 60-second read timeout, and the outer recursive transfer retries only expired credentials.

Megatron and DeepSpeed also stage directories, but their checkpoint libraries discover metadata and shard ownership from the directory. Their file requirements are not equivalent to the explicit three-file FSDP2 format.

## Hypothesis 1

FSDP2 can stage exactly the three paths determined by its world size and rank without changing checkpoint semantics.

## Changes to make

Add a context manager that stages an explicit list of files, then use it at the FSDP2 load boundary. Preserve the local-path behavior and optional optimizer state.

## Results

Confirmed. A mocked cloud load requested only `model_world_size_8_rank_3.pt`, `optim_world_size_8_rank_3.pt`, and `extra_state_world_size_8_rank_3.pt`. The existing two-rank local save/load contract remained green.

## Hypothesis 2

An explicit S3 client read timeout plus transfer-level retry catches the observed `FSTimeoutError`. Retrying only `fs.open()` would be insufficient because a streamed read can fail after the context manager has returned the file object.

## Changes to make

Configure the cached S3 filesystem with explicit connect/read timeouts and adaptive SDK retries. Retry an entire staged-file transfer with exponential backoff when fsspec or botocore reports a transient timeout or connection error.

## Results

Confirmed. Both the observed fsspec `FSTimeoutError` and botocore `ReadTimeoutError` retried with exponential backoff. The client is constructed with a 60-second connect timeout, 300-second read timeout, and ten adaptive SDK attempts; the explicit transfer layer permits five attempts. The full launcher and trainer CPU gate passed with 1,250 tests and 21 skips.

Megatron and DeepSpeed continue to stage checkpoint directories because their checkpoint libraries consume directory metadata and determine shard ownership internally. They benefit from the shared S3 timeout and retry changes, but eliminating their read amplification requires a format-aware remote-storage backend or coordinated node-local staging rather than the FSDP2 filename rule.
