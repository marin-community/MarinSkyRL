# Measure cold Hugging Face model mirroring

[PR #715](https://github.com/marin-community/MarinSkyRL/pull/715) added a cache where one task mirrors a pinned model to S3 and other tasks hit or wait. Measure this path before changing its transfer method.

- Log hit, wait, and publisher roles; monotonic time for download, manifest hashing, S3 publication, and the full call; and file count or bytes from the manifest. Start markers should identify the active phase if a task stops early.
- On a cold publication, log only `HF_TOKEN` presence and a safe class for the environment's S3 endpoint hint: `coreweave_in_cluster`, `other`, `unset`, or `unknown`. The hint is not proof of the storage client's actual endpoint. Never log credentials or full URLs.
- Preserve cache behavior. Do not add a transfer method, telemetry service, or costly validation run. Use the next ordinary cold run to decide what to optimize; until then, production phase timings remain unmeasured.
- Validate with local Iris tests and changed-file lint, review once, and hand off the draft PR after its current CI and feedback are resolved.
