# Debugging log for the Harbor image gate

The GPU image build must prove that Harbor bounds a stalled artifact upload and keeps its writer usable without making ordinary builder scheduling latency a test failure.

## Initial status

The ARM64 standard image reached the Harbor validation layer. The intended cloud upload timed out after 50 ms and the follow-up local artifact succeeded, but the validator process exited after its two-second outer deadline. The Iris retry reproduced the same image context; the pinned Harbor source and package installation were unchanged.

## Hypothesis 1

The two-second validation deadline conflates the Harbor upload bound with process and thread scheduling on a loaded cross-architecture Kaniko host. The upload timeout itself is the behavior under test and remains 50 ms.

## Changes to make

Increase only the validator's outer completion deadline from two seconds to ten seconds. Keep the simulated object-store stall at 30 seconds, so a Harbor implementation that fails to release its worker still fails the gate.

## Results

The exact pinned Harbor commit passes the validator locally while logging one expected failed upload and a successful follow-up write. Rebuild both architectures and variants from the corrected MarinSkyRL commit to exercise the same gate under native builders.

## Future work

- [ ] None.
