# Distributed debug artifact acceptance

The opt-in two-node test in `distributed_debug_artifact_contract.py` runs one
healthy gang and one rank non-arrival gang. It checks that both finish within
bounded deadlines and that the declared debug artifacts are present under the
chosen durable root.

Run it only on an otherwise idle two-node allocation in the policy runtime.
See `docs/debug-modes.md` for the command and acceptance criteria. The test is
outside ordinary pytest discovery and does not run in PR CI.
