# Debugging log for Jupiter Ray artifact sync

Make one explicitly selected Slurm job produce a complete, bounded local Ray-log bundle.

## Initial status

Syncing job `1253429` copied 4.0 GB and 5,481 regular files, but returned `0 directories synced`. Both Ray
transfers exited with rsync code 23 after attempting to recreate Ray's Unix-domain `sockets/raylet` entry.
The destination also contained `ray_1249877`, even though only job `1253429` was selected.

## Hypothesis 1

Ray session sockets are ephemeral process endpoints, not diagnostic files. Excluding every directory named
`sockets` from Ray-log rsync should let valid files transfer without turning the stage into an error.

## Changes to make

Add a boundary test whose fake rsync reproduces code 23 unless the Ray transfer excludes socket directories.
Pass the exclusion only for Ray logs; Slurm logs and traces keep their current transfer policy.

## Results

The boundary test failed before the change and passed after job-specific Ray directories received the socket
exclusion. A live retry then exposed a separate macOS compatibility failure before any transfer began.

## Hypothesis 2

The Ray sync selects the experiment's entire `ray_logs` roots instead of the requested Slurm job's
`ray_<job-id>` and `ray_<job-id>_workers` directories. Selecting those two names under each supported root
will prevent a single-job sync from copying sibling allocations.

## Changes to make

Strengthen the explicit-GPFS-subtree test so only job-specific Ray directories may cross the rsync boundary.

## Results

The original implementation selected both complete Ray roots. The regression test failed until the sync named
only `ray_1253429` and `ray_1253429_workers` under each supported root.

## Hypothesis 3

Apple's bundled openrsync implements protected arguments as `-s` even though it rejects the GNU long spelling
`--protect-args`.

## Changes to make

Try the short option against `/usr/bin/rsync` and add a boundary test that models openrsync rejecting the long
spelling.

## Results

Refuted. Jupiter's launch environment selected `/usr/bin/rsync`, whose openrsync build rejects both
`--protect-args` and `-s`. That explains why the same helper worked when Homebrew rsync preceded `/usr/bin` and
failed from a clean worktree.

## Hypothesis 4

Rsync 2.6.9 can transfer the selected artifacts without a protected-arguments option when the remote path is
shell-quoted inside the single `HOST:PATH` argument.

## Changes to make

Remove the unsupported option and quote the validated remote path with `shlex.quote` before constructing the
remote source argument.

## Results

The focused test passed, `/usr/bin/rsync` transferred a real Jupiter Ray directory with the same arguments,
and the public sync completed against job `1253429` with `4 directories synced` and no errors.

## Future work

- [ ] Decide whether identical launcher and preserved-worker copies should be deduplicated after their
  completeness guarantees are documented.
