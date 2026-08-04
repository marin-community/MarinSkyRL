# Debugging log for Ray spill-directory startup

Ensure every Iris task prepares its configured node-local Ray spill directory before starting Ray.

## Initial status

An Iris RL gang reached rendezvous, then every worker failed during `ray start` because
`/tmp/skyrl-ray-spill` did not exist. The launcher resolved and passed the local path correctly.

## Hypothesis 1

`LocalRaySpillTarget` validates path syntax and emits `--object-spilling-directory`, but the node runtime
never creates the directory. Both head and worker startup therefore depend on pre-existing pod filesystem
state.

## Changes to make

Add regression coverage requiring the configured directory to exist at the external `ray start` boundary
for both roles, whether the directory is initially absent or already present. Use a blocked real filesystem
path to require an actionable creation error before Ray is invoked.

## Results

The new test failed for both head and worker when the directory was initially absent: the mocked external
process observed `False` for `is_dir()`. The pre-existing cases passed. A real blocked parent path also reached
the mocked Ray process instead of raising locally. Hypothesis 1 is confirmed.

## Hypothesis 2

Giving each spill target an explicit per-node preparation contract will keep local filesystem ownership beside
its flag construction while preserving the remote backend. Calling it at both Ray startup boundaries should
make directory creation idempotent and fail before subprocess dispatch.

## Changes to make

Add `prepare_node()` to the spill-target protocol. The local target performs `mkdir -p` and wraps filesystem
errors with the configured path; the R2 target has no node-local preparation. Invoke the method before building
or running the head and worker commands.

## Results

All nine focused spill-policy tests pass. A missing nested directory exists when the external Ray process is
invoked for both head and worker; an existing directory remains valid; and a blocked filesystem path raises a
`RuntimeError` containing the configured spill directory before subprocess dispatch. Hypothesis 2 is confirmed.

## Future work

- [x] The image build validates imports, dependency metadata, and Docker assertions but does not start a
  multi-node Ray cluster. The CPU launcher tests also stop at the external-process boundary. The new regression
  covers the missing node-local side effect at that boundary without requiring a cluster allocation.

## Reopened status

Three jobs launched from the fixed revision failed with the same Ray fallback-directory error. Source and bundle
inspection showed the fixed modules were present, but the first regression only mocked the external process and
therefore did not validate Ray's own startup behavior.

## Hypothesis 3

Ray may interpret `--object-spilling-directory` as a parent for a generated session path rather than as the exact
directory prepared by MarinSkyRL.

## Results

Refuted. Ray 2.51.1 turns the CLI value into a filesystem object-spilling configuration and passes that exact path
to `determine_plasma_store_config` as its fallback directory. A disposable real `ray start --head` probe succeeded
with a freshly created spill directory and showed the same path in the raylet's `--fallback_directory` argument.

## Hypothesis 4

Preparation inside the Python controller is not a reliable deployed boundary. Preparing the path in the Iris task
shell, before Python starts, gives every pod an independent filesystem preflight while retaining the controller's
immediate pre-Ray check.

## Changes to make

For local spilling, make the generated task shell create and validate the configured directory before changing into
the application workspace or starting the controller. Fail with the exact path before controller execution when the
directory cannot be created. Exercise the generated shell with a real temporary filesystem and an external
controller probe; do not mock the shell or the preparation helper.

## Results

On unchanged `main`, the controller probe exited because the directory was absent, and a blocked path reached the
controller without naming the path. Both red tests pass after adding the shell preflight. The focused spill and
launcher suite passes 40 tests, and the complete Iris suite passes 192 tests with one existing conditional skip.
Hypothesis 4 is confirmed at the generated task-shell boundary; the next operational validation is a single-node
Iris bring-up before another multi-node E9 allocation.

## Root-cause correction

The post-fix launches did not contain either fix. `build_runtime_bundle()` copied files from the imported
`cloud.iris` package, and the `marinskyrl` console script imported a physical snapshot under `site-packages`
rather than the checkout. The bundle reported the checkout's current Git revision separately, so its provenance
looked current even when its Python files were stale. A task traceback proved the mismatch: its
`task_runtime.py` line numbers existed only in the installed snapshot, not at the reported revision.

## Hypothesis 5

Selecting the checkout independently of Python import resolution, requiring the typed request's launcher commit
to equal that checkout's `HEAD`, and rejecting uncommitted bundle inputs will make the bundle identity describe
the bytes actually submitted. A manifest read from the selected checkout also prevents an old console-script
snapshot from silently retaining an obsolete runtime file list.

## Changes to make

Resolve the source checkout from the current Git worktree or the installation's `direct_url.json`. Copy only the
paths declared by that checkout, hash every copied file into a bundle identity record, and fail before Iris
submission on a commit mismatch or dirty runtime input. Cover the original console-script shape with a separate
temporary checkout whose runtime bytes differ from the imported package.

## Results

The console-script regression failed against the old implementation because it copied the imported package and
accepted no expected commit. It now bundles the selected checkout's distinct marker and records matching hashes.
Commit mismatch and dirty-runtime cases fail before launch side effects. The complete Iris suite passes 195 tests
with one conditional skip, and a built wheel contains both the hardening module and the checkout-owned file
manifest. Hypothesis 5 is confirmed at the launcher boundary.
