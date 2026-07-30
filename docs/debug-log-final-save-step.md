# Debugging log for final-save global step

Ensure train-end checkpoints, exports, and callbacks use the number of completed optimizer steps rather
than the next step index maintained by the fully asynchronous loop.

## Initial status

After completing step N, `FullyAsyncRayPPOTrainer._train_loop` increments `global_step` to N+1 and
notifies the staleness manager. The train-end block then creates callback state and saves artifacts
without restoring the completed-step value. In-loop saves are correctly named because they occur before
the increment, but the final checkpoint and export are stamped N+1.

## Hypothesis 1

Passing the last completed step explicitly into a shared train-end finalizer will keep staleness
bookkeeping unchanged while making callback state and artifact paths truthful.

## Changes to make

Add a regression test that enters finalization with a next-step value of 17 and a completed-step value
of 16. Require the train-end callback, checkpoint save, model export, and `on_save` callback to all
observe step 16.

## Results

The regression failed because no completed-step finalization boundary existed; the loop proceeded
directly from the N+1 next-step value to train-end callbacks. A shared finalizer now restores the last
completed step before constructing callback state or saving. The same finalizer is used for
resume-at-max and dispatches `on_save` after checkpoint persistence.

## Future work

- [x] Verify the final checkpoint dispatches `on_save` so data and generation-buffer state are included.
- [x] Keep resume-at-max behavior at the loaded completed step.
