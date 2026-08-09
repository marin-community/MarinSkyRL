# Debugging log for dataset-free HF export execution

Remove the training-data dependency from standalone checkpoint exports and keep Harbor's full and standalone
configuration packages on one source commit.

## Initial status

An export-only Iris job staged `DCAgent/exp_rpt_curriculum-easy`, restored offline mode, and then attempted to
load the Hub dataset again before reaching checkpoint export. `export_hf_checkpoint.py` supplied that dataset
only to satisfy normal trainer construction. Harbor and `harbor-config` also resolved from different commits in
the frozen environment.

## Hypothesis 1

Export execution can bypass train/eval dataset construction and dataloader scheduling because it resumes an
exact completed step, runs finalization, and exits without consuming a batch.

## Changes to make

Remove `--train_data` from the export CLI and job specification. Make the shared experiment and trainer
lifecycle represent export execution without datasets, and reject a checkpoint whose loaded step differs from
the requested export step.

## Results

The red tests reproduced both discarded dependencies: the command still passed `--train_data`, and export
experiment construction called `get_train_dataset`. After the change, synchronous and fully asynchronous
trainers construct an export schedule with no dataloader; experiment construction loads neither train nor
evaluation data; generator startup is skipped; and a checkpoint-step mismatch fails before finalization.

## Hypothesis 2

The Harbor release contract can be enforced from the frozen lock: the Git fragment on `harbor` and the commit
embedded in the `harbor-config` release tag must be identical.

## Changes to make

Resolve both packages at Harbor `24e5e67ac93d21c1a2202d6906f6093e0a57b86c` and add a packaging contract that
fails whenever their lockfile commits diverge.

## Results

Harbor's commit-tagged release exists and its wheel digest is
`f5046e5b1131c739a128afd23174c4c79d7aa432ab58e3f8582cb357cdcab047`. The packaging contract reads
both resolved sources from `uv.lock` and passes only when their 40-character commits match.

## Future work

- [ ] Decide separately whether normal training should accept Hub dataset IDs while trainer ranks run offline.
