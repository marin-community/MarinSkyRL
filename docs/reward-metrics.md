# Reward and verifier score metrics

`reward/avg_raw_reward` is the mean reward used for optimization. It includes
reward shaping and retains each environment's native scale. It can exceed one
when a batch contains GenRM comparisons, whose judge rates responses from one
to five. `eval/all/avg_score` likewise reports the native aggregate score.
Neither metric is a pass rate.

`reward/avg_verifier_score` and `eval/all/avg_verifier_score` average bounded
verifier scores. Each verified result is mapped from its declared native range
to `[0, 1]`; a verifier without declared bounds uses `[0, 1]`. GenRM declares
`[1, 5]`. Scores outside the declared range are clipped for these diagnostics.
Only verified results enter the average, including verified scores of zero.
Verifier errors, unavailable or skipped results, and rows without a verifier
result are omitted. The corresponding
`verifier_score_coverage` metric reports the fraction of rows that entered the
average. Training also reports `reward/agent/<agent>/avg_verifier_score` for up
to 32 verifier agents, using the same normalization.

Evaluation reports `eval/all/num_attempted` for returned trajectory rows,
including ungraded rows, and `eval/all/num_scored` for verified scores. Returned
rows with infrastructure failures reduce scored coverage; they do not become
zero-valued task outcomes.

For a trajectory with multiple scored Gym turns, the terminal verifier score
is the mean of their individually normalized scores. The trajectory passes
only when every scored turn passes. Turns without a verdict, such as tool
execution turns, do not enter that mean. Per-turn diagnostics remain attached
to the terminal verifier result.

These score metrics do not rescale the optimization reward. Use the bounded
verifier score to compare task outcomes across sources, `avg_raw_reward` to
inspect the optimizer signal, and pass@k for the fraction of prompt groups
with at least one successful completion.
