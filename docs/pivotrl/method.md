# PivotRL with a frozen selection

Train GRPO on task-disjoint prefixes selected from saved initial-policy samples.
For binary outcomes, a group with success rate p has empirical reward variance
p(1-p). A selected group must contain both successes and failures. Selection is
specific to the initial checkpoint, reward, tokenizer, and generation cap of the
training arm. A group selected for tool-name reward can be uniform under NeMo;
a Grug-selected group can be uniform under Snowball.

`infra.rl_data.pivot` prepares pinned release records and selects prefixes from
retained, fully verified step-zero groups with immutable model provenance.
`infra.rl_data.pivot_pilot` prepares the task-disjoint holdout, count-matched random
control, and frozen training subsets. The final experiment snapshots and selection
receipts identify the exact prepared datasets used by each arm.

The launch compiler maps `pivot.mode` to teacher-forced SFT, sampled GRPO, or frozen
sampling. GRPO keeps binary rewards unshaped and uses a differentiable importance-
weighted estimate of forward KL to the frozen reference. Synchronous runs allow no
stale groups. Trajectory retention stores the request, outcome, verifier diagnostics,
model identity, and policy step; grade tables and exposure counters make the
accepted training and evaluation data inspectable.

Existing saved outcomes determine the selections in the recorded experiments.
Reproduction uses those immutable selected datasets directly; it does not require
another sampling or grading pass. The archived statistics record capped failures
and exclude ambiguous cap/EOS boundaries and overlong prompts. This is an empirical
eight-draw selection invariant: future stochastic groups can still be all-pass or
all-fail. Check retained first-update groups to measure that change, rather than
assuming their variance must remain positive.

Full heldout evaluations cover all 256 prefixes at initialization and updates
5, 10, 15, and 20. Quick evaluations use a fixed subset between full evaluations.
SFT evaluates at loss-token milestones. Accept a result only with complete grading,
complete heldout coverage, and retained training samples. An ungraded member can
alter the GRPO baseline even when its own loss is masked.

The recipe templates are inputs to the preparation helpers. Use the separately
recorded final configs for reproduction of the accepted runs.
