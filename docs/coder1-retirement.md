# coder1 and GeneralReactTask retirement

`coder1`, `GeneralReactTask`, and `DummyReactTask` are deprecated and removed from
the dormant `skyrl-agent` snapshot. Their implementations, code execution backends,
and dependent example configurations and launch scripts have been deleted. Package
retirement notices reject old imports with `ModuleNotFoundError` and a migration
message, including imports of the former execution backends.

The task's exclusive math and question-answering verifiers are also removed.
Retain the two import notices until the dormant `skyrl-agent` snapshot is deleted:
they provide actionable rejection for old configurations, with no scoring or
compatibility implementation.

`coder1` executed arbitrary Python tests alongside candidate objects. Identity
checks, class and module introspection, callbacks, and custom objects cannot be
preserved by translating inputs and outputs to isolated execution. The partial
adapter described in [issue #880](https://github.com/marin-community/MarinSkyRL/issues/880)
does not establish compatibility for that contract. `coder1` is excluded from
verifier unification; there is no supported adapter or execution-mode override.

## Migrate code tasks

Use the maintained `lcb` SkyRL-Gym environment for Python tasks with explicit
standard-input/output tests or function arguments and expected return values.
Set dataset rows to `env_class: lcb` and encode the test cases as JSON in
`reward_model.ground_truth`. The existing
[`normalize_lcb_ground_truth`](../skyrl-gym/skyrl_gym/envs/lcb/livecodebench.py)
accepts APPS `inputs`/`outputs` mappings, canonical case lists, and open-r1
`test_cases` mappings. See the
[dataset builder](../skyrl-train/examples/livecodebench/lcb_dataset.py) and
[launch example](../skyrl-train/examples/livecodebench/run_lcb.sh).

Old `coder1` `inputs`/`outputs` cases can use this route when they describe
standard-input/output behavior. Rewrite `pytest`, `functional`, and
`solution_file` cases into explicit supported cases where possible. Do not merely
rename `data_source: codegen*` or the verifier: compare expected results and reward
aggregation after conversion. Python-object behavior requires a redesigned task.

Use the maintained
[Harbor trajectory runner](../skyrl-train/skyrl_train/trajectory_runners/harbor/runner.py)
for tasks that need filesystem setup, dependencies, tools, or a test suite in a
sandbox. Package the problem and tests as a Harbor task and configure its verifier
there. Harbor is not a drop-in implementation of `coder1`'s Python-object contract.

## Migrate other GeneralReactTask configurations

The removed examples used `GeneralReactTask` for MemAgent, Search-R1, BrowseComp,
and OpenAI ReAct. These recipes are retired along with the task. For math and
question-answering workloads, select a maintained SkyRL-Gym environment with the
appropriate data contract, such as `aime`, `gsm8k`, or `search`; multi-step agent
tasks need a supported trajectory runner such as Harbor. Convert the dataset,
tool interface, and reward policy explicitly. There is no automatic migration
of the removed ReAct recipes.
