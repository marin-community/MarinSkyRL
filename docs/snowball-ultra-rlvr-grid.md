# Snowball Ultra-style RLVR grid

This grid adapts NVIDIA's two-phase general RLVR cold-start recipe to the Snowball 67B-A2B checkpoint and a
64-H100 Iris allocation. It is a launch proposal, not a GPU-validated recipe: the checkpoint has been served on
eight H100s at 8K context, but neither 65,536-token serving nor these training geometries has completed a smoke
run.

| topology | RLVR1 | RLVR2 | allocation |
| --- | --- | --- | --- |
| split64 | `snowball_ultra_rlvr1_split64.yaml` | `snowball_ultra_rlvr2_split64.yaml` | 32 learner + 32 rollout GPUs |
| colocated64 | `snowball_ultra_rlvr1_colocated64.yaml` | `snowball_ultra_rlvr2_colocated64.yaml` | 64 GPUs shared by learning and rollout |

Start with `split64`. It is closest to the Snowball geometry already exercised in this repository. `colocated64`
retains the Ultra recipe's CP8 learner geometry but has greater memory and lifecycle risk because vLLM and
Megatron share every GPU.

## Shared recipe

Both topologies use synchronous GRPO with 512 prompts and 16 responses per prompt, AdamW at `4e-6`, one update
epoch, token-mean loss, symmetric PPO clipping at 0.2, and no entropy or KL loss. The request window is 65,536
tokens and each model turn may generate at most 6,528 tokens. `max_turns: 999999` is the current representation
of no fixed turn limit. The launcher derives `trainer.max_prompt_length=59008`; the derived field is deliberately
absent from the authored YAML.

RLVR1 stops at step 128. RLVR2 resumes the same checkpoint directory and stops at cumulative step 178. Its
`restore_dataloader_state: false` setting retains model, optimizer, scheduler, and global-step state while starting
the replacement RLVR2 dataset at its first row. Both phases disable dataloader shuffling.

Although the configs set `colocate_policy_ref: true`, the trainer does not instantiate a reference model while
both KL controls are disabled. The placement remains explicit so enabling a reference-dependent objective later
does not silently allocate a separate node group.

## Inputs required at launch

Pass the Snowball model from
[`open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.11`](https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.11).
Pin the resolved model artifact before launching rather than relying on the mutable repository head.

Prepare RLVR1 and RLVR2 independently from NVIDIA's pinned
[`nvidia/Nemotron-RL-Ultra-Training-Blends`](https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/tree/79f8eda15ea12e1adf7bb14dcb338a29d391b80e)
revision. The upstream files are JSONL. Marin's `infra.rl_data` tools extract their embedded records.
Then use `marinskyrl.nemotron_preparation` to write bare TaskSpecs.
The trainer accepts these prepared records. It does not accept raw Nemotron rows.
See the [preparation guide](../skyrl-train/examples/nemotron/README.md) for the image and verifier inputs.

Keep the released order, reserve the final 100 prepared rows for validation, and pass the remaining rows through
`--train_data`. RLVR2 must contain all 99,016 training rows, yielding 194 batches at batch size 512. The trainer's
step target is absolute after resume, so a replacement artifact with fewer than 178 batches would terminate before
the intended cumulative step 178. Convert the SWE packages to Harbor TaskSpecs before Nemotron preparation.
Pass that artifact through `--terminal-tasks` and keep its execution settings in the run config.
The trainer does not require the original package directories.
Non-SWE tasks use `NemotronTaskSession` and optional task-local Shellbox machines.
Declare verifier parameters during preparation. Select machine backends and execution limits in the run config.

Use the same Iris job name and checkpoint path for each topology's two phases so `resume_mode: latest` finds the
RLVR1 checkpoint. Run a bounded startup and 65K-context memory smoke before committing the full 64-GPU gang.
