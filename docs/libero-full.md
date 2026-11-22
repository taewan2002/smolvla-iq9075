# Full LIBERO evaluation

The evaluation covers Spatial, Object, Goal, and Long (`libero_10`). Each suite has ten tasks. Each task runs once from each of its 50 published initial states, giving 500 episodes per suite and 2,000 episodes per policy. This is the four-suite, 50-episodes-per-task protocol. [LeRobot's full LIBERO reporting example](https://github.com/huggingface/lerobot/blob/main/docs/source/evo1.mdx).

All runs use native LeRobot inference with two 256×256 RGB views, LIBERO's relative controller, and the same published initial states, noise seeds, and action budgets (280 Spatial, 280 Object, 300 Goal, 520 Long). The original checkpoint runs in two modes. Its saved settings predict and execute 50 actions with ten denoising steps. The deployment configuration predicts 50 actions and executes the first ten per query.

The batch runner starts a fresh MuJoCo environment for every episode and batches observations from active workers. It renders the two cameras at policy observation ticks while keeping the simulation's observation clocks and control steps. Batch composition and floating-point kernel selection can affect individual rollouts. Both modes use the same runner and simulator settings.

## Success rates

| Suite | Checkpoint default (10-step, execute 50) | Original 10-step (execute 10) |
| --- | ---: | ---: |
| Spatial | 70.20% (351/500) | 87.80% (439/500) |
| Object | 66.80% (334/500) | 90.60% (453/500) |
| Goal | 78.40% (392/500) | 88.20% (441/500) |
| Long | 47.00% (235/500) | 65.60% (328/500) |
| **All** | 65.60% (1,312/2,000) | 83.05% (1,661/2,000) |

Both runs match the initial RGB and robot-state hashes on all 2,000 episode IDs. One pair was rerun on the same GPU with fresh simulators to match its initial observation.

[Checkpoint defaults](../results/libero-full/default.jsonl) · [Original 10-step](../results/libero-full/teacher.jsonl) · [Matched reruns](../results/libero-full/matched-reruns.json).

## Reproduce

Use the [workstation environment](reproduce.md). The commands below run twelve simulator workers per policy. Run them on separate GPUs if available.

```bash
LIBERO_ROOT=upstream/LIBERO/libero/libero

python -m smolvla_iq9075.libero_batch --libero-root "$LIBERO_ROOT" \
  --batch-size 12 --execute 50 --inits 50 --fresh-env --resume \
  --output runs/libero-full/default.jsonl
python -m smolvla_iq9075.libero_batch --libero-root "$LIBERO_ROOT" \
  --batch-size 12 --execute 10 --inits 50 --flow-steps 10 --fresh-env --resume \
  --output runs/libero-full/teacher.jsonl
```

To shard the work, run disjoint suites with `--suites`, or disjoint task ranges with `--task-start` and `--tasks`. Use separate output files for each shard and concatenate them for comparison. See the [board protocol](results.md) for IQ-9075 timing and the [SnapFlow evaluation](snapflow.md) for a matched comparison with the one-step checkpoint.

The measurement ran on NVIDIA RTX PRO 6000 Blackwell GPUs. For targeted reruns, use `--episode-list PATH` with the same planned suites and initial states, plus `--fresh-env`. The [recorded protocol](../results/libero-full/protocol.json) lists the software and seed formula.
