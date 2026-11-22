# Full LIBERO evaluation

The evaluation covers Spatial, Object, Goal, and Long (`libero_10`). Each suite has ten tasks. Each task runs once from each of its 50 published initial states, giving 500 episodes per suite and 2,000 episodes per policy. This is the four-suite, 50-episodes-per-task protocol. [LeRobot's full LIBERO reporting example](https://github.com/huggingface/lerobot/blob/main/docs/source/evo1.mdx).

All runs use native LeRobot inference with two 256×256 RGB views, LIBERO's relative controller, and the same published initial states, noise seeds, and action budgets (280 Spatial, 280 Object, 300 Goal, 520 Long). The original checkpoint runs in two modes. Its saved settings predict and execute 50 actions with ten denoising steps. The deployment comparison predicts 50 actions and executes the first ten per query for all three policies. The undistilled control overrides the denoising schedule to two steps on the original weights. The distilled run uses the separate student weights from the [200-update experiment](distillation.md).

The batch runner starts a fresh MuJoCo environment for every episode and batches observations from active workers. It renders the two cameras at policy observation ticks while keeping the simulation's observation clocks and control steps. Batch composition and floating-point kernel selection can affect individual rollouts. All four modes use the same runner and simulator settings.

## Success rates

| Suite | Checkpoint default (10-step, execute 50) | Original 10-step (execute 10) | Undistilled 2-step (execute 10) | Distilled 2-step (execute 10) |
| --- | ---: | ---: | ---: | ---: |
| Spatial | 70.20% (351/500) | 87.80% (439/500) | 85.60% (428/500) | 82.00% (410/500) |
| Object | 66.80% (334/500) | 90.60% (453/500) | 91.20% (456/500) | 91.20% (456/500) |
| Goal | 78.40% (392/500) | 88.20% (441/500) | 89.20% (446/500) | 86.00% (430/500) |
| Long | 47.00% (235/500) | 65.60% (328/500) | 67.40% (337/500) | 63.60% (318/500) |
| **All** | 65.60% (1,312/2,000) | 83.05% (1,661/2,000) | 83.35% (1,667/2,000) | 80.70% (1,614/2,000) |

All four runs match the initial RGB and robot-state hashes on all 2,000 episode IDs. We reran all four policies for 1 IDs where the original observation hashes differed. Each rerun group used one GPU and fresh simulators. Selection used the initial observation hashes.

[Checkpoint defaults](../results/libero-full/default.jsonl) · [Original 10-step](../results/libero-full/teacher.jsonl) · [Undistilled 2-step](../results/libero-full/undistilled.jsonl) · [Distilled 2-step](../results/libero-full/distilled.jsonl) · [Summary](../results/libero-full/summary.json) · [Matched reruns](../results/libero-full/matched-reruns.json).

## Reproduce

Use the [workstation environment](reproduce.md). The commands below run twelve simulator workers per policy. Run them on separate GPUs if available. For the student checkpoint, follow the [distillation example](distillation.md).

```bash
LIBERO_ROOT=upstream/LIBERO/libero/libero
VLM_DIR="$(python -c 'from smolvla_iq9075.native import snapshots; print(snapshots()[1])')"

python -m smolvla_iq9075.libero_batch --libero-root "$LIBERO_ROOT" \
  --batch-size 12 --execute 50 --inits 50 --fresh-env --resume \
  --output runs/libero-full/default.jsonl
python -m smolvla_iq9075.libero_batch --libero-root "$LIBERO_ROOT" \
  --batch-size 12 --execute 10 --inits 50 --flow-steps 10 --fresh-env --resume \
  --output runs/libero-full/teacher.jsonl
python -m smolvla_iq9075.libero_batch --libero-root "$LIBERO_ROOT" \
  --batch-size 12 --execute 10 --inits 50 --flow-steps 2 --fresh-env --resume \
  --output runs/libero-full/undistilled.jsonl
python -m smolvla_iq9075.libero_batch --libero-root "$LIBERO_ROOT" \
  --batch-size 12 --execute 10 --inits 50 --fresh-env --resume \
  --checkpoint checkpoints/libero-2step --vlm "$VLM_DIR" \
  --output runs/libero-full/distilled.jsonl

python -m smolvla_iq9075.compare runs/libero-full/teacher.jsonl runs/libero-full/undistilled.jsonl
python -m smolvla_iq9075.compare runs/libero-full/teacher.jsonl runs/libero-full/distilled.jsonl \
  --distillation checkpoints/libero-2step/distillation.json
```

To shard the work, run disjoint suites with `--suites`, or disjoint task ranges with `--task-start` and `--tasks`. Use separate output files for each shard and concatenate them for comparison. For the three ten-action runs, the comparator verifies the episode IDs, initial observation hashes, seeds, simulator asset hashes, execution length, and checkpoint lineage. The IQ-9075 timing and closed-loop pilot use the [board protocol](results.md).

The measurement ran on NVIDIA RTX PRO 6000 Blackwell GPUs. For targeted reruns, use `--episode-list PATH` with the same planned suites and initial states, plus `--fresh-env`. The [recorded protocol](../results/libero-full/protocol.json) lists the software and seed formula.
