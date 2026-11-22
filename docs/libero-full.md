# Full LIBERO evaluation

The evaluation covers Spatial, Object, Goal, and Long (`libero_10`). Each suite has ten tasks. Each task runs once from each of its 50 published initial states, giving 500 episodes per suite and 2,000 episodes per policy. This is the four-suite, 50-episodes-per-task protocol. [LeRobot's full LIBERO reporting example](https://github.com/huggingface/lerobot/blob/main/docs/source/evo1.mdx).

All runs use the original checkpoint weights, native LeRobot inference with two 256×256 RGB views, LIBERO's relative controller, and the same published initial states, noise seeds, and action budgets (280 Spatial, 280 Object, 300 Goal, 520 Long). The saved configuration predicts and executes 50 actions with ten denoising steps. The solver comparison uses ten, two, or one denoising step and executes the first ten of 50 predicted actions per query.

The batch runner starts a fresh MuJoCo environment for every episode and batches observations from active workers. It renders the two cameras at policy observation ticks while keeping the simulation's observation clocks and control steps. Batch composition and floating-point kernel selection can affect individual rollouts. All modes use the same runner and simulator settings.

## Success rates

| Suite | Checkpoint default (10 steps, execute 50) | 10 steps (execute 10) | 2 steps (execute 10) | 1 step (execute 10) |
| --- | ---: | ---: | ---: | ---: |
| Spatial | 70.20% (351/500) | 87.80% (439/500) | 85.60% (428/500) | 85.00% (425/500) |
| Object | 66.80% (334/500) | 90.60% (453/500) | 91.20% (456/500) | 87.20% (436/500) |
| Goal | 78.40% (392/500) | 88.20% (441/500) | 89.20% (446/500) | 85.00% (425/500) |
| Long | 47.00% (235/500) | 65.60% (328/500) | 67.40% (337/500) | 63.60% (318/500) |
| **All** | 65.60% (1,312/2,000) | 83.05% (1,661/2,000) | 83.35% (1,667/2,000) | 80.20% (1,604/2,000) |

All four runs match the initial RGB and robot-state hashes on 2,000 episode IDs and use the same environment assets and noise seeds. One episode in the one-step run was rerun in isolation to match its initial observation hash.

Against two steps, one step gained 94 successes and lost 157 on the paired episodes, a drop of **3.15 percentage points** overall.

[Checkpoint defaults](../results/libero-full/default.jsonl) · [Original 10-step](../results/libero-full/teacher.jsonl) · [Original 2-step](../results/libero-full/undistilled.jsonl) · [Original 1-step](../results/libero-full/naive-1nfe.jsonl) · [Paired results](../results/libero-full/solver-steps-paired.json).

## Reproduce

Use the [workstation environment](reproduce.md). The commands below run twelve simulator workers per policy. Run them on separate GPUs if available.

```bash
LIBERO_ROOT=upstream/LIBERO/libero/libero

python -m smolvla_iq9075.libero_batch --libero-root "$LIBERO_ROOT" \
  --batch-size 12 --execute 50 --inits 50 --fresh-env --resume \
  --output runs/libero-full/default.jsonl
for steps in 10 2 1; do
  python -m smolvla_iq9075.libero_batch --libero-root "$LIBERO_ROOT" \
    --batch-size 12 --execute 10 --inits 50 --flow-steps "$steps" --fresh-env --resume \
    --output "runs/libero-full/step-$steps.jsonl"
done
python -m smolvla_iq9075.compare runs/libero-full/step-2.jsonl runs/libero-full/step-1.jsonl
```

To shard the work, run disjoint suites with `--suites`, task ranges with `--task-start` and `--tasks`, or initial-state ranges with `--init-start` and `--inits`. Use separate output files for each shard and concatenate them for comparison. The one-step run used four initial-state ranges covering all tasks (0–12, 13–25, 26–37, 38–49), one range per GPU. See the [board protocol](results.md) for IQ-9075 timing and the [SnapFlow evaluation](snapflow.md) for the trained one-step variant.

The measurement ran on NVIDIA RTX PRO 6000 Blackwell GPUs. For targeted reruns, use `--episode-list PATH` with the same planned suites and initial states, plus `--fresh-env`. The [recorded protocol](../results/libero-full/protocol.json) lists the software and seed formula.
