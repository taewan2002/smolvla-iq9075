# Full LIBERO evaluation

The evaluation covers Spatial, Object, Goal, and Long (`libero_10`). Each suite has ten tasks. Each task runs once from each of its 50 published initial states, giving 500 episodes per suite and 2,000 episodes per policy. This is the four-suite, 50-episodes-per-task protocol. [LeRobot's full LIBERO reporting example](https://github.com/huggingface/lerobot/blob/main/docs/source/evo1.mdx).

All three policies use native LeRobot inference. They predict 50 actions and execute ten per query, at 256×256 per view, with the same relative controller, seeds, raw initial states, and action budgets (280 Spatial, 280 Object, 300 Goal, 520 Long). The checkpoint's saved default is one executed action per query. The 10-step and undistilled 2-step policies use the same public checkpoint. The distilled example uses the Action Expert from the [200-update run](distillation.md).

The parallel runner hosts MuJoCo in separate worker processes and batches observations from active episodes for native inference. It renders camera images at policy observation ticks and keeps the simulation's observable clocks and control steps. The parity check compared pixels and robot state with the serial runner at query ticks for three initial states and an identical action sequence. Batch composition and floating-point kernel selection can change individual rollouts, so the three headline scores use the same batched runner and evaluation settings.

## Success rates

| Suite | Original 10-step | Undistilled 2-step | Distilled 2-step |
| --- | ---: | ---: | ---: |
| Spatial | 66.80% (334/500) | 80.40% (402/500) | 59.00% (295/500) |
| Object | 89.80% (449/500) | 97.20% (486/500) | 90.20% (451/500) |
| Goal | 76.60% (383/500) | 86.60% (433/500) | 74.60% (373/500) |
| Long | 37.00% (185/500) | 49.40% (247/500) | 41.20% (206/500) |
| **All** | 67.55% (1,351/2,000) | **78.40% (1,568/2,000)** | 66.25% (1,325/2,000) |

The final records match the initial RGB and robot-state hashes across all 2,000 episode IDs. For 110 IDs with different initial observation hashes, we reran all three policies on the same GPU with freshly created simulators and checked the match again. The selected rerun IDs are listed below.

[Original 10-step](../results/libero-full/teacher.jsonl) · [Undistilled 2-step](../results/libero-full/undistilled.jsonl) · [Distilled 2-step](../results/libero-full/distilled.jsonl) · [Summary](../results/libero-full/summary.json) · [Matched reruns](../results/libero-full/matched-reruns.json).

## Reproduce

Use the [workstation environment](reproduce.md). The commands below run twelve simulator workers per policy. Run them on separate GPUs if available. For the student checkpoint, follow the [distillation example](distillation.md).

```bash
LIBERO_ROOT=upstream/LIBERO/libero/libero
VLM_DIR="$(python -c 'from smolvla_iq9075.native import snapshots; print(snapshots()[1])')"

python -m smolvla_iq9075.libero_batch --libero-root "$LIBERO_ROOT" \
  --batch-size 12 --execute 10 --inits 50 --flow-steps 10 --resume \
  --output runs/libero-full/teacher.jsonl
python -m smolvla_iq9075.libero_batch --libero-root "$LIBERO_ROOT" \
  --batch-size 12 --execute 10 --inits 50 --flow-steps 2 --resume \
  --output runs/libero-full/undistilled.jsonl
python -m smolvla_iq9075.libero_batch --libero-root "$LIBERO_ROOT" \
  --batch-size 12 --execute 10 --inits 50 --resume \
  --checkpoint checkpoints/libero-2step --vlm "$VLM_DIR" \
  --output runs/libero-full/distilled.jsonl

python -m smolvla_iq9075.compare runs/libero-full/teacher.jsonl runs/libero-full/undistilled.jsonl
python -m smolvla_iq9075.compare runs/libero-full/teacher.jsonl runs/libero-full/distilled.jsonl \
  --distillation checkpoints/libero-2step/distillation.json
```

To shard the work, run disjoint suites with `--suites`, or disjoint task ranges with `--task-start` and `--tasks`. Use separate output files for each shard and concatenate them for comparison. The comparator verifies the episode IDs, initial observation hashes, seeds, simulator asset hashes, execution length, and checkpoint lineage. The IQ-9075 timing and closed-loop pilot use the [board protocol](results.md).

The measurement ran on NVIDIA RTX PRO 6000 Blackwell GPUs. For targeted reruns, use `--episode-list PATH` with the same planned suites and initial states, plus `--fresh-env`. The [recorded protocol](../results/libero-full/protocol.json) lists the software and seed formula.
