# Two-step distillation example

This example starts from the published SmolVLA LIBERO checkpoint. A frozen teacher runs the original ten Euler steps from the sampled noise at `t=1` to the final action chunk at `t=0`. The student makes two updates, at `t=1` and `t=0.5`, and learns to match the teacher's midpoint and final chunk. Training updates the Action Expert and its action and time projections. The loss emphasizes the first ten actions and also covers the rest of the chunk. The vision and language backbone, state projection, and normalization statistics are shared with the teacher.

The model predicts 50 actions. The comparisons below execute 10 actions per query. The two denoising steps are two evaluations of the Action Expert per query.

## Train the student

Use the [workstation setup](reproduce.md) and install the HDF5 reader.

```bash
python -m pip install h5py==3.14.0
python upstream/LIBERO/benchmark_scripts/download_libero_datasets.py \
  --use-huggingface --download-dir "$PWD/datasets"
python -m smolvla_iq9075.distill --demos datasets \
  --output checkpoints/libero-2step --updates 200 --batch 8 --validation 64
```

The dataset directory contains LIBERO suite subdirectories with the original `*_demo.hdf5` files. The trainer takes RGB, task text and robot state from the recordings, and supervises the action chunk with teacher outputs. It holds out whole demonstrations with index divisible by five in each task. The validation measures the effect of distillation on unseen demonstrations for the student. The public teacher may have been trained on these demonstrations. [LIBERO data and download instructions](https://github.com/Lifelong-Robot-Learning/LIBERO).

The trainer writes a complete LeRobot checkpoint with `num_steps: 2`, along with `distillation.json` containing the paired checkpoint hashes and the fixed-observation errors before and after training. For an offline run, pass `--checkpoint` and `--vlm` with the two snapshots already available.

## Compare three policies in LIBERO

Keep the same simulator, camera processing, suite, initial state index, seed and executed action count for all three runs.

```bash
VLM_DIR="$(python -c 'from smolvla_iq9075.native import snapshots; print(snapshots()[1])')"
LIBERO_ROOT=upstream/LIBERO/libero/libero
python -m smolvla_iq9075.libero --engine native --libero-root "$LIBERO_ROOT" \
  --execute 10 --inits 1 --save-fixtures runs/teacher-fixtures --output runs/teacher-10.jsonl
python -m smolvla_iq9075.libero --engine native --libero-root "$LIBERO_ROOT" \
  --execute 10 --inits 1 --flow-steps 2 --output runs/undistilled-2.jsonl
python -m smolvla_iq9075.libero --engine native --libero-root "$LIBERO_ROOT" \
  --execute 10 --inits 1 --checkpoint checkpoints/libero-2step --vlm "$VLM_DIR" \
  --output runs/distilled-2.jsonl

python -m smolvla_iq9075.compare runs/teacher-10.jsonl runs/undistilled-2.jsonl
python -m smolvla_iq9075.compare runs/teacher-10.jsonl runs/distilled-2.jsonl \
  --distillation checkpoints/libero-2step/distillation.json
```

Run the teacher with its saved 10-step schedule. The untrained 2-step control uses the same checkpoint with a larger Euler step. The distilled student uses its saved 2-step schedule and trained weights.

## Export and time the student on IQ-9075

```bash
python -m smolvla_iq9075.export --checkpoint checkpoints/libero-2step \
  --vlm "$VLM_DIR" --output bundles/libero-2step
scp -r bundles/libero-2step runs/teacher-fixtures user@iq9075:~/
```

With the [board runtime](reproduce.md#latency-on-iq-9075), time the student against the same teacher fixtures used for the ten-step reference.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m smolvla_iq9075.bench --bundle ~/libero-2step --fixtures ~/teacher-fixtures \
  --output runs/iq9075-2step.json --engines dual --warmup 3 --limit 20 --execute 10
```

The latency report covers RGB arrays in memory through action chunk output. Its action error uses the original 10-step teacher output stored in each fixture.

## Example results

The example sampled eight observations per update for 200 updates from four LIBERO suites. The split contained 1,600 training demonstrations and 400 held-out demonstrations. The fixed validation set samples 64 observations from the held-out demonstrations. The errors below use normalized actions and the same initial noise for every policy. [Training configuration and results](../results/distillation/recipe.json).

| Relative L2 against the 10-step teacher | Undistilled 2-step | Distilled 2-step |
| --- | ---: | ---: |
| First 10 actions | 22.35% | **14.35%** |
| Full 50-action chunk | 25.55% | **19.26%** |

### IQ-9075 latency

The dual-HTP FP16 measurements use the same 20 simulator observations and warm-up as the 10-step release benchmark.

| Model | Median total | p95 total | Median Action Expert and output scaling |
| --- | ---: | ---: | ---: |
| Original 10-step | 394.1 ms | 405.2 ms | 183.0 ms |
| Distilled 2-step | **248.2 ms** | **253.5 ms** | **36.7 ms** |

Total median latency fell by **37.0%**, while the Action Expert stage ran about **5×** faster. On these board inputs, the distilled 2-step output differed from the original native 10-step output by 6.14% relative L2 for the first ten denormalized actions and 13.42% for the whole chunk. [Board timing traces](../results/distillation/iq9075.json) · [Export manifest](../results/distillation/export-manifest.json).

### Closed-loop LIBERO pilot

The model-level comparison is also available across [2,000 episodes per policy](libero-full.md).

All three policies ran via native LeRobot on the same 40 LIBERO initial conditions and matched noise seeds, with 50 actions predicted and ten executed per query.

| Suite | Original 10-step | Undistilled 2-step | Distilled 2-step |
| --- | ---: | ---: | ---: |
| Spatial | 8/10 | 9/10 | 5/10 |
| Object | 9/10 | 10/10 | 8/10 |
| Goal | 6/10 | 9/10 | 6/10 |
| Long | 3/10 | 5/10 | 4/10 |
| **All** | 26/40 | 33/40 | 23/40 |

Distillation improved fixed-observation agreement with the teacher. The undistilled 2-step control had the highest task success in this pilot, and the distilled student finished three episodes behind the 10-step teacher.

[Undistilled episode log](../results/distillation/undistilled.jsonl) · [Distilled episode log](../results/distillation/distilled.jsonl) · [Undistilled paired counts](../results/distillation/undistilled-paired.json) · [Distilled paired counts](../results/distillation/distilled-paired.json).
