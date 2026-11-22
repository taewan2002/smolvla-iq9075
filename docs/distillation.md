# Two-step distillation example

This example starts from the pinned `lerobot/smolvla_libero` checkpoint. A frozen teacher runs the original ten Euler steps from the sampled noise at `t=1` to the final action chunk at `t=0`. The student makes two updates, at `t=1` and `t=0.5`, and learns to match the teacher's midpoint and final chunk. Training updates the Action Expert and its action and time projections. The loss emphasizes the first ten actions and also covers the rest of the chunk. The vision and language backbone, state projection, and normalization statistics are shared with the teacher. [Checkpoint](https://huggingface.co/lerobot/smolvla_libero/tree/a24f1f380461b665158d52acd140152c79e8b36d).

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

Use the [full matched LIBERO protocol](libero-full.md) to evaluate the original ten-step teacher, the undistilled two-step control and the distilled student. The undistilled control changes only the Euler schedule. The student also updates the Action Expert and its projections.

## Export and time both two-step variants on IQ-9075

```bash
VLM_DIR="$(python -c 'from smolvla_iq9075.native import snapshots; print(snapshots()[1])')"
python -m smolvla_iq9075.export --flow-steps 2 --output bundles/libero-2step-undistilled
python -m smolvla_iq9075.export --checkpoint checkpoints/libero-2step \
  --vlm "$VLM_DIR" --output bundles/libero-2step
scp -r bundles/libero-2step bundles/libero-2step-undistilled runs/fixtures user@iq9075:~/
```

With the [board runtime](reproduce.md#latency-on-iq-9075), time each two-step variant against the same teacher fixtures used for the ten-step reference.

```bash
for model in libero-2step-undistilled libero-2step; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python -m smolvla_iq9075.bench --bundle "$HOME/$model" --fixtures ~/fixtures \
    --output "runs/iq9075-$model.json" --engines dual --warmup 3 --limit 20 --execute 10
done
```

The latency report covers RGB arrays in memory through action chunk output. Its action error uses the original 10-step teacher output stored in each fixture.

## Example results

The example sampled eight observations per update for 200 updates from four LIBERO suites. The split contained 1,600 training demonstrations and 400 held-out demonstrations. The fixed validation set samples 64 observations from the held-out demonstrations. The errors below use normalized actions and the same initial noise for every policy. [Training configuration and results](../results/distillation/recipe.json).

| Relative L2 against the 10-step teacher | Undistilled 2-step | Distilled 2-step |
| --- | ---: | ---: |
| First 10 actions | 7.07% | 10.30% |
| Full 50-action chunk | 6.43% | 8.57% |

### IQ-9075 latency and action agreement

The dual-HTP FP16 measurements use the same 20 simulator observations and warm-up as the ten-step release benchmark. Board action errors are measured on denormalized actions against the original ten-step CUDA output.

| Dual-HTP model | Median | p95 | Median Action Expert and scaling | Relative L2, first 10 | Relative L2, all 50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original 10-step | 249.7 ms | 254.4 ms | 98.0 ms | 0.46% | 3.08% |
| Undistilled 2-step | 173.1 ms | 176.9 ms | 20.3 ms | 3.88% | 4.04% |
| Distilled 2-step | 172.4 ms | 178.0 ms | 20.1 ms | 4.37% | 8.98% |

[Undistilled timing and error](../results/distillation/iq9075-undistilled.json) · [Undistilled export](../results/distillation/export-undistilled-manifest.json) · [Student timing and error](../results/distillation/iq9075.json) · [Student export](../results/distillation/export-manifest.json).

### Closed-loop LIBERO

Over the same 2,000 LIBERO episodes with ten actions executed per query, the original teacher reached 83.05% (1,661/2,000), the undistilled two-step policy reached 83.35% (1,667/2,000), and the distilled student reached 80.70% (1,614/2,000). For this checkpoint, the 200-update student raised the first-ten-action error against the teacher on the held-out demonstrations.

[Full suite breakdown and episode logs](libero-full.md).
