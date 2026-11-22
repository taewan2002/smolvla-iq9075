# SnapFlow one-step SmolVLA

This adaptation starts from the pinned `lerobot/smolvla_libero` checkpoint. It freezes the vision-language prefix and trains the Action Expert and a zero-initialized destination-time projection. Each batch mixes supervised flow matching on demonstration actions with shortcut targets built from two stop-gradient half steps of the same model. The three expert calls share one frozen prefix cache. At deployment the source time is 1, the destination time is 0, and one expert forward pass predicts 50 actions. The evaluation applies ten actions per query. [SnapFlow v2, sections 3.4–3.6 and appendix B](https://arxiv.org/html/2604.05656v2).

The paper reports SmolVLA results on PushT and closed-loop LIBERO results on π0 and π0.5. This recipe uses SmolVLA and LIBERO together. SmolVLA's destination projection is added to the source-time embedding before the first time MLP. The deployable checkpoint folds the fixed destination embedding into that layer's bias and sets `num_steps: 1`. The training script checks the folded output against the explicit one-step implementation on held-out inputs. The checkpoint works with the same LeRobot evaluator and HTP exporter.

## Pilot results

The pilot used 400 updates, batch size 4, and 128 observations sampled from held-out demonstrations. Offline MSE is computed in normalized action space and masked to recorded actions.

| Policy | First 10 action MSE | Full chunk MSE |
| --- | ---: | ---: |
| Original 10-NFE | 0.0160 | 0.0231 |
| Unchanged checkpoint, 1-NFE | 0.0177 | 0.0255 |
| SnapFlow, 1-NFE | 0.0202 | 0.0279 |

The paired closed-loop evaluation used the first two published initial states per task with native LeRobot on the same GPU. All 80 episode keys and initial observation hashes match across conditions.

| Suite | Original 10-NFE | Unchanged checkpoint, 1-NFE | SnapFlow, 1-NFE |
| --- | ---: | ---: | ---: |
| Spatial | 19/20 | 19/20 | 19/20 |
| Object | 18/20 | 19/20 | 18/20 |
| Goal | 17/20 | 17/20 | 16/20 |
| Long | 16/20 | 15/20 | 16/20 |
| **All** | 70/80 | 70/80 | 69/80 |

On the IQ-9075, the optimized dual-HTP export took **164.1 ms median / 167.2 ms p95** over the same 20 input observations as the 10-NFE benchmark. Median latency fell **34.3%** from 249.7 ms. The Action Expert and output scaling took **10.4 ms** median. The first ten actions differed from native SnapFlow on CUDA by **0.79%** relative L2. The input arrays, task text and initial noise are identical to the 10-NFE fixtures. The stored reference output uses the corresponding SnapFlow checkpoint.

Two additional pilot runs used the original LIBERO HDF5 recordings. The 3,000-update run at `2.5e-5` succeeded in 25/80, and the 2,000-update run at `5e-6` succeeded in 60/80. These runs differ in both learning rate and update count. Their offline MSE is measured on a different set of recordings from the LeRobot pilot. [HDF5 3,000-update report](../results/snapflow/hdf5-3000-training.json) · [HDF5 2,000-update report](../results/snapflow/hdf5-2000-training.json).

A one-episode integration check ran the SnapFlow bundle on IQ-9075 with the 25 Hz asynchronous simulator client. Spatial task 0 succeeded with zero empty ticks and zero skipped control slots. [Integration check](../results/snapflow/async-smoke.jsonl).

[Training report](../results/snapflow/training.json) · [Paired counts](../results/snapflow/paired.json) · [Original rollouts](../results/snapflow/original.jsonl) · [Unchanged 1-NFE rollouts](../results/snapflow/naive.jsonl) · [SnapFlow rollouts](../results/snapflow/snapflow.jsonl) · [Board timings](../results/snapflow/iq9075.json) · [Fixture hashes](../results/snapflow/fixture-lineage.json).

## Train

Use the [workstation environment](reproduce.md) with PyTorch, LeRobot and a video backend such as TorchCodec. The trainer holds out every fifth demonstration in each task and masks action targets past the end of the recording. The starting checkpoint may already have seen these demonstrations. [LeRobot LIBERO dataset](https://huggingface.co/datasets/lerobot/libero).

```bash
DATA_REV=a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4
hf download lerobot/libero --repo-type dataset --revision "$DATA_REV" --local-dir datasets/lerobot-libero
python -m smolvla_iq9075.snapflow --demos datasets/lerobot-libero --dataset-revision "$DATA_REV" \
  --output checkpoints/libero-snapflow --updates 400 --warmup 100 --lr 3e-6 --batch 4 --data-workers 12 --bf16
```

The command writes a LeRobot checkpoint and a `snapflow.json` report containing its source hash, optimization settings and masked action MSE for the original 10-NFE policy, the unchanged checkpoint at 1-NFE, and the trained 1-NFE policy. Use `--checkpoint` and `--vlm` to load locally cached model snapshots. The trainer also accepts the original LIBERO HDF5 suite directories with `h5py` installed.

## Evaluate and deploy

Compare the three model conditions using the same initial-state indices, episode seeds and execution window.

```bash
LIBERO_ROOT=upstream/LIBERO/libero/libero
VLM_DIR="$(python -c 'from smolvla_iq9075.native import snapshots; print(snapshots()[1])')"
python -m smolvla_iq9075.libero_batch --libero-root "$LIBERO_ROOT" \
  --batch-size 12 --execute 10 --inits 2 --fresh-env --flow-steps 10 --output runs/10nfe.jsonl
python -m smolvla_iq9075.libero_batch --libero-root "$LIBERO_ROOT" \
  --batch-size 12 --execute 10 --inits 2 --fresh-env --flow-steps 1 --output runs/naive-1nfe.jsonl
python -m smolvla_iq9075.libero_batch --libero-root "$LIBERO_ROOT" \
  --batch-size 12 --execute 10 --inits 2 --fresh-env --checkpoint checkpoints/libero-snapflow \
  --vlm "$VLM_DIR" --output runs/snapflow.jsonl
python -m smolvla_iq9075.compare runs/10nfe.jsonl runs/naive-1nfe.jsonl
python -m smolvla_iq9075.compare runs/10nfe.jsonl runs/snapflow.jsonl \
  --snapflow checkpoints/libero-snapflow/snapflow.json
python -m smolvla_iq9075.export --checkpoint checkpoints/libero-snapflow \
  --vlm "$VLM_DIR" --output bundles/libero-snapflow
```

Time the exported bundle using the [same 20 IQ-9075 fixtures and board procedure](reproduce.md#latency-on-iq-9075). For timed rollouts use the [chunk pacing runner](chunk-pacing.md). The execution window is independent of the model's denoising evaluation count. Use the folded checkpoint with one evaluation and use the original checkpoint for Euler step-count sweeps.
