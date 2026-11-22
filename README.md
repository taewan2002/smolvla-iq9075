# SmolVLA on Qualcomm IQ-9075

An ONNX exporter, dual-HTP runtime and paired LIBERO benchmark for `lerobot/smolvla_libero`. The pinned checkpoint stores 450,046,176 parameters, with 16 language layers and 16 Action Expert layers. [Checkpoint](https://huggingface.co/lerobot/smolvla_libero/tree/a24f1f380461b665158d52acd140152c79e8b36d).

[Reproduce the evaluation](docs/reproduce.md) covers installation, model export, board latency measurement and closed-loop evaluation. The board accepts two RGB arrays, task text, low-dimensional state and reproducible initial noise. It returns a full action chunk that LIBERO can execute in simulation.

## Default IQ-9075 deployment

The pinned export uses the original weights and two Euler steps. It predicts 50 actions and records a ten-action execution window.

```bash
python -m smolvla_iq9075.export --output bundles/libero
```

The default paced example uses **25 Hz**. With an eight-action refill lead, the control-rate pilot completed eight simulated rollouts at this rate with zero empty queue ticks and zero skipped control slots. [Control-rate sweep and commands](docs/chunk-pacing.md#two-step-deployment-rates).

## IQ-9075 HTP inference

The reference configuration is the original 10-step LeRobot graph on a single QNN HTP in FP16. The installed QNN provider rejected some operators in that graph during session creation. The measured runs each use the same 20 LIBERO observations and time one prediction from two RGB images in memory through 50 denormalized actions, after three warm-up calls.

| Runtime on IQ-9075 | Denoising steps | Median | p95 |
| --- | ---: | ---: | ---: |
| QNN single HTP, FP16, unoptimized reference | 10 | Unavailable | — |
| QNN single HTP, FP16, optimized export | 10 | 323.2 ms | 329.6 ms |
| QNN dual HTP, FP16, optimized export | 10 | 249.7 ms | 254.4 ms |
| QNN dual HTP, FP16, original weights | 2 | 173.6 ms | 182.1 ms |
| QNN dual HTP, FP16, original weights | 1 | 162.9 ms | 173.4 ms |
| QNN dual HTP, FP16, SnapFlow export | 1 | 164.1 ms | 167.2 ms |

Using the same optimized export and ten denoising steps, dual HTP reduced median prediction latency from **323.2 to 249.7 ms (−22.7%)**. [Benchmark protocol and raw timings](docs/results.md).

## Timed asynchronous rollout

The simulator runs at 25 Hz while IQ-9075 dual HTP predicts 50 actions per query. Each chunk supplies ten commands. In the two-step pilot, every mode used three-step cosine blending for aligned position and rotation commands. Gripper commands came from the new chunk.

| Two-step execution | Success | Median chunk wait | Empty ticks |
| --- | ---: | ---: | ---: |
| Synchronous | 7/8 | 320 ms | 1,288 |
| Async, request with 3 actions left | 7/8 | 160 ms | 804 |
| Async, request with 5 actions left | 5/8 | 120 ms | 637 |

The three-action lead halved median chunk wait in this shared-board pilot. Sync and the three-action lead each solved one episode the other missed. Timing covers completed chunk handoffs after startup. [Matched episodes, board activity and reproduction](docs/chunk-pacing.md#two-step-refill-lead-at-25-hz).

### Ten-step comparison

The earlier pilot used ten denoising steps and an eight-action refill lead in both async modes.

| Execution | Success | Median chunk wait | Empty ticks |
| --- | ---: | ---: | ---: |
| Synchronous | 31/40 | 320 ms | 7,130 (44.72%) |
| Asynchronous | 23/40 | 0 ms | 479 (4.72%) |
| Asynchronous + 3-step cosine | 26/40 | 0 ms | 836 (8.22%) |

The paired trial covers 40 matching LIBERO initial conditions per mode. Chunk wait and empty ticks exclude startup. The control rate is configurable. [Timing definitions, command continuity and reproduction](docs/chunk-pacing.md).

The cosine blend reduced the p95 position-command jump from 0.850 to 0.235 relative to plain async and recovered three successes. Both async settings reduced waiting substantially, and sync retained the highest success count in this pilot.

## What changed

- **Static ONNX export and QNN FP16.** Export fixed-shape graphs for a single camera view, the language prefill partitions and one Action Expert update. Resize, tokenize, normalize and scale actions on the host. Run the graphs on HTP via the QNN execution provider and rescale RMSNorm activations to stay within FP16 range.
- **Dual-HTP scheduling.** Encode the two camera views concurrently, one per HTP. Vision latency fell from **148.4 to 76.2 ms (−48.6%)** compared with single HTP. Full inference fell from **323.2 to 249.7 ms (−22.7%)**.
- **Partitioned language prefill.** Split the 16 language layers into groups of 6, 6 and 4, run successive groups on alternating HTPs, and export the per-layer attention K/V cache. Prefill on dual HTP took **35.8 ms**.
- **Reused Action Expert inputs.** Precompute flow timestep embeddings at export. Build the cross-attention K/V projections, position encodings and masks once per observation and reuse them in each Euler update.

## Full LIBERO success rate

Each column evaluates **2,000 episodes** across four LIBERO suites, using 50 published initial states per task and native LeRobot inference. All modes predict 50 actions. The published checkpoint's saved default executes all 50 before re-observing. The 10-, 2-, and 1-step solver settings execute ten actions per query with the original weights.

| Suite | Checkpoint default (10 steps, execute 50) | 10 steps (execute 10) | 2 steps (execute 10) | 1 step (execute 10) |
| --- | ---: | ---: | ---: | ---: |
| Spatial | 70.20% (351/500) | 87.80% (439/500) | 85.60% (428/500) | 85.00% (425/500) |
| Object | 66.80% (334/500) | 90.60% (453/500) | 91.20% (456/500) | 87.20% (436/500) |
| Goal | 78.40% (392/500) | 88.20% (441/500) | 89.20% (446/500) | 85.00% (425/500) |
| Long | 47.00% (235/500) | 65.60% (328/500) | 67.40% (337/500) | 63.60% (318/500) |
| **All** | 65.60% (1,312/2,000) | 83.05% (1,661/2,000) | 83.35% (1,667/2,000) | 80.20% (1,604/2,000) |

One step cut median latency by **6.2%** on matched inputs. Aggregate LIBERO success fell by **3.15 percentage points** from two steps. [Matched episodes and reproduction](docs/libero-full.md).

The original 10-step model's first ten actions had 0.46% relative L2 error on dual HTP against native LeRobot on CUDA. In a separate matched 40-episode closed-loop pilot, native LeRobot succeeded in 33/40 episodes and dual HTP in 36/40. [Full LIBERO protocol and episode logs](docs/libero-full.md) · [Board pilot and action agreement](docs/results.md).

## SnapFlow one-step action generation

SnapFlow adapts the original checkpoint with flow matching and shortcut targets from the same model. The trained policy calls the Action Expert once, reaching **164.1 ms** median and **167.2 ms** p95 on dual HTP. On 80 paired LIBERO episodes, the original 10-NFE policy succeeded in **70/80**, the unchanged checkpoint at 1-NFE in **70/80**, and SnapFlow in **69/80**. Each predicts 50 actions and executes ten per query. [Training and matched results](docs/snapflow.md).

## Use another checkpoint

The exporter derives camera order, model dimensions, action chunk size and normalization statistics from the checkpoint. Pass `--checkpoint` and `--vlm` for a compatible SmolVLA snapshot. Local snapshots use their saved denoising and execution counts unless you pass `--flow-steps` or `--execute`. For another architecture, provide an exporter and policy backend with the following inference contract.

```python
action_shape = (1, chunk_length, padded_action_width)
actions = policy(rgb_views, state, instruction, initial_noise)
```

Inputs are ordered HWC uint8 RGB arrays, a raw float32 state vector, a string and float32 noise of shape `action_shape`. Outputs are denormalized actions with dimensions `[batch, chunk_length, action_width]`. Plug a backend into the LIBERO `episode()` driver directly, or give it a `backend` label and a `manifest["checkpoint_sha256"]` value and host it with `smolvla_iq9075.remote.serve()`. The same remote evaluator and paired-results comparator work with these backends. The ONNX bundle manifest defines the supported graph and preprocessing layout for the included runtime.

## Attribution

Released under [Apache 2.0](LICENSE). See [NOTICE](NOTICE) for upstream attribution. Checkpoints, third-party runtime binaries and benchmark assets are provided by their respective publishers. [SmolVLA paper](https://arxiv.org/abs/2506.01844), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [ONNX Runtime QNN](https://github.com/onnxruntime/onnxruntime-qnn).
