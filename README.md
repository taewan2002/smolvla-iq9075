# SmolVLA on Qualcomm IQ-9075

An ONNX exporter, dual-HTP runtime and paired LIBERO benchmark for `lerobot/smolvla_libero`. The pinned checkpoint stores 450,046,176 parameters, with 16 language layers and 16 Action Expert layers. [Checkpoint](https://huggingface.co/lerobot/smolvla_libero/tree/a24f1f380461b665158d52acd140152c79e8b36d).

[Reproduce the evaluation](docs/reproduce.md) covers installation, model export, board latency measurement and closed-loop evaluation. The board accepts two RGB arrays, task text, low-dimensional state and reproducible initial noise. It returns a full action chunk that LIBERO can execute in simulation.

## IQ-9075 speedup

The original 10-step checkpoint went from **28,749.1 ms on PyTorch CPU to 249.7 ms on dual HTP**, a **115.1× speedup**. Against ONNX Runtime CPU on the same exported graphs, dual HTP was **23.3× faster**. All rows use the same 20 LIBERO observations, measured from two RGB images in memory through 50 denormalized actions after three warm-up calls.

| Runtime on IQ-9075 | Denoising steps | Median | p95 | Speedup vs. PyTorch CPU |
| --- | ---: | ---: | ---: | ---: |
| Native LeRobot, PyTorch CPU | 10 | 28,749.1 ms | 29,915.4 ms | 1.0× |
| ONNX Runtime CPU, FP32 | 10 | 5,812.8 ms | 6,001.3 ms | 4.9× |
| QNN Adreno 663 GPU, FP32 recheck | 10 | 27,891.6 ms | 27,919.3 ms | 1.03× |
| QNN Adreno 663 GPU, FP16 | 10 | 21,540.7 ms | 21,577.8 ms | 1.33× |
| QNN single HTP, FP16 | 10 | 323.2 ms | 329.6 ms | 89.0× |
| QNN dual HTP, FP16 | 10 | 249.7 ms | 254.4 ms | 115.1× |
| QNN dual HTP, FP16, undistilled | 2 | **173.1 ms** | **176.9 ms** | **166.1×** |

The 10-step rows use the same checkpoint weights and denoising schedule. The last row keeps the weights and cuts the Euler schedule from ten updates to two. [Benchmark protocol and raw timings](docs/results.md).

The Adreno FP32 recheck reproduced the original **27,900.8 ms** median with a **0.03%** difference on the same 20 inputs. FP16 reduced two-view vision latency from **20,649.7 to 15,777.5 ms (23.6%)** and total latency by **22.8%**. The FP16 GPU pass still took **66.6×** as long as single HTP and **86.3×** as long as dual HTP. The Adreno runs used the GPU device with graph CPU fallback disabled. [Adreno benchmark details](docs/results.md#adreno-gpu-comparison).

## What changed

- **Static ONNX export and QNN FP16.** Export fixed-shape graphs for a single camera view, the language prefill partitions and one Action Expert update. Resize, tokenize, normalize and scale actions on the host. Run the graphs on HTP via the QNN execution provider and rescale RMSNorm activations to stay within FP16 range.
- **Dual-HTP scheduling.** Encode the two camera views concurrently, one per HTP. Vision latency fell from **148.4 to 76.2 ms (−48.6%)** compared with single HTP. Full inference fell from **323.2 to 249.7 ms (−22.7%)**.
- **Partitioned language prefill.** Split the 16 language layers into groups of 6, 6 and 4, run successive groups on alternating HTPs, and export the per-layer attention K/V cache. Prefill on dual HTP took **35.8 ms**, compared with **561.2 ms** for the same partitions on ORT CPU.
- **Reused Action Expert inputs.** Precompute flow timestep embeddings at export. Build the cross-attention K/V projections, position encodings and masks once per observation and reuse them in each Euler update. With the two-step schedule, median Action Expert latency including output scaling fell from **98.0 to 20.3 ms**, and full inference fell **30.7%** to **173.1 ms**. [Two-step benchmark](docs/distillation.md).

## Full LIBERO success rate

Each column evaluates **2,000 episodes** across four LIBERO suites, using 50 published initial states per task and native LeRobot inference. All modes predict 50 actions. The published checkpoint's saved default executes all 50 before re-observing. The three deployment variants execute ten actions per query.

| Suite | Checkpoint default (10-step, execute 50) | Original 10-step (execute 10) | Undistilled 2-step (execute 10) | Distilled 2-step (execute 10) |
| --- | ---: | ---: | ---: | ---: |
| Spatial | 70.20% (351/500) | 87.80% (439/500) | 85.60% (428/500) | 82.00% (410/500) |
| Object | 66.80% (334/500) | 90.60% (453/500) | 91.20% (456/500) | 91.20% (456/500) |
| Goal | 78.40% (392/500) | 88.20% (441/500) | 89.20% (446/500) | 86.00% (430/500) |
| Long | 47.00% (235/500) | 65.60% (328/500) | 67.40% (337/500) | 63.60% (318/500) |
| **All** | 65.60% (1,312/2,000) | 83.05% (1,661/2,000) | 83.35% (1,667/2,000) | 80.70% (1,614/2,000) |

The original 10-step model's first ten actions had 0.46% relative L2 error on dual HTP against native LeRobot on CUDA. In a separate matched 40-episode closed-loop pilot, native LeRobot succeeded in 33/40 episodes and dual HTP in 36/40. [Full LIBERO protocol and episode logs](docs/libero-full.md) · [Board pilot and action agreement](docs/results.md).

## Two-step distillation

Train an Action Expert that samples in two calls and compare it with the original ten-step policy and an undistilled two-step control. See the [distillation recipe](docs/distillation.md).

## Use another checkpoint

The exporter derives camera order, model dimensions, action chunk size, normalization statistics and denoising count from the checkpoint. Pass `--checkpoint` and `--vlm` for a compatible SmolVLA snapshot. Use `--flow-steps` to export a different Euler schedule with the same weights. For another architecture, provide an exporter and policy backend with the following inference contract.

```python
action_shape = (1, chunk_length, padded_action_width)
actions = policy(rgb_views, state, instruction, initial_noise)
```

Inputs are ordered HWC uint8 RGB arrays, a raw float32 state vector, a string and float32 noise of shape `action_shape`. Outputs are denormalized actions with dimensions `[batch, chunk_length, action_width]`. Plug a backend into the LIBERO `episode()` driver directly, or give it a `backend` label and a `manifest["checkpoint_sha256"]` value and host it with `smolvla_iq9075.remote.serve()`. The same remote evaluator and paired-results comparator work with these backends. The ONNX bundle manifest defines the supported graph and preprocessing layout for the included runtime.

## Attribution

Released under [Apache 2.0](LICENSE). See [NOTICE](NOTICE) for upstream attribution. Checkpoints, third-party runtime binaries and benchmark assets are provided by their respective publishers. [SmolVLA paper](https://arxiv.org/abs/2506.01844), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [ONNX Runtime QNN](https://github.com/onnxruntime/onnxruntime-qnn).
