# SmolVLA on Qualcomm IQ-9075

An ONNX exporter, dual-HTP runtime and paired LIBERO benchmark for `lerobot/smolvla_libero`. The pinned checkpoint stores 450,046,176 parameters, with 16 language layers and 16 Action Expert layers. [Checkpoint](https://huggingface.co/lerobot/smolvla_libero/tree/a24f1f380461b665158d52acd140152c79e8b36d).

[Reproduce the evaluation](docs/reproduce.md) covers installation, model export, board latency measurement and closed-loop evaluation. The board accepts two RGB arrays, task text, low-dimensional state and reproducible initial noise. It returns a full action chunk that LIBERO can execute in simulation.

## Full LIBERO success rate

Each column evaluates **2,000 episodes** across four LIBERO suites, using 50 published initial states per task and native LeRobot inference. All modes predict 50 actions. The published checkpoint's saved default executes all 50 before re-observing. The three deployment variants execute ten actions per query.

| Suite | Checkpoint default (10-step, execute 50) | Original 10-step (execute 10) | Undistilled 2-step (execute 10) | Distilled 2-step (execute 10) |
| --- | ---: | ---: | ---: | ---: |
| Spatial | 70.20% (351/500) | 87.80% (439/500) | 85.60% (428/500) | 82.00% (410/500) |
| Object | 66.80% (334/500) | 90.60% (453/500) | 91.20% (456/500) | 91.20% (456/500) |
| Goal | 78.40% (392/500) | 88.20% (441/500) | 89.20% (446/500) | 86.00% (430/500) |
| Long | 47.00% (235/500) | 65.60% (328/500) | 67.40% (337/500) | 63.60% (318/500) |
| **All** | 65.60% (1,312/2,000) | 83.05% (1,661/2,000) | 83.35% (1,667/2,000) | 80.70% (1,614/2,000) |

[Full protocol and episode logs](docs/libero-full.md).

## Measured on IQ-9075

On 20 matched LIBERO observations, the original ten-step model ran in **249.7 ms** median and **254.4 ms** p95 on dual HTP. That was **115.1×** faster than PyTorch CPU and **23.3×** faster than ONNX Runtime CPU on the same board. For the undistilled two-step model, the dual-HTP median was **173.1 ms**. Timing covers in-memory RGB to all 50 actions.

The first ten actions had 0.46% relative L2 error against native LeRobot on CUDA for the original model. In the paired closed-loop pilot, native LeRobot succeeded in 33/40 episodes and dual HTP in 36/40.

[Stage timings, action agreement and board pilot](docs/results.md) · [Two-step results](docs/distillation.md).

## What runs where

| Stage | Implementation |
| --- | --- |
| Observation and text | Host-side tokenizer, separable bilinear resize, checkpoint normalization and embedding lookup |
| Vision encoder | Static single-view ONNX graph, with two camera views scheduled concurrently on separate HTPs |
| Language prefill | Static partitions of up to six layers with grouped-query attention and per-layer K/V outputs |
| Action expert | Static one-step graph reused across the configured denoising steps, reusing prefix projections and positional inputs |
| Action output | Host-side action denormalization from the saved checkpoint statistics |

The runtime uses QNN's FP16 path on HTP. Export preserves shapes and parameter values as loaded from the public checkpoint, serializes the flow timestep embeddings, and uses rescaled RMSNorm to keep its squared activations in FP16 range. The FP32 ONNX CPU timing uses the same exported graphs.

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
