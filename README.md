# SmolVLA on Qualcomm IQ-9075

An ONNX exporter, dual-HTP runtime and paired LIBERO benchmark for the public [SmolVLA LIBERO checkpoint](https://huggingface.co/HuggingFaceVLA/smolvla_libero).

[Reproduce the evaluation](docs/reproduce.md) covers installation, model export, board latency measurement and closed-loop evaluation. The board accepts two RGB arrays, task text, low-dimensional state and a reproducible initial noise tensor. It returns a full action chunk that LIBERO can execute in simulation.

## Full LIBERO success rate

Each policy evaluated **2,000 episodes** across four LIBERO suites, with 50 published initial states per task. All three use native LeRobot inference with 50 predicted actions and ten executed per query.

| Suite | Original 10-step | Undistilled 2-step | Distilled 2-step |
| --- | ---: | ---: | ---: |
| Spatial | 66.80% (334/500) | 80.40% (402/500) | 59.00% (295/500) |
| Object | 89.80% (449/500) | 97.20% (486/500) | 90.20% (451/500) |
| Goal | 76.60% (383/500) | 86.60% (433/500) | 74.60% (373/500) |
| Long | 37.00% (185/500) | 49.40% (247/500) | 41.20% (206/500) |
| **All** | 67.55% (1,351/2,000) | **78.40% (1,568/2,000)** | 66.25% (1,325/2,000) |

[Full protocol and episode logs](docs/libero-full.md).

## Measured on IQ-9075

On 20 matched LIBERO observations, dual HTP ran in **394.1 ms** median and **405.2 ms** p95. On the same IQ-9075, eager LeRobot took **42.7 s** median (**108.3×** slower), and ORT CPU took **5,958.1 ms** with the same exported graphs (**15.1×** slower). Running the two camera views on separate HTPs reduced median vision latency from 148.8 to 76.5 ms compared with a single HTP.

On the 20 fixed observations, relative L2 error for the first ten actions was **1.44%**. The board also ran a separate 40-episode closed-loop pilot. [IQ-9075 measurements](docs/results.md).

See [benchmark results](docs/results.md) for stage timings, suite results, full-chunk action error and raw measurements.

## What runs where

| Stage | Implementation |
| --- | --- |
| Observation and text | Host-side tokenizer, separable bilinear resize, checkpoint normalization and embedding lookup |
| Vision encoder | Static single-view ONNX graph, with two camera views scheduled concurrently on separate HTPs |
| Language prefill | Static partitions of up to six layers with grouped-query attention and per-layer K/V outputs |
| Action expert | Static one-step graph reused across ten denoising steps, reusing prefix projections and positional inputs |
| Action output | Host-side action denormalization from the saved checkpoint statistics |

The runtime uses QNN's FP16 path on HTP. Export preserves shapes and parameter values as loaded from the public checkpoint, serializes the flow timestep embeddings, and uses rescaled RMSNorm to keep its squared activations in FP16 range. The FP32 ONNX CPU timing uses the same exported graphs.

## Two-step distillation

Train an Action Expert that samples in two calls and compare it with the original ten-step policy and an untrained two-step control. See the [distillation recipe](docs/distillation.md).

## Use another checkpoint

The exporter derives camera order, model dimensions, action chunk size, normalization statistics and denoising count from the checkpoint. Pass `--checkpoint` and `--vlm` for a compatible SmolVLA snapshot. For another architecture, provide an exporter and policy backend with the following inference contract.

```python
action_shape = (1, chunk_length, padded_action_width)
actions = policy(rgb_views, state, instruction, initial_noise)
```

Inputs are ordered HWC uint8 RGB arrays, a raw float32 state vector, a string and float32 noise of shape `action_shape`. Outputs are denormalized actions with dimensions `[batch, chunk_length, action_width]`. Plug a backend into the LIBERO `episode()` driver directly, or give it a `backend` label and a `manifest["checkpoint_sha256"]` value and host it with `smolvla_iq9075.remote.serve()`. The same remote evaluator and paired-results comparator work with these backends. The ONNX bundle manifest defines the supported graph and preprocessing layout for the included runtime.

## Attribution

Released under [Apache 2.0](LICENSE). See [NOTICE](NOTICE) for upstream attribution. Checkpoints, third-party runtime binaries and benchmark assets are provided by their respective publishers. [SmolVLA paper](https://arxiv.org/abs/2506.01844), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [ONNX Runtime QNN](https://github.com/onnxruntime/onnxruntime-qnn).
