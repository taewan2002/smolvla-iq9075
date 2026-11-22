# SmolVLA on Qualcomm IQ-9075

An ONNX exporter, dual-HTP runtime and paired LIBERO benchmark for the public [SmolVLA LIBERO checkpoint](https://huggingface.co/HuggingFaceVLA/smolvla_libero).

[Reproduce the evaluation](docs/reproduce.md) covers installation, model export, board latency measurement and closed-loop evaluation. The board accepts two RGB arrays, task text, low-dimensional state and a reproducible initial noise tensor. It returns a full action chunk that LIBERO can execute in simulation.

## Measured on IQ-9075

On 20 matched LIBERO observations, dual HTP ran in **394.1 ms** median and **405.2 ms** p95. On the same IQ-9075, eager LeRobot took **42.7 s** median (**108.3×** slower), and ORT CPU took **5,958.1 ms** with the same exported graphs (**15.1×** slower). Running the two camera views on separate HTPs reduced median vision latency from 148.8 to 76.5 ms compared with a single HTP.

Closed-loop LIBERO success was **26/40 (65%)** for native LeRobot and **26/40 (65%)** for IQ-9075 dual HTP. The pilot used one published initial state per task across four suites. Both backends predicted 50 actions and executed 10 per query. On the 20 fixed observations, relative L2 error for those first 10 actions was **1.44%**.

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

## Use another checkpoint

The exporter derives camera order, model dimensions, action chunk size, normalization statistics and denoising count from the checkpoint. Pass `--checkpoint` and `--vlm` for a compatible SmolVLA snapshot. For another architecture, provide an exporter and policy backend with the following inference contract.

```python
action_shape = (1, chunk_length, padded_action_width)
actions = policy(rgb_views, state, instruction, initial_noise)
```

Inputs are ordered HWC uint8 RGB arrays, a raw float32 state vector, a string and float32 noise of shape `action_shape`. Outputs are denormalized actions with dimensions `[batch, chunk_length, action_width]`. Plug a backend into the LIBERO `episode()` driver directly, or give it a `backend` label and a `manifest["checkpoint_sha256"]` value and host it with `smolvla_iq9075.remote.serve()`. The same remote evaluator and paired-results comparator work with these backends. The ONNX bundle manifest defines the supported graph and preprocessing layout for the included runtime.

## Attribution

Released under [Apache 2.0](LICENSE). See [NOTICE](NOTICE) for upstream attribution. Checkpoints, third-party runtime binaries and benchmark assets are provided by their respective publishers. [SmolVLA paper](https://arxiv.org/abs/2506.01844), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [ONNX Runtime QNN](https://github.com/onnxruntime/onnxruntime-qnn).
