# IQ-9075 benchmark results

The reference latency and closed-loop board tests use `lerobot/smolvla_libero` at revision `a24f1f3`, with 50 predicted actions and ten denoising steps. The export and native reference use the same checkpoint weights. The manifest records file checksums and library versions. This export loaded a local copy of the pinned snapshot, and its checkpoint SHA-256 agrees with the evaluation protocol. [Checkpoint](https://huggingface.co/lerobot/smolvla_libero/tree/a24f1f380461b665158d52acd140152c79e8b36d) · [Export manifest](../results/export-manifest.json) · [Evaluation protocol](../results/libero-full/protocol.json).

## On-device latency

Measurements used a Qualcomm IQ-9075 EVK with eight Cortex-A78C CPU cores and two HTPs. All engines received the same first observation from task IDs 0–4 in each LIBERO suite, for 20 observations total. The board ran three warm-ups per engine. ORT and PyTorch used four inference threads, with one OpenBLAS thread for the host projections. Timing starts with both RGB arrays in memory and ends with 50 denormalized actions. Camera capture, transport and motor control are outside this measurement. [ONNX and QNN timings](../results/iq9075.json) · [Native PyTorch timings](../results/iq9075-native.json).

| IQ-9075 engine | Median | p95 | Maximum | Cold initialization |
| --- | ---: | ---: | ---: | ---: |
| Native LeRobot (PyTorch CPU) | 28,749.1 ms | 29,915.4 ms | 29,964.8 ms | 29.6 s |
| ORT CPU FP32 | 5,812.8 ms | 6,001.3 ms | 6,056.4 ms | 3.0 s |
| QNN single HTP FP16 | 323.2 ms | 329.6 ms | 329.7 ms | 64.2 s |
| QNN dual HTP FP16 | 249.7 ms | 254.4 ms | 255.1 ms | 78.6 s |

| Transition | Median latency before → after | Speedup | What changes |
| --- | ---: | ---: | --- |
| PyTorch CPU → ORT CPU | 28,749.1 → 5,812.8 ms | 4.9× | Static ONNX graphs and ORT execution on CPU |
| ORT CPU → single HTP | 5,812.8 → 323.2 ms | 18.0× | The same graphs move from FP32 on the CPU to QNN FP16 on one HTP |
| Single HTP → dual HTP | 323.2 → 249.7 ms | 1.3× | The camera views run concurrently and the prefill groups alternate between HTPs |
| Dual HTP, 10-step → undistilled 2-step | 249.7 → 173.1 ms | 1.4× | The Euler schedule runs two updates with the original weights |

Across the 10-step configurations, dual HTP was **115.1×** faster than eager LeRobot on the board CPU and **23.3×** faster than ORT CPU. Both HTP device nodes opened during the dual run. [Two-step timings](distillation.md#iq-9075-latency-and-action-agreement).

| Median per stage | ORT CPU | Single HTP | Dual HTP |
| --- | ---: | ---: | ---: |
| Preprocessing | 18.4 ms | 18.1 ms | 18.0 ms |
| Vision, two views | 4,069.7 ms | 148.4 ms | 76.2 ms |
| Language prefill | 561.2 ms | 35.6 ms | 35.8 ms |
| Host K/V projections and masks | 21.2 ms | 22.3 ms | 21.3 ms |
| Action Expert and output scaling | 1,154.1 ms | 98.3 ms | 98.0 ms |

From single to dual HTP, vision latency fell **48.6%** and prefill stayed near 36 ms. The ORT CPU and HTP runs use the same 6+6+4 prefill partition and reuse cross-attention K/V projections and masks between denoising steps. Each stage has its own median. Cold initialization includes session creation and HTP graph finalization outside the warm inference timers.

## Action agreement on fixed observations

Relative L2 divides the pooled L2 norm of the error by the pooled L2 norm of the native CUDA output. Each comparison uses the same RGB images, state, task text and initial noise. Action values are denormalized.

| Against native LeRobot on CUDA | First 10 actions | Full 50-action chunk |
| --- | ---: | ---: |
| Native LeRobot (PyTorch CPU) | 0.23% | 0.35% |
| ORT CPU FP32 | 0.29% | 0.28% |
| QNN single HTP FP16 | 0.46% | 3.08% |
| QNN dual HTP FP16 | 0.46% | 3.08% |

The dual HTP mean absolute error was 0.00122 for the first ten actions and 0.00148 over all 50. The closed-loop pilot evaluates the first ten actions from each prediction.

## Closed-loop IQ-9075 pilot

The native comparison runs LeRobot v0.4.4 on an NVIDIA RTX PRO 6000 Blackwell. The IQ-9075 candidate uses the dual-HTP export from the timing study. Both runs render the same LIBERO tasks on the GPU host and use initial-state index 0 for all 40 tasks. The action budgets for Spatial, Object, Goal and Long are 280, 280, 300 and 520 ticks, respectively, following ten warm-up ticks. Both policies predict 50 actions and execute ten before the next observation. For the full LIBERO score, see the [2,000-episode native evaluations](libero-full.md).

| Suite | Native LeRobot | IQ-9075 dual HTP | Change |
| --- | ---: | ---: | ---: |
| Spatial | 9/10 | 10/10 | +10 pp |
| Object | 9/10 | 10/10 | +10 pp |
| Goal | 9/10 | 9/10 | 0 pp |
| Long | 6/10 | 7/10 | +10 pp |
| **All** | 33/40 | 36/40 | +7.5 pp |

Both engines succeeded in 32 episodes and failed in three. Native LeRobot recorded one unique success and dual HTP recorded four.

[Native episode log](../results/libero/native.jsonl) · [IQ-9075 episode log](../results/libero/dual-htp.jsonl) · [Paired counts](../results/libero/paired.json).
