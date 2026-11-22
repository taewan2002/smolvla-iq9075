# IQ-9075 benchmark results

The study used the public [SmolVLA LIBERO checkpoint at revision `6721902`](https://huggingface.co/HuggingFaceVLA/smolvla_libero/tree/6721902bc4d61e50a3bfdb11dfb4cb626f05d102), with 50 predicted actions per inference and ten denoising steps. The exported model and all rollouts use identical checkpoint weights. The [export manifest](../results/export-manifest.json) records file checksums and tool versions.

## On-device latency

Measurements were collected on a Qualcomm IQ-9075 EVK with eight Cortex-A78C CPU cores and two HTPs. All engines received the first policy observation from the same 20 LIBERO episodes, covering task IDs 0–4 in each of the four suites. The board processed one observation at a time after three warm-ups per engine, with four ORT intra-op threads and one OpenBLAS thread for the host projections. Time starts with both RGB arrays in memory and ends with all 50 denormalized actions. The eager LeRobot reference used four PyTorch CPU threads and the same inputs. See the [ONNX and QNN raw timings](../results/iq9075.json) and [native PyTorch raw timings](../results/iq9075-native.json).

| IQ-9075 engine | Median | p95 | Maximum | Cold initialization |
| --- | ---: | ---: | ---: | ---: |
| Native LeRobot (PyTorch CPU) | 42,698.3 ms | 43,880.6 ms | 44,117.1 ms | 26.4 s |
| ORT CPU FP32 | 5,958.1 ms | 6,346.1 ms | 6,393.3 ms | 4.3 s |
| QNN single HTP FP16 | 463.8 ms | 473.1 ms | 474.0 ms | 96.7 s |
| QNN dual HTP FP16 | **394.1 ms** | **405.2 ms** | **406.7 ms** | 105.7 s |

The dual schedule was **108.3×** faster than eager LeRobot on the board CPU and reduced median latency by **93.4%** versus ORT CPU, a **15.1×** speedup on the same exported graphs. It reduced median latency by **15.0%** versus single HTP. Both HTP device nodes opened during the dual run. QNN FP16 produced identical aggregate action errors with both HTP schedules.

| Median per stage | ORT CPU | Single HTP | Dual HTP |
| --- | ---: | ---: | ---: |
| Preprocessing | 18.2 ms | 18.9 ms | 18.5 ms |
| Vision, two views | 3,574.0 ms | 148.8 ms | 76.5 ms |
| Language prefill | 1,025.4 ms | 72.5 ms | 73.4 ms |
| Host K/V projections and masks | 38.4 ms | 43.1 ms | 42.9 ms |
| Action Expert and output scaling | 1,194.4 ms | 180.8 ms | 183.0 ms |

Each stage reports its own median. Cold initialization includes ONNX session creation and HTP graph finalization and sits outside the warm inference timers.

## Action agreement on fixed observations

Relative L2 is the pooled L2 norm of the output difference divided by the pooled L2 norm of the native LeRobot actions. This compares denormalized values with identical RGB, state, instruction and initial noise.

| Against native LeRobot on CUDA | First 10 actions | Full 50-action chunk |
| --- | ---: | ---: |
| ORT CPU FP32 | 0.96% | 9.83% |
| QNN dual HTP FP16 | 1.44% | 13.37% |

The dual HTP mean absolute error was 0.00328 for the first 10 actions and 0.00657 across all 50. The full-chunk error is sensitive to large gripper differences in the later part of the prediction. In the FP32 export check, for example, the largest difference was 1.99 on the gripper dimension at action index 39 for Goal task 1 (zero-based). The paired rollouts execute the first 10 actions, and the full-chunk results matter for applications using a longer execution horizon.

## Closed-loop IQ-9075 pilot

See the [full native LeRobot evaluation](libero-full.md) for the 2,000-episode policy scores.

The native comparison uses LeRobot v0.4.4 on an NVIDIA RTX PRO 6000 Blackwell. The IQ-9075 candidate uses dual HTP through the same export as the latency study. The same GPU host renders both runs. The separate PyTorch CPU timing run uses the board. Both backends receive the same task, initial state, action noise seed and simulator settings. Each suite uses all ten tasks with initial-state index 0. Actions use LIBERO's relative controller, and each run allows 280, 280, 300 or 520 action ticks for Spatial, Object, Goal or Long respectively, after ten warm-up ticks.

The model predicts 50 actions and both runners execute ten per query before taking another observation. The checkpoint's default is one action per query. The fixed-state pilot and the execution cadence define the scope of the score.

| Suite | Native LeRobot | IQ-9075 dual HTP | Change |
| --- | ---: | ---: | ---: |
| Spatial | 8/10 | 9/10 | +10 pp |
| Object | 9/10 | 8/10 | -10 pp |
| Goal | 6/10 | 5/10 | -10 pp |
| Long | 3/10 | 4/10 | +10 pp |
| **All** | 26/40 | 26/40 | +0 pp |

Both engines succeeded in 22 episodes and failed in 10. The remaining eight outcomes split evenly, with four successes unique to each engine. The equal 65% aggregate applies to this matched 40-episode pilot.

[Native episode log](../results/libero/native.jsonl), [IQ-9075 episode log](../results/libero/dual-htp.jsonl), [paired counts](../results/libero/paired.json).
