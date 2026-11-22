# Timed LIBERO rollouts on IQ-9075

The workstation runs LIBERO and issues simulated commands on a 25 Hz monotonic clock for the paired pilots. IQ-9075 predicts action chunks with the original SmolVLA weights on dual HTP. Each query predicts 50 actions. Each accepted chunk supplies ten commands, up to the episode's terminal condition. Within each pilot, all modes share the model, simulated controller, control frequency, published initial states, episode seeds and caps on policy actions. The original LIBERO evaluation uses its 20 Hz default. [LIBERO environment wrapper](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/env_wrapper.py).

## Chunk scheduling and smoothing

- **Sync.** Take a fresh observation after the tenth action and request the next chunk. On an empty tick, step the simulated controller with zero pose delta and the last gripper command.
- **Async.** Request the next chunk in a background worker with eight actions left to execute. The request records the number of policy actions already applied at capture time. At the boundary, use the slice in the forecast aligned to the current action index. The request at offset two therefore supplies forecast entries 8–17, with zero-based indexing. The worker holds one in-flight request at a time.
- **Async + cosine blend.** Blend the next three position and rotation deltas with the previous forecast at the same action indices. The new forecast weights are 0.25, 0.75 and 1.0. The gripper uses the new forecast throughout, and each chunk still supplies ten executed policy actions.

LeRobot's asynchronous inference overlaps execution with prediction. ALOHA Lightning (Flash ALOHA) blends overlapping joint-position targets with a sigmoid schedule. This runner applies a three-step cosine schedule to LIBERO's relative pose commands. [LeRobot asynchronous inference](https://huggingface.co/blog/async-robot-inference) · [ALOHA Lightning, Test-Time Smoothing](https://arxiv.org/html/2611.21919v1).

## Timing definitions

- A command tick is one slot on the 25 Hz host clock. Eight queued actions take 320 ms to apply. An entire ten-action execution window spans 400 ms. The model's prediction time, the capture and RPC path, and the number of remaining commands determine the refill deadline.
- An **empty tick** is a visited control slot after startup where the policy queue is empty. The simulator advances with the hold command. A **scheduler miss** is a slot skipped because the host reached a later deadline. Skipped slots never cause a burst of catch-up commands.
- **Chunk wait** is the number of clock intervals from the expected next chunk boundary to its first applied action, multiplied by 40 ms. The median includes all completed boundaries across the episodes, with zero for an immediate handoff. **Startup** covers the first prediction separately. Inference time includes client observation preparation, RPC and board computation.
- **Boundary jump** is the L2 difference between the last position or rotation command of one chunk and the first of the next. It measures command continuity in LIBERO's relative action units.

The action rate is configurable with `--control-hz`. At 50 Hz the ten-action window would span 200 ms, shorter than the measured 10-step dual-HTP prediction alone. A higher-frequency low-level controller can interpolate between policy setpoints while preserving the policy's action timing. These measurements concern simulator command ticks.

## Paired 25 Hz results

Each mode ran 40 episodes with the first published initial state for each of the ten tasks in all four LIBERO suites. The matched hashes include the first RGB frames and robot state for each episode after warm-up. The command clock and camera sampler ran on the simulator host and the board ran the 10-step QNN dual-HTP graph.

| Execution | Success | Median chunk wait | Empty ticks |
| --- | ---: | ---: | ---: |
| Synchronous | 31/40 | 320 ms | 7,130 (44.72%) |
| Asynchronous | 23/40 | 0 ms | 479 (4.72%) |
| Asynchronous + 3-step cosine | 26/40 | 0 ms | 836 (8.22%) |

| Execution | Late boundaries | Scheduler misses | Median episode excluding startup | Median / p95 query |
| --- | ---: | ---: | ---: | ---: |
| Synchronous | 849/849 | 0 | 12.05 s | 339.0 ms / 403.4 ms |
| Asynchronous | 219/940 | 0 | 10.41 s | 337.2 ms / 450.0 ms |
| Asynchronous + 3-step cosine | 286/906 | 0 | 8.15 s | 342.9 ms / 471.3 ms |

| Suite | Synchronous | Asynchronous | Async + cosine |
| --- | ---: | ---: | ---: |
| Spatial | 8/10 | 7/10 | 7/10 |
| Object | 9/10 | 5/10 | 7/10 |
| Goal | 8/10 | 6/10 | 7/10 |
| Long | 6/10 | 5/10 | 5/10 |
| **All** | 31/40 | 23/40 | 26/40 |

The following command jumps compare the last command of one chunk with the first command of the following chunk. Values are in the policy's relative action units.

| Execution | Position jump, p95 | Rotation jump, p95 |
| --- | ---: | ---: |
| Synchronous | 0.508 | 0.070 |
| Asynchronous | 0.850 | 0.065 |
| Asynchronous + 3-step cosine | 0.235 | 0.021 |

Compared with plain async, the cosine setting reduced p95 boundary jumps by 72% for translation and 67% for rotation. It succeeded on five episodes that plain async missed and missed two that plain async solved. The fixed eight-action refill lead uses an observation taken eight commands before the handoff. These success counts apply to that lead, command rate and action-slice alignment.

[Sync episode logs](../results/chunk-pacing/sync.jsonl) · [Async episode logs](../results/chunk-pacing/async.jsonl) · [Blend episode logs](../results/chunk-pacing/blend.jsonl) · [Paired summary](../results/chunk-pacing/summary.json).

## Reproduce

Start the IQ-9075 dual-HTP server and forward it to the simulator host as described for the [closed-loop pilot](reproduce.md#run-closed-loop-libero-with-the-board). Use the same pinned LIBERO assets and export the ten-step bundle with `--flow-steps 10 --execute 50`. On the simulation workstation:

```bash
export PYTHONHASHSEED=0
for mode in sync async blend; do
  case "$mode" in
    sync) options=(--refill-actions 0 --overlap 0) ;;
    async) options=(--refill-actions 8 --overlap 0) ;;
    blend) options=(--refill-actions 8 --overlap 3) ;;
  esac
  python -m smolvla_iq9075.libero --engine remote --url http://127.0.0.1:18075 \
    --libero-root upstream/LIBERO/libero/libero --control-hz 25 --execute 10 \
    --output "runs/paced-$mode.jsonl" "${options[@]}"
done
python -m smolvla_iq9075.compare_pacing runs/paced-sync.jsonl runs/paced-async.jsonl runs/paced-blend.jsonl
```

The comparison verifies the per-episode initial observation hash, simulator asset hashes, seeds, checkpoint and cadence. The benchmark uses initial-state index 0 for each task across all four suites.

## Two-step deployment rates

The sweep runs the default 2-step export on dual HTP FP16 with the original weights. The model predicts 50 actions, and the client applies ten per accepted chunk. The simulator requests the next chunk with eight commands left and uses three-step cosine blending. Every rate uses task 0 and its first published initial state in each LIBERO suite. The 25 and 30 Hz runs also use task 1 and its first initial state. The rate sets both the paced client clock and the simulator control frequency, so the fixed warm-up tick count can produce different initial observations across rates.

| Command rate | Ten-command window | Rollout success | Empty ticks | Skipped ticks | Query p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 Hz | 500.0 ms | 3/4 | 0 | 0 | 311.6 ms |
| 25 Hz | 400.0 ms | 6/8 | 0 | 0 | 283.5 ms |
| 30 Hz | 333.3 ms | 2/8 | 56 | 0 | 330.8 ms |
| 35 Hz | 285.7 ms | 1/4 | 112 | 0 | 343.6 ms |
| 40 Hz | 250.0 ms | 1/4 | 298 | 0 | 295.3 ms |
| 50 Hz | 200.0 ms | 0/4 | 657 | 0 | 314.5 ms |

**25 Hz** was the highest rate with zero empty queue ticks and zero scheduler misses in this sweep. The query p95 excludes startup and includes observation preparation, network transport and board inference. The board-only 2-step p95 of 182.1 ms would give a budget of about 55 actions/s at ten actions per query if prediction were the only delay. The rate sweep measures the workstation-to-board path and simulator command ticks. [Board-only trace](../results/solver-steps/iq9075-2nfe.json) · [Episode logs](../results/control-rate-2nfe/episodes.jsonl) · [Rate summary](../results/control-rate-2nfe/summary.json).

Use the default two-step export and [board service](reproduce.md#run-closed-loop-libero-with-the-board). Forward the board port to the simulator workstation and set the paced client to 25 Hz.

```bash
python -m smolvla_iq9075.libero --engine remote --url http://127.0.0.1:18075 \
  --libero-root upstream/LIBERO/libero/libero --execute 10 --control-hz 25 \
  --refill-actions 8 --overlap 3 --tasks 2 --output runs/dual-2step-25hz.jsonl
```

For the sweep, use task 0 with `--tasks 1` at 20, 25, 30, 35, 40 and 50 Hz. Add a second run with `--task-start 1 --tasks 1` at 25 and 30 Hz. Use a fresh output file for each rate and task group.

## Two-step refill lead at 25 Hz

The paired pilot used tasks 0 and 1 in each of the four LIBERO suites, with published initial-state index 0. IQ-9075 ran the original two-step FP16 model on dual HTP. Every mode predicted 50 actions, executed ten per chunk and used three-step cosine blending. The simulator ran at 25 Hz, with its episode horizon counting both policy and hold commands. The first RGB and robot-state hashes matched across all modes.

| Refill schedule | Success | Median chunk wait | Completed handoffs | Empty ticks |
| --- | ---: | ---: | ---: | ---: |
| Sync | 7/8 | 320 ms | 156 | 1,288 |
| Async with 3 actions left | 7/8 | 160 ms | 159 | 804 |
| Async with 5 actions left | 5/8 | 120 ms | 176 | 637 |

All modes recorded zero scheduler misses. The three-action lead succeeded on six of the same episodes as sync, recovered LIBERO Long task 1 and missed LIBERO Object task 0. A separate process briefly used one HTP, and camera and display services started during the sweep. Their effect on timing was not isolated. [Episode logs](../results/two-step-cosine-refill/episodes.jsonl) · [Paired summary](../results/two-step-cosine-refill/summary.json).

The recorded sweep interleaved the three modes for each initial state. To run each profile from the CLI with the default dual-HTP two-step bundle and the same simulator assets, run the following on the simulation workstation.

```bash
export PYTHONHASHSEED=0
for lead in 0 3 5; do
  python -m smolvla_iq9075.libero --engine remote --url http://127.0.0.1:18075 \
    --libero-root upstream/LIBERO/libero/libero --control-hz 25 --execute 10 \
    --refill-actions "$lead" --overlap 3 --tasks 2 --output "runs/dual-2step-lead-$lead.jsonl"
done
```
