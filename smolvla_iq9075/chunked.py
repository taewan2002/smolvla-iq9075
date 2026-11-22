"""Paced, single-request action chunk execution in a simulator."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter, sleep

import numpy as np


@dataclass(frozen=True)
class Pacing:
    control_hz: int
    refill_actions: int
    overlap: int


@dataclass(frozen=True)
class Prediction:
    origin: int
    actions: np.ndarray
    inference_ms: float


def select_actions(prediction, previous, applied, count, overlap):
    """Take the forecast for the next logical action ticks, blending matched ticks."""
    offset = applied - prediction.origin
    result = prediction.actions[offset:offset + count].copy()
    if offset < 0 or len(result) != count:
        raise ValueError("Prediction does not cover the next execution window")
    if previous is not None and overlap:
        offset = applied - previous.origin
        old = previous.actions[offset:offset + overlap]
        if offset < 0 or len(old) != overlap:
            raise ValueError("Previous prediction does not cover the overlap window")
        alpha = ((1 - np.cos(np.pi * np.arange(1, overlap + 1) / overlap)) / 2)[:, None]
        # LIBERO actions contain three position deltas, three rotation deltas and gripper.
        result[:overlap, :6] = (1 - alpha) * old[:, :6] + alpha * result[:overlap, :6]
    return result


def execute_episode(env, raw, observe, camera, policy, instruction, noise_rng, execute, pacing, limit):
    period = 1 / pacing.control_hz
    applied = empty = missed = submitted = slot = 0
    ready_window = None
    previous = None
    last_action = None
    hold = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    waits, inference, offsets, position_jumps, rotation_jumps = [], [], [], [], []

    def predict(origin, obs, noise, started):
        rgb, state = obs
        actions = policy(rgb, state, instruction, noise)[0]
        if actions.shape != (policy.action_shape[1], 7) or not np.isfinite(actions).all():
            raise ValueError("Policy returned an invalid LIBERO action chunk")
        return Prediction(origin, actions, (perf_counter() - started) * 1000)

    with ThreadPoolExecutor(max_workers=1) as worker:
        def submit(obs, origin):
            nonlocal submitted
            submitted += 1
            started = perf_counter()
            noise = noise_rng.standard_normal(policy.action_shape, dtype=np.float32)
            return worker.submit(predict, origin, observe(obs), noise, started)

        startup = submit(raw, 0).result()
        future = None
        ready = startup
        epoch = perf_counter()
        next_boundary = 0
        index = execute

        while applied < limit:
            remaining_time = epoch + slot * period - perf_counter()
            if remaining_time > 0:
                sleep(remaining_time)
            actual_slot = int((perf_counter() - epoch) / period)
            if actual_slot > slot:
                missed += actual_slot - slot
                slot = actual_slot

            if index == execute and ready is None and future.done():
                ready, future = future.result(), None
            if index == execute and ready is not None:
                ready_window = select_actions(ready, previous, applied, execute, pacing.overlap)
                if previous is not None:
                    waits.append(round((slot - next_boundary) * period * 1000, 4))
                    position_jumps.append(float(np.linalg.norm(ready_window[0, :3] - last_action[:3])))
                    rotation_jumps.append(float(np.linalg.norm(ready_window[0, 3:6] - last_action[3:6])))
                offsets.append(applied - ready.origin)
                inference.append(round(ready.inference_ms, 4))
                previous, ready = ready, None
                index = 0

            if index == execute:
                empty += 1
                camera.capture = False
                command = hold
                request = False
            else:
                command = ready_window[index]
                hold[-1] = command[-1]
                last_action = command
                index += 1
                applied += 1
                request = future is None and execute - index == pacing.refill_actions
                camera.capture = request
                if index == execute:
                    next_boundary = slot + 1
            raw, _, done, _ = env.step(command)
            success = bool(env.check_success())
            if success or done or applied == limit:
                break
            if request:
                future = submit(raw, applied)
            slot += 1

        elapsed = perf_counter() - epoch
        if future is not None:
            future.cancel()

    return {
        "steps": applied, "queries": submitted, "success": success,
        "control_seconds": round(elapsed, 4), "startup_ms": round(startup.inference_ms, 4),
        "empty_ticks": empty, "scheduler_missed_ticks": missed,
        "chunk_wait_ms": waits, "inference_ms": inference, "forecast_offsets": offsets,
        "position_jumps": position_jumps, "rotation_jumps": rotation_jumps,
    }
