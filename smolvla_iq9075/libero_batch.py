"""Batched native LIBERO evaluation with separate MuJoCo workers."""
import argparse
from collections import deque
import hashlib
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import random
import tempfile
from time import perf_counter
import traceback

import numpy as np

from .compare import read
from .libero import SUITES, CameraSampler, configure, observation
from .native import NativePolicy, sha256, snapshots

def sim_worker(pipe, libero_root, fresh_env=False):
    import torch

    torch.set_num_threads(1)
    env = None
    active_asset = None
    try:
        with tempfile.TemporaryDirectory(prefix="libero-worker-") as config:
            configure(libero_root, config)
            from libero.libero.envs import OffScreenRenderEnv

            while True:
                message = pipe.recv()
                if message is None:
                    return
                if message[0] == "reset":
                    _, asset, init_states, row = message
                    if fresh_env or asset != active_asset:
                        random.seed(row["seed"])
                        np.random.seed(row["seed"])
                        if env is not None:
                            env.close()
                        env = OffScreenRenderEnv(bddl_file_name=str(asset), camera_heights=256,
                                                 camera_widths=256, render_gpu_device_id=0)
                        initial_states = torch.load(init_states, map_location="cpu", weights_only=False)
                        active_asset = asset
                    random.seed(row["seed"])
                    env.seed(row["seed"])
                    env.reset()
                    camera = CameraSampler(env)
                    env.set_init_state(initial_states[row["init"]])
                    for index in range(10):
                        camera.capture = index == 9
                        raw, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                    for robot in env.robots:
                        robot.controller.use_delta = True
                    steps = 0
                    pipe.send(("ok", (observation(raw), steps, False, False)))
                    continue

                _, actions = message
                success = False
                done = False
                for index, action in enumerate(actions):
                    camera.capture = index == len(actions) - 1
                    raw, _, done, _ = env.step(action)
                    steps += 1
                    success = bool(env.check_success())
                    if success or done:
                        break
                terminal = success or done or steps == SUITES[row["suite"]]
                pipe.send(("ok", (None if terminal else observation(raw), steps, terminal, success)))
    except EOFError:
        return
    except Exception:
        pipe.send(("error", traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        pipe.close()


def receive(pipe):
    if not pipe.poll(300):
        raise TimeoutError("LIBERO worker did not return an observation")
    status, result = pipe.recv()
    if status == "error":
        raise RuntimeError(result)
    return result


def task_jobs(args, checkpoint_sha, denoising_steps):
    with tempfile.TemporaryDirectory(prefix="libero-plan-") as config:
        benchmark = configure(args.libero_root, config)
        for suite_name in args.suites:
            suite = benchmark.get_benchmark_dict()[suite_name]()
            for task_id in range(args.task_start, args.task_start + args.tasks):
                task = suite.get_task(task_id)
                asset = args.libero_root / "bddl_files" / task.problem_folder / task.bddl_file
                states = args.libero_root / "init_files" / task.problem_folder / task.init_states_file
                shared = {
                    "suite": suite_name, "task": task_id, "instruction": task.language,
                    "engine": "native", "backend": f"torch-{args.device}",
                    "bddl_sha256": sha256(asset), "init_states_sha256": sha256(states),
                    "checkpoint_sha256": checkpoint_sha, "execute": args.execute,
                    "denoising_steps": denoising_steps,
                }
                for init in range(args.init_start, args.init_start + args.inits):
                    seed = 47000 + list(SUITES).index(suite_name) * 10000 + task_id * 100 + init
                    yield asset, states, {**shared, "init": init, "seed": seed}


def run(args, checkpoint, vlm):
    import torch

    torch.set_num_threads(1)
    policy = NativePolicy(checkpoint, vlm, args.device)
    if args.flow_steps is not None:
        policy.policy.model.config.num_steps = args.flow_steps
    config = policy.policy.model.config
    if config.n_obs_steps != 1 or args.execute > config.chunk_size:
        raise ValueError("Batched runner expects one observation and a valid execution count")
    plan = {(row["suite"], row["task"], row["init"]): (asset, states, row)
            for asset, states, row in task_jobs(args, sha256(checkpoint / "model.safetensors"), config.num_steps)}
    if args.episode_list:
        requested = json.loads(args.episode_list.read_text())
        selected = {(entry["suite"], entry["task"], entry["init"]) for entry in requested}
        if not selected or len(selected) != len(requested) or not selected.issubset(plan):
            raise ValueError("Episode list contains duplicates or entries outside the planned tasks")
        plan = {key: value for key, value in plan.items() if key in selected}
    mode = "x"
    if args.resume and args.output.is_file():
        for key, recorded in read(args.output).items():
            if key not in plan or any(recorded.get(field) != value for field, value in plan[key][2].items()):
                raise ValueError(f"Resume file does not match the requested protocol for {key}")
            del plan[key]
        mode = "a"
    jobs = deque(plan.values())
    workers = {}
    # Spawned LIBERO interpreters must agree on any set traversal used in scene construction.
    os.environ["PYTHONHASHSEED"] = "0"
    ctx = mp.get_context("spawn")
    queued = {}
    active = {}
    remaining = len(jobs)
    completed = 0
    clock_start = perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def reset(pipe):
        if not jobs:
            return
        asset, states, row = jobs.popleft()
        pipe.send(("reset", asset, states, row))
        queued[pipe] = row

    try:
        with args.output.open(mode) as output:
            for _ in range(min(args.batch_size, len(jobs))):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=sim_worker, args=(child, args.libero_root, args.fresh_env))
                process.start()
                child.close()
                workers[parent] = process
                reset(parent)

            while queued or active:
                ready = [pipe for pipe in queued if pipe.poll()]
                if not ready and not active:
                    ready = wait(tuple(queued), timeout=300)
                    if not ready:
                        raise TimeoutError("LIBERO workers did not complete reset")
                for pipe in ready:
                    row = queued.pop(pipe)
                    obs, steps, _, _ = receive(pipe)
                    rgb, state = obs
                    row["initial_sha256"] = hashlib.sha256(
                        b"".join(frame.tobytes() for frame in rgb) + state.tobytes()).hexdigest()
                    active[pipe] = (row, obs, steps, np.random.default_rng(row["seed"]), 0, perf_counter())
                pipes = list(active)
                rgb, state, task, noise = [], [], [], []
                for pipe in pipes:
                    row, observation_data, _, rng, _, _ = active[pipe]
                    images, pose = observation_data
                    rgb.append(images)
                    state.append(pose)
                    task.append(row["instruction"])
                    noise.append(rng.standard_normal(policy.action_shape[1:], dtype=np.float32))
                actions = policy.predict_batch(rgb, np.stack(state), task, np.stack(noise))
                if actions.shape != (len(pipes), config.chunk_size, config.action_feature.shape[0]):
                    raise ValueError(f"Policy returned an unexpected action shape: {actions.shape}")
                if not np.isfinite(actions).all():
                    raise FloatingPointError("Policy returned a non-finite action")
                for pipe, chunk in zip(pipes, actions, strict=True):
                    steps = active[pipe][2]
                    budget = SUITES[active[pipe][0]["suite"]] - steps
                    pipe.send(("step", chunk[:min(args.execute, budget)]))
                for pipe in pipes:
                    row, _, _, rng, queries, start = active[pipe]
                    obs, steps, terminal, success = receive(pipe)
                    if terminal:
                        record = {**row, "steps": steps, "queries": queries + 1, "success": success,
                                  "wall_seconds": round(perf_counter() - start, 3)}
                        print(json.dumps(record), file=output, flush=True)
                        completed += 1
                        if completed % 25 == 0 or completed == remaining:
                            print(f"episodes {completed}/{remaining} in {perf_counter() - clock_start:.1f} s", flush=True)
                        del active[pipe]
                        reset(pipe)
                    else:
                        active[pipe] = row, obs, steps, rng, queries + 1, start
    finally:
        for pipe, process in workers.items():
            if process.is_alive():
                try:
                    pipe.send(None)
                except (BrokenPipeError, OSError):
                    pass
            pipe.close()
        for process in workers.values():
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vlm", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flow-steps", type=int)
    parser.add_argument("--execute", type=int, default=10)
    parser.add_argument("--inits", type=int, default=50)
    parser.add_argument("--init-start", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--suites", nargs="+", choices=tuple(SUITES), default=list(SUITES))
    parser.add_argument("--episode-list", type=Path, help="JSON array of suite, task and init records")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Keep complete matching rows and run the remaining episodes")
    parser.add_argument("--fresh-env", action="store_true", help="Construct a new simulator environment per episode")
    args = parser.parse_args()
    args.libero_root = args.libero_root.resolve()
    if (min(args.execute, args.inits, args.tasks, args.batch_size) < 1
            or args.task_start < 0 or args.init_start < 0 or args.task_start + args.tasks > 10):
        parser.error("Invalid task, initial-state, execution or batch count")
    if args.flow_steps is not None and args.flow_steps < 1:
        parser.error("--flow-steps must be positive")
    if bool(args.checkpoint) != bool(args.vlm):
        parser.error("Supply both --checkpoint and --vlm or neither")
    checkpoint, vlm = (args.checkpoint, args.vlm) if args.checkpoint else snapshots()
    run(args, checkpoint, vlm)


if __name__ == "__main__":
    main()
