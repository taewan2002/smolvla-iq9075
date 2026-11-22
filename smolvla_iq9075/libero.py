# Portions adapted from LeRobot.
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run paired LIBERO episodes with the native, local ONNX or remote ONNX backend."""
import argparse
from collections import Counter
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from time import perf_counter

import numpy as np

from .native import NativePolicy, sha256, snapshots

SUITES = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")


class CameraSampler:
    """Reuse unused frames while retaining the simulator's observation clocks."""

    def __init__(self, env):
        self.capture = True
        self.frames = {}
        for name in CAMERAS:
            observable = env.env._observables[name]
            observable._sensor = self.wrap(name, observable._sensor)

    def wrap(self, name, original):
        @wraps(original)
        def sensor(cache):
            if self.capture or name not in self.frames:
                self.frames[name] = original(cache)
            return self.frames[name]

        return sensor


def observation(raw):
    rgb = tuple(raw[name][::-1, ::-1].copy() for name in CAMERAS)
    quat = raw["robot0_eef_quat"].astype(np.float32)
    w = np.float32(np.clip(quat[3], -1, 1))
    denominator = np.sqrt(max(np.float32(0), np.float32(1) - w * w))
    axis_angle = quat[:3] * (2 * np.arccos(w) / denominator) if denominator > 1e-10 else np.zeros(3)
    state = np.concatenate((raw["robot0_eef_pos"], axis_angle, raw["robot0_gripper_qpos"])).astype(np.float32)
    return rgb, state


def episode(asset, init_states, row, policy, execute, fixture=None, *, pacing=None):
    import torch
    from libero.libero.envs import OffScreenRenderEnv

    options = {}
    if pacing is not None:
        random.seed(row["seed"])
        np.random.seed(row["seed"])
        options["control_freq"] = pacing.control_hz
    env = OffScreenRenderEnv(bddl_file_name=str(asset), camera_heights=256,
                             camera_widths=256, render_gpu_device_id=0, **options)
    noise_rng = np.random.default_rng(row["seed"])
    start = perf_counter()
    queries = 0
    try:
        env.seed(row["seed"])
        env.reset()
        raw = env.set_init_state(torch.load(init_states, map_location="cpu", weights_only=False)[row["init"]])
        for _ in range(10):
            raw, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        for robot in env.robots:
            robot.controller.use_delta = True
        rgb, state = observation(raw)
        initial = hashlib.sha256(b"".join(frame.tobytes() for frame in rgb) + state.tobytes()).hexdigest()
        if pacing is not None:
            from .chunked import execute_episode

            timed = execute_episode(env, raw, observation, CameraSampler(env), policy, row["instruction"],
                                    noise_rng, execute, pacing, SUITES[row["suite"]])
            return {**row, "execute": execute, "initial_sha256": initial,
                    "control_hz": pacing.control_hz, "predicted_actions": policy.action_shape[1],
                    "refill_actions": pacing.refill_actions, "overlap": pacing.overlap,
                    **timed, "wall_seconds": round(perf_counter() - start, 3)}
        actions = np.empty((0,))
        success = False
        for tick in range(SUITES[row["suite"]]):
            if tick % execute == 0:
                rgb, state = observation(raw)
                noise = noise_rng.standard_normal(policy.action_shape, dtype=np.float32)
                actions = policy(rgb, state, row["instruction"], noise)[0]
                if actions.ndim != 2 or len(actions) < execute or not np.isfinite(actions).all():
                    raise ValueError("Policy returned an invalid action chunk")
                if fixture is not None and queries == 0:
                    np.savez_compressed(fixture, rgb=np.stack(rgb), state=state,
                                        task=row["instruction"], noise=noise, native=actions)
                queries += 1
            raw, _, done, _ = env.step(actions[tick % execute])
            success = bool(env.check_success())
            if success or done:
                break
        return {**row, "execute": execute, "initial_sha256": initial, "steps": tick + 1,
                "queries": queries, "success": success, "wall_seconds": round(perf_counter() - start, 3)}
    finally:
        env.close()


def configure(root, directory):
    import yaml

    root = root.resolve()
    if not (root / "bddl_files").is_dir():
        raise ValueError("LIBERO root must be the directory containing bddl_files, init_files and assets")
    paths = {"benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
             "init_states": str(root / "init_files"), "assets": str(root / "assets"),
             "datasets": str(Path(directory) / "datasets")}
    (Path(directory) / "config.yaml").write_text(yaml.safe_dump(paths))
    os.environ["LIBERO_CONFIG_PATH"] = directory
    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero import benchmark
    import robosuite.macros as macros

    macros.IMAGE_CONVENTION = "opengl"
    return benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--engine", choices=["native", "cpu", "htp", "dual", "remote"], default="native")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vlm", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--url")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flow-steps", type=int, help="Override Euler steps for a native control run")
    parser.add_argument("--execute", type=int, default=1)
    parser.add_argument("--control-hz", type=int, help="Pace commands and the simulator at this frequency")
    parser.add_argument("--refill-actions", type=int, default=0, help="Request a new chunk with this many actions left")
    parser.add_argument("--overlap", type=int, default=0, help="Number of boundary actions to blend")
    parser.add_argument("--inits", type=int, default=1)
    parser.add_argument("--init-start", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--suites", nargs="+", choices=tuple(SUITES), default=list(SUITES))
    parser.add_argument("--save-fixtures", type=Path)
    args = parser.parse_args()
    if (not 1 <= args.tasks <= 10 or args.task_start < 0 or args.task_start + args.tasks > 10
            or args.inits < 1 or args.init_start < 0 or args.execute < 1):
        parser.error("Invalid task, initial-state or execution count")
    if not 0 <= args.refill_actions < args.execute or not 0 <= args.overlap <= args.execute:
        parser.error("Refill and overlap must fit the execution window")
    pacing = None
    if args.control_hz is not None:
        if args.control_hz < 1 or args.save_fixtures:
            parser.error("Paced evaluation requires a positive control rate and a separate fixture run")
        from .chunked import Pacing

        pacing = Pacing(args.control_hz, args.refill_actions, args.overlap)
    elif args.refill_actions or args.overlap:
        parser.error("Refill and overlap require --control-hz")
    if args.flow_steps is not None and (args.engine != "native" or args.flow_steps < 1):
        parser.error("--flow-steps requires the native engine and a positive step count")
    if args.engine == "native":
        if bool(args.checkpoint) != bool(args.vlm):
            parser.error("Supply both --checkpoint and --vlm or neither")
        checkpoint, vlm = (args.checkpoint, args.vlm) if args.checkpoint else snapshots()
        policy = NativePolicy(checkpoint, vlm, args.device)
        if args.flow_steps is not None:
            policy.policy.model.config.num_steps = args.flow_steps
        denoising_steps = policy.policy.model.config.num_steps
        checkpoint_sha = sha256(checkpoint / "model.safetensors")
        backend = f"torch-{args.device}"
    elif args.engine == "remote":
        from .remote import RemotePolicy

        if not args.url:
            parser.error("--url is required in remote mode")
        policy = RemotePolicy(args.url)
        denoising_steps = policy.manifest.get("denoising_steps")
        checkpoint_sha = policy.manifest["checkpoint_sha256"]
        backend = policy.manifest["backend"]
    else:
        from .runtime import EdgePolicy

        if not args.bundle:
            parser.error("--bundle is required with a local ONNX engine")
        policy = EdgePolicy(args.bundle, args.engine)
        denoising_steps = policy.manifest.get("denoising_steps")
        checkpoint_sha = policy.manifest["checkpoint_sha256"]
        backend = policy.backend
    if args.execute + args.refill_actions + args.overlap > policy.action_shape[1]:
        parser.error("The aligned execution and overlap windows exceed the predicted chunk length")
    if args.save_fixtures:
        args.save_fixtures.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    try:
        with tempfile.TemporaryDirectory(prefix="libero-eval-") as config, args.output.open("x") as log:
            benchmark = configure(args.libero_root, config)
            suites = {name: benchmark.get_benchmark_dict()[name]() for name in args.suites}
            for init in range(args.init_start, args.init_start + args.inits):
                for task_id in range(args.task_start, args.task_start + args.tasks):
                    for name, suite in suites.items():
                        task = suite.get_task(task_id)
                        root = args.libero_root
                        asset = root / "bddl_files" / task.problem_folder / task.bddl_file
                        states = root / "init_files" / task.problem_folder / task.init_states_file
                        seed = 47000 + list(SUITES).index(name) * 10000 + task_id * 100 + init
                        row = {"suite": name, "task": task_id, "init": init, "seed": seed,
                               "instruction": task.language, "engine": args.engine, "backend": backend,
                               "checkpoint_sha256": checkpoint_sha,
                               "bddl_sha256": sha256(asset), "init_states_sha256": sha256(states)}
                        if denoising_steps is not None:
                            row["denoising_steps"] = denoising_steps
                        file = args.save_fixtures / f"{name}-{task_id:02d}-{init:02d}.npz" if args.save_fixtures else None
                        result = episode(asset, states, row, policy, args.execute, file, pacing=pacing)
                        print(json.dumps(result), file=log, flush=True)
                        print(json.dumps(result), flush=True)
                        counts[result["success"]] += 1
    finally:
        if hasattr(policy, "close"):
            policy.close()
    print(f"successes {counts[True]}/{sum(counts.values())}", flush=True)


if __name__ == "__main__":
    main()
