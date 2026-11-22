"""Train a two-step SmolVLA Action Expert on public LIBERO demonstrations."""
import argparse
import json
from pathlib import Path
import shutil
from time import monotonic
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from .libero import SUITES
from .native import NativePolicy, sha256, snapshots

EXPERT_PREFIXES = (
    "model.vlm_with_expert.lm_expert.",
    "model.action_in_proj.",
    "model.action_time_mlp_in.",
    "model.action_time_mlp_out.",
    "model.action_out_proj.",
)


class Demonstrations:
    def __init__(self, root):
        import h5py

        self.h5py = h5py
        self.train = {}
        self.validation = {}
        files = sorted(file for suite in SUITES for file in (root / suite).glob("*_demo.hdf5"))
        for file in files:
            with h5py.File(file, "r") as source:
                data = source["data"]
                if data.attrs["macros_image_convention"].lower() not in ("opengl", b"opengl"):
                    raise ValueError(f"Expected OpenGL camera convention in {file.name}")
                task = json.loads(data.attrs["problem_info"])["language_instruction"]
                train, validation = [], []
                for key, demo in data.items():
                    length = len(demo["obs"]["agentview_rgb"])
                    if length:
                        unit = (file, key, length, task)
                        (validation if int(key.rsplit("_", 1)[1]) % 5 == 0 else train).append(unit)
                if train and validation:
                    self.train[file] = train
                    self.validation[file] = validation
        if not self.train:
            raise ValueError("Expected LIBERO suite directories containing *_demo.hdf5 files")

    def sample(self, partition, rng, count):
        groups = list(partition.values())
        examples = []
        for _ in range(count):
            demos = groups[rng.integers(len(groups))]
            file, demo, length, task = demos[rng.integers(len(demos))]
            index = rng.integers(length)
            with self.h5py.File(file, "r") as source:
                obs = source["data"][demo]["obs"]
                rgb = [obs[name][index][::-1, ::-1].copy()
                       for name in ("agentview_rgb", "eye_in_hand_rgb")]
                state = np.concatenate([obs[name][index]
                                        for name in ("ee_pos", "ee_ori", "gripper_states")])
            examples.append((rgb, state, task))
        return examples


def prepare_batch(handler, examples):
    batch = {}
    for index, key in enumerate(handler.image_keys):
        camera = np.stack([rgb[index] for rgb, _, _ in examples])
        batch[key] = torch.from_numpy(camera).permute(0, 3, 1, 2).float().div_(255)
    batch["observation.state"] = torch.from_numpy(np.stack([state for _, state, _ in examples])).float()
    batch["task"] = [task for _, _, task in examples]
    return handler.pre(batch)


@torch.no_grad()
def teacher_targets(teacher, examples, noise):
    core = teacher.policy.model
    native_step = core.denoise_step
    position = 0
    targets = {}

    def record(prefix_pad_masks, past_key_values, x_t, timestep):
        nonlocal position
        velocity = native_step(prefix_pad_masks, past_key_values, x_t, timestep)
        if position == 0:
            targets["context"] = (prefix_pad_masks, past_key_values)
        if position == 5:
            targets["midpoint"] = x_t
        if position == 9:
            targets["final"] = x_t - 0.1 * velocity
        position += 1
        return velocity

    with patch.object(core, "denoise_step", record):
        action = teacher.policy.predict_action_chunk(prepare_batch(teacher, examples), noise=noise)
    if position != 10:
        raise RuntimeError(f"Expected 10 teacher steps, observed {position}")
    torch.testing.assert_close(targets["final"][..., :action.shape[-1]], action)
    return noise, targets["context"], targets["midpoint"], targets["final"]


def two_steps(core, x, context):
    times = torch.tensor([1.0, 0.5], device=x.device, dtype=torch.float32)
    midpoint = x - 0.5 * core.denoise_step(*context, x, times[0].expand(len(x)))
    final = midpoint - 0.5 * core.denoise_step(*context, midpoint, times[1].expand(len(x)))
    return midpoint, final


def loss(estimate, target, action_dim):
    error = F.smooth_l1_loss(estimate.float(), target, reduction="none", beta=0.1)
    active = error[..., :action_dim]
    channel_mask = error[..., action_dim:]
    padded = 0.02 * channel_mask.mean() if channel_mask.numel() else 0.0
    return 0.7 * active[:, :10].mean() + 0.3 * active.mean() + padded


@torch.no_grad()
def measure(core, records, action_dim):
    predicted, expected = [], []
    for noise, context, _, final in records:
        predicted.append(two_steps(core, noise, context)[-1][..., :action_dim])
        expected.append(final[..., :action_dim])
    predicted, expected = torch.cat(predicted), torch.cat(expected)
    error = (predicted - expected).float()
    return {
        "observations": len(error),
        "first10_relative_l2": (error[:, :10].norm() / expected[:, :10].norm().clamp_min(1e-12)).item(),
        "full_relative_l2": (error.norm() / expected.norm().clamp_min(1e-12)).item(),
    }


def save_checkpoint(original, output, weights):
    from safetensors.torch import load_file, save_file

    config = json.loads((original / "config.json").read_text())
    config["num_steps"] = 2
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        source = original / filename
        shutil.copyfile(source, output / filename)
        for step in json.loads(source.read_text())["steps"]:
            if "state_file" in step:
                name = Path(step["state_file"]).name
                shutil.copyfile(original / name, output / name)

    tensors = load_file(original / "model.safetensors")
    if not weights or any(name not in tensors for name in weights):
        raise ValueError("Action Expert weights do not match the teacher checkpoint layout")
    tensors.update({name: value.detach().cpu().contiguous() for name, value in weights.items()})
    save_file(tensors, output / "model.safetensors", metadata={"format": "pt"})


def train(args, checkpoint, vlm):
    started = monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    teacher = NativePolicy(checkpoint, vlm, str(device))
    config = teacher.policy.config
    if (config.num_steps != 10 or config.n_obs_steps != 1 or not config.use_cache
            or config.adapt_to_pi_aloha or teacher.policy.model._rtc_enabled()
            or config.chunk_size < 10 or len(teacher.image_keys) != 2):
        raise ValueError("Expected a two-camera cached SmolVLA policy with a 10-step teacher")
    teacher.policy.requires_grad_(False)
    student = NativePolicy(checkpoint, vlm, str(device))
    student.policy.requires_grad_(False)
    core = student.policy.model
    for module in (core.vlm_with_expert.lm_expert, core.action_in_proj, core.action_out_proj,
                   core.action_time_mlp_in, core.action_time_mlp_out):
        module.float().requires_grad_(True)
    parameters = {key: param for key, param in student.policy.named_parameters()
                  if key.startswith(EXPERT_PREFIXES)}
    optimizer = torch.optim.AdamW(parameters.values(), lr=args.lr, weight_decay=0.01)
    data = Demonstrations(args.demos)
    train_rng = np.random.default_rng(args.seed)
    val_rng = np.random.default_rng(args.seed + 1)
    noise_rng = torch.Generator(device=device).manual_seed(args.seed)
    validation_noise_rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    action_dim = config.action_feature.shape[0]

    def record(partition, rng, generator, count):
        obs = data.sample(partition, rng, count)
        noise = torch.randn((count, *teacher.action_shape[1:]), generator=generator, device=device)
        return teacher_targets(teacher, obs, noise)

    val = [record(data.validation, val_rng, validation_noise_rng, min(args.batch, args.validation - offset))
           for offset in range(0, args.validation, args.batch)]
    baseline = measure(teacher.policy.model, val, action_dim)
    print(json.dumps({"undistilled_2step": baseline}), flush=True)
    for update in range(1, args.updates + 1):
        noise, context, midpoint, final = record(data.train, train_rng, noise_rng, args.batch)
        optimizer.zero_grad(set_to_none=True)
        halfway, result = two_steps(core, noise, context)
        objective = loss(result, final, action_dim) + 0.35 * loss(halfway, midpoint, action_dim)
        if not torch.isfinite(objective):
            raise FloatingPointError(f"Non-finite training loss at update {update}")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(parameters.values(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if update == 1 or update % 25 == 0 or update == args.updates:
            print(json.dumps({"update": update, "loss": objective.item(),
                              "seconds": round(monotonic() - started, 1)}), flush=True)

    summary = {
        "teacher_checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
        "teacher_steps": 10, "student_steps": 2, "updates": args.updates, "batch": args.batch,
        "learning_rate": args.lr, "seed": args.seed, "prediction_actions": config.chunk_size,
        "demo_files": len(data.train),
        "train_demos": sum(map(len, data.train.values())),
        "heldout_demos": sum(map(len, data.validation.values())),
        "undistilled_2step": baseline, "distilled_2step": measure(core, val, action_dim),
    }
    save_checkpoint(checkpoint, args.output, parameters)
    summary["student_checkpoint_sha256"] = sha256(args.output / "model.safetensors")
    (args.output / "distillation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, required=True, help="Directory containing the LIBERO suite directories")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vlm", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--validation", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=47000)
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.vlm):
        parser.error("Supply both --checkpoint and --vlm or neither")
    if min(args.updates, args.batch, args.validation) < 1 or not 0 < args.lr < float("inf"):
        parser.error("Use positive training counts and a finite positive learning rate")
    checkpoint, vlm = (args.checkpoint, args.vlm) if args.checkpoint else snapshots()
    train(args, checkpoint, vlm)


if __name__ == "__main__":
    main()
