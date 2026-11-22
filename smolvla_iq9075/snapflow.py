"""Adapt a pretrained SmolVLA to SnapFlow one-evaluation action generation."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
import json
import multiprocessing as mp
from pathlib import Path
import shutil
from time import monotonic

import numpy as np
import torch
from torch import nn

from .libero import SUITES
from .native import NativePolicy, sha256, snapshots


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

    def sample(self, partition, rng, count, horizon):
        groups = list(partition.values())
        examples = []
        for _ in range(count):
            demos = groups[rng.integers(len(groups))]
            file, demo, length, task = demos[rng.integers(len(demos))]
            index = rng.integers(length)
            with self.h5py.File(file, "r") as source:
                record = source["data"][demo]
                obs = record["obs"]
                rgb = [obs[name][index][::-1, ::-1].copy()
                       for name in ("agentview_rgb", "eye_in_hand_rgb")]
                state = np.concatenate([obs[name][index]
                                        for name in ("ee_pos", "ee_ori", "gripper_states")])
                actions = np.asarray(record["actions"][index:index + horizon], dtype=np.float32)
            valid = np.arange(horizon) < len(actions)
            actions = np.pad(actions, ((0, horizon - len(actions)), (0, 0)), mode="edge")
            examples.append((rgb, state, task, actions, valid))
        return examples


class LeRobotDemonstrations:
    cameras = ("observation.images.image", "observation.images.image2")

    def __init__(self, root, horizon, revision):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        info = json.loads((root / "meta/info.json").read_text())
        fps = info["fps"]
        self.dataset = LeRobotDataset("lerobot/libero", root=root, revision=revision,
                                      delta_timestamps={"action": [i / fps for i in range(horizon)]})
        self.train, self.validation = {}, {}
        counts = {}
        episodes = self.dataset.meta.episodes
        starts = [episode["dataset_from_index"] for episode in episodes]
        task_indices = self.dataset.hf_dataset.select(starts)["task_index"]
        for episode, task_index in zip(episodes, task_indices, strict=True):
            task = int(task_index)
            index = counts.get(task, 0)
            counts[task] = index + 1
            partition = self.validation if index % 5 == 0 else self.train
            partition.setdefault(task, []).append((episode["dataset_from_index"], episode["dataset_to_index"]))
        if self.train.keys() != self.validation.keys():
            raise ValueError("Expected both training and validation episodes for each LIBERO task")

    def indices(self, partition, rng, count):
        groups = list(partition.values())
        for _ in range(count):
            episodes = groups[rng.integers(len(groups))]
            start, stop = episodes[rng.integers(len(episodes))]
            yield int(rng.integers(start, stop))

    def read(self, indices):
        examples = []
        for index in indices:
            item = self.dataset[index]
            rgb = [(item[key].permute(1, 2, 0) * 255).round().to(torch.uint8).numpy() for key in self.cameras]
            examples.append((rgb, item["observation.state"].numpy(), item["task"],
                             item["action"].numpy(), ~item["action_is_pad"].numpy()))
        return examples

    def sample(self, partition, rng, count, horizon):
        return self.read(self.indices(partition, rng, count))


def open_video_dataset(root, horizon, revision):
    global video_dataset
    torch.set_num_threads(1)
    video_dataset = LeRobotDemonstrations(root, horizon, revision)


def read_video_batch(indices):
    return video_dataset.read(indices)


def batches(data, partition, rng, sizes, horizon, pool, workers):
    if pool is None:
        for size in sizes:
            yield data.sample(partition, rng, size, horizon)
        return
    indices = (list(data.indices(partition, rng, size)) for size in sizes)
    pending = deque(pool.submit(read_video_batch, batch) for batch in islice(indices, 2 * workers))
    while pending:
        yield pending.popleft().result()
        batch = next(indices, None)
        if batch is not None:
            pending.append(pool.submit(read_video_batch, batch))


def prepare_batch(handler, examples):
    batch = {}
    for index, key in enumerate(handler.image_keys):
        camera = np.stack([row[0][index] for row in examples])
        batch[key] = torch.from_numpy(camera).permute(0, 3, 1, 2).float().div_(255)
    batch["observation.state"] = torch.from_numpy(np.stack([row[1] for row in examples])).float()
    batch["task"] = [row[2] for row in examples]
    batch["action"] = torch.from_numpy(np.stack([row[3] for row in examples]))
    valid = torch.from_numpy(np.stack([row[4] for row in examples])).to(handler.device)
    return handler.pre(batch), valid


class SnapFlow:
    def __init__(self, core):
        self.core = core
        width = core.vlm_with_expert.expert_hidden_size
        self.destination = nn.Linear(width, width, bias=False, device=core.action_time_mlp_in.weight.device)
        nn.init.zeros_(self.destination.weight)

    def time_embedding(self, time):
        from lerobot.policies.smolvla.modeling_smolvla import create_sinusoidal_pos_embedding

        return create_sinusoidal_pos_embedding(time, self.destination.in_features,
                                               self.core.config.min_period, self.core.config.max_period,
                                               device=time.device).float()

    def velocity(self, context, noisy, source, destination):
        delta = self.destination(self.time_embedding(destination))

        def add_destination(_module, inputs):
            action, time = inputs[0].chunk(2, dim=-1)
            return (torch.cat((action, time + delta[:, None].to(time.dtype)), dim=-1),)

        hook = self.core.action_time_mlp_in.register_forward_pre_hook(add_destination)
        try:
            return self.core.denoise_step(*context, noisy, source)
        finally:
            hook.remove()

    @torch.no_grad()
    def prefix(self, handler, batch):
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        policy = handler.policy
        images, image_masks = policy.prepare_images(batch)
        embeds, padding, attention = self.core.embed_prefix(
            images, image_masks, batch["observation.language.tokens"],
            batch["observation.language.attention_mask"], state=policy.prepare_state(batch),
        )
        _, cache = self.core.vlm_with_expert(
            attention_mask=make_att_2d_masks(padding, attention),
            position_ids=torch.cumsum(padding, dim=1) - 1,
            past_key_values=None, inputs_embeds=[embeds, None], use_cache=True, fill_kv_cache=True,
        )
        return padding, cache

    def loss(self, context, actions, valid, alpha, shortcut_weight):
        batch = len(actions)
        source = self.core.sample_time(batch, actions.device)
        noise = torch.randn_like(actions)
        noisy = torch.lerp(actions, noise, source[:, None, None])
        local = torch.rand(batch, device=actions.device) < alpha
        destination = torch.where(local, source, 0.0)
        with torch.no_grad():
            first = self.velocity(context, noisy, source, source)
            midpoint = source * 0.5
            second = self.velocity(context, noisy - midpoint[:, None, None] * first, midpoint, midpoint)
            shortcut = ((first.float() + second.float()) * 0.5).clamp(-20, 20)
            target = torch.where(local[:, None, None], noise - actions, shortcut)
        predicted = self.velocity(context, noisy, source, destination)
        weights = torch.where(local, 1.0, shortcut_weight)
        error = (predicted.float() - target).square().mean(dim=-1)
        return (error * weights[:, None] * valid).sum() / valid.sum()

    @torch.no_grad()
    def predict(self, context, noise):
        source = noise.new_ones(len(noise))
        destination = noise.new_zeros(len(noise))
        return noise - self.velocity(context, noise, source, destination)

    @torch.no_grad()
    def fold_destination(self):
        """Absorb the constant s=0 projection into the first time MLP bias."""
        affine = self.core.action_time_mlp_in
        zero = affine.weight.new_zeros(1)
        shift = self.destination(self.time_embedding(zero))[0].to(affine.weight)
        affine.bias.add_(affine.weight[:, self.destination.in_features:] @ shift)


def save_checkpoint(original, output, weights):
    from safetensors.torch import load_file, save_file

    config = json.loads((original / "config.json").read_text())
    config["num_steps"] = 1
    config["n_action_steps"] = 10
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
        raise ValueError("Action Expert weights do not match the source checkpoint layout")
    tensors.update({name: value.detach().cpu().contiguous() for name, value in weights.items()})
    save_file(tensors, output / "model.safetensors", metadata={"format": "pt"})



@torch.no_grad()
def metrics(records, predictor, action_dim):
    first, full, early, total = 0.0, 0.0, 0, 0
    for context, actions, valid, noise in records:
        predicted = predictor(context, noise)[..., :action_dim].float()
        error = (predicted - actions[..., :action_dim]).square().mean(dim=-1) * valid
        first += error[:, :10].sum().item()
        full += error.sum().item()
        early += valid[:, :10].sum().item()
        total += valid.sum().item()
    return {"first10_mse": first / early, "full_chunk_mse": full / total}


def train(args, checkpoint, vlm):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    handler = NativePolicy(checkpoint, vlm, str(device))
    config = handler.policy.config
    if (config.num_steps != 10 or config.chunk_size != 50 or config.n_obs_steps != 1
            or not config.use_cache or config.adapt_to_pi_aloha
            or handler.policy.model._rtc_enabled() or len(handler.image_keys) != 2):
        raise ValueError("Expected a cached two-camera SmolVLA with 50 actions and 10 Euler evaluations")
    handler.policy.requires_grad_(False)
    core = handler.policy.model
    trainable = (core.vlm_with_expert.lm_expert, core.action_in_proj, core.action_out_proj,
                 core.action_time_mlp_in, core.action_time_mlp_out)
    for module in trainable:
        module.float().requires_grad_(True)
    model = SnapFlow(core)
    parameters = {name: p for name, p in handler.policy.named_parameters() if p.requires_grad}
    trainable_params = [*parameters.values(), *model.destination.parameters()]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    lerobot_data = (args.demos / "meta/info.json").is_file()
    data = (LeRobotDemonstrations(args.demos, config.chunk_size, args.dataset_revision)
            if lerobot_data else Demonstrations(args.demos))
    train_rng = np.random.default_rng(args.seed)
    val_rng = np.random.default_rng(args.seed + 1)
    val_noise = torch.Generator(device=device).manual_seed(args.seed + 1)
    action_dim = config.action_feature.shape[0]
    started = monotonic()

    def sample(examples):
        batch, valid = prepare_batch(handler, examples)
        return model.prefix(handler, batch), handler.policy.prepare_action(batch), valid

    @torch.no_grad()
    def original(context, noise):
        x = noise
        for i in range(10):
            time = x.new_full((len(x),), 1 - i / 10)
            x = x - 0.1 * core.denoise_step(*context, x, time)
        return x

    pool = None
    if lerobot_data and args.data_workers:
        pool = ProcessPoolExecutor(args.data_workers, mp_context=mp.get_context("spawn"),
                                   initializer=open_video_dataset,
                                   initargs=(args.demos, config.chunk_size, args.dataset_revision))
    try:
        records = []
        validation_sizes = (min(args.validation - offset, args.batch)
                            for offset in range(0, args.validation, args.batch))
        validation = batches(data, data.validation, val_rng, validation_sizes,
                             config.chunk_size, pool, args.data_workers)
        for examples in validation:
            context, actions, valid = sample(examples)
            noise = torch.randn(actions.shape, device=device, generator=val_noise)
            records.append((context, actions, valid, noise))
        native = metrics(records, original, action_dim)
        naive = metrics(records, model.predict, action_dim)
        print(json.dumps({"original_10nfe": native, "naive_1nfe": naive}), flush=True)

        args.output.mkdir(parents=True, exist_ok=False)
        training = batches(data, data.train, train_rng, [args.batch] * args.updates,
                           config.chunk_size, pool, args.data_workers)
        for update, examples in enumerate(training, 1):
            optimizer.param_groups[0]["lr"] = args.lr * min(update / max(args.warmup, 1), 1.0)
            optimizer.zero_grad(set_to_none=True)
            # Casting the no-gradient targets must not detach the expert's training weights.
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16, cache_enabled=False):
                context, actions, valid = sample(examples)
                objective = model.loss(context, actions, valid, args.alpha, args.shortcut_weight)
            objective.backward()
            if update == 1:
                expert_weight = core.vlm_with_expert.lm_expert.layers[0].self_attn.q_proj.weight
                if expert_weight.grad is None or model.destination.weight.grad is None:
                    raise RuntimeError("Expected gradients through the Action Expert and destination projection")
            norm = nn.utils.clip_grad_norm_(trainable_params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            if update == 1 or update % 25 == 0 or update == args.updates:
                print(json.dumps({"update": update, "loss": objective.item(), "grad_norm": norm.item(),
                                  "seconds": round(monotonic() - started, 1)}), flush=True)
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)

    result = metrics(records, model.predict, action_dim)
    context, _, _, noise = records[0]
    expected = model.predict(context, noise)
    model.fold_destination()
    with torch.no_grad():
        actual = noise - core.denoise_step(*context, noise, noise.new_ones(len(noise)))
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=3e-4)
    save_checkpoint(checkpoint, args.output, parameters)
    summary = {
        "starting_checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
        "checkpoint_sha256": sha256(args.output / "model.safetensors"),
        "updates": args.updates, "batch": args.batch, "learning_rate": args.lr, "warmup": args.warmup,
        "alpha": args.alpha, "shortcut_weight": args.shortcut_weight, "seed": args.seed,
        "training_precision": "bf16 autocast" if args.bf16 else "fp32",
        "denoising_steps": 1, "predicted_actions": config.chunk_size, "execution_actions": 10,
        "dataset": "lerobot/libero" if lerobot_data else "original LIBERO HDF5",
        "dataset_revision": args.dataset_revision,
        "data_workers": args.data_workers if lerobot_data else 0,
        "tasks": len(data.train), "train_demos": sum(map(len, data.train.values())),
        "heldout_demos": sum(map(len, data.validation.values())), "validation_samples": args.validation,
        "original_10nfe": native, "naive_1nfe": naive, "snapflow_1nfe": result,
        "fold_max_abs_error": (actual - expected).abs().max().item(),
        "seconds": round(monotonic() - started, 1),
    }
    (args.output / "snapflow.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, required=True, help="Local LeRobot or original LIBERO dataset root")
    parser.add_argument("--dataset-revision", help="Revision of the local lerobot/libero snapshot")
    parser.add_argument("--data-workers", type=int, default=0, help="Parallel workers for LeRobot video decoding")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vlm", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=15_000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--validation", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--shortcut-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=47000)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.vlm):
        parser.error("Supply both --checkpoint and --vlm or neither")
    if (min(args.updates, args.batch, args.validation) < 1 or min(args.warmup, args.data_workers) < 0 or not 0 < args.alpha < 1
            or not 0 < args.lr < float("inf") or not 0 < args.shortcut_weight < float("inf")):
        parser.error("Invalid training counts, learning rate or loss weights")
    checkpoint, vlm = (args.checkpoint, args.vlm) if args.checkpoint else snapshots()
    train(args, checkpoint, vlm)


if __name__ == "__main__":
    main()
