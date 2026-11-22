# Portions adapted from LeRobot.
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a fixed-shape, batch-one SmolVLA ONNX bundle on an x86 workstation."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .native import MODEL, REVISION, VLM, VLM_REVISION, NativePolicy, sha256, snapshots


class StableRMSNorm(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.weight = source.weight
        self.eps = source.variance_epsilon

    def forward(self, value):
        # Keep the squared activations within the FP16 range on HTP.
        scale = (value.abs().amax(-1, keepdim=True) / 32.0).clamp(min=1.0)
        bounded = value / scale
        variance = (bounded * bounded).mean(-1, keepdim=True)
        return self.weight * (bounded * torch.rsqrt(variance + (self.eps / scale) / scale))


def stabilize(module):
    for name, child in module.named_children():
        if type(child).__name__.endswith("RMSNorm"):
            setattr(module, name, StableRMSNorm(child))
        else:
            stabilize(child)


def heads(value, count, width):
    return value.reshape(1, value.shape[1], count, width).transpose(1, 2)


def rope(value, sine, cosine):
    left, right = value.chunk(2, dim=-1)
    return torch.cat((left * cosine - right * sine, right * cosine + left * sine), dim=-1)


def attend(q, k, v, bias):
    count, length, width = q.shape[1:]
    groups = count // k.shape[1]
    grouped = q.reshape(1, k.shape[1], groups * length, width)
    scores = torch.matmul(grouped, k.transpose(-2, -1)) * (width ** -0.5)
    out = torch.matmul((scores + bias).softmax(-1), v)
    return out.reshape(1, count, length, width).transpose(1, 2).reshape(1, length, count * width)


def residual(layer, hidden, attended):
    hidden = hidden + layer.self_attn.o_proj(attended)
    return hidden + layer.mlp(layer.post_attention_layernorm(hidden))


class Vision(nn.Module):
    def __init__(self, model, size):
        super().__init__()
        native = model.vision_model
        self.patch = native.embeddings.patch_embedding
        self.layers = native.encoder.layers
        self.norm = native.post_layernorm
        self.connector = model.connector
        # Capture the upstream interpolation/position indexing for this static shape.
        pixels = torch.zeros(1, 3, *size)
        patch_hw = self.patch(pixels).shape[-2:]
        mask = torch.ones(1, *patch_hw, dtype=torch.bool)
        positions = []
        hook = native.embeddings.position_embedding.register_forward_hook(
            lambda _layer, _inputs, output: positions.append(output.detach())
        )
        try:
            native.embeddings(pixels, mask)
        finally:
            hook.remove()
        self.register_buffer("positions", positions[0])

    def forward(self, pixels):
        hidden = self.patch(pixels).flatten(2).transpose(1, 2) + self.positions
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=None)[0]
        return self.connector(self.norm(hidden))


class Prefill(nn.Module):
    def __init__(self, layers, width, q_heads, kv_heads, last):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.width = width
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.last = last

    def forward(self, hidden, mask, sine, cosine):
        keys, values = [], []
        for index, layer in enumerate(self.layers):
            norm = layer.input_layernorm(hidden)
            attn = layer.self_attn
            key = rope(heads(attn.k_proj(norm), self.kv_heads, self.width), sine, cosine)
            value = heads(attn.v_proj(norm), self.kv_heads, self.width)
            keys.append(key)
            values.append(value)
            if self.last and index == len(self.layers) - 1:
                break
            query = rope(heads(attn.q_proj(norm), self.q_heads, self.width), sine, cosine)
            hidden = residual(layer, hidden, attend(query, key, value, mask))
        return hidden, torch.stack(keys), torch.stack(values)


class ActionStep(nn.Module):
    def __init__(self, core, width, q_heads, kv_heads, cross):
        super().__init__()
        self.layers = core.vlm_with_expert.lm_expert.layers
        self.norm = core.vlm_with_expert.lm_expert.norm
        self.action_in_proj = core.action_in_proj
        self.time_in = core.action_time_mlp_in
        self.time_out = core.action_time_mlp_out
        self.action_out_proj = core.action_out_proj
        self.width, self.q_heads, self.kv_heads = width, q_heads, kv_heads
        self.cross = cross
        self.dt = -1.0 / core.config.num_steps

    def forward(self, noise, time, self_sine, self_cosine, cross_sine, cross_cosine,
                self_mask, cross_mask, keys, values):
        action = self.action_in_proj(noise)
        hidden = self.time_out(F.silu(self.time_in(torch.cat((action, time[:, None].expand_as(action)), -1))))
        for index, layer in enumerate(self.layers):
            attn = layer.self_attn
            norm = layer.input_layernorm(hidden)
            query = heads(attn.q_proj(norm), self.q_heads, self.width)
            key, value = keys[index], values[index]
            if index in self.cross:
                query = rope(query, cross_sine, cross_cosine)
                mask = cross_mask
            else:
                query = rope(query, self_sine, self_cosine)
                suffix_k = rope(heads(attn.k_proj(norm), self.kv_heads, self.width), self_sine, self_cosine)
                suffix_v = heads(attn.v_proj(norm), self.kv_heads, self.width)
                key = torch.cat((key, suffix_k), dim=-2)
                value = torch.cat((value, suffix_v), dim=-2)
                mask = self_mask
            hidden = residual(layer, hidden, attend(query, key, value, mask))
        return noise + self.dt * self.action_out_proj(self.norm(hidden))


def emit(output, name, model, inputs, outputs):
    import onnx

    path = output / (name + ".onnx")
    torch.onnx.export(model, tuple(inputs.values()), str(path), input_names=list(inputs),
                      output_names=outputs, opset_version=17, dynamo=False)
    onnx.checker.check_model(str(path))
    print(f"ready {path.name} {path.stat().st_size:,} bytes", flush=True)


def processor_stats(directory, filename, feature):
    from safetensors.numpy import load_file

    config = json.loads((directory / filename).read_text())
    processor = next(item for item in config["steps"] if "state_file" in item)
    tensors = load_file(str(directory / processor["state_file"]))
    return (tensors[f"{feature}.mean"], tensors[f"{feature}.std"],
            processor["config"].get("eps", 1e-8))


def input_contract(config, wrapper):
    if (config.add_image_special_tokens or config.prefix_length or config.adapt_to_pi_aloha
            or config.empty_cameras or not config.use_cache or config.n_obs_steps != 1
            or wrapper.num_vlm_layers != wrapper.num_expert_layers):
        raise ValueError("Unsupported SmolVLA prefix layout. Provide a compatible exporter for this policy.")
    if config.rtc_config is not None and config.rtc_config.enabled:
        raise ValueError("RTC is unsupported by this exporter")
    resize = config.resize_imgs_with_padding
    if resize is None or resize[0] != resize[1]:
        raise ValueError("The reference exporter requires a fixed square vision input")
    return resize


@torch.inference_mode()
def export(checkpoint, vlm, output, part_size, provenance):
    from lerobot.policies.smolvla.modeling_smolvla import create_sinusoidal_pos_embedding

    torch.set_num_threads(4)
    native = NativePolicy(checkpoint, vlm, "cpu")
    native.policy.float()
    core = native.policy.model
    config = native.policy.config
    wrapper = core.vlm_with_expert
    size = input_contract(config, wrapper)
    model = wrapper.get_vlm_model()
    text = model.text_model
    layers = list(text.layers)
    q_heads, kv_heads = wrapper.num_attention_heads, wrapper.num_key_value_heads
    width = layers[0].self_attn.head_dim
    cross = [i for i in range(len(layers)) if "cross" in wrapper.attention_mode and
             (wrapper.self_attn_every_n_layers <= 0 or i % wrapper.self_attn_every_n_layers)]
    groups = q_heads // kv_heads
    output.mkdir(parents=True, exist_ok=False)

    def save_tensor(name, tensor):
        np.save(output / f"{name}.npy", tensor.detach().float().numpy())

    vision = Vision(model, size).eval()
    pixels = torch.zeros(1, 3, *size)
    image_tokens = vision(pixels)
    torch.testing.assert_close(image_tokens, wrapper.embed_image(pixels), atol=1e-5, rtol=1e-5)
    emit(output, "vision", vision, {"pixels": pixels}, ["tokens"])
    save_tensor("token_embedding", text.embed_tokens.weight)
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(vlm).backend_tokenizer.save(str(output / "tokenizer.json"))
    save_tensor("state_weight", core.state_proj.weight)
    save_tensor("state_bias", core.state_proj.bias)

    # Use torch's resize kernel to derive separable weights for the serialized input size.
    image_shapes = [list(config.input_features[key].shape) for key in native.image_keys]
    if len({tuple(shape) for shape in image_shapes}) != 1 or image_shapes[0][0] != 3:
        raise ValueError("Camera feeds must share the same RGB shape")
    _, height, image_width = image_shapes[0]
    scale = max(image_width / size[0], height / size[1])
    inner = (int(height / scale), int(image_width / scale))

    def resize_axis(source, target):
        identity = torch.eye(source).reshape(source, 1, 1, source)
        matrix = F.interpolate(identity, (1, target), mode="bilinear",
                               align_corners=False)[..., 0, :].squeeze(1).T
        weights, indices = torch.topk(matrix, 2, dim=-1)
        return weights.numpy(), indices.numpy()

    x_weight, x_index = resize_axis(image_width, inner[1])
    y_weight, y_index = resize_axis(height, inner[0])
    np.savez(output / "resize.npz", x_weight=x_weight, x_index=x_index,
             y_weight=y_weight, y_index=y_index, output_size=size)

    length = image_tokens.shape[1] * len(image_shapes) + config.tokenizer_max_length + 1
    hidden = torch.zeros(1, length, text.embed_tokens.embedding_dim)
    prefix_angles = torch.zeros(1, 1, length, width // 2)
    prefix_mask = torch.zeros(1, 1, length * groups, length)
    parts = []
    for start in range(0, len(layers), part_size):
        end = min(start + part_size, len(layers))
        part = Prefill(layers[start:end], width, q_heads, kv_heads, end == len(layers)).eval()
        stabilize(part)
        name = f"prefill_{start:02d}"
        emit(output, name, part, {"hidden": hidden, "mask": prefix_mask, "sine": prefix_angles,
                                  "cosine": prefix_angles + 1}, ["next_hidden", "keys", "values"])
        parts.append(name)

    projections = {}
    for index in cross:
        attn = wrapper.lm_expert.layers[index].self_attn
        for key, projection in (("k", attn.k_proj), ("v", attn.v_proj)):
            projections[f"{key}{index}"] = projection.weight.detach().float().numpy()
            if projection.bias is not None:
                projections[f"{key}{index}_bias"] = projection.bias.detach().float().numpy()
    np.savez(output / "cross_projection.npz", **projections)
    chunk = config.chunk_size
    shape = (len(layers), 1, kv_heads, length, width)
    times = torch.stack([
        create_sinusoidal_pos_embedding(torch.tensor([1 - i / config.num_steps]),
                                        wrapper.expert_hidden_size, config.min_period,
                                        config.max_period, device=torch.device("cpu")).float()
        for i in range(config.num_steps)
    ])
    save_tensor("time", times)
    suffix_angles = torch.zeros(1, 1, chunk, width // 2)
    step = ActionStep(core, width, q_heads, kv_heads, cross).eval()
    stabilize(step)
    emit(output, "action", step, {
        "noise": torch.zeros(1, chunk, config.max_action_dim), "time": times[0],
        "self_sine": suffix_angles, "self_cosine": suffix_angles + 1,
        "cross_sine": suffix_angles, "cross_cosine": suffix_angles + 1,
        "self_mask": torch.zeros(1, 1, chunk * groups, length + chunk),
        "cross_mask": torch.zeros(1, 1, chunk * groups, length),
        "keys": torch.zeros(shape), "values": torch.zeros(shape),
    }, ["next_noise"])

    sm, ss, eps = processor_stats(checkpoint, "policy_preprocessor.json", "observation.state")
    am, ast, _ = processor_stats(checkpoint, "policy_postprocessor.json", "action")
    np.savez(output / "stats.npz", state_mean=sm, state_scale=ss + np.float32(eps),
             action_mean=am, action_scale=ast)
    manifest = {
        "format": 1, "model": provenance, "images": image_shapes, "image_tokens": image_tokens.shape[1],
        "text_length": config.tokenizer_max_length, "prefix_length": length, "head_width": width,
        "heads_per_group": groups, "state_dim": config.max_state_dim, "action_dim": len(am),
        "action_shape": [1, chunk, config.max_action_dim], "execute": config.n_action_steps,
        "denoising_steps": config.num_steps, "prefill": parts, "cross_layers": cross,
        "token_scale": math.sqrt(text.embed_tokens.embedding_dim),
        "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
        "versions": {key: version(key) for key in ("torch", "transformers", "onnx")},
    }
    manifest["files"] = {p.name: sha256(p) for p in sorted(output.iterdir())}
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vlm", type=Path)
    parser.add_argument("--prefill-layers", type=int, default=6)
    args = parser.parse_args()
    if args.prefill_layers <= 0 or bool(args.checkpoint) != bool(args.vlm):
        parser.error("Use a positive partition size and supply both local paths or neither")
    if args.checkpoint:
        checkpoint, vlm = args.checkpoint, args.vlm
        provenance = {"source": "local"}
    else:
        checkpoint, vlm = snapshots()
        provenance = {"id": MODEL, "revision": REVISION, "vlm": VLM, "vlm_revision": VLM_REVISION}
    export(checkpoint, vlm, args.output, args.prefill_layers, provenance)


if __name__ == "__main__":
    main()
