"""Batch-one ONNX policy runtime with CPU, single-HTP and dual-HTP schedules."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import os
from pathlib import Path
from time import perf_counter_ns

import numpy as np


def milliseconds(start):
    return (perf_counter_ns() - start) / 1e6


def rotary(position, width):
    frequency = np.float32(10000) ** (np.arange(width // 2, dtype=np.float32) * (2.0 / width))
    phase = position.astype(np.float32)[None, None, :, None] / frequency
    return np.sin(phase), np.cos(phase)


def bias(valid, groups):
    return np.where(np.tile(valid, (groups, 1)), np.float32(0), np.float32(-10000))[None, None]


def opened_htp_nodes():
    paths = set()
    fd_dir = Path("/proc/self/fd")
    if fd_dir.is_dir():
        for entry in fd_dir.iterdir():
            try:
                path = os.readlink(entry)
            except OSError:
                continue
            if path.startswith("/dev/fastrpc-cdsp"):
                paths.add(path)
    return sorted(paths)


@lru_cache(maxsize=1)
def qnn_setup():
    from importlib.metadata import version

    import onnxruntime as ort
    import onnxruntime_qnn as qnn

    ort.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())
    devices = [
        device for device in ort.get_ep_devices()
        if device.ep_name == "QNNExecutionProvider"
        and device.device.type == ort.OrtHardwareDeviceType.NPU
    ][:1]
    if not devices:
        raise RuntimeError("No QNN HTP device registered")
    return version("onnxruntime-qnn"), Path(qnn.__file__).parent / "libQnnHtp.so", devices


class SessionFactory:
    def __init__(self, threads, use_qnn):
        import onnxruntime as ort

        self.ort = ort
        self.threads = threads
        self.qnn_version = None
        if use_qnn:
            self.qnn_version, self.backend, self.devices = qnn_setup()

    def create(self, path, device=None):
        ort = self.ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.threads
        opts.inter_op_num_threads = 1
        if device is None:
            return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

        opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        opts.add_provider_for_devices(self.devices, {
            "backend_path": str(self.backend), "device_id": str(device), "htp_arch": "73",
            "enable_htp_fp16_precision": "1", "htp_performance_mode": "burst",
            "offload_graph_io_quantization": "0", "htp_graph_finalization_optimization_mode": "3",
        })
        session = ort.InferenceSession(str(path), opts)
        session.disable_fallback()
        if session.get_providers()[0] != "QNNExecutionProvider":
            raise RuntimeError(f"Could not place {path.name} on HTP {device}")
        return session


class EdgePolicy:
    def __init__(self, bundle, backend="dual", threads=4):
        from tokenizers import Tokenizer

        bundle = Path(bundle)
        self.manifest = meta = json.loads((bundle / "bundle.json").read_text())
        if meta["format"] != 1:
            raise ValueError(f"Unsupported bundle format {meta['format']}")
        if backend not in ("cpu", "htp", "dual"):
            raise ValueError(f"Unknown backend: {backend}")
        self.backend = backend
        self.action_shape = tuple(meta["action_shape"])
        self.width = meta["head_width"]
        self.groups = meta["heads_per_group"]
        self.tokenizer = Tokenizer.from_file(str(bundle / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=meta["text_length"])
        self.table = np.load(bundle / "token_embedding.npy", mmap_mode="r")
        self.time = np.load(bundle / "time.npy")
        self.state_weight = np.load(bundle / "state_weight.npy")
        self.state_bias = np.load(bundle / "state_bias.npy")
        for field, name in (("resize", "resize"), ("stats", "stats"), ("projection", "cross_projection")):
            with np.load(bundle / f"{name}.npz") as archive:
                setattr(self, field, {key: archive[key] for key in archive})

        maker = SessionFactory(threads, backend != "cpu")
        primary = None if backend == "cpu" else 0
        self.ort_version = maker.ort.__version__
        self.qnn_version = maker.qnn_version
        self.vision = [maker.create(bundle / "vision.onnx", primary)]
        if backend == "dual":
            self.vision.append(maker.create(bundle / "vision.onnx", 1))
        self.prefill = []
        for index, part in enumerate(meta["prefill"]):
            device = index % 2 if backend == "dual" else primary
            self.prefill.append(maker.create(bundle / f"{part}.onnx", device))
        self.action = maker.create(bundle / "action.onnx", primary)
        self.action_inputs = {item.name for item in self.action.get_inputs()}
        self.devices = opened_htp_nodes()
        self.pool = ThreadPoolExecutor(max_workers=2) if backend == "dual" else None

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()

    def prepare_rgb(self, rgb):
        params = self.resize
        image = rgb.astype(np.float32) / np.float32(255.0)
        xi, xw = params["x_index"], params["x_weight"]
        yi, yw = params["y_index"], params["y_weight"]
        along_x = image[:, xi[:, 0]] * xw[None, :, 0, None] + image[:, xi[:, 1]] * xw[None, :, 1, None]
        inner = along_x[yi[:, 0]] * yw[:, 0, None, None] + along_x[yi[:, 1]] * yw[:, 1, None, None]
        height, width = params["output_size"]
        result = np.full((1, 3, height, width), -1, dtype=np.float32)
        result[0, :, height - inner.shape[0]:, width - inner.shape[1]:] = inner.transpose(2, 0, 1) * 2 - 1
        return result

    def __call__(self, rgb, state, task, noise):
        return self.predict(rgb, state, task, noise)[0]

    def predict(self, rgb, state, task, noise):
        total = perf_counter_ns()
        stage = total
        meta = self.manifest
        stats = self.stats
        expected = [(shape[1], shape[2], shape[0]) for shape in meta["images"]]
        if len(rgb) != len(expected) or any(
            image.shape != shape or image.dtype != np.uint8 for image, shape in zip(rgb, expected)
        ):
            raise ValueError(f"Expected uint8 NHWC RGB images with per-view shapes {expected}")
        if state.shape != stats["state_mean"].shape or noise.shape != self.action_shape:
            raise ValueError("Observation state or initial noise shape does not match the bundle")
        if noise.dtype != np.float32 or not np.isfinite(state).all() or not np.isfinite(noise).all():
            raise ValueError("Use finite state values and finite float32 initial noise")

        frames = [self.prepare_rgb(image) for image in rgb]
        ids = self.tokenizer.encode(task if task.endswith("\n") else task + "\n").ids
        language = np.zeros((1, meta["text_length"], self.table.shape[-1]), dtype=np.float32)
        language[:, :len(ids)] = self.table[ids]
        state_input = np.zeros((1, meta["state_dim"]), dtype=np.float32)
        state_input[0, :state.size] = (state - stats["state_mean"]) / stats["state_scale"]
        state_token = (state_input @ self.state_weight.T + self.state_bias)[:, None]
        times = {"preprocess_ms": milliseconds(stage)}

        stage = perf_counter_ns()

        def encode(index):
            return self.vision[index % len(self.vision)].run(None, {"pixels": frames[index]})[0]

        indexes = range(len(frames))
        # One active job per HTP. Other camera configurations retain the serial path.
        if self.pool is not None and len(frames) == 2:
            tokens = list(self.pool.map(encode, indexes))
        else:
            tokens = [encode(index) for index in indexes]
        times["vision_ms"] = milliseconds(stage)

        stage = perf_counter_ns()
        visual = np.concatenate(tokens, axis=1)
        hidden = np.concatenate((visual * meta["token_scale"], language * meta["token_scale"], state_token), axis=1)
        valid = np.ones(meta["prefix_length"], dtype=bool)
        valid[visual.shape[1] + len(ids):-1] = False
        allowed = valid[:, None] & valid[None, :]
        allowed[:-1, -1] = False
        mask = bias(allowed, self.groups)
        sine, cosine = rotary(valid.cumsum() - 1, self.width)
        keys, values = [], []
        for session in self.prefill:
            candidates = {"hidden": hidden, "mask": mask, "sine": sine, "cosine": cosine}
            hidden, key, value = session.run(None, {item.name: candidates[item.name] for item in session.get_inputs()})
            keys.append(key)
            values.append(value)
        keys, values = np.concatenate(keys), np.concatenate(values)
        times["prefill_ms"] = milliseconds(stage)

        stage = perf_counter_ns()
        for index in meta["cross_layers"]:
            for symbol, cache in (("k", keys), ("v", values)):
                entry = cache[index].transpose(0, 2, 1, 3)
                name = f"{symbol}{index}"
                flat = entry.reshape(1, entry.shape[1], -1)
                result = flat @ self.projection[name].T
                if name + "_bias" in self.projection:
                    result += self.projection[name + "_bias"]
                cache[index] = result.reshape(entry.shape).transpose(0, 2, 1, 3)
        chunk = self.action_shape[1]
        positions = np.arange(chunk)
        self_sine, self_cosine = rotary(valid.sum() + positions, self.width)
        cross_sine, cross_cosine = rotary(positions, self.width)
        cross_mask = np.broadcast_to(valid, (chunk, len(valid)))
        self_mask = np.concatenate((cross_mask, positions[:, None] >= positions[None, :]), axis=-1)
        context = {"self_sine": self_sine, "self_cosine": self_cosine,
                   "cross_sine": cross_sine, "cross_cosine": cross_cosine,
                   "self_mask": bias(self_mask, self.groups), "cross_mask": bias(cross_mask, self.groups),
                   "keys": keys, "values": values}
        context = {key: value for key, value in context.items() if key in self.action_inputs}
        times["cache_ms"] = milliseconds(stage)

        stage = perf_counter_ns()
        result = noise
        for step in self.time:
            result = self.action.run(None, {"noise": result, "time": step, **context})[0]
        action = result[..., :meta["action_dim"]] * stats["action_scale"] + stats["action_mean"]
        if not np.isfinite(action).all():
            raise FloatingPointError("Inference returned a non-finite action")
        times["action_ms"] = milliseconds(stage)
        times["total_ms"] = milliseconds(total)
        return action, times
