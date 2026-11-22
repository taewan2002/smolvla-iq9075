"""Measure on-device latency and numerical agreement on identical saved LIBERO observations."""
import argparse
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from .native import NativePolicy, sha256, snapshots
from .runtime import EDGE_BACKENDS, EdgePolicy


def statistics(values):
    return {field: round(float(p), 4) for field, p in
            zip(("median", "p95", "max"), np.percentile(values, [50, 95, 100]))}


def compare(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    error = left - right
    return {
        "max_absolute": float(np.abs(error).max()),
        "mean_absolute": float(np.abs(error).mean()),
        "relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(right), 1e-12)),
        "mean_absolute_by_action_dim": np.abs(error).reshape(-1, left.shape[-1]).mean(0).tolist(),
    }


def measure(predict, fixtures, repeats, warmup):
    observations = []
    for file in fixtures:
        with np.load(file, allow_pickle=False) as data:
            observations.append((data["rgb"], data["state"], str(data["task"].item()), data["noise"]))
    for index in range(warmup):
        predict(*observations[index % len(observations)])
    timings, outputs = [], []
    for repeat in range(repeats):
        for file, args in zip(fixtures, observations, strict=True):
            result, timing = predict(*args)
            timings.append({"fixture": file.name, "repeat": repeat, **timing})
            if repeat == 0:
                outputs.append(result[0])
    aggregate = {key: statistics([t[key] for t in timings]) for key in timings[0] if key.endswith("_ms")}
    return np.stack(outputs), timings, aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--engines", nargs="+", choices=("native", *EDGE_BACKENDS), default=["cpu", "htp", "dual"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vlm", type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--execute", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--serve-port", type=int, help="Keep the final ONNX engine ready for LIBERO")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.threads < 1 or args.execute < 1:
        parser.error("Invalid repeats, warm-up, thread or execution count")
    if bool(args.checkpoint) != bool(args.vlm):
        parser.error("Supply both --checkpoint and --vlm or neither")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.serve_port is not None and not 1 <= args.serve_port <= 65535:
        parser.error("--serve-port must be between 1 and 65535")
    if args.serve_port and args.engines[-1] == "native":
        parser.error("--serve-port requires an ONNX engine as the final engine")
    files = sorted(args.fixtures.glob("*.npz"),
                   key=lambda path: (path.stem.rsplit("-", 2)[-2:], path.name))[:args.limit]
    if not files:
        parser.error("No fixtures found")
    native = []
    for file in files:
        with np.load(file, allow_pickle=False) as fixture:
            native.append(fixture["native"])
    native = np.stack(native)
    report = {
        "bundle_manifest_sha256": sha256(args.bundle / "bundle.json"),
        "machine": platform.machine(), "kernel": platform.release(),
        "warmup": args.warmup, "repeats": args.repeats, "threads": args.threads,
        "executed_slice": args.execute, "fixtures": {file.name: sha256(file) for file in files},
        "runs": {},
    }
    if args.execute > native.shape[1]:
        parser.error("Execution slice exceeds fixture chunk length")
    baseline = None
    for index, engine in enumerate(args.engines):
        begin = perf_counter()
        if engine == "native":
            import torch

            checkpoint, vlm = (args.checkpoint, args.vlm) if args.checkpoint else snapshots()
            manifest = json.loads((args.bundle / "bundle.json").read_text())
            if sha256(checkpoint / "model.safetensors") != manifest["checkpoint_sha256"]:
                raise ValueError("Native checkpoint and exported bundle differ")
            torch.set_num_threads(args.threads)
            policy = NativePolicy(checkpoint, vlm, "cpu")
            details = {"torch": torch.__version__}

            def predict(*obs):
                start = perf_counter()
                result = policy(*obs)
                return result, {"total_ms": (perf_counter() - start) * 1000}
        else:
            policy = EdgePolicy(args.bundle, engine, args.threads)
            predict = policy.predict
            details = {"htp_nodes": policy.devices, "onnxruntime": policy.ort_version,
                       "qnn_provider": policy.qnn_version}
            if engine == "gpu":
                details["gpu_nodes"] = policy.gpu_devices
                details["graph_precision"] = policy.graph_precision
        startup_ms = (perf_counter() - begin) * 1000
        try:
            outputs, timings, aggregate = measure(predict, files, args.repeats, args.warmup)
            row = {"startup_ms": round(startup_ms, 4), **details,
                   "aggregate_ms": aggregate, "raw_ms": timings,
                   "vs_native": compare(outputs, native),
                   "executed_slice_vs_native": compare(outputs[:, :args.execute], native[:, :args.execute])}
            if engine == "cpu":
                baseline = outputs
            if baseline is not None and engine != "cpu":
                row["vs_cpu"] = compare(outputs, baseline)
            report["runs"][engine] = row
            print(json.dumps({engine: aggregate, "vs_native": row["vs_native"]}), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(args.output)
            if args.serve_port and index == len(args.engines) - 1:
                from .remote import serve

                serve(policy, args.serve_port)
        finally:
            if hasattr(policy, "close"):
                policy.close()
            del predict, policy



if __name__ == "__main__":
    main()
