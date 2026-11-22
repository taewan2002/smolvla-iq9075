"""Capture detailed QNN GPU events for both vision views in a saved observation."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .runtime import SessionFactory, opened_device_nodes, prepare_rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output CSV path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.bundle / "bundle.json").read_text())
    dtype = np.float16 if manifest.get("graph_precision") == "fp16" else np.float32
    with (
        np.load(args.bundle / "resize.npz", allow_pickle=False) as resize,
        np.load(args.fixture, allow_pickle=False) as fixture,
    ):
        views = [prepare_rgb(view, resize).astype(dtype, copy=False) for view in fixture["rgb"]]
    session = SessionFactory(4, "gpu").create(args.bundle / "vision.onnx", 0, qnn_profile=args.output)
    devices = opened_device_nodes("/dev/kgsl-", "/dev/dri/renderD")
    if not devices:
        raise RuntimeError("QNN GPU vision session did not open an Adreno device node")
    print("GPU devices", devices, flush=True)

    # The first execution is a warm-up and is also present in the QNN trace.
    for index, view in enumerate((views[0], *views)):
        start = perf_counter()
        output = session.run(None, {"pixels": view})[0]
        elapsed_ms = (perf_counter() - start) * 1000
        if not np.isfinite(output).all():
            raise FloatingPointError("Vision returned a non-finite output")
        print(json.dumps({"warmup": index == 0, "ms": elapsed_ms, "shape": output.shape}), flush=True)


if __name__ == "__main__":
    main()
