"""Convert an Adreno-compatible FP32 bundle to FP16 with the same checkpoint and schedule."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnxconverter_common.float16 import convert_float_to_float16

from .native import sha256


def stabilize_fp16_rmsnorm(model):
    producers = {name: node for node in model.graph.node for name in node.output}
    nodes = {node.name: node for node in model.graph.node}
    after_division = {}
    inverse_epsilon_scale = "adreno_fp16_rmsnorm_inverse_epsilon_scale"
    for node in model.graph.node:
        if node.op_type != "Clip" or len(node.input) < 2:
            continue
        scale = producers.get(node.input[0])
        floor = producers.get(node.input[1])
        if scale is None or scale.op_type != "Div" or floor is None or floor.op_type != "Constant":
            continue
        maximum = producers.get(scale.input[0])
        divisor = producers.get(scale.input[1])
        if maximum is None or maximum.op_type != "ReduceMax" or divisor is None or divisor.op_type != "Constant":
            continue
        limit = numpy_helper.to_array(divisor.attribute[0].t)
        lower_bound = floor.attribute[0].t
        bound = numpy_helper.to_array(lower_bound)
        if limit.size != 1 or bound.size != 1 or limit.item() != 32 or bound.item() != 1:
            continue
        scope = node.name.rsplit("/", 1)[0]
        epsilon = nodes[f"{scope}/Constant_2"].attribute[0].t
        epsilon_term = nodes[f"{scope}/Div_2"]
        # Compute the same epsilon / scale**2 with a normal FP16 constant,
        # and with scale small enough that zero rows still have a normal epsilon term.
        lower_bound.CopyFrom(numpy_helper.from_array(np.full_like(bound, 0.25), lower_bound.name))
        value = numpy_helper.to_array(epsilon)
        epsilon.CopyFrom(numpy_helper.from_array(value * np.float32(16), epsilon.name))
        output = epsilon_term.output[0]
        epsilon_term.output[0] = f"{output}_scaled"
        after_division[epsilon_term.name] = helper.make_node(
            "Mul", [epsilon_term.output[0], inverse_epsilon_scale], [output],
            name=f"{scope}/RestoreEpsilonScale",
        )
    if after_division:
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(1 / 16, dtype=np.float32), inverse_epsilon_scale)
        )
        ordered = []
        for node in model.graph.node:
            ordered.append(node)
            if node.name in after_division:
                ordered.append(after_division[node.name])
        del model.graph.node[:]
        model.graph.node.extend(ordered)


def replace_reciprocal_with_div(model):
    reciprocals = [node for node in model.graph.node if node.op_type == "Reciprocal"]
    if not reciprocals:
        return
    # QNN GPU's Reciprocal builder uses a four-byte constant for this two-byte dtype.
    # An explicit FP16 numerator gives the same reciprocal via ordinary division.
    numerator = "adreno_fp16_reciprocal_one"
    model.graph.initializer.append(numpy_helper.from_array(np.ones(1, dtype=np.float16), numerator))
    for node in reciprocals:
        node.op_type = "Div"
        node.input.insert(0, numerator)


def convert(bundle, output):
    manifest_path = bundle / "bundle.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest["format"] != 1 or manifest.get("prefill_cache_rank") != 4
            or manifest.get("graph_precision", "fp32") != "fp32"):
        raise ValueError("Use an FP32 bundle exported with --adreno-compatible")
    for name, digest in manifest["files"].items():
        if sha256(bundle / name) != digest:
            raise ValueError(f"Bundle checksum differs for {name}")

    output.mkdir(parents=True, exist_ok=False)
    for name in manifest["files"]:
        source, target = bundle / name, output / name
        if source.suffix == ".onnx":
            model = onnx.load(source)
            stabilize_fp16_rmsnorm(model)
            model = convert_float_to_float16(
                model, keep_io_types=False, op_block_list=[], min_positive_val=0.0, max_finite_val=65504.0,
            )
            replace_reciprocal_with_div(model)
            onnx.checker.check_model(model)
            onnx.save(model, target)
        else:
            shutil.copyfile(source, target)
        print(f"ready {name}", flush=True)

    manifest["graph_precision"] = "fp16"
    manifest["source_bundle_sha256"] = sha256(manifest_path)
    manifest["versions"]["onnxconverter-common"] = version("onnxconverter-common")
    manifest["files"] = {name: sha256(output / name) for name in manifest["files"]}
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    convert(args.bundle, args.output)


if __name__ == "__main__":
    main()
