"""Use QNN GPU's pointwise convolution kernels for the FP16 vision linear layers."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference

from .native import sha256


def rewrite_vision(model):
    graph = model.graph
    inferred = shape_inference.infer_shapes(model).graph
    shapes = {
        item.name: tuple(dim.dim_value for dim in item.type.tensor_type.shape.dim)
        for item in (*inferred.input, *inferred.value_info)
    }
    if shapes.get("pixels") != (1, 3, 512, 512):
        raise ValueError("Expected the batch-one 512×512 SmolVLA vision graph")

    weights = {value.name: value for value in graph.initializer}
    consumers = defaultdict(list)
    for node in graph.node:
        for name in node.input:
            consumers[name].append(node)

    replacements = {}
    bias_nodes = set()
    replaced_weights = set()
    for node in graph.node:
        if (node.op_type != "MatMul" or not node.name.startswith("/layers.")
                or not ("/mlp/" in node.name or "/self_attn/" in node.name)
                or node.input[1] not in weights):
            continue
        following = consumers[node.output[0]]
        if len(following) != 1 or following[0].op_type != "Add":
            continue
        add = following[0]
        bias = next(name for name in add.input if name != node.output[0])
        weight = numpy_helper.to_array(weights[node.input[1]])
        if weight.ndim != 2 or bias not in weights:
            continue
        channels_in, channels_out = weight.shape
        if (shapes.get(node.input[0], ())[-2:] != (1024, channels_in)
                or tuple(weights[bias].dims) != (channels_out,)):
            continue

        name = f"{node.name}/conv1x1"
        kernel = f"{name}/weights"
        spatial_shape = f"{name}/spatial_shape"
        sequence_shape = f"{name}/sequence_shape"
        graph.initializer.extend([
            numpy_helper.from_array(np.ascontiguousarray(weight.T[:, :, None, None]), kernel),
            numpy_helper.from_array(np.array([1, channels_in, 32, 32], dtype=np.int64), spatial_shape),
            numpy_helper.from_array(np.array([1, channels_out, 1024], dtype=np.int64), sequence_shape),
        ])
        transposed, spatial, convolved, flattened = (f"{name}/tmp{i}" for i in range(4))
        replacements[node.name] = [
            helper.make_node("Transpose", [node.input[0]], [transposed], name=f"{name}/ToNCHW", perm=[0, 2, 1]),
            helper.make_node("Reshape", [transposed, spatial_shape], [spatial], name=f"{name}/Spatial"),
            helper.make_node("Conv", [spatial, kernel, bias], [convolved], name=name, kernel_shape=[1, 1]),
            helper.make_node("Reshape", [convolved, sequence_shape], [flattened], name=f"{name}/Sequence"),
            helper.make_node("Transpose", [flattened], [add.output[0]], name=f"{name}/ToOriginal", perm=[0, 2, 1]),
        ]
        bias_nodes.add(add.name)
        replaced_weights.add(node.input[1])

    if not replacements:
        raise ValueError("No supported FP16 vision linear layers found")
    nodes = []
    for node in graph.node:
        if node.name not in bias_nodes:
            nodes.extend(replacements.get(node.name, [node]))
    del graph.node[:]
    graph.node.extend(nodes)
    used = {name for node in graph.node for name in node.input} | {output.name for output in graph.output}
    parameters = [value for value in graph.initializer if value.name not in replaced_weights or value.name in used]
    del graph.initializer[:]
    graph.initializer.extend(parameters)
    onnx.checker.check_model(model)
    return len(replacements)


def convert(bundle, output):
    manifest_path = bundle / "bundle.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest["format"] != 1 or manifest.get("graph_precision") != "fp16"
            or manifest.get("prefill_cache_rank") != 4):
        raise ValueError("Use an Adreno FP16 bundle")
    for name, digest in manifest["files"].items():
        if sha256(bundle / name) != digest:
            raise ValueError(f"Bundle checksum differs for {name}")
    vision = onnx.load(bundle / "vision.onnx")
    count = rewrite_vision(vision)
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest["files"]:
        if name == "vision.onnx":
            onnx.save(vision, output / name)
        else:
            shutil.copyfile(bundle / name, output / name)
    manifest["source_bundle_sha256"] = sha256(manifest_path)
    manifest["vision_linear_conv_count"] = count
    manifest["files"]["vision.onnx"] = sha256(output / "vision.onnx")
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Rewrote {count} vision linear layers", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    convert(args.bundle, args.output)


if __name__ == "__main__":
    main()
