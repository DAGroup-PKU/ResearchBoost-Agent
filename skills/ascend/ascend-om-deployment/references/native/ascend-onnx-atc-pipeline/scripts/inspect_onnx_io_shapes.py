#!/usr/bin/env python3
"""
Inspect ONNX model input/output shapes and optionally patch output shapes when
runtime probes provide enough evidence.

Usage:
    python3 inspect_onnx_io_shapes.py --model model.onnx
    python3 inspect_onnx_io_shapes.py --model model.onnx --probe-outputs
    python3 inspect_onnx_io_shapes.py --model model.onnx --probe-outputs \
        --patch-output-model model_patched.onnx
"""

import argparse
import json
import os
import sys


def _load_onnx():
    try:
        import onnx
        from onnx import TensorProto
    except ImportError:
        print("Error: onnx not installed. Install with: pip install onnx", file=sys.stderr)
        sys.exit(1)
    return onnx, TensorProto


def _dim_to_repr(dim):
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    if dim.HasField("dim_param"):
        return dim.dim_param
    return "?"


def _value_info_to_dict(value_info, tensor_proto):
    tensor_type = value_info.type.tensor_type
    shape = [_dim_to_repr(dim) for dim in tensor_type.shape.dim]
    return {
        "name": value_info.name,
        "elem_type": tensor_proto.DataType.Name(tensor_type.elem_type),
        "shape": shape,
        "is_dynamic": any(not isinstance(dim, int) for dim in shape),
    }


def _input_value_infos(model):
    initializer_names = {init.name for init in model.graph.initializer}
    return [
        value_info
        for value_info in model.graph.input
        if value_info.name not in initializer_names
    ]


def inspect_model_io(input_path):
    onnx, tensor_proto = _load_onnx()
    model = onnx.load(input_path, load_external_data=True)
    inputs = [
        _value_info_to_dict(value_info, tensor_proto)
        for value_info in _input_value_infos(model)
    ]
    outputs = [
        _value_info_to_dict(value_info, tensor_proto) for value_info in model.graph.output
    ]
    all_inputs_static = all(not item["is_dynamic"] for item in inputs)
    has_dynamic_output = any(item["is_dynamic"] for item in outputs)
    return {
        "input_path": input_path,
        "ir_version": int(model.ir_version),
        "opsets": [
            {
                "domain": opset.domain or "",
                "version": int(opset.version),
            }
            for opset in model.opset_import
        ],
        "inputs": inputs,
        "outputs": outputs,
        "all_inputs_static": all_inputs_static,
        "has_dynamic_output": has_dynamic_output,
    }


def parse_dim_list(raw_shape):
    dims = []
    for raw_dim in raw_shape.split(","):
        raw_dim = raw_dim.strip()
        if not raw_dim:
            raise ValueError(f"Invalid empty dimension in shape: {raw_shape}")
        dim = int(raw_dim)
        if dim <= 0:
            raise ValueError(f"Probe dimensions must be positive: {raw_shape}")
        dims.append(dim)
    return dims


def parse_shape_profile(raw_profile):
    profile = {}
    for item in raw_profile.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid profile item {item!r}; expected name:dim0,dim1,..."
            )
        name, raw_shape = item.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid empty input name in profile: {raw_profile}")
        if name in profile:
            raise ValueError(f"Duplicate input name {name!r} in profile")
        profile[name] = parse_dim_list(raw_shape)
    if not profile:
        raise ValueError("Shape profile is empty")
    return profile


def parse_shape_profiles(raw_profiles):
    return [parse_shape_profile(raw_profile) for raw_profile in raw_profiles or []]


def _numpy_dtype_from_onnx(elem_type, tensor_proto):
    import numpy as np

    mapping = {
        tensor_proto.FLOAT: np.float32,
        tensor_proto.FLOAT16: np.float16,
        tensor_proto.DOUBLE: np.float64,
        tensor_proto.INT64: np.int64,
        tensor_proto.INT32: np.int32,
        tensor_proto.INT16: np.int16,
        tensor_proto.INT8: np.int8,
        tensor_proto.UINT64: np.uint64,
        tensor_proto.UINT32: np.uint32,
        tensor_proto.UINT16: np.uint16,
        tensor_proto.UINT8: np.uint8,
        tensor_proto.BOOL: np.bool_,
    }
    if elem_type not in mapping:
        raise ValueError(
            f"Unsupported ONNX elem_type: {tensor_proto.DataType.Name(elem_type)}"
        )
    return mapping[elem_type]


def _shape_from_profile_or_graph(value_info, profile):
    graph_shape = [_dim_to_repr(dim) for dim in value_info.type.tensor_type.shape.dim]
    if value_info.name not in profile:
        if any(not isinstance(dim, int) for dim in graph_shape):
            raise ValueError(
                f"Cannot auto-probe dynamic input {value_info.name} with shape "
                f"{graph_shape}; provide --probe-shape-profile"
            )
        return graph_shape

    shape = profile[value_info.name]
    if len(shape) != len(graph_shape):
        raise ValueError(
            f"Profile shape rank mismatch for {value_info.name}: got {shape}, "
            f"graph rank is {len(graph_shape)}"
        )
    for index, (graph_dim, profile_dim) in enumerate(zip(graph_shape, shape)):
        if isinstance(graph_dim, int) and graph_dim != profile_dim:
            raise ValueError(
                f"Profile shape mismatch for {value_info.name} dim {index}: "
                f"got {profile_dim}, graph requires {graph_dim}"
            )
    return shape


def _validate_profile_names(profiles, graph_inputs):
    input_names = {value_info.name for value_info in graph_inputs}
    for index, profile in enumerate(profiles):
        unknown = sorted(set(profile) - input_names)
        if unknown:
            raise ValueError(
                f"Profile {index} contains unknown graph inputs: {unknown}"
            )


def _make_feeds(model, tensor_proto, profile):
    import numpy as np

    graph_inputs = _input_value_infos(model)
    _validate_profile_names([profile], graph_inputs)

    feeds = {}
    for value_info in graph_inputs:
        tensor_type = value_info.type.tensor_type
        shape = _shape_from_profile_or_graph(value_info, profile)
        dtype = _numpy_dtype_from_onnx(tensor_type.elem_type, tensor_proto)
        feeds[value_info.name] = np.zeros(shape, dtype=dtype)
    return feeds


def probe_output_shapes_with_onnxruntime(input_path, profiles=None):
    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "Error: onnxruntime not installed. Install with: pip install onnxruntime",
            file=sys.stderr,
        )
        sys.exit(1)

    onnx, tensor_proto = _load_onnx()
    model = onnx.load(input_path, load_external_data=True)
    profiles = profiles or [{}]
    _validate_profile_names(profiles, _input_value_infos(model))
    session = ort.InferenceSession(input_path, providers=["CPUExecutionProvider"])

    runs = []
    for profile_index, profile in enumerate(profiles):
        feeds = _make_feeds(model, tensor_proto, profile)
        output_values = session.run(None, feeds)
        output_meta = session.get_outputs()
        outputs = []
        for meta, value in zip(output_meta, output_values):
            outputs.append(
                {
                    "name": meta.name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            )
        runs.append(
            {
                "profile_index": profile_index,
                "profile": profile,
                "feeds": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in feeds.items()
                },
                "outputs": outputs,
            }
        )

    return {
        "input_path": input_path,
        "runs": runs,
    }


def output_shape_template_from_probe(probe):
    runs = probe["runs"]
    if not runs:
        return {}

    by_name = {}
    for run in runs:
        for output in run["outputs"]:
            by_name.setdefault(output["name"], []).append(output["shape"])

    templates = {}
    for name, shapes in by_name.items():
        ranks = {len(shape) for shape in shapes}
        if len(ranks) != 1:
            raise ValueError(f"Output {name} rank varies across profiles: {shapes}")
        rank = ranks.pop()
        dims = []
        for axis in range(rank):
            values = [shape[axis] for shape in shapes]
            if len(set(values)) == 1:
                dims.append({"kind": "fixed", "value": int(values[0])})
            else:
                dims.append(
                    {
                        "kind": "symbolic",
                        "param": f"{name}_dim{axis}",
                        "values": [int(value) for value in values],
                    }
                )
        templates[name] = dims
    return templates


def profile_coverage_for_dynamic_inputs(inspect_summary, profiles):
    if not profiles:
        return {
            "profile_count": 0,
            "uncovered_dynamic_axes": [],
            "covered_dynamic_axes": [],
        }

    profile_names = set(profiles[0])
    for index, profile in enumerate(profiles[1:], start=1):
        if set(profile) != profile_names:
            raise ValueError(f"Profile {index} inputs do not match profile 0 inputs")

    covered = []
    uncovered = []
    for item in inspect_summary["inputs"]:
        name = item["name"]
        shape = item["shape"]
        if name not in profile_names:
            if item["is_dynamic"]:
                uncovered.append({"input": name, "axis": "*", "reason": "missing"})
            continue
        profile_shapes = [profile[name] for profile in profiles]
        ranks = {len(profile_shape) for profile_shape in profile_shapes}
        if len(ranks) != 1:
            raise ValueError(f"Profiles for {name} have varying ranks: {profile_shapes}")
        rank = ranks.pop()
        if rank != len(shape):
            raise ValueError(
                f"Profile rank mismatch for {name}: got {rank}, graph rank is {len(shape)}"
            )
        for axis, graph_dim in enumerate(shape):
            if isinstance(graph_dim, int):
                continue
            values = [profile_shape[axis] for profile_shape in profile_shapes]
            entry = {
                "input": name,
                "axis": axis,
                "graph_dim": graph_dim,
                "values": values,
            }
            if len(set(values)) > 1:
                covered.append(entry)
            else:
                uncovered.append(entry)

    return {
        "profile_count": len(profiles),
        "covered_dynamic_axes": covered,
        "uncovered_dynamic_axes": uncovered,
    }


def patch_decision(inspect_summary, profiles, output_templates):
    if not output_templates:
        return {"patch": False, "reason": "no_output_templates"}
    if inspect_summary["all_inputs_static"]:
        return {"patch": True, "reason": "all_inputs_static"}
    if len(profiles) < 2:
        return {
            "patch": False,
            "reason": "dynamic_inputs_need_at_least_two_profiles",
        }

    coverage = profile_coverage_for_dynamic_inputs(inspect_summary, profiles)
    if coverage["uncovered_dynamic_axes"]:
        return {
            "patch": False,
            "reason": "profiles_do_not_cover_all_dynamic_input_axes",
            "coverage": coverage,
        }
    return {
        "patch": True,
        "reason": "profiles_cover_all_dynamic_input_axes",
        "coverage": coverage,
    }


def patch_graph_output_shapes(input_path, output_path, output_templates):
    onnx, _ = _load_onnx()
    model = onnx.load(input_path, load_external_data=True)
    patched = 0

    for output in model.graph.output:
        template = output_templates.get(output.name)
        if template is None:
            continue
        output.type.tensor_type.shape.ClearField("dim")
        for dim in template:
            new_dim = output.type.tensor_type.shape.dim.add()
            if dim["kind"] == "fixed":
                new_dim.dim_value = int(dim["value"])
            else:
                new_dim.dim_param = dim["param"]
        patched += 1

    onnx.save_model(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(output_path) + ".data",
        size_threshold=1024,
    )
    return {
        "input_path": input_path,
        "output_path": output_path,
        "patched_outputs": patched,
        "requested_outputs": len(output_templates),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect ONNX input/output shapes and optionally probe or patch "
            "output shapes when runtime probes provide enough evidence."
        )
    )
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument(
        "--probe-outputs",
        action="store_true",
        help="Run ONNX Runtime to probe output shapes from static inputs or profiles",
    )
    parser.add_argument(
        "--probe-shape-profile",
        action="append",
        default=[],
        help=(
            "Concrete input profile for ORT probing, formatted as "
            "'input0:1,3,1024,1024;input1:1,256'. Repeat to cover dynamic axes."
        ),
    )
    parser.add_argument(
        "--patch-output-model",
        help=(
            "Write a patched ONNX with probed graph output shapes. "
            "Requires --probe-outputs."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    summary = inspect_model_io(args.model)
    print(json.dumps({"inspect": summary}, ensure_ascii=False, indent=2))

    profiles = parse_shape_profiles(args.probe_shape_profile)

    if not args.probe_outputs and not args.patch_output_model:
        return

    if not summary["all_inputs_static"] and not profiles:
        raise SystemExit(
            "Refusing ORT probe because graph inputs are still dynamic. "
            "Provide --probe-shape-profile for dynamic-input models."
        )

    probe = probe_output_shapes_with_onnxruntime(args.model, profiles)
    print(json.dumps({"probe": probe}, ensure_ascii=False, indent=2))

    if not args.patch_output_model:
        return

    output_templates = output_shape_template_from_probe(probe)
    decision = patch_decision(summary, profiles, output_templates)
    print(
        json.dumps(
            {"output_shape_template": output_templates, "patch_decision": decision},
            ensure_ascii=False,
            indent=2,
        )
    )
    if not decision["patch"]:
        return

    patch_result = patch_graph_output_shapes(
        args.model,
        args.patch_output_model,
        output_templates,
    )
    print(
        json.dumps(
            {"patch": patch_result},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
