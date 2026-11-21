#!/usr/bin/env python3
"""
ONNX vs OM Precision Comparison Script

Compare inference outputs between CPU ONNX and NPU OM models to verify
conversion accuracy.

Usage:
    python3 compare_precision.py --onnx model.onnx --om model.om --input test.npy
    python3 compare_precision.py --onnx model.onnx --om model.om --input test.npy --atol 1e-3 --rtol 1e-2
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime not installed.")
    print("Install with: pip install onnxruntime")
    sys.exit(1)

try:
    from ais_bench.infer.interface import InferSession
except ImportError:
    print("Error: ais_bench package not installed.")
    print("Run the unified environment check and repair first:")
    print(f"  bash {SCRIPT_DIR / 'check_env_enhanced.sh'}")
    print(
        "  ASCEND_ONNX_ATC_PIPELINE_OM_WHEEL_DIR=/path/to/wheels "
        f"bash {SCRIPT_DIR / 'repair_python_env.sh'}"
    )
    sys.exit(1)


def get_device_id():
    """Get available NPU device ID."""
    return int(os.environ.get("ASCEND_DEVICE_ID", 0))


def load_input(input_path, dtype=np.float32):
    """Load input data from file."""
    if input_path.endswith(".npy"):
        return np.load(input_path)
    elif input_path.endswith(".bin"):
        return np.fromfile(input_path, dtype=dtype)
    else:
        raise ValueError(f"Unsupported input format: {input_path}")


def safe_dtype_text(array_like):
    """Return a printable dtype string across patched NumPy builds."""
    dtype = getattr(array_like, "dtype", None)
    if dtype is None:
        return "unknown"

    array_interface = getattr(array_like, "__array_interface__", None)
    if isinstance(array_interface, dict):
        typestr = array_interface.get("typestr")
        if typestr:
            return str(typestr)

    try:
        return np.asarray(array_like).dtype.str
    except Exception:
        pass

    try:
        return object.__repr__(dtype)
    except Exception:
        return "unknown"


def to_plain_ndarray(array_like):
    """Normalize array-like outputs before numeric comparison."""
    arr = np.asarray(array_like)
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return np.array(arr, copy=False)


def run_onnx_inference(model_path, inputs):
    """Run ONNX model inference on CPU."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Get input names
    input_names = [inp.name for inp in session.get_inputs()]

    # Prepare inputs dict
    if isinstance(inputs, np.ndarray):
        inputs = [inputs]

    inputs_dict = {name: inp for name, inp in zip(input_names, inputs)}

    # Run inference
    outputs = session.run(None, inputs_dict)

    return outputs, session


def run_om_inference(model_path, inputs, device_id=0, mode="static", custom_sizes=100000):
    """Run OM model inference on NPU."""
    session = InferSession(device_id=device_id, model_path=model_path)

    # Get input info
    model_inputs = session.get_inputs()
    print("\nOM Model Inputs:")
    for i, inp in enumerate(model_inputs):
        print(f"  [{i}] {inp.name}: shape={inp.shape}, type={inp.datatype}")

    model_outputs = session.get_outputs()
    print("\nOM Model Outputs:")
    for i, out in enumerate(model_outputs):
        print(f"  [{i}] {out.name}: shape={out.shape}, type={out.datatype}")

    # Prepare inputs
    if isinstance(inputs, np.ndarray):
        inputs = [inputs]

    # Run inference
    outputs = session.infer(inputs, mode=mode, custom_sizes=custom_sizes)

    return outputs, session


def compare_outputs(onnx_outputs, om_outputs, atol=1e-5, rtol=1e-3):
    """
    Compare ONNX and OM outputs with detailed metrics.

    Returns dict with comparison results for each output tensor.
    """
    results = []

    for i, (onnx_out, om_out) in enumerate(zip(onnx_outputs, om_outputs)):
        onnx_out = to_plain_ndarray(onnx_out)
        om_out = to_plain_ndarray(om_out)

        # Ensure same shape
        if onnx_out.shape != om_out.shape:
            print(
                f"Warning: Output {i} shape mismatch: ONNX {onnx_out.shape} vs OM {om_out.shape}"
            )
            # Try to reshape if total elements match
            if onnx_out.size == om_out.size:
                om_out = om_out.reshape(onnx_out.shape)
                print(f"  Reshaped OM output to {onnx_out.shape}")
            else:
                results.append(
                    {
                        "index": i,
                        "shape_match": False,
                        "error": f"Shape mismatch: {onnx_out.shape} vs {om_out.shape}",
                    }
                )
                continue

        # Calculate metrics
        diff = np.abs(onnx_out - om_out)
        rel_diff = diff / (np.abs(onnx_out) + 1e-8)

        # Check if values are close
        is_close = np.allclose(onnx_out, om_out, atol=atol, rtol=rtol)

        # Cosine similarity
        onnx_flat = onnx_out.flatten()
        om_flat = om_out.flatten()
        cos_sim = np.dot(onnx_flat, om_flat) / (
            np.linalg.norm(onnx_flat) * np.linalg.norm(om_flat) + 1e-8
        )

        result = {
            "index": i,
            "shape_match": True,
            "shape": onnx_out.shape,
            "is_close": is_close,
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "max_rel_diff": float(rel_diff.max()),
            "mean_rel_diff": float(rel_diff.mean()),
            "cosine_similarity": float(cos_sim),
            "onnx_min": float(onnx_out.min()),
            "onnx_max": float(onnx_out.max()),
            "om_min": float(om_out.min()),
            "om_max": float(om_out.max()),
        }

        # Count outliers (values differing by more than threshold)
        outlier_threshold = atol + rtol * np.abs(onnx_out)
        outlier_mask = diff > outlier_threshold
        result["outlier_count"] = int(outlier_mask.sum())
        result["outlier_ratio"] = float(outlier_mask.sum() / onnx_out.size)

        results.append(result)

    return results


def print_comparison_report(results, atol, rtol):
    """Print detailed comparison report."""
    print("\n" + "=" * 70)
    print("SUBGRAPH PRECISION REFERENCE REPORT")
    print("=" * 70)
    print(f"Tolerances: atol={atol}, rtol={rtol}")
    print("Note: this report is only a subgraph-level numeric reference.")
    print("Final business accuracy must be decided in end-to-end validation.")
    print("-" * 70)

    all_passed = True

    for r in results:
        status = "WITHIN_TOLERANCE" if r.get("is_close", False) else "EXCEEDS_TOLERANCE"
        if not r.get("is_close", False):
            all_passed = False

        print(f"\n[Output {r['index']}] {status}")

        if not r.get("shape_match", True):
            print(f"  ERROR: {r.get('error', 'Shape mismatch')}")
            continue

        print(f"  Shape:              {r['shape']}")
        print(f"  Close enough:       {r['is_close']}")
        print(f"  Cosine similarity:  {r['cosine_similarity']:.6f}")
        print(f"  Max absolute diff:  {r['max_abs_diff']:.6e}")
        print(f"  Mean absolute diff: {r['mean_abs_diff']:.6e}")
        print(f"  Max relative diff:  {r['max_rel_diff']:.6e}")
        print(f"  Mean relative diff: {r['mean_rel_diff']:.6e}")
        print(
            f"  Outlier ratio:      {r['outlier_ratio'] * 100:.2f}% ({r['outlier_count']}/{np.prod(r['shape'])})"
        )
        print(f"  ONNX range:         [{r['onnx_min']:.4f}, {r['onnx_max']:.4f}]")
        print(f"  OM range:           [{r['om_min']:.4f}, {r['om_max']:.4f}]")

    print("\n" + "=" * 70)
    if all_passed:
        print("OVERALL: all compared outputs are within tolerance")
    else:
        print("OVERALL: some compared outputs exceed tolerance")
    print("Interpretation: use this result to screen obvious conversion issues first.")
    print("Do not treat it as the final task-level accuracy conclusion.")
    print("=" * 70 + "\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Compare ONNX and OM model precision")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--om", required=True, help="Path to OM model")
    parser.add_argument("--input", required=True, help="Path to input data file (.npy)")
    parser.add_argument("--device", type=int, default=None, help="NPU device ID")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance")
    parser.add_argument(
        "--mode",
        default="static",
        choices=["static", "dymbatch", "dymhw", "dymdims", "dymshape"],
        help="OM inference mode",
    )
    parser.add_argument(
        "--custom-sizes",
        type=int,
        default=100000,
        help="ais_bench custom_sizes value for dynamic inference modes",
    )
    parser.add_argument("--output", help="Save comparison results to JSON file")
    parser.add_argument("--save-diff", help="Save difference arrays to .npy files")

    args = parser.parse_args()

    # Check files exist
    if not os.path.exists(args.onnx):
        print(f"Error: ONNX model not found: {args.onnx}")
        sys.exit(1)
    if not os.path.exists(args.om):
        print(f"Error: OM model not found: {args.om}")
        sys.exit(1)
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    device_id = args.device if args.device is not None else get_device_id()

    # Load input
    print(f"Loading input: {args.input}")
    inputs = load_input(args.input)
    print(f"Input shape: {inputs.shape}")

    # Run ONNX inference
    print(f"\nRunning ONNX inference on CPU: {args.onnx}")
    onnx_outputs, onnx_session = run_onnx_inference(args.onnx, inputs)
    print(f"ONNX outputs: {len(onnx_outputs)} tensors")
    for i, out in enumerate(onnx_outputs):
        print(f"  [{i}] shape={out.shape}, dtype={safe_dtype_text(out)}")

    # Run OM inference
    print(
        f"\nRunning OM inference on NPU (device {device_id}, mode={args.mode}, "
        f"custom_sizes={args.custom_sizes}): {args.om}"
    )
    om_outputs, om_session = run_om_inference(
        args.om,
        inputs,
        device_id=device_id,
        mode=args.mode,
        custom_sizes=args.custom_sizes,
    )
    print(f"OM outputs: {len(om_outputs)} tensors")
    for i, out in enumerate(om_outputs):
        print(f"  [{i}] shape={out.shape}, dtype={safe_dtype_text(out)}")

    # Compare outputs
    results = compare_outputs(onnx_outputs, om_outputs, atol=args.atol, rtol=args.rtol)

    # Print report
    all_passed = print_comparison_report(results, args.atol, args.rtol)

    # Save results if requested
    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(
                {
                    "tolerances": {"atol": args.atol, "rtol": args.rtol},
                    "passed": all_passed,
                    "outputs": results,
                },
                f,
                indent=2,
            )
        print(f"Results saved to: {args.output}")

    # Save diff arrays if requested
    if args.save_diff:
        os.makedirs(args.save_diff, exist_ok=True)
        for i, (onnx_out, om_out) in enumerate(zip(onnx_outputs, om_outputs)):
            diff = np.abs(onnx_out - om_out)
            np.save(os.path.join(args.save_diff, f"diff_{i}.npy"), diff)
            np.save(os.path.join(args.save_diff, f"onnx_out_{i}.npy"), onnx_out)
            np.save(os.path.join(args.save_diff, f"om_out_{i}.npy"), om_out)
        print(f"Difference arrays saved to: {args.save_diff}")

    # Cleanup
    om_session.free_resource()

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
