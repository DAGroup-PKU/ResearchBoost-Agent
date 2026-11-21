#!/usr/bin/env python3
"""
Generic PyTorch to ONNX Export Script

Export any PyTorch model (.pt/.pth) to ONNX format for subsequent ATC conversion.
Supports standard torch.save models, TorchScript models, and torchvision pretrained models.

Usage:
    # Export a saved PyTorch model
    python3 export_onnx.py --pt_model model.pt --output model.onnx --input_shape 1,3,224,224

    # Specify opset version (default: 11, safest for CANN compatibility)
    python3 export_onnx.py --pt_model model.pt --output model.onnx --input_shape 1,3,224,224 --opset 13

    # With dynamic axes (JSON string)
    python3 export_onnx.py --pt_model model.pt --output model.onnx --input_shape 1,3,224,224 \
        --dynamic_axes '{"input": {0: "batch"}, "output": {0: "batch"}}'

    # Export a torchvision pretrained model by name
    python3 export_onnx.py --torchvision resnet50 --output resnet50.onnx --input_shape 1,3,224,224

    # Custom input/output names
    python3 export_onnx.py --pt_model model.pt --output model.onnx --input_shape 1,3,640,640 \
        --input_names images --output_names output0
"""

import argparse
import inspect
import json
import os
import sys


def parse_input_shape(shape_str):
    """Parse comma-separated shape string into a tuple of ints.

    Args:
        shape_str: e.g. "1,3,224,224"

    Returns:
        Tuple of ints, e.g. (1, 3, 224, 224)
    """
    try:
        return tuple(int(x.strip()) for x in shape_str.split(","))
    except ValueError:
        print(f"Error: Invalid input_shape format: '{shape_str}'")
        print("Expected format: comma-separated integers, e.g. '1,3,224,224'")
        sys.exit(1)


def parse_dynamic_axes(axes_str):
    """Parse dynamic axes from JSON string.

    Args:
        axes_str: JSON string, e.g. '{"input": {0: "batch"}, "output": {0: "batch"}}'

    Returns:
        Dict for torch.onnx.export dynamic_axes parameter
    """
    if not axes_str:
        return None
    try:
        axes = json.loads(axes_str)
        # JSON keys are always strings, but torch.onnx.export expects int keys
        # for the inner dict. Convert string digit keys to int.
        parsed = {}
        for name, dim_map in axes.items():
            parsed[name] = {int(k): v for k, v in dim_map.items()}
        return parsed
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        print(f"Error: Invalid dynamic_axes JSON: {e}")
        print('Expected format: \'{"input": {"0": "batch"}, "output": {"0": "batch"}}\'')
        sys.exit(1)


def load_pytorch_model(model_path):
    """Load a PyTorch model from file with robust error handling.

    Supports:
    - torch.save(model, path) — full model
    - torch.save(model.state_dict(), path) — state dict (will fail gracefully)
    - TorchScript models (torch.jit.save)

    Args:
        model_path: Path to .pt or .pth file

    Returns:
        Loaded model in eval mode
    """
    import torch

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    # Try loading as TorchScript first
    try:
        model = torch.jit.load(model_path, map_location="cpu")
        model.eval()
        print(f"Loaded TorchScript model: {model_path}")
        return model
    except Exception:
        pass

    # Try loading as full model
    try:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(model, dict):
            # This is likely a state_dict or checkpoint dict
            if "model" in model:
                # Common checkpoint format: {"model": state_dict, "optimizer": ...}
                print(
                    "Warning: Loaded a checkpoint dict with 'model' key. "
                    "Attempting to extract the model."
                )
                model = model["model"]
            elif "state_dict" in model:
                print(
                    "Error: Loaded a state_dict checkpoint. Cannot export without "
                    "the model architecture."
                )
                print(
                    "Hint: Load the model architecture first, then load state_dict "
                    "into it, and save the full model:"
                )
                print("  model = YourModelClass()")
                print("  model.load_state_dict(torch.load('model.pt')['state_dict'])")
                print("  torch.save(model, 'full_model.pt')")
                sys.exit(1)
            else:
                print(
                    "Error: Loaded a dict (likely a state_dict). Cannot export "
                    "without the model architecture."
                )
                print("Hint: Save the full model with: torch.save(model, 'model.pt')")
                sys.exit(1)

        if not hasattr(model, "forward"):
            print(f"Error: Loaded object is not a PyTorch model (type: {type(model).__name__})")
            sys.exit(1)

        model.eval()
        print(f"Loaded PyTorch model: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def load_torchvision_model(model_name):
    """Load a torchvision pretrained model by name.

    Args:
        model_name: e.g. "resnet50", "mobilenet_v2", "efficientnet_b0"

    Returns:
        Loaded model in eval mode
    """
    try:
        import torchvision.models as models
    except ImportError:
        print("Error: torchvision not installed. Install with: pip install torchvision")
        sys.exit(1)

    # Get model constructor
    model_fn = getattr(models, model_name, None)
    if model_fn is None:
        available = [n for n in dir(models) if not n.startswith("_") and callable(getattr(models, n))]
        print(f"Error: Unknown torchvision model: '{model_name}'")
        print(f"Available models (partial): {', '.join(sorted(available)[:20])}...")
        sys.exit(1)

    try:
        # Try loading with default weights
        weights_enum_name = model_name.replace("_", " ").title().replace(" ", "_") + "_Weights"
        weights_enum = getattr(models, weights_enum_name, None)
        if weights_enum and hasattr(weights_enum, "DEFAULT"):
            model = model_fn(weights=weights_enum.DEFAULT)
            print(f"Loaded torchvision model '{model_name}' with default pretrained weights")
        else:
            model = model_fn(pretrained=False)
            print(f"Loaded torchvision model '{model_name}' (no pretrained weights)")
    except Exception:
        model = model_fn(pretrained=False)
        print(f"Loaded torchvision model '{model_name}' (no pretrained weights)")

    model.eval()
    return model


def export_to_onnx(
    model,
    input_shape,
    output_path,
    opset_version=11,
    input_names=None,
    output_names=None,
    dynamic_axes=None,
):
    """Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model (nn.Module or ScriptModule)
        input_shape: Tuple of ints, e.g. (1, 3, 224, 224)
        output_path: Output .onnx file path
        opset_version: ONNX opset version (default: 11)
        input_names: List of input tensor names
        output_names: List of output tensor names
        dynamic_axes: Dict for dynamic axes specification
    """
    import torch

    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    print(f"Dummy input shape: {list(input_shape)}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Export
    print(f"Exporting to ONNX (opset={opset_version})...")
    try:
        export_kwargs = dict(
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            # Newer PyTorch versions may default to the torch.export/dynamo ONNX
            # exporter, which adds an onnxscript dependency and changes export
            # behavior. Keep the legacy exporter as the default ATC-friendly path.
            export_kwargs["dynamo"] = False

        torch.onnx.export(model, dummy_input, output_path, **export_kwargs)
    except Exception as e:
        print(f"Error during ONNX export: {e}")
        print("\nTroubleshooting tips:")
        print(f"  - Try a different opset version (current: {opset_version})")
        print("  - Ensure the model is compatible with torch.onnx.export")
        print("  - Check if the model uses unsupported dynamic operations")
        sys.exit(1)

    # Verify output
    file_size = os.path.getsize(output_path)
    print("\nExport successful!")
    print(f"  Output: {output_path}")
    print(f"  Size:   {file_size / 1024 / 1024:.2f} MB")

    # Validate ONNX model
    try:
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("  ONNX validation: PASSED")

        # Print model I/O info
        graph = onnx_model.graph
        print("\n  Inputs:")
        for inp in graph.input:
            shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
            print(f"    {inp.name}: {shape}")
        print("  Outputs:")
        for out in graph.output:
            shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
            print(f"    {out.name}: {shape}")
    except ImportError:
        print("  ONNX validation: skipped (onnx package not installed)")
    except Exception as e:
        print(f"  ONNX validation: WARNING - {e}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch models to ONNX format for Ascend ATC conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export a saved model
  python3 export_onnx.py --pt_model model.pt --output model.onnx --input_shape 1,3,224,224

  # Export with specific opset for CANN compatibility
  python3 export_onnx.py --pt_model model.pt --output model.onnx --input_shape 1,3,640,640 --opset 11

  # Export torchvision model
  python3 export_onnx.py --torchvision resnet50 --output resnet50.onnx --input_shape 1,3,224,224

  # With dynamic batch size
  python3 export_onnx.py --pt_model model.pt --output model.onnx --input_shape 1,3,224,224 \\
      --dynamic_axes '{"input": {"0": "batch"}, "output": {"0": "batch"}}'

Notes:
  - 默认优先选择与目标 CANN 兼容的较稳妥 opset。
  - 不要只凭经验固定 opset，仍需结合真实环境验证。
  - 导出后应继续检查 ONNX 输入输出契约，再进入后续优化与 ATC。
""",
    )

    # Model source (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--pt_model", help="Path to PyTorch model file (.pt/.pth)")
    model_group.add_argument(
        "--torchvision",
        help="Name of torchvision model (e.g. resnet50, mobilenet_v2)",
    )

    # Required parameters
    parser.add_argument("--output", required=True, help="Output ONNX file path")
    parser.add_argument(
        "--input_shape",
        required=True,
        help="Input tensor shape, comma-separated (e.g. '1,3,224,224')",
    )

    # Optional parameters
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version (default: 11, safest for CANN)",
    )
    parser.add_argument(
        "--dynamic_axes",
        help='Dynamic axes as JSON string (e.g. \'{"input": {"0": "batch"}}\')',
    )
    parser.add_argument(
        "--input_names",
        nargs="+",
        default=None,
        help="Input tensor names (default: ['input'])",
    )
    parser.add_argument(
        "--output_names",
        nargs="+",
        default=None,
        help="Output tensor names (default: ['output'])",
    )

    args = parser.parse_args()

    # Parse parameters
    input_shape = parse_input_shape(args.input_shape)
    dynamic_axes = parse_dynamic_axes(args.dynamic_axes)

    # Load model
    if args.pt_model:
        model = load_pytorch_model(args.pt_model)
    else:
        model = load_torchvision_model(args.torchvision)

    # Export
    export_to_onnx(
        model=model,
        input_shape=input_shape,
        output_path=args.output,
        opset_version=args.opset,
        input_names=args.input_names,
        output_names=args.output_names,
        dynamic_axes=dynamic_axes,
    )

    # Print next steps
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("\n--- Next Steps ---")
    print("1. Inspect the ONNX model:")
    print(
        f"   python3 -u {os.path.join(script_dir, 'inspect_onnx_io_shapes.py')} "
        f"--model {args.output}"
    )
    print("2. Prepare optimization candidates:")
    print(
        f"   python3 -u {os.path.join(script_dir, 'optimize_onnx.py')} "
        f"--model {args.output} --output-dir {os.path.splitext(args.output)[0]}_opt"
    )
    print("3. Convert to OM with recorded fallback order:")
    print(
        f"   python3 -u {os.path.join(script_dir, 'pipeline_onnx_to_om.py')} "
        f"--artifacts-dir {os.path.splitext(args.output)[0]}_opt "
        f"--output-dir {os.path.splitext(args.output)[0]}_om "
        f"--base-name {os.path.basename(os.path.splitext(args.output)[0])} "
        "--soc-version <soc_version> "
        f"--input-shape input:{args.input_shape}"
    )


if __name__ == "__main__":
    main()
