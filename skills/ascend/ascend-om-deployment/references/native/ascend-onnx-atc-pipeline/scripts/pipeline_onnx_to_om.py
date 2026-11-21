#!/usr/bin/env python3
"""Run ATC with recorded fallback order over optimized ONNX candidates."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_PRIORITY = ["auto_optimizer", "onnxslim", "original"]


def load_summary(artifacts_dir: Path) -> dict:
    summary_path = artifacts_dir / "optimization_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def candidate_path(artifacts_dir: Path, summary: dict, label: str) -> Path:
    summary_candidates = summary.get("candidates", {})
    if label in summary_candidates and summary_candidates[label].get("path"):
        return Path(summary_candidates[label]["path"]).resolve()
    return (artifacts_dir / f"{label}.onnx").resolve()


def candidate_allowed(summary: dict, label: str) -> tuple[bool, str | None]:
    record = summary.get("candidates", {}).get(label)
    if not record:
        return True, None
    if record.get("success", False):
        return True, None
    reason = record.get("reason") or record.get("error") or "candidate_marked_invalid"
    return False, reason


def build_command(args: argparse.Namespace, model_path: Path, output_prefix: Path) -> list[str]:
    command = [
        "atc",
        f"--model={model_path}",
        f"--framework={args.framework}",
        f"--output={output_prefix}",
        f"--soc_version={args.soc_version}",
    ]
    optional_map = {
        "input_format": args.input_format,
        "input_shape": args.input_shape,
        "dynamic_batch_size": args.dynamic_batch_size,
        "dynamic_image_size": args.dynamic_image_size,
        "dynamic_dims": args.dynamic_dims,
        "precision_mode": args.precision_mode,
        "output_type": args.output_type,
        "log": args.log,
        "auto_tune_mode": args.auto_tune_mode,
        "fusion_switch_file": args.fusion_switch_file,
    }
    for key, value in optional_map.items():
        if value:
            command.append(f"--{key}={value}")
    for extra in args.atc_extra:
        command.append(extra)
    return command


def validate_shape_strategy(args: argparse.Namespace) -> None:
    strategy_flags = {
        "dynamic_batch_size": args.dynamic_batch_size,
        "dynamic_image_size": args.dynamic_image_size,
        "dynamic_dims": args.dynamic_dims,
    }
    enabled = [name for name, value in strategy_flags.items() if value]
    if len(enabled) > 1:
        raise ValueError(
            "Conflicting ATC shape strategies: "
            + ", ".join(enabled)
            + ". Choose only one primary dynamic strategy per ATC attempt."
        )


def run_attempt(command: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"COMMAND: {' '.join(shlex.quote(part) for part in command)}",
                "",
                "[STDOUT]",
                result.stdout,
                "",
                "[STDERR]",
                result.stderr,
                "",
                f"RETURN_CODE: {result.returncode}",
            ]
        ),
        encoding="utf-8",
    )
    return result


def resolve_om_path(output_prefix: Path) -> Path | None:
    exact_path = output_prefix.with_suffix(".om")
    if exact_path.exists():
        return exact_path.resolve()

    candidates = sorted(output_prefix.parent.glob(f"{output_prefix.name}*.om"))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0].resolve()

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0].resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Try ATC over auto_optimizer / onnxslim / original candidates."
    )
    parser.add_argument("--artifacts-dir", required=True, help="Directory containing ONNX artifacts")
    parser.add_argument("--output-dir", required=True, help="Directory for ATC outputs and logs")
    parser.add_argument("--base-name", required=True, help="Base name for generated OM files")
    parser.add_argument(
        "--soc-version",
        required=True,
        help="ATC soc_version, e.g. Ascend910A or Ascend910B4",
    )
    parser.add_argument("--framework", default="5", help="ATC framework value, default 5 for ONNX")
    parser.add_argument("--input-format", help="ATC input_format")
    parser.add_argument(
        "--input-shape",
        help="ATC input_shape. Dynamic routes may also require it; if it contains -1 and no dynamic_* option is set, that means dymshape.",
    )
    parser.add_argument("--dynamic-batch-size", help="ATC dynamic_batch_size")
    parser.add_argument("--dynamic-image-size", help="ATC dynamic_image_size")
    parser.add_argument("--dynamic-dims", help="ATC dynamic_dims")
    parser.add_argument("--precision-mode", help="ATC precision_mode")
    parser.add_argument("--output-type", help="ATC output_type")
    parser.add_argument("--log", help="ATC log level")
    parser.add_argument("--auto-tune-mode", help="ATC auto_tune_mode")
    parser.add_argument("--fusion-switch-file", help="ATC fusion_switch_file")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Optional explicit candidate order label, can be repeated",
    )
    parser.add_argument(
        "--atc-extra",
        action="append",
        default=[],
        help="Additional raw ATC argument such as --enable_single_stream=true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_shape_strategy(args)
    artifacts_dir = Path(args.artifacts_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    summary = load_summary(artifacts_dir)

    candidate_order = args.candidate or summary.get("candidate_priority") or DEFAULT_PRIORITY
    attempts = []

    for label in candidate_order:
        allowed, blocked_reason = candidate_allowed(summary, label)
        if not allowed:
            attempts.append(
                {
                    "candidate": label,
                    "model_path": None,
                    "attempted": False,
                    "success": False,
                    "reason": blocked_reason,
                }
            )
            continue

        model_path = candidate_path(artifacts_dir, summary, label)
        if not model_path.exists():
            attempts.append(
                {
                    "candidate": label,
                    "model_path": str(model_path),
                    "attempted": False,
                    "success": False,
                    "reason": "model_missing",
                }
            )
            continue

        output_prefix = output_dir / f"{args.base_name}_{label}"
        command = build_command(args, model_path, output_prefix)
        result = run_attempt(command, logs_dir / f"{label}.log")

        resolved_om_path = resolve_om_path(output_prefix)
        record = {
            "candidate": label,
            "model_path": str(model_path),
            "command": command,
            "attempted": True,
            "returncode": result.returncode,
            "log_path": str((logs_dir / f"{label}.log").resolve()),
            "om_path": str(resolved_om_path) if resolved_om_path else None,
            "success": result.returncode == 0 and resolved_om_path is not None,
        }
        attempts.append(record)

        if record["success"]:
            payload = {
                "selected_candidate": label,
                "selected_model": str(model_path),
                "selected_om": str(resolved_om_path),
                "attempts": attempts,
            }
            (output_dir / "atc_selection.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

    payload = {
        "selected_candidate": None,
        "selected_model": None,
        "selected_om": None,
        "attempts": attempts,
    }
    (output_dir / "atc_selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1


if __name__ == "__main__":
    sys.exit(main())
