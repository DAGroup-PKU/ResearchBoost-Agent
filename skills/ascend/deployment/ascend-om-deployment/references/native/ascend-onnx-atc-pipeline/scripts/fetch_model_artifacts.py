#!/usr/bin/env python3
"""Fetch model artifacts from local paths, URLs, or ModelScope."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch model artifacts from local paths, URLs, or ModelScope."
    )
    parser.add_argument("--target-dir", required=True, help="Directory to store downloaded files")
    parser.add_argument(
        "--source-type",
        required=True,
        choices=["local-dir", "local-file", "url", "modelscope-repo", "modelscope-file"],
        help="Artifact source type",
    )
    parser.add_argument("--source", required=True, help="Source path, URL, or ModelScope model id")
    parser.add_argument("--revision", help="Optional ModelScope revision")
    parser.add_argument("--repo-file-path", help="Single file path inside the ModelScope repo")
    parser.add_argument(
        "--output-name",
        help="Optional output file name for url/local-file sources",
    )
    parser.add_argument(
        "--local-policy",
        choices=["reuse", "copy"],
        default="reuse",
        help="How to handle local-dir/local-file sources. Defaults to reuse.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Optional allow pattern, can be repeated",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[],
        help="Optional ignore pattern, can be repeated",
    )
    return parser.parse_args()


def ensure_modelscope_available() -> None:
    try:
        import modelscope  # noqa: F401
    except ImportError:
        print("ModelScope is not installed. Installing modelscope via pip...", file=sys.stderr)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "modelscope"]
        )
        import modelscope  # noqa: F401


def reuse_local_path(source: Path) -> str:
    return str(source.resolve())


def copy_local_dir(source: Path, target_dir: Path) -> str:
    destination = target_dir / source.name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return str(destination)


def copy_local_file(source: Path, target_dir: Path, output_name: str | None) -> str:
    destination = target_dir / (output_name or source.name)
    shutil.copy2(source, destination)
    return str(destination)


def download_url(source: str, target_dir: Path, output_name: str | None) -> str:
    if output_name:
        destination = target_dir / output_name
    else:
        destination = target_dir / Path(urllib.parse.urlparse(source).path).name
        if not destination.name:
            raise ValueError("--output-name is required when URL path has no file name")
    urllib.request.urlretrieve(source, destination)
    return str(destination)


def fetch_modelscope_repo(
    model_id: str,
    target_dir: Path,
    revision: str | None,
    allow_patterns: list[str],
    ignore_patterns: list[str],
) -> str:
    ensure_modelscope_available()
    from modelscope.hub.snapshot_download import snapshot_download

    return str(
        snapshot_download(
            model_id=model_id,
            local_dir=str(target_dir),
            revision=revision,
            allow_patterns=allow_patterns or None,
            ignore_patterns=ignore_patterns or None,
        )
    )


def fetch_modelscope_file(
    model_id: str,
    target_dir: Path,
    file_path: str,
    revision: str | None,
) -> str:
    ensure_modelscope_available()
    from modelscope.hub.file_download import model_file_download

    return str(
        model_file_download(
            model_id=model_id,
            file_path=file_path,
            local_dir=str(target_dir),
            revision=revision,
        )
    )


def main() -> int:
    args = parse_args()
    target_dir = Path(args.target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.source).expanduser()

    if args.source_type == "local-dir":
        if not source.is_dir():
            print(f"Local directory not found: {source}", file=sys.stderr)
            return 1
        if args.local_policy == "reuse":
            output = reuse_local_path(source)
        else:
            output = copy_local_dir(source.resolve(), target_dir)
    elif args.source_type == "local-file":
        if not source.is_file():
            print(f"Local file not found: {source}", file=sys.stderr)
            return 1
        if args.local_policy == "reuse" and not args.output_name:
            output = reuse_local_path(source)
        else:
            output = copy_local_file(source.resolve(), target_dir, args.output_name)
    elif args.source_type == "url":
        output = download_url(args.source, target_dir, args.output_name)
    elif args.source_type == "modelscope-repo":
        output = fetch_modelscope_repo(
            model_id=args.source,
            target_dir=target_dir,
            revision=args.revision,
            allow_patterns=args.allow_pattern,
            ignore_patterns=args.ignore_pattern,
        )
    else:
        if not args.repo_file_path:
            print("--repo-file-path is required for modelscope-file", file=sys.stderr)
            return 1
        output = fetch_modelscope_file(
            model_id=args.source,
            target_dir=target_dir,
            file_path=args.repo_file_path,
            revision=args.revision,
        )

    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
