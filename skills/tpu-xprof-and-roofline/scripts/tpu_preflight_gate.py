#!/usr/bin/env python3
"""Zero-TPU-initialization preflight validator for Cloud TPU VMs.

What problem this solves:
-------------------------
On multi-host or multi-tenant Cloud TPU slices (v5p, v6e), launching a training
or evaluation job without preflight validation frequently fails or hangs due to:
  1. Another process holding `/dev/vfio/*` or `/dev/accel*` (causing PJRT init
     to block or fail with `RESOURCE_EXHAUSTED` / device busy).
  2. A leftover `/tmp/libtpu_lockfile` from a crashed run (or conversely, a live
     process holding an open file descriptor to `/tmp/libtpu_lockfile` whose
     inode must be verified before assuming the lock is stale).
  3. Typoed or version-skewed flags in `LIBTPU_INIT_ARGS` (which cause `libtpu`
     to abort during C++ initialization across all workers after waiting on
     coordinator barriers). Scanning `libtpu.so` via `mmap` catches invalid
     flags in <100 ms without opening the TPU.
  4. Host resource exhaustion: low `MemAvailable` in `/proc/meminfo`, full
     `/dev/shm` or `/` root filesystem, high memory/IO Pressure Stall
     Information (`/proc/pressure/{memory,io}`), non-zero core dump limits
     (`ulimit -c != 0` filling the root disk on abort), or heavy foreign CPU
     load from zombie/other-tenant processes starving host-side XLA tracing.

When to use it:
---------------
Run on every TPU VM worker *before* launching `torch_xla` or `jax` workloads.
This script uses ONLY the Python standard library and never initializes the TPU.

Example CLI invocations:
------------------------
  # Full preflight check with default v6e resource thresholds:
  python3 scripts/tpu_preflight_gate.py --profile v6e

  # Validate LIBTPU_INIT_ARGS against installed libtpu.so and rename stale lockfile:
  python3 scripts/tpu_preflight_gate.py --check-flags "$LIBTPU_INIT_ARGS" --rename-stale-lock

  # Statically audit .sh / .py launch scripts offline without executing them:
  python3 scripts/tpu_preflight_gate.py --audit-script "scripts/launch_*.sh" --strict-audit

  # Run built-in CPU/mock self-tests (works on macOS and Linux without a TPU):
  python3 scripts/tpu_preflight_gate.py --self-test
"""
from __future__ import annotations

import argparse
import ast
from collections.abc import Sequence
from datetime import datetime, timezone
import glob
from importlib import metadata
import json
import mmap
import os
from pathlib import Path
import pwd
import re
import resource
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


DEFAULT_LOCKFILE = "/tmp/libtpu_lockfile"

RESOURCE_PROFILES = {
    "v6e": {
        "min_mem_available_gib": 64.0,
        "min_shm_free_gib": 5.0,
        "min_root_free_gib": 5.0,
        "psi_memory_avg10_limit": 0.20,
        "psi_io_avg10_limit": 0.20,
        "foreign_cpu_cores_limit": 4.0,
    },
    "v5p": {
        "min_mem_available_gib": 200.0,
        "min_shm_free_gib": 5.0,
        "min_root_free_gib": 5.0,
        "psi_memory_avg10_limit": 0.20,
        "psi_io_avg10_limit": 0.20,
        "foreign_cpu_cores_limit": 4.0,
    },
}

DEVICE_PATH_RE = re.compile(
    r"^/dev/(?:vfio(?:/|$)|accel(?:/|[0-9]|$)|tpu(?:/|[0-9]|$))"
)
IDENT_TOKEN_RE = re.compile(rb"[A-Za-z0-9_]+")


def is_tpu_device_path(path_str: str) -> bool:
    """Return True if path_str points to a VFIO or accel/TPU device node."""
    cleaned = path_str.removesuffix(" (deleted)")
    return bool(DEVICE_PATH_RE.match(cleaned))


def flag_name_of(flag: str) -> str:
    """Extract the bare flag identifier from `--flag_name=value` or `--flag_name`."""
    return flag.lstrip("-").split("=", 1)[0].strip()


def _sudo_available() -> bool:
    try:
        return (
            subprocess.run(
                ["sudo", "-n", "true"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def _fuser_held(path: str, allow_sudo: bool = True) -> bool | None:
    """Return True if `fuser` reports a holder for `path`, False if idle, None if unknown."""
    cmd = ["fuser", "-s", path]
    if os.geteuid() != 0 and allow_sudo and _sudo_available():
        cmd = ["sudo", "-n", *cmd]
    try:
        code = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).returncode
    except (OSError, subprocess.TimeoutExpired):
        return None
    return {0: True, 1: False}.get(code)


def _read_cmdline(proc_dir: Path) -> str:
    try:
        raw = (proc_dir / "cmdline").read_bytes()
        parts = [p.decode(errors="replace") for p in raw.split(b"\0") if p]
        return " ".join(parts)[:400]
    except OSError:
        return ""


def _uid_name(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def check_device_holders(
    proc_root: str = "/proc",
    dev_root: str = "/dev",
    use_fuser: bool = True,
    exclude_pids: set[int] | None = None,
) -> dict:
    """Inspect `/dev/vfio/*` and `/dev/accel*` holders via `/proc/<pid>/fd` & `fuser`.

    Never opens any device node. Permission errors when scanning `/proc/<pid>/fd`
    are recorded in `unreadable_pids` so the caller never mistakes missing
    permissions for an idle TPU.
    """
    exclude = set(exclude_pids or ()) | {os.getpid()}
    vfio_nodes = sorted(
        p
        for p in glob.glob(os.path.join(dev_root, "vfio", "*"))
        if os.path.basename(p).isdigit()
    )
    accel_nodes = sorted(
        p
        for p in glob.glob(os.path.join(dev_root, "accel*"))
        if re.fullmatch(r"accel[0-9]+", os.path.basename(p))
    )
    device_nodes = vfio_nodes + accel_nodes

    holders = []
    unreadable_pids = set()
    proc_path = Path(proc_root)

    if proc_path.is_dir():
        for entry in proc_path.iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid in exclude:
                continue
            dev_fds = set()
            dev_maps = set()

            # Check process-level and thread-level fd directories.
            fd_dirs = [entry / "fd"]
            task_dir = entry / "task"
            if task_dir.is_dir():
                try:
                    for tid_entry in task_dir.iterdir():
                        if tid_entry.name.isdigit():
                            fd_dirs.append(tid_entry / "fd")
                except PermissionError:
                    unreadable_pids.add(pid)
                except OSError:
                    pass

            for fd_dir in fd_dirs:
                try:
                    for fd_name in os.listdir(fd_dir):
                        try:
                            target = os.readlink(fd_dir / fd_name)
                            if is_tpu_device_path(target):
                                dev_fds.add(target.removesuffix(" (deleted)"))
                        except (FileNotFoundError, ProcessLookupError):
                            pass
                        except PermissionError:
                            unreadable_pids.add(pid)
                except PermissionError:
                    unreadable_pids.add(pid)
                except (FileNotFoundError, ProcessLookupError, OSError):
                    pass

            try:
                maps_text = (entry / "maps").read_text(errors="replace")
                for line in maps_text.splitlines():
                    parts = line.split(None, 5)
                    if len(parts) == 6 and is_tpu_device_path(parts[5].strip()):
                        dev_maps.add(parts[5].strip().removesuffix(" (deleted)"))
            except PermissionError:
                unreadable_pids.add(pid)
            except (FileNotFoundError, ProcessLookupError, OSError):
                pass

            if dev_fds or dev_maps:
                try:
                    uid = entry.stat().st_uid
                except OSError:
                    uid = -1
                holders.append(
                    {
                        "pid": pid,
                        "uid": uid,
                        "user": _uid_name(uid) if uid >= 0 else "unknown",
                        "device_fds": sorted(dev_fds),
                        "device_maps": sorted(dev_maps),
                        "cmdline": _read_cmdline(entry),
                    }
                )

    fuser_busy_nodes = []
    fuser_checked = False
    if use_fuser and device_nodes:
        fuser_results = {node: _fuser_held(node) for node in device_nodes}
        if any(v is not None for v in fuser_results.values()):
            fuser_checked = True
            fuser_busy_nodes = sorted(
                node for node, is_busy in fuser_results.items() if is_busy
            )

    if holders or fuser_busy_nodes:
        idle = False
    elif unreadable_pids and not fuser_checked:
        idle = None
    else:
        idle = True

    return {
        "vfio_nodes": vfio_nodes,
        "accel_nodes": accel_nodes,
        "holders": sorted(holders, key=lambda r: r["pid"]),
        "unreadable_pids": sorted(unreadable_pids),
        "fuser_checked": fuser_checked,
        "fuser_busy_nodes": fuser_busy_nodes,
        "idle": idle,
    }


def check_lockfile(
    lock_path: str = DEFAULT_LOCKFILE,
    proc_root: str = "/proc",
    apply_stale_rename: bool = False,
    exclude_pids: set[int] | None = None,
) -> dict:
    """Verify `/tmp/libtpu_lockfile` holder PIDs and open FD inode consistency.

    Compares `stat(/proc/<pid>/fd/<n>)` `(st_dev, st_ino)` against
    `stat(/tmp/libtpu_lockfile)` in addition to resolved path matching.
    A lockfile is classified as `stale` only when:
      - It is a regular file (not a symlink) of size 0 bytes,
      - Zero processes hold an open FD matching its inode or resolved path, and
      - No `/proc/<pid>/fd` directories failed with `PermissionError`.
    """
    exclude = set(exclude_pids or ()) | {os.getpid()}
    exists = os.path.lexists(lock_path)
    if not exists:
        return {
            "lock_path": lock_path,
            "present": False,
            "held": False,
            "stale": False,
            "inode_consistent": True,
            "holders": [],
            "deleted_inode_holders": [],
            "unreadable_pids": [],
        }

    lstat_info = os.lstat(lock_path)
    is_regular = os.path.isfile(lock_path) and not os.path.islink(lock_path)
    real_lock = os.path.realpath(lock_path)
    lock_dev_ino = (lstat_info.st_dev, lstat_info.st_ino)

    holders = []
    deleted_inode_holders = []
    unreadable_pids = set()
    proc_path = Path(proc_root)

    if proc_path.is_dir():
        for entry in proc_path.iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid in exclude:
                continue
            fd_dir = entry / "fd"
            try:
                fd_names = os.listdir(fd_dir)
            except PermissionError:
                unreadable_pids.add(pid)
                continue
            except OSError:
                continue

            for fd_name in fd_names:
                fd_entry = fd_dir / fd_name
                try:
                    link_target = os.readlink(fd_entry)
                    fd_stat = os.stat(fd_entry)
                except PermissionError:
                    unreadable_pids.add(pid)
                    continue
                except OSError:
                    continue

                fd_dev_ino = (fd_stat.st_dev, fd_stat.st_ino)
                is_deleted_link = link_target.endswith(" (deleted)")
                clean_target = link_target.removesuffix(" (deleted)")
                path_matches = (
                    clean_target == lock_path
                    or os.path.realpath(clean_target) == real_lock
                )
                inode_matches = fd_dev_ino == lock_dev_ino

                if inode_matches:
                    holders.append(
                        {
                            "pid": pid,
                            "fd": fd_name,
                            "link": link_target,
                            "inode": f"{fd_stat.st_dev}:{fd_stat.st_ino}",
                            "inode_matches_lockfile": True,
                        }
                    )
                    break
                if path_matches and is_deleted_link and not inode_matches:
                    deleted_inode_holders.append(
                        {
                            "pid": pid,
                            "fd": fd_name,
                            "link": link_target,
                            "fd_inode": f"{fd_stat.st_dev}:{fd_stat.st_ino}",
                            "lockfile_inode": f"{lstat_info.st_dev}:{lstat_info.st_ino}",
                        }
                    )

    held = bool(holders)
    stale = (
        is_regular
        and lstat_info.st_size == 0
        and not held
        and not unreadable_pids
    )
    out = {
        "lock_path": lock_path,
        "present": True,
        "is_regular_file": is_regular,
        "uid": lstat_info.st_uid,
        "owned_by_current_uid": lstat_info.st_uid == os.getuid(),
        "size_bytes": lstat_info.st_size,
        "inode": f"{lstat_info.st_dev}:{lstat_info.st_ino}",
        "held": held,
        "stale": stale,
        "inode_consistent": len(deleted_inode_holders) == 0,
        "holders": holders,
        "deleted_inode_holders": deleted_inode_holders,
        "unreadable_pids": sorted(unreadable_pids),
    }

    if apply_stale_rename and stale:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = f"{lock_path}.stale-uid{lstat_info.st_uid}-{stamp}"
        os.rename(lock_path, dest)
        out["moved_to"] = dest
        out["present_after_rename"] = os.path.lexists(lock_path)

    return out


def find_libtpu_so(python_exe: str | None = None) -> str | None:
    """Locate installed `libtpu.so` via `importlib.metadata` or site-packages glob."""
    for dist_name in ("libtpu", "libtpu-nightly"):
        try:
            files = metadata.files(dist_name) or []
            for f in files:
                if f.name.startswith("libtpu") and f.name.endswith(".so"):
                    located = Path(f.locate())
                    if located.is_file():
                        return str(located.resolve())
        except metadata.PackageNotFoundError:
            pass

    exe = Path(python_exe or sys.executable).resolve()
    venv_root = exe.parent.parent
    candidates = sorted(
        glob.glob(
            str(venv_root / "lib" / "python*" / "site-packages" / "libtpu*" / "**" / "libtpu*.so"),
            recursive=True,
        )
    )
    return candidates[0] if candidates else None


ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
VALID_FLAG_IDENT_RE = re.compile(r"^[A-Za-z0-9_]+$")
SIZE_SUFFIX_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*([KMGTkmgt])(?:i?[Bb])?$")
SHELL_VAR_TOKEN_RE = re.compile(r"^\$(?:\{[A-Za-z_][A-Za-z0-9_]*[^}]*\}|[A-Za-z_][A-Za-z0-9_]*)$")


def validate_libtpu_init_args_syntax(
    raw_args: str | Sequence[str],
    allow_shell_var_tokens: bool = False,
) -> dict[str, Any]:
    """Statically validate `LIBTPU_INIT_ARGS` token hygiene without `libtpu.so`.

    Flags:
      - Tokens starting with `--jax_` or `-jax_` (JAX Python flags inside `LIBTPU_INIT_ARGS`)
      - Bare `KEY=VALUE` environment assignments without a leading `-`
      - Malformed tokens not starting with `-` (or with invalid C++ flag identifiers)
      - Duplicate `--flag=val1 ... --flag=val2` overrides with conflicting values
    """
    if isinstance(raw_args, str):
        try:
            tokens = [tok for tok in shlex.split(raw_args) if tok.strip()]
        except ValueError as exc:
            issue = {
                "code": "unmatched_quote",
                "token": raw_args,
                "message": f"Malformed shell quoting in LIBTPU_INIT_ARGS: {exc}",
            }
            return {
                "valid": False,
                "tokens": [],
                "checked_flags": [],
                "jax_flags": [],
                "env_assignments": [],
                "malformed_tokens": [raw_args],
                "conflicting_duplicates": [],
                "issues": [issue],
            }
    else:
        tokens = [str(tok).strip() for tok in raw_args if str(tok).strip()]

    checked_flags: list[str] = []
    jax_flags: list[str] = []
    env_assignments: list[str] = []
    malformed_tokens: list[str] = []
    conflicting_duplicates: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    seen_values: dict[str, list[str]] = {}

    for tok in tokens:
        if allow_shell_var_tokens and SHELL_VAR_TOKEN_RE.fullmatch(tok):
            continue
        if tok.startswith("--jax_") or tok.startswith("-jax_"):
            jax_flags.append(tok)
            issues.append(
                {
                    "code": "jax_flag_in_libtpu_init_args",
                    "token": tok,
                    "message": (
                        f"JAX Python flag {tok!r} placed inside LIBTPU_INIT_ARGS; "
                        "configure via JAX_* environment variables or jax.config instead."
                    ),
                }
            )
            continue

        if not tok.startswith("-"):
            if ENV_ASSIGNMENT_RE.fullmatch(tok):
                env_assignments.append(tok)
                issues.append(
                    {
                        "code": "bare_env_assignment",
                        "token": tok,
                        "message": (
                            f"Bare KEY=VALUE environment assignment {tok!r} placed inside "
                            "LIBTPU_INIT_ARGS; export it as a standalone environment variable."
                        ),
                    }
                )
            else:
                malformed_tokens.append(tok)
                issues.append(
                    {
                        "code": "malformed_token",
                        "token": tok,
                        "message": (
                            f"Malformed token {tok!r} in LIBTPU_INIT_ARGS does not start with '-'."
                        ),
                    }
                )
            continue

        body = tok.lstrip("-")
        if not body or body.startswith("="):
            malformed_tokens.append(tok)
            issues.append(
                {
                    "code": "malformed_token",
                    "token": tok,
                    "message": f"Malformed empty flag token {tok!r} in LIBTPU_INIT_ARGS.",
                }
            )
            continue

        raw_name, has_eq, raw_val = body.partition("=")
        raw_name = raw_name.strip()
        if not VALID_FLAG_IDENT_RE.fullmatch(raw_name):
            malformed_tokens.append(tok)
            issues.append(
                {
                    "code": "malformed_token",
                    "token": tok,
                    "message": (
                        f"Malformed flag identifier {raw_name!r} in token {tok!r}."
                    ),
                }
            )
            continue

        checked_flags.append(raw_name)
        if has_eq:
            canon_name = raw_name
            norm_val = raw_val
        elif raw_name.startswith("noxla_") and len(raw_name) > 2:
            canon_name = raw_name[2:]
            norm_val = "false"
        else:
            canon_name = raw_name
            norm_val = "true"
        seen_values.setdefault(canon_name, []).append(norm_val)

    for canon_name, vals in seen_values.items():
        distinct = list(dict.fromkeys(vals))
        if len(distinct) > 1:
            conflicting_duplicates.append({"flag": canon_name, "values": vals})
            issues.append(
                {
                    "code": "conflicting_duplicate_flag",
                    "flag": canon_name,
                    "values": vals,
                    "message": (
                        f"Duplicate --{canon_name} overrides with conflicting values: {vals}."
                    ),
                }
            )

    return {
        "valid": len(issues) == 0,
        "tokens": tokens,
        "checked_flags": checked_flags,
        "jax_flags": jax_flags,
        "env_assignments": env_assignments,
        "malformed_tokens": malformed_tokens,
        "conflicting_duplicates": conflicting_duplicates,
        "issues": issues,
    }


def scan_libtpu_flags(so_path: str, init_args: str | list[str]) -> dict:
    """Validate every flag in `LIBTPU_INIT_ARGS` against syntax rules and `libtpu.so`.

    Always runs `validate_libtpu_init_args_syntax` first, then extracts all
    `[A-Za-z0-9_]+` identifier tokens from `libtpu.so` via `mmap` so typos or
    removed flags fail in <1s without initializing PJRT or opening `/dev/vfio/*`.
    """
    t0 = time.perf_counter()
    syntax = validate_libtpu_init_args_syntax(init_args)
    flag_names = list(syntax["checked_flags"])
    path_obj = Path(so_path)
    if not path_obj.is_file() or path_obj.stat().st_size == 0:
        return {
            "libtpu_so": so_path,
            "valid": False,
            "error": "libtpu.so missing or empty",
            "present": {},
            "missing": flag_names,
            "syntax": syntax,
            "syntax_issues": syntax["issues"],
            "elapsed_ms": round(1000.0 * (time.perf_counter() - t0), 2),
        }

    present = {}
    with open(path_obj, "rb") as stream, mmap.mmap(
        stream.fileno(), 0, access=mmap.ACCESS_READ
    ) as blob:
        known_identifiers = {
            m.group(0).decode("ascii", errors="ignore")
            for m in IDENT_TOKEN_RE.finditer(blob)
        }
        for name in flag_names:
            present[name] = name in known_identifiers

    missing = [name for name, ok in present.items() if not ok]
    valid = len(missing) == 0 and bool(syntax["valid"])
    out: dict[str, Any] = {
        "libtpu_so": str(path_obj),
        "libtpu_size_bytes": path_obj.stat().st_size,
        "checked_flags": flag_names,
        "present": present,
        "missing": missing,
        "syntax": syntax,
        "syntax_issues": syntax["issues"],
        "valid": valid,
        "elapsed_ms": round(1000.0 * (time.perf_counter() - t0), 2),
    }
    if not syntax["valid"]:
        out["error"] = "; ".join(item["message"] for item in syntax["issues"])
    return out


def _safe_eval_int_expr(expr_text: str) -> int | None:
    """Safely evaluate simple integer arithmetic expressions without `eval()`."""
    try:
        tree = ast.parse(expr_text.strip(), mode="eval")
    except SyntaxError:
        return None

    def _walk(node: ast.AST) -> int | None:
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return int(node.value)
        if isinstance(node, ast.UnaryOp):
            operand = _walk(node.operand)
            if operand is None:
                return None
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
            return None
        if isinstance(node, ast.BinOp):
            left = _walk(node.left)
            right = _walk(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv) and right != 0:
                return left // right
            if isinstance(node.op, ast.Pow) and 0 <= right <= 64:
                return left ** right
            if isinstance(node.op, ast.LShift) and 0 <= right <= 60:
                return left << right
        return None

    return _walk(tree)


def _classify_cache_max_size(raw_val: str) -> tuple[bool, str | None]:
    """Return `(is_positive_finite_cap, warning_message)` for `JAX_COMPILATION_CACHE_MAX_SIZE`."""
    cleaned = raw_val.strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()

    # Unwrap shell parameter fallback `${JAX_COMPILATION_CACHE_MAX_SIZE:-1073741824}`
    param_match = re.fullmatch(
        r"\$\{JAX_COMPILATION_CACHE_MAX_SIZE(?::-|:=|=)([^}]*)\}", cleaned
    )
    if param_match:
        cleaned = param_match.group(1).strip()

    if cleaned.startswith("$(( ") or cleaned.startswith("$(("):
        if cleaned.endswith("))"):
            inner = cleaned[3:-2].strip()
            val = _safe_eval_int_expr(inner)
            if val is not None and val > 0:
                return (
                    True,
                    f"JAX_COMPILATION_CACHE_MAX_SIZE={val} ({raw_val.strip()}) sets a positive "
                    "finite byte cap that triggers persistent cache eviction; use -1 for unbounded.",
                )
            return False, None

    int_val = _safe_eval_int_expr(cleaned.replace("_", ""))
    if int_val is not None:
        if int_val > 0:
            return (
                True,
                f"JAX_COMPILATION_CACHE_MAX_SIZE={int_val} sets a positive finite byte cap "
                "instead of -1 (unbounded persistent compilation cache).",
            )
        return False, None

    size_match = SIZE_SUFFIX_RE.fullmatch(cleaned)
    if size_match:
        qty = float(size_match.group(1))
        if qty > 0:
            return (
                True,
                f"JAX_COMPILATION_CACHE_MAX_SIZE={cleaned!r} sets a positive finite size cap "
                "instead of -1 (unbounded persistent compilation cache).",
            )

    return False, None


def _logical_shell_lines(text: str) -> list[tuple[int, str]]:
    """Join multiline shell continuations and open quotes while retaining start line numbers."""
    physical = text.splitlines()
    out: list[tuple[int, str]] = []
    buf: list[str] = []
    start_line = 1
    quote_char: str | None = None

    for idx, raw_line in enumerate(physical, start=1):
        if not buf:
            start_line = idx

        # Strip unquoted `#` comments before tracking quote continuation.
        cleaned_chars: list[str] = []
        i = 0
        escaped = False
        while i < len(raw_line):
            ch = raw_line[i]
            if escaped:
                cleaned_chars.append(ch)
                escaped = False
                i += 1
                continue
            if ch == "\\" and quote_char != "'":
                # Trailing line-continuation backslash
                if i == len(raw_line) - 1:
                    escaped = True
                    break
                cleaned_chars.append(ch)
                escaped = True
                i += 1
                continue
            if quote_char is None:
                if ch in ("'", '"'):
                    quote_char = ch
                    cleaned_chars.append(ch)
                elif ch == "#" and (i == 0 or raw_line[i - 1].isspace()):
                    break
                else:
                    cleaned_chars.append(ch)
            else:
                if ch == quote_char:
                    quote_char = None
                cleaned_chars.append(ch)
            i += 1

        segment = "".join(cleaned_chars)
        buf.append(segment)
        has_trailing_backslash = escaped
        if not has_trailing_backslash and quote_char is None:
            out.append((start_line, " ".join(buf)))
            buf = []

    if buf:
        out.append((start_line, " ".join(buf)))
    return out


def _read_shell_rhs(text: str, pos: int) -> tuple[str, int]:
    """Read a shell assignment RHS value starting at `pos` up to unquoted delimiter."""
    chars: list[str] = []
    quote_char: str | None = None
    paren_depth = 0
    i = pos
    while i < len(text):
        ch = text[i]
        if quote_char is not None:
            if ch == "\\" and quote_char == '"' and i + 1 < len(text):
                chars.append(ch)
                chars.append(text[i + 1])
                i += 2
                continue
            if ch == quote_char:
                quote_char = None
            chars.append(ch)
            i += 1
            continue

        if ch in ("'", '"'):
            quote_char = ch
            chars.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < len(text):
            chars.append(text[i + 1])
            i += 2
            continue
        if ch == "(":
            paren_depth += 1
            chars.append(ch)
            i += 1
            continue
        if ch == ")" and paren_depth > 0:
            paren_depth -= 1
            chars.append(ch)
            i += 1
            continue
        if paren_depth == 0 and (ch.isspace() or ch in ";|&"):
            break
        chars.append(ch)
        i += 1

    raw = "".join(chars).strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        raw = raw[1:-1]
    return raw, i


def _audit_shell_script(
    path: Path, text: str, so_path: str | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Statically audit a `.sh` script for `LIBTPU_INIT_ARGS` & `JAX_COMPILATION_CACHE_MAX_SIZE`."""
    assignments: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    shell_arrays: dict[str, str] = {}
    prev_libtpu_val = ""

    assign_re = re.compile(
        r"(?:^|[;&|]\s*|(?:export|env)\s+|\s+)"
        r"(LIBTPU_INIT_ARGS|JAX_COMPILATION_CACHE_MAX_SIZE)(\+?=)"
    )
    array_re = re.compile(
        r"(?:^|[;&|]\s*|(?:declare\s+-a\s+)?)"
        r"([A-Za-z_][A-Za-z0-9_]*)=\(([^)]*)\)"
    )
    default_assign_re = re.compile(
        r"\$\{(LIBTPU_INIT_ARGS|JAX_COMPILATION_CACHE_MAX_SIZE):?=([^}]*)\}"
    )

    for lineno, stmt in _logical_shell_lines(text):
        for arr_match in array_re.finditer(stmt):
            arr_name = arr_match.group(1)
            arr_body = arr_match.group(2)
            try:
                elems = shlex.split(arr_body)
                shell_arrays[arr_name] = " ".join(elems)
            except ValueError:
                shell_arrays[arr_name] = arr_body

        extracted_pairs: list[tuple[str, str, str]] = []
        pos = 0
        while pos < len(stmt):
            m = assign_re.search(stmt, pos)
            if not m:
                break
            var_name = m.group(1)
            op = m.group(2)
            rhs, next_pos = _read_shell_rhs(stmt, m.end())
            extracted_pairs.append((var_name, op, rhs))
            pos = max(next_pos, m.end())

        for dm in default_assign_re.finditer(stmt):
            extracted_pairs.append((dm.group(1), ":=", dm.group(2).strip()))

        for var_name, op, raw_rhs in extracted_pairs:
            expanded = raw_rhs
            for arr_name, arr_joined in shell_arrays.items():
                expanded = expanded.replace(f"${{{arr_name}[*]}}", arr_joined)
                expanded = expanded.replace(f"${{{arr_name}[@]}}", arr_joined)

            assignments.append(
                {
                    "file": str(path),
                    "line": lineno,
                    "variable": var_name,
                    "raw_value": raw_rhs,
                }
            )

            if var_name == "LIBTPU_INIT_ARGS":
                references_self = (
                    op == "+="
                    or "$LIBTPU_INIT_ARGS" in expanded
                    or "${LIBTPU_INIT_ARGS" in expanded
                )
                # Strip self-expansions from delta segment while keeping fallback defaults
                delta_str = re.sub(
                    r"\$\{LIBTPU_INIT_ARGS(?::-|:=)([^}]*)\}", r"\1", expanded
                )
                delta_str = re.sub(r"\$\{LIBTPU_INIT_ARGS[^}]*\}", " ", delta_str)
                delta_str = re.sub(r"\$LIBTPU_INIT_ARGS\b", " ", delta_str)

                combined_str = (
                    f"{prev_libtpu_val} {delta_str}".strip()
                    if references_self and prev_libtpu_val
                    else delta_str.strip()
                )
                prev_libtpu_val = combined_str

                delta_syntax = validate_libtpu_init_args_syntax(
                    delta_str, allow_shell_var_tokens=True
                )
                combined_syntax = validate_libtpu_init_args_syntax(
                    combined_str, allow_shell_var_tokens=True
                )
                # Emit non-duplicate token errors from delta + duplicate conflicts from combined
                for item in delta_syntax["issues"]:
                    if item["code"] != "conflicting_duplicate_flag":
                        issues.append(
                            {
                                "file": str(path),
                                "line": lineno,
                                "severity": "error",
                                "variable": "LIBTPU_INIT_ARGS",
                                **item,
                            }
                        )
                for item in combined_syntax["issues"]:
                    if item["code"] == "conflicting_duplicate_flag":
                        issues.append(
                            {
                                "file": str(path),
                                "line": lineno,
                                "severity": "error",
                                "variable": "LIBTPU_INIT_ARGS",
                                **item,
                            }
                        )
                if so_path:
                    bin_scan = scan_libtpu_flags(so_path, delta_syntax["checked_flags"])
                    for missing_flag in bin_scan.get("missing", []):
                        issues.append(
                            {
                                "file": str(path),
                                "line": lineno,
                                "severity": "error",
                                "variable": "LIBTPU_INIT_ARGS",
                                "code": "unknown_libtpu_flag",
                                "flag": missing_flag,
                                "message": (
                                    f"Flag --{missing_flag} not found in {so_path}."
                                ),
                            }
                        )

            elif var_name == "JAX_COMPILATION_CACHE_MAX_SIZE":
                is_bounded, msg = _classify_cache_max_size(expanded)
                if is_bounded and msg:
                    issues.append(
                        {
                            "file": str(path),
                            "line": lineno,
                            "severity": "warning",
                            "variable": "JAX_COMPILATION_CACHE_MAX_SIZE",
                            "code": "bounded_compilation_cache_max_size",
                            "value": raw_rhs,
                            "message": msg,
                        }
                    )

    return assignments, issues


def _resolve_ast_value(node: ast.AST, symtab: dict[str, Any]) -> Any:
    """Resolve static Python AST constants, lists/tuples, joins, f-strings, and ints."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return symtab.get(node.id)
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [_resolve_ast_value(elt, symtab) for elt in node.elts]
        if all(item is not None for item in items):
            return items
        return None
    if isinstance(node, ast.UnaryOp):
        val = _resolve_ast_value(node.operand, symtab)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if isinstance(node.op, ast.USub):
                return -val
            if isinstance(node.op, ast.UAdd):
                return +val
        return None
    if isinstance(node, ast.BinOp):
        left = _resolve_ast_value(node.left, symtab)
        right = _resolve_ast_value(node.right, symtab)
        if isinstance(left, str) and isinstance(right, str) and isinstance(node.op, ast.Add):
            return left + right
        if isinstance(left, list) and isinstance(right, list) and isinstance(node.op, ast.Add):
            return left + right
        if (
            isinstance(left, int)
            and not isinstance(left, bool)
            and isinstance(right, int)
            and not isinstance(right, bool)
        ):
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv) and right != 0:
                return left // right
            if isinstance(node.op, ast.Pow) and 0 <= right <= 64:
                return left ** right
            if isinstance(node.op, ast.LShift) and 0 <= right <= 60:
                return left << right
        return None
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for val_node in node.values:
            if isinstance(val_node, ast.Constant) and isinstance(val_node.value, str):
                parts.append(val_node.value)
            elif isinstance(val_node, ast.FormattedValue):
                resolved = _resolve_ast_value(val_node.value, symtab)
                parts.append(str(resolved) if resolved is not None else "${PY_EXPR}")
        return "".join(parts)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "str" and len(node.args) == 1:
            inner = _resolve_ast_value(node.args[0], symtab)
            return str(inner) if inner is not None else None
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "join"
            and len(node.args) == 1
        ):
            sep = _resolve_ast_value(node.func.value, symtab)
            seq = _resolve_ast_value(node.args[0], symtab)
            if isinstance(sep, str) and isinstance(seq, list):
                return sep.join(str(x) for x in seq)
    return None


def _audit_python_script(
    path: Path, text: str, so_path: str | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Statically audit a `.py` script via `ast.parse` without executing it."""
    assignments: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        issues.append(
            {
                "file": str(path),
                "line": exc.lineno or 1,
                "severity": "error",
                "variable": "SYNTAX",
                "code": "python_syntax_error",
                "message": f"Python syntax error while parsing {path}: {exc.msg}",
            }
        )
        return assignments, issues

    symtab: dict[str, Any] = {}
    recorded_keys: set[tuple[int, str, str]] = set()

    def _record_setting(lineno: int, var_name: str, resolved_val: Any, is_append: bool = False) -> None:
        if resolved_val is None:
            return
        if isinstance(resolved_val, list):
            str_val = " ".join(str(x) for x in resolved_val)
        else:
            str_val = str(resolved_val)

        dedup_key = (lineno, var_name, str_val)
        if dedup_key in recorded_keys:
            return
        recorded_keys.add(dedup_key)

        if is_append and var_name in symtab and isinstance(symtab[var_name], str):
            combined = f"{symtab[var_name]} {str_val}".strip()
        else:
            combined = str_val
        symtab[var_name] = combined

        assignments.append(
            {
                "file": str(path),
                "line": lineno,
                "variable": var_name,
                "raw_value": str_val,
            }
        )
        if var_name == "LIBTPU_INIT_ARGS":
            delta_syntax = validate_libtpu_init_args_syntax(
                str_val, allow_shell_var_tokens=True
            )
            combined_syntax = validate_libtpu_init_args_syntax(
                combined, allow_shell_var_tokens=True
            )
            for item in delta_syntax["issues"]:
                if item["code"] != "conflicting_duplicate_flag":
                    issues.append(
                        {
                            "file": str(path),
                            "line": lineno,
                            "severity": "error",
                            "variable": "LIBTPU_INIT_ARGS",
                            **item,
                        }
                    )
            for item in combined_syntax["issues"]:
                if item["code"] == "conflicting_duplicate_flag":
                    issues.append(
                        {
                            "file": str(path),
                            "line": lineno,
                            "severity": "error",
                            "variable": "LIBTPU_INIT_ARGS",
                            **item,
                        }
                    )
            if so_path:
                bin_scan = scan_libtpu_flags(so_path, delta_syntax["checked_flags"])
                for missing_flag in bin_scan.get("missing", []):
                    issues.append(
                        {
                            "file": str(path),
                            "line": lineno,
                            "severity": "error",
                            "variable": "LIBTPU_INIT_ARGS",
                            "code": "unknown_libtpu_flag",
                            "flag": missing_flag,
                            "message": f"Flag --{missing_flag} not found in {so_path}.",
                        }
                    )
        elif var_name == "JAX_COMPILATION_CACHE_MAX_SIZE":
            is_bounded, msg = _classify_cache_max_size(str_val)
            if is_bounded and msg:
                issues.append(
                    {
                        "file": str(path),
                        "line": lineno,
                        "severity": "warning",
                        "variable": "JAX_COMPILATION_CACHE_MAX_SIZE",
                        "code": "bounded_compilation_cache_max_size",
                        "value": str_val,
                        "message": msg,
                    }
                )

    def _subscript_key(target: ast.AST) -> str | None:
        if isinstance(target, ast.Subscript):
            slc = target.slice
            if isinstance(slc, ast.Constant) and isinstance(slc.value, str):
                return slc.value
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            resolved = _resolve_ast_value(node.value, symtab)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if resolved is not None:
                        symtab[target.id] = resolved
                    if target.id in ("LIBTPU_INIT_ARGS", "JAX_COMPILATION_CACHE_MAX_SIZE"):
                        _record_setting(node.lineno, target.id, resolved)
                sub_key = _subscript_key(target)
                if sub_key in ("LIBTPU_INIT_ARGS", "JAX_COMPILATION_CACHE_MAX_SIZE"):
                    _record_setting(node.lineno, sub_key, resolved)
        elif isinstance(node, ast.AnnAssign):
            resolved = (
                _resolve_ast_value(node.value, symtab) if node.value is not None else None
            )
            if isinstance(node.target, ast.Name):
                if resolved is not None:
                    symtab[node.target.id] = resolved
                if node.target.id in ("LIBTPU_INIT_ARGS", "JAX_COMPILATION_CACHE_MAX_SIZE"):
                    _record_setting(node.lineno, node.target.id, resolved)
            sub_key = _subscript_key(node.target)
            if sub_key in ("LIBTPU_INIT_ARGS", "JAX_COMPILATION_CACHE_MAX_SIZE"):
                _record_setting(node.lineno, sub_key, resolved)
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
            resolved = _resolve_ast_value(node.value, symtab)
            if isinstance(node.target, ast.Name) and node.target.id == "LIBTPU_INIT_ARGS":
                _record_setting(node.lineno, "LIBTPU_INIT_ARGS", resolved, is_append=True)
            sub_key = _subscript_key(node.target)
            if sub_key == "LIBTPU_INIT_ARGS":
                _record_setting(node.lineno, "LIBTPU_INIT_ARGS", resolved, is_append=True)
        elif isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id

            if func_name in ("setdefault", "putenv") and len(node.args) >= 2:
                key_val = _resolve_ast_value(node.args[0], symtab)
                if key_val in ("LIBTPU_INIT_ARGS", "JAX_COMPILATION_CACHE_MAX_SIZE"):
                    _record_setting(
                        node.lineno,
                        str(key_val),
                        _resolve_ast_value(node.args[1], symtab),
                    )
            elif func_name == "update" and len(node.args) >= 2:
                key_val = _resolve_ast_value(node.args[0], symtab)
                if key_val == "jax_compilation_cache_max_size":
                    _record_setting(
                        node.lineno,
                        "JAX_COMPILATION_CACHE_MAX_SIZE",
                        _resolve_ast_value(node.args[1], symtab),
                    )
        elif isinstance(node, ast.Dict):
            for k_node, v_node in zip(node.keys, node.values):
                if k_node is None:
                    continue
                key_val = _resolve_ast_value(k_node, symtab)
                if key_val in ("LIBTPU_INIT_ARGS", "JAX_COMPILATION_CACHE_MAX_SIZE"):
                    _record_setting(
                        k_node.lineno,
                        str(key_val),
                        _resolve_ast_value(v_node, symtab),
                    )

    return assignments, issues


def audit_launch_scripts(
    patterns: Sequence[str],
    libtpu_so: str | None = None,
    strict_audit: bool = False,
) -> dict[str, Any]:
    """Statically scan `.sh` and `.py` launch scripts without executing them."""
    resolved_files: list[Path] = []
    seen_paths: set[Path] = set()
    issues: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []

    for pat in patterns:
        p = Path(pat)
        matched: list[Path] = []
        if p.is_file():
            matched = [p]
        elif p.is_dir():
            matched = sorted(
                cand
                for cand in p.rglob("*")
                if cand.is_file() and cand.suffix.lower() in (".sh", ".bash", ".zsh", ".py")
            )
        else:
            matched = sorted(
                Path(hit)
                for hit in glob.glob(pat, recursive=True)
                if Path(hit).is_file()
            )
        if not matched:
            issues.append(
                {
                    "file": pat,
                    "line": 0,
                    "severity": "error",
                    "variable": "PATH",
                    "code": "audit_target_not_found",
                    "message": f"No launch script files matched pattern {pat!r}.",
                }
            )
            continue
        for item in matched:
            canon = item.resolve()
            if canon not in seen_paths:
                seen_paths.add(canon)
                resolved_files.append(item)

    so_path = libtpu_so or find_libtpu_so()
    for script_path in resolved_files:
        try:
            text = script_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            issues.append(
                {
                    "file": str(script_path),
                    "line": 0,
                    "severity": "error",
                    "variable": "IO",
                    "code": "unreadable_script",
                    "message": f"Failed to read {script_path}: {exc}",
                }
            )
            continue

        suffix = script_path.suffix.lower()
        first_line = text.splitlines()[0] if text else ""
        is_python = suffix == ".py" or (
            suffix == "" and "python" in first_line
        )
        if is_python:
            file_assigns, file_issues = _audit_python_script(
                script_path, text, so_path=so_path
            )
        else:
            file_assigns, file_issues = _audit_shell_script(
                script_path, text, so_path=so_path
            )
        assignments.extend(file_assigns)
        issues.extend(file_issues)

    return {
        "ok": len(issues) == 0,
        "strict_audit": bool(strict_audit),
        "libtpu_so": so_path,
        "files_scanned": [str(f) for f in resolved_files],
        "assignment_count": len(assignments),
        "issue_count": len(issues),
        "assignments": assignments,
        "issues": issues,
    }


def format_audit_markdown(report: dict[str, Any]) -> str:
    """Format `audit_launch_scripts` output as a Markdown table."""
    status = "PASS" if report.get("ok") else "ISSUES_FOUND"
    lines = [
        "### Launch Script Static Audit",
        "",
        f"- Status: `{status}`",
        f"- Files scanned: `{len(report.get('files_scanned', []))}`",
        f"- Assignments inspected: `{report.get('assignment_count', 0)}`",
        f"- Issues found: `{report.get('issue_count', 0)}`",
        "",
    ]
    issues = report.get("issues") or []
    if not issues:
        lines.append("No launch script configuration issues detected.")
        return "\n".join(lines)

    lines.extend(
        [
            "| File | Line | Severity | Variable | Code | Message |",
            "| --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for item in issues:
        msg = str(item.get("message", "")).replace("|", "\\|")
        lines.append(
            f"| `{item.get('file', '')}` | {item.get('line', 0)} | "
            f"`{item.get('severity', 'error')}` | `{item.get('variable', '')}` | "
            f"`{item.get('code', '')}` | {msg} |"
        )
    return "\n".join(lines)


def _parse_psi_file(psi_file: Path) -> dict:
    """Parse Linux `/proc/pressure/{memory,io,cpu}` into `{some: {avg10, ...}, full: {...}}`."""
    if not psi_file.is_file():
        return {}
    out = {}
    try:
        for line in psi_file.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            kind = parts[0]
            kv = {}
            for item in parts[1:]:
                if "=" in item:
                    k, v = item.split("=", 1)
                    try:
                        kv[k] = float(v) if "." in v else int(v)
                    except ValueError:
                        kv[k] = v
            out[kind] = kv
    except OSError:
        return {}
    return out


def _read_proc_pid_ticks(proc_root: Path, exclude_pids: set[int]) -> dict[int, tuple[int, int, str]]:
    """Return `{pid: (utime_plus_stime_ticks, uid, comm)}` from `/proc/<pid>/stat`."""
    snap = {}
    if not proc_root.is_dir():
        return snap
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in exclude_pids:
            continue
        try:
            uid = entry.stat().st_uid
            raw = (entry / "stat").read_text(errors="replace")
            lpar = raw.find("(")
            rpar = raw.rfind(")")
            if lpar == -1 or rpar == -1:
                continue
            comm = raw[lpar + 1 : rpar]
            after = raw[rpar + 2 :].split()
            # In /proc/<pid>/stat, after `(comm)`: index 11 is utime (field 14), index 12 is stime (field 15).
            utime = int(after[11])
            stime = int(after[12])
            snap[pid] = (utime + stime, uid, comm)
        except (OSError, ValueError, IndexError):
            continue
    return snap


def sample_foreign_cpu_load(
    proc_root: str = "/proc",
    interval_s: float = 0.20,
    own_uid: int | None = None,
    clk_tck: int | None = None,
    _snap_pair: tuple[dict, dict, float] | None = None,
) -> dict:
    """Measure foreign CPU load via two `/proc/<pid>/stat` tick samples.

    Computes CPU core equivalents consumed by processes belonging to UIDs other
    than `own_uid` (and total non-self CPU core usage) over `interval_s`.
    """
    uid_self = os.getuid() if own_uid is None else own_uid
    hz = clk_tck or (os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100)
    proc_path = Path(proc_root)

    if _snap_pair is not None:
        s0, s1, dt = _snap_pair
    else:
        exclude = {os.getpid()}
        t0 = time.perf_counter()
        s0 = _read_proc_pid_ticks(proc_path, exclude)
        time.sleep(max(0.01, interval_s))
        s1 = _read_proc_pid_ticks(proc_path, exclude)
        dt = max(1e-6, time.perf_counter() - t0)

    total_ticks_per_core = dt * hz
    foreign_ticks = 0
    own_other_ticks = 0
    offenders = []

    for pid, (ticks1, uid, comm) in s1.items():
        if pid not in s0:
            continue
        ticks0, _, _ = s0[pid]
        delta = max(0, ticks1 - ticks0)
        if delta == 0:
            continue
        cores = delta / total_ticks_per_core
        if uid != uid_self:
            foreign_ticks += delta
            offenders.append(
                {
                    "pid": pid,
                    "uid": uid,
                    "user": _uid_name(uid),
                    "comm": comm,
                    "cores": round(cores, 3),
                }
            )
        else:
            own_other_ticks += delta

    offenders.sort(key=lambda r: r["cores"], reverse=True)
    return {
        "sample_interval_s": round(dt, 3),
        "clk_tck": hz,
        "foreign_cpu_cores": round(foreign_ticks / total_ticks_per_core, 3),
        "own_background_cpu_cores": round(own_other_ticks / total_ticks_per_core, 3),
        "top_foreign_processes": offenders[:10],
    }


def check_host_health(
    proc_root: str = "/proc",
    shm_path: str = "/dev/shm",
    root_path: str = "/",
    cpu_sample_s: float = 0.20,
    own_uid: int | None = None,
    _snap_pair: tuple[dict, dict, float] | None = None,
) -> dict:
    """Inspect `/proc/meminfo`, `/dev/shm`, `/`, `/proc/pressure/*`, `ulimit -c`, and foreign CPU."""
    proc_path = Path(proc_root)
    meminfo = {}
    meminfo_file = proc_path / "meminfo"
    if meminfo_file.is_file():
        for line in meminfo_file.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                parts = v.split()
                if parts and parts[0].isdigit():
                    meminfo[k.strip()] = int(parts[0])

    gib = 1024**3
    mem_total_gib = (
        round(meminfo["MemTotal"] / 1048576.0, 2) if "MemTotal" in meminfo else None
    )
    mem_avail_gib = (
        round(meminfo["MemAvailable"] / 1048576.0, 2)
        if "MemAvailable" in meminfo
        else None
    )

    try:
        shm_free_gib = (
            round(shutil.disk_usage(shm_path).free / gib, 2)
            if os.path.exists(shm_path)
            else None
        )
    except OSError:
        shm_free_gib = None

    try:
        root_free_gib = (
            round(shutil.disk_usage(root_path).free / gib, 2)
            if os.path.exists(root_path)
            else None
        )
    except OSError:
        root_free_gib = None

    psi_memory = _parse_psi_file(proc_path / "pressure" / "memory")
    psi_io = _parse_psi_file(proc_path / "pressure" / "io")
    psi_cpu = _parse_psi_file(proc_path / "pressure" / "cpu")

    soft_core, hard_core = resource.getrlimit(resource.RLIMIT_CORE)
    cpu_load = sample_foreign_cpu_load(
        proc_root=proc_root,
        interval_s=cpu_sample_s,
        own_uid=own_uid,
        _snap_pair=_snap_pair,
    )

    return {
        "mem_total_gib": mem_total_gib,
        "mem_available_gib": mem_avail_gib,
        "shm_free_gib": shm_free_gib,
        "root_free_gib": root_free_gib,
        "psi_memory": psi_memory,
        "psi_io": psi_io,
        "psi_cpu": psi_cpu,
        "ulimit_core_soft": soft_core,
        "ulimit_core_hard": hard_core,
        "ulimit_core_zero": soft_core == 0,
        "cpu_load": cpu_load,
    }


def evaluate_gate(
    device_report: dict,
    lock_report: dict,
    flag_report: dict | None,
    health_report: dict,
    thresholds: dict,
    require_ulimit_core_zero: bool = True,
) -> list[str]:
    """Return a list of human-readable failure reasons (empty list means PASS)."""
    failures = []

    if device_report.get("idle") is False:
        failures.append(
            f"TPU devices busy: holders={device_report.get('holders')} "
            f"fuser_busy={device_report.get('fuser_busy_nodes')}"
        )
    elif device_report.get("idle") is None:
        failures.append(
            f"TPU device idle state unverifiable (unreadable PIDs: {device_report.get('unreadable_pids')})"
        )

    if lock_report.get("present") and not lock_report.get("moved_to"):
        if lock_report.get("held"):
            failures.append(
                f"Lockfile {lock_report['lock_path']} is actively held by {lock_report['holders']}"
            )
        elif lock_report.get("stale"):
            failures.append(
                f"Stale 0-byte lockfile {lock_report['lock_path']} present (pass --rename-stale-lock to clear)"
            )
        else:
            failures.append(
                f"Lockfile {lock_report['lock_path']} present and not provably stale: {lock_report}"
            )

    if not lock_report.get("inode_consistent", True):
        failures.append(
            f"Deleted-inode lockfile FDs detected: {lock_report.get('deleted_inode_holders')}"
        )

    if flag_report is not None and not flag_report.get("valid", False):
        syntax_msgs = [
            item.get("message", str(item))
            for item in (flag_report.get("syntax_issues") or [])
        ]
        detail_parts = []
        if flag_report.get("missing"):
            detail_parts.append(f"missing={flag_report.get('missing')}")
        if syntax_msgs:
            detail_parts.append(f"syntax_issues={syntax_msgs}")
        elif flag_report.get("error"):
            detail_parts.append(f"error={flag_report.get('error')}")
        failures.append(
            "Invalid LIBTPU_INIT_ARGS flags: "
            + (" ".join(detail_parts) if detail_parts else str(flag_report))
        )

    mem_avail = health_report.get("mem_available_gib")
    if mem_avail is not None and mem_avail < thresholds["min_mem_available_gib"]:
        failures.append(
            f"MemAvailable {mem_avail} GiB < {thresholds['min_mem_available_gib']} GiB"
        )

    shm_free = health_report.get("shm_free_gib")
    if shm_free is not None and shm_free < thresholds["min_shm_free_gib"]:
        failures.append(
            f"/dev/shm free {shm_free} GiB < {thresholds['min_shm_free_gib']} GiB"
        )

    root_free = health_report.get("root_free_gib")
    if root_free is not None and root_free < thresholds["min_root_free_gib"]:
        failures.append(
            f"Root filesystem free {root_free} GiB < {thresholds['min_root_free_gib']} GiB"
        )

    psi_mem_some = (health_report.get("psi_memory", {}).get("some") or {}).get("avg10")
    if psi_mem_some is not None and psi_mem_some > thresholds["psi_memory_avg10_limit"]:
        failures.append(
            f"Memory PSI some avg10 {psi_mem_some} > {thresholds['psi_memory_avg10_limit']}"
        )

    psi_io_some = (health_report.get("psi_io", {}).get("some") or {}).get("avg10")
    if psi_io_some is not None and psi_io_some > thresholds["psi_io_avg10_limit"]:
        failures.append(
            f"IO PSI some avg10 {psi_io_some} > {thresholds['psi_io_avg10_limit']}"
        )

    if require_ulimit_core_zero and not health_report.get("ulimit_core_zero", False):
        failures.append(
            f"ulimit -c soft limit is {health_report.get('ulimit_core_soft')} (expected 0 to prevent core dumps filling /)"
        )

    foreign_cores = (health_report.get("cpu_load") or {}).get("foreign_cpu_cores", 0.0)
    if foreign_cores > thresholds["foreign_cpu_cores_limit"]:
        failures.append(
            f"Foreign CPU load {foreign_cores} cores > {thresholds['foreign_cpu_cores_limit']} cores"
        )

    return failures


def run_self_test() -> int:
    """Run deterministic self-tests on mock binary, lockfile, and `/proc` trees."""
    print("Running tpu_preflight_gate.py --self-test ...")
    with tempfile.TemporaryDirectory(prefix="tpu_preflight_test_") as tmp:
        root = Path(tmp)

        # 1. Test libtpu.so mmap token scanner
        fake_so = root / "libtpu.so"
        fake_so.write_bytes(
            b"\x00\x7fELF\x00"
            b"xla_tpu_enable_latency_hiding_scheduler\x00"
            b"xla_enable_async_all_gather\x00"
            b"xla_tpu_spmd_threshold_for_allgather_cse\x00"
        )
        good_scan = scan_libtpu_flags(
            str(fake_so),
            "--xla_tpu_enable_latency_hiding_scheduler=true --xla_enable_async_all_gather=true",
        )
        assert good_scan["valid"] is True and not good_scan["missing"], good_scan

        # Partial prefix `xla_enable_async` must NOT match `xla_enable_async_all_gather`
        bad_scan = scan_libtpu_flags(
            str(fake_so),
            "--xla_enable_async=true --xla_nonexistent_flag=1",
        )
        assert bad_scan["valid"] is False, bad_scan
        assert set(bad_scan["missing"]) == {
            "xla_enable_async",
            "xla_nonexistent_flag",
        }, bad_scan

        # 2. Test lockfile inode verifier & stale lockfile rename
        proc_dir = root / "proc"
        (proc_dir / "101" / "fd").mkdir(parents=True)
        lock_file = root / "libtpu_lockfile"
        lock_file.write_bytes(b"")

        # 2a. Unheld 0-byte lockfile -> stale=True
        st_unheld = check_lockfile(str(lock_file), proc_root=str(proc_dir))
        assert st_unheld["stale"] is True and st_unheld["held"] is False, st_unheld

        # 2b. Live holder with matching inode -> held=True, stale=False
        os.symlink(lock_file, proc_dir / "101" / "fd" / "3")
        st_held = check_lockfile(str(lock_file), proc_root=str(proc_dir))
        assert st_held["held"] is True and st_held["stale"] is False, st_held
        assert st_held["holders"][0]["pid"] == 101, st_held

        # 2c. Remove holder symlink and test safe rename of stale lock
        os.unlink(proc_dir / "101" / "fd" / "3")
        st_renamed = check_lockfile(
            str(lock_file), proc_root=str(proc_dir), apply_stale_rename=True
        )
        assert st_renamed["stale"] is True and "moved_to" in st_renamed, st_renamed
        assert not lock_file.exists() and Path(st_renamed["moved_to"]).exists()

        # 2d. Non-empty lockfile -> never marked stale
        lock_file.write_text("active")
        st_nonempty = check_lockfile(str(lock_file), proc_root=str(proc_dir))
        assert st_nonempty["stale"] is False, st_nonempty
        lock_file.unlink()

        # 3. Test /proc device holder detection
        dev_dir = root / "dev"
        (dev_dir / "vfio").mkdir(parents=True)
        vfio0 = dev_dir / "vfio" / "0"
        vfio0.write_text("")
        (proc_dir / "101" / "cmdline").write_bytes(b"python3\x00train.py\x00")
        (proc_dir / "101" / "maps").write_text("")
        os.symlink("/dev/vfio/0", proc_dir / "101" / "fd" / "4")

        dev_busy = check_device_holders(
            proc_root=str(proc_dir), dev_root=str(dev_dir), use_fuser=False
        )
        assert dev_busy["idle"] is False and len(dev_busy["holders"]) == 1, dev_busy
        assert dev_busy["holders"][0]["pid"] == 101, dev_busy

        os.unlink(proc_dir / "101" / "fd" / "4")
        dev_idle = check_device_holders(
            proc_root=str(proc_dir), dev_root=str(dev_dir), use_fuser=False
        )
        assert dev_idle["idle"] is True and not dev_idle["holders"], dev_idle

        # 4. Test /proc/meminfo, /proc/pressure/*, and foreign CPU tick parser
        (proc_dir / "meminfo").write_text(
            "MemTotal:       196608000 kB\nMemAvailable:   131072000 kB\n"
        )
        (proc_dir / "pressure").mkdir(parents=True)
        (proc_dir / "pressure" / "memory").write_text(
            "some avg10=0.05 avg60=0.02 avg300=0.00 total=100\n"
            "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
        )
        (proc_dir / "pressure" / "io").write_text(
            "some avg10=0.01 avg60=0.01 avg300=0.00 total=50\n"
            "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
        )

        # Synthetic two-sample tick snapshots: foreign UID 9999 used 60 ticks in 0.2s at 100 Hz = 3.0 cores
        snap0 = {101: (1000, os.getuid(), "self_worker"), 202: (500, 9999, "foreign_job")}
        snap1 = {101: (1010, os.getuid(), "self_worker"), 202: (560, 9999, "foreign_job")}
        health = check_host_health(
            proc_root=str(proc_dir),
            shm_path=str(root),
            root_path=str(root),
            own_uid=os.getuid(),
            _snap_pair=(snap0, snap1, 0.20),
        )
        assert health["mem_available_gib"] == 125.0, health
        assert health["psi_memory"]["some"]["avg10"] == 0.05, health
        assert health["cpu_load"]["foreign_cpu_cores"] == 3.0, health
        assert health["cpu_load"]["own_background_cpu_cores"] == 0.5, health

        failures = evaluate_gate(
            dev_idle,
            {"present": False, "inode_consistent": True},
            good_scan,
            health,
            RESOURCE_PROFILES["v6e"],
            require_ulimit_core_zero=False,
        )
        assert not failures, f"Expected zero gate failures, got: {failures}"

        # 5. Test static LIBTPU_INIT_ARGS syntax hygiene without libtpu.so
        clean_syntax = validate_libtpu_init_args_syntax(
            "--xla_tpu_enable_async_collective_fusion=true "
            "--xla_tpu_enable_latency_hiding_scheduler=true"
        )
        assert clean_syntax["valid"] is True and not clean_syntax["issues"], clean_syntax

        polluted_syntax = validate_libtpu_init_args_syntax(
            "--jax_threefry_partitionable=true "
            "TPU_MEGACORE=MEGACORE_DENSE "
            "stray_token "
            "--xla_enable_async_all_gather=true "
            "--xla_enable_async_all_gather=false"
        )
        assert polluted_syntax["valid"] is False, polluted_syntax
        issue_codes = {item["code"] for item in polluted_syntax["issues"]}
        assert issue_codes == {
            "jax_flag_in_libtpu_init_args",
            "bare_env_assignment",
            "malformed_token",
            "conflicting_duplicate_flag",
        }, polluted_syntax

        # 6. Test offline launch script auditor (.sh and .py) without executing user code
        bad_sh = root / "launch_bad.sh"
        bad_sh.write_text(
            "#!/usr/bin/env bash\n"
            "export LIBTPU_INIT_ARGS=\"\\\n"
            "  --xla_tpu_enable_latency_hiding_scheduler=true \\\n"
            "  --jax_compilation_cache_dir=/tmp/xla_cache \\\n"
            "  TPU_MEGACORE=MEGACORE_DENSE \\\n"
            "  --xla_enable_async_all_gather=true \\\n"
            "  --xla_enable_async_all_gather=false\"\n"
            "export JAX_COMPILATION_CACHE_MAX_SIZE=1073741824\n",
            encoding="utf-8",
        )
        bad_py = root / "launch_bad.py"
        bad_py.write_text(
            "import os\n"
            "raise RuntimeError('must never execute during static audit')\n"
            "FLAGS = ['--xla_enable_async_all_gather=true', '--jax_enable_x64=true', 'bare_word']\n"
            "os.environ['LIBTPU_INIT_ARGS'] = ' '.join(FLAGS)\n"
            "os.environ['JAX_COMPILATION_CACHE_MAX_SIZE'] = str(2 * 1024**3)\n",
            encoding="utf-8",
        )
        good_sh = root / "launch_good.sh"
        good_sh.write_text(
            "#!/usr/bin/env bash\n"
            "export LIBTPU_INIT_ARGS=\"--xla_tpu_enable_latency_hiding_scheduler=true ${EXTRA_ARGS:-}\"\n"
            "export JAX_COMPILATION_CACHE_MAX_SIZE=-1\n",
            encoding="utf-8",
        )

        clean_audit = audit_launch_scripts([str(good_sh)], libtpu_so=str(fake_so))
        assert clean_audit["ok"] is True and clean_audit["issue_count"] == 0, clean_audit

        bad_audit = audit_launch_scripts(
            [str(bad_sh), str(bad_py)], libtpu_so=str(fake_so), strict_audit=True
        )
        assert bad_audit["ok"] is False and bad_audit["issue_count"] >= 6, bad_audit
        audit_codes = {item["code"] for item in bad_audit["issues"]}
        assert "jax_flag_in_libtpu_init_args" in audit_codes, bad_audit
        assert "bare_env_assignment" in audit_codes, bad_audit
        assert "conflicting_duplicate_flag" in audit_codes, bad_audit
        assert "malformed_token" in audit_codes, bad_audit
        assert "bounded_compilation_cache_max_size" in audit_codes, bad_audit
        md_text = format_audit_markdown(bad_audit)
        assert "| File | Line | Severity |" in md_text, md_text

    print("PASS: All tpu_preflight_gate self-tests succeeded.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Zero-TPU-init preflight validator for Cloud TPU VMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--profile",
        choices=sorted(RESOURCE_PROFILES),
        default="v6e",
        help="Hardware resource threshold profile (default: v6e).",
    )
    ap.add_argument(
        "--lockfile",
        default=DEFAULT_LOCKFILE,
        help="Path to libtpu lockfile (default: /tmp/libtpu_lockfile).",
    )
    ap.add_argument(
        "--rename-stale-lock",
        action="store_true",
        help="Safely rename a 0-byte unheld lockfile aside instead of failing.",
    )
    ap.add_argument(
        "--check-flags",
        default=os.environ.get("LIBTPU_INIT_ARGS", ""),
        help="LIBTPU_INIT_ARGS string to validate for token hygiene and against libtpu.so.",
    )
    ap.add_argument(
        "--libtpu-so",
        default="",
        help="Explicit path to libtpu.so (auto-detected if omitted).",
    )
    ap.add_argument(
        "--audit-script",
        action="append",
        default=[],
        metavar="PATH_OR_GLOB",
        help="Repeatable path or glob of .sh / .py launch scripts to audit statically without executing.",
    )
    ap.add_argument(
        "--strict-audit",
        action="store_true",
        help="Exit with code 1 when --audit-script finds any issues or warnings.",
    )
    ap.add_argument(
        "--markdown",
        action="store_true",
        help="Render --audit-script output as a Markdown table instead of JSON.",
    )
    ap.add_argument(
        "--allow-nonzero-core-ulimit",
        action="store_true",
        help="Do not fail if ulimit -c != 0.",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in CPU/mock self-tests and exit.",
    )
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if args.audit_script:
        audit_rep = audit_launch_scripts(
            args.audit_script,
            libtpu_so=args.libtpu_so or None,
            strict_audit=args.strict_audit,
        )
        if args.markdown:
            print(format_audit_markdown(audit_rep), flush=True)
        else:
            print(json.dumps(audit_rep, indent=2, sort_keys=True), flush=True)
        if args.strict_audit and not audit_rep["ok"]:
            return 1
        if any(item.get("code") == "audit_target_not_found" for item in audit_rep["issues"]):
            return 1
        return 0

    device_rep = check_device_holders()
    lock_rep = check_lockfile(args.lockfile, apply_stale_rename=args.rename_stale_lock)

    flag_rep = None
    if args.check_flags.strip():
        syntax = validate_libtpu_init_args_syntax(args.check_flags)
        if args.libtpu_so:
            flag_rep = scan_libtpu_flags(args.libtpu_so, args.check_flags)
        else:
            so_path = find_libtpu_so()
            if so_path:
                flag_rep = scan_libtpu_flags(so_path, args.check_flags)
            else:
                flag_rep = {
                    "libtpu_so": None,
                    "symbol_scan_skipped": True,
                    "checked_flags": syntax["checked_flags"],
                    "present": {},
                    "missing": [],
                    "syntax": syntax,
                    "syntax_issues": syntax["issues"],
                    "valid": syntax["valid"],
                    "error": (
                        "; ".join(item["message"] for item in syntax["issues"])
                        if not syntax["valid"]
                        else None
                    ),
                }

    health_rep = check_host_health()
    thresholds = RESOURCE_PROFILES[args.profile]
    failures = evaluate_gate(
        device_rep,
        lock_rep,
        flag_rep,
        health_rep,
        thresholds,
        require_ulimit_core_zero=not args.allow_nonzero_core_ulimit,
    )

    payload = {
        "ok": len(failures) == 0,
        "profile": args.profile,
        "failures": failures,
        "device": device_rep,
        "lockfile": lock_rep,
        "flag_scan": flag_rep,
        "host_health": health_rep,
    }
    print("PREFLIGHT_GATE " + json.dumps(payload, sort_keys=True), flush=True)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
