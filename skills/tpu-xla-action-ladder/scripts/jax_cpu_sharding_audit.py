#!/usr/bin/env python3
"""Zero-TPU CPU 64-Device Parameter & Optimizer Sharding Auditor.

WHY THIS SCRIPT EXISTS
----------------------
In JAX/Flax SPMD training, parameter sharding functions commonly traverse a
PyTree using `jax.tree_util.tree_map_with_path` and match parameter names
against suffix or regex rules (e.g., `name.endswith(("kernel", "embedding"))`).

The Classic `str(DictKey)` Silent Replication Bug:
If a sharding helper constructs leaf paths via `"/".join(str(p) for p in path)`:
  - `str(DictKey('kernel'))` evaluates to `"['kernel']"` (with brackets/quotes)
  - `str(GetAttrKey('mu'))` evaluates to `".mu"`
  - `str(SequenceKey(0))` evaluates to `"[0]"`
Every joined path then looks like `"['decoder']/['layers_0']/['q_proj']/['kernel']"`.
Because `"['kernel']".endswith("kernel")` is `False`, every suffix rule misses,
silently falling back to `PartitionSpec()` (100% replicated across all chips).
On a `dp=4 x fsdp=16` mesh, this silent bug inflated per-device persistent HBM
by 16x (`7.81 GiB/device` instead of `0.49 GiB/device`) without raising a
single warning.

WHAT THIS SCRIPT DOES
---------------------
1. Forces `JAX_PLATFORMS=cpu` and `--xla_force_host_platform_device_count=N`
   (e.g., 64 virtual CPU devices) BEFORE importing `jax`, auditing full-scale
   64-chip meshes (`dp * fsdp * tensor`) in seconds on any laptop or login node.
2. Audits real model PyTrees or serialized leaf manifests via `--pytree-json`
   or `--factory module_or_file:callable` (preserving user-supplied `PartitionSpec`
   assignments when present rather than overwriting them with heuristic rules).
3. Provides `_clean_key_name(key)` and `join_clean_path(path)` to strip
   `DictKey`, `GetAttrKey`, `SequenceKey`, and `Flax` key wrappers cleanly.
4. Computes total logical bytes, per-device sharded bytes, replicated bytes,
   `replicated_pct`, and fails closed if `fsdp * tp > 1` and any non-whitelisted
   tensor >= `--fail-on-replicated-mib` (or `--max-unsharded-mib`) stays replicated.

EXAMPLE CLI INVOCATIONS
-----------------------
  # Run self-test verifying _clean_key_name, --pytree-json, and --factory adapters:
  python3 scripts/jax_cpu_sharding_audit.py --self-test

  # Audit a serialized leaf + PartitionSpec manifest from a real model:
  python3 scripts/jax_cpu_sharding_audit.py --dp 4 --fsdp 16 --tensor 1 --pytree-json /tmp/pytree_specs.json --fail-on-replicated-mib 4.0

  # Audit a Python factory returning (params_pytree, specs_pytree) on virtual CPU mesh:
  python3 scripts/jax_cpu_sharding_audit.py --dp 4 --fsdp 16 --factory my_pkg.sharding:build_model_and_specs --fail-on-replicated-mib 4.0

  # Reproduce the legacy str(DictKey) bug on the synthetic parameter tree:
  python3 scripts/jax_cpu_sharding_audit.py --dp 4 --fsdp 16 --legacy-bug-demo
"""
from __future__ import annotations

import argparse
import collections
from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

MIB = float(1024 ** 2)
GIB = float(1024 ** 3)

_DTYPE_ITEMSIZE: dict[str, int] = {
    "bool": 1,
    "bool_": 1,
    "int8": 1,
    "i1": 1,
    "i8": 1,
    "uint8": 1,
    "u1": 1,
    "u8": 1,
    "float8_e4m3fn": 1,
    "float8_e4m3fnuz": 1,
    "float8_e4m3b11fnuz": 1,
    "float8_e5m2": 1,
    "float8_e5m2fnuz": 1,
    "fp8": 1,
    "bfloat16": 2,
    "bf16": 2,
    "float16": 2,
    "f2": 2,
    "f16": 2,
    "half": 2,
    "int16": 2,
    "i2": 2,
    "i16": 2,
    "uint16": 2,
    "u2": 2,
    "u16": 2,
    "float32": 4,
    "f4": 4,
    "f32": 4,
    "float": 4,
    "int32": 4,
    "i4": 4,
    "i32": 4,
    "uint32": 4,
    "u4": 4,
    "u32": 4,
    "complex64": 8,
    "float64": 8,
    "f8": 8,
    "f64": 8,
    "double": 8,
    "int64": 8,
    "i8_64": 8,
    "i64": 8,
    "uint64": 8,
    "u8_64": 8,
    "u64": 8,
    "complex128": 16,
}

_DICTKEY_BRACKET_RE = re.compile(
    r"\['[^']+'\]|\[\"[^\"]+\"\]|(?:^|/)\.[A-Za-z_][A-Za-z0-9_]*"
)


def configure_cpu_virtual_devices(num_devices: int) -> None:
    """Force `num_devices` virtual CPU devices before importing `jax`."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_force_host_platform_device_count" not in flags:
        os.environ["XLA_FLAGS"] = (
            f"{flags} --xla_force_host_platform_device_count={int(num_devices)}"
        ).strip()


def load_callable_factory(spec: str) -> Any:
    """Load a callable from `'pkg.module:callable'` or `'path/to/file.py:callable'`.

    Supports dotted attribute chains after `':'` (for example `'pkg.mod:Builder.make'`).
    """
    if not spec or ":" not in spec:
        raise ValueError(
            f"Invalid factory spec {spec!r}; expected 'pkg.module:fn' or 'path/to/file.py:fn'."
        )
    mod_or_path, _, attr_chain = spec.rpartition(":")
    mod_or_path = mod_or_path.strip()
    attr_chain = attr_chain.strip()
    if not mod_or_path or not attr_chain:
        raise ValueError(
            f"Invalid factory spec {spec!r}; both module/file and callable name must be non-empty."
        )

    cwd = os.getcwd()
    if cwd not in sys.path and "" not in sys.path:
        sys.path.insert(0, cwd)

    candidate_path = Path(mod_or_path)
    is_file_spec = (
        mod_or_path.endswith(".py")
        or "/" in mod_or_path
        or os.sep in mod_or_path
        or candidate_path.is_file()
    )
    if is_file_spec:
        resolved = candidate_path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Factory file {mod_or_path!r} (resolved: {resolved}) does not exist."
            )
        digest = hashlib.md5(str(resolved).encode("utf-8")).hexdigest()[:10]
        mod_name = f"_tpu_kit_factory_{resolved.stem}_{digest}"
        mod_spec = importlib.util.spec_from_file_location(mod_name, str(resolved))
        if mod_spec is None or mod_spec.loader is None:
            raise ImportError(f"Could not create module spec from {resolved}.")
        module = importlib.util.module_from_spec(mod_spec)
        sys.modules[mod_name] = module
        parent_dir = str(resolved.parent)
        added_parent = False
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            added_parent = True
        try:
            mod_spec.loader.exec_module(module)
        finally:
            if added_parent and parent_dir in sys.path:
                sys.path.remove(parent_dir)
    else:
        module = importlib.import_module(mod_or_path)

    target: Any = module
    for part in attr_chain.split("."):
        if not hasattr(target, part):
            raise AttributeError(
                f"Factory target {attr_chain!r} missing attribute {part!r} in {mod_or_path!r}."
            )
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f"Resolved factory target {spec!r} ({target!r}) is not callable.")
    return target


def parse_factory_kwargs(raw_json: str | None) -> dict[str, Any]:
    """Parse `--factory-kwargs '<json>'` into a Python keyword-argument dictionary."""
    if not raw_json or not raw_json.strip():
        return {}
    parsed = json.loads(raw_json)
    if not isinstance(parsed, dict):
        raise ValueError(
            f"--factory-kwargs must decode to a JSON object, got {type(parsed).__name__}."
        )
    return parsed


def dtype_to_name_and_itemsize(dtype_obj: Any) -> tuple[str, int]:
    """Resolve dtype name and byte width without requiring NumPy/JAX at import time."""
    if dtype_obj is None:
        return "float32", 4
    if hasattr(dtype_obj, "dtype") and not isinstance(dtype_obj, (str, type)):
        dtype_obj = dtype_obj.dtype
    itemsize = getattr(dtype_obj, "itemsize", None)
    name = getattr(dtype_obj, "name", None) or str(dtype_obj)
    # Strip class wrappers like "<class 'jax.numpy.bfloat16'>" or "dtype('float32')"
    if name.startswith("<class '") and name.endswith("'>"):
        name = name[8:-2].rsplit(".", 1)[-1]
    elif name.startswith("dtype(") and name.endswith(")"):
        name = name[6:-1].strip("'\"")
    clean_lower = name.lower().strip()
    if isinstance(itemsize, int) and itemsize > 0:
        return clean_lower, int(itemsize)
    if clean_lower in _DTYPE_ITEMSIZE:
        return clean_lower, _DTYPE_ITEMSIZE[clean_lower]
    try:
        import numpy as np  # type: ignore[import-not-found]
        dt = np.dtype(dtype_obj)
        return str(dt.name), int(dt.itemsize)
    except Exception:
        raise ValueError(f"Unrecognized dtype {dtype_obj!r}; supply a standard dtype name such as 'float32' or 'bfloat16'.")


# =============================================================================
# 1. Key Normalization: Fixing the str(DictKey('kernel')) -> "['kernel']" Trap
# =============================================================================

def _clean_key_name(key: Any) -> str:
    """Extract the bare string identifier from a JAX PyTree path key entry.

    Handles:
      - `jax.tree_util.DictKey('kernel')` -> `'kernel'` (whereas `str(key)` is `"['kernel']"`)
      - `jax.tree_util.GetAttrKey('mu')`  -> `'mu'`     (whereas `str(key)` is `".mu"`)
      - `jax.tree_util.SequenceKey(0)`    -> `'0'`      (whereas `str(key)` is `"[0]"`)
      - Plain strings or ints             -> `str(key)`
    """
    for attr in ("key", "name", "idx"):
        if hasattr(key, attr):
            return str(getattr(key, attr))
    raw = str(key)
    # Fallback regex strip if an unknown wrapper formats as "['foo']" or ".foo"
    if raw.startswith("['") and raw.endswith("']"):
        return raw[2:-2]
    if raw.startswith('["') and raw.endswith('"]'):
        return raw[2:-2]
    if raw.startswith(".") and len(raw) > 1:
        return raw[1:]
    if raw.startswith("[") and raw.endswith("]"):
        return raw[1:-1]
    return raw


def join_clean_path(path: Iterable[Any]) -> str:
    """Join a JAX PyTree key path into a clean `'/'`-separated string."""
    return "/".join(_clean_key_name(k) for k in path)


def join_legacy_buggy_path(path: Iterable[Any]) -> str:
    """Reproduce the classic buggy `"/".join(str(k) for k in path)` anti-pattern."""
    return "/".join(str(k) for k in path)


def has_dictkey_bracket_bug(path_str: str) -> bool:
    """Detect whether a serialized path string contains `['key']` or `.attr` `str(KeyEntry)` artifacts."""
    return bool(_DICTKEY_BRACKET_RE.search(path_str or ""))


# =============================================================================
# 2. Logical Sharding Rule Engine & Explicit PartitionSpec Normalizer
# =============================================================================

DEFAULT_AXIS_RULES: tuple[tuple[str, str | None], ...] = (
    ("embed", "fsdp"),
    ("mlp", "fsdp"),
    ("kv", None),
    ("vocab", "fsdp"),
    ("hidden", "fsdp"),
)

DEFAULT_REPLICATED_WHITELIST_RE = re.compile(
    r"(^|/)(norm|layernorm|rmsnorm|ln_[0-9]+|input_layernorm|post_attention_layernorm|"
    r"q_norm|k_norm|bias|scale|count|step)(/|$)",
    re.IGNORECASE,
)

_AXIS_SYNONYMS: dict[str, tuple[str, ...]] = {
    "data": ("data", "dp", "batch", "replica"),
    "dp": ("dp", "data", "batch", "replica"),
    "batch": ("batch", "data", "dp"),
    "fsdp": ("fsdp", "zero", "sharded"),
    "tensor": ("tensor", "tp", "model", "mp", "megatron"),
    "tp": ("tp", "tensor", "model", "mp", "megatron"),
    "model": ("model", "tensor", "tp", "mp"),
    "mp": ("mp", "tensor", "tp", "model"),
}


def _resolve_axis_degree(axis_name: str, mesh_axes: dict[str, int]) -> int:
    if axis_name in mesh_axes:
        return int(mesh_axes[axis_name])
    for cand in _AXIS_SYNONYMS.get(axis_name, ()):
        if cand in mesh_axes:
            return int(mesh_axes[cand])
    return 1


def normalize_explicit_spec(spec_obj: Any) -> tuple[Any, ...]:
    """Normalize a JSON list, tuple, `PartitionSpec`, or `NamedSharding` into a spec tuple."""
    if spec_obj is None:
        return ()
    if hasattr(spec_obj, "spec") and not isinstance(spec_obj, (str, dict, list, tuple)):
        spec_obj = spec_obj.spec
    if hasattr(spec_obj, "_partitions") and not isinstance(spec_obj, (str, dict, list, tuple)):
        spec_obj = spec_obj._partitions
    if isinstance(spec_obj, str):
        s = spec_obj.strip()
        if s in ("", "P()", "PartitionSpec()", "None", "null"):
            return ()
        for prefix in ("PartitionSpec(", "P("):
            if s.startswith(prefix) and s.endswith(")"):
                inner = s[len(prefix):-1].strip()
                if not inner:
                    return ()
                parts: list[str | None] = []
                for tok in inner.split(","):
                    t = tok.strip().strip("'\"")
                    if not t:
                        continue
                    parts.append(None if t in ("None", "null") else t)
                return tuple(parts)
        return (s,)
    if isinstance(spec_obj, (list, tuple)):
        norm: list[Any] = []
        for elem in spec_obj:
            if elem is None:
                norm.append(None)
            elif isinstance(elem, (list, tuple)):
                sub = tuple(str(x) for x in elem if x is not None)
                norm.append(sub if len(sub) > 1 else (sub[0] if len(sub) == 1 else None))
            else:
                s_elem = str(elem).strip()
                norm.append(None if s_elem in ("", "None", "null", "UNCONSTRAINED") else s_elem)
        return tuple(norm)
    raise ValueError(f"Unsupported PartitionSpec representation: {spec_obj!r}")


def format_pspec(pspec: Sequence[Any]) -> str:
    """Format a normalized PartitionSpec tuple as `P(...)`."""
    return f"P({', '.join(repr(x) for x in pspec)})"


def infer_logical_spec(
    leaf_name: str,
    shape: tuple[int, ...],
) -> tuple[str | None, ...] | None:
    """Infer a logical axis spec for a parameter leaf from its normalized path and shape."""
    if not shape or len(shape) == 0:
        return None
    if leaf_name.endswith(("embedding", "tok_embeddings/weight", "embed_tokens/weight")):
        return ("vocab", None) if len(shape) == 2 else ("vocab",)
    if leaf_name.endswith("kernel") or leaf_name.endswith("weight"):
        if DEFAULT_REPLICATED_WHITELIST_RE.search(leaf_name) and len(shape) == 1:
            return None
        if len(shape) == 2:
            if any(k in leaf_name for k in ("gate_proj", "up_proj", "fc1", "mlp_in")):
                return ("embed", "mlp")
            if any(k in leaf_name for k in ("down_proj", "fc2", "mlp_out")):
                return ("mlp", "embed")
            return ("embed", "kv")
        if len(shape) >= 3:
            return ("embed",) + (None,) * (len(shape) - 1)
    return None


def map_logical_to_pspec(
    lspec: tuple[str | None, ...] | None,
    axis_rules: dict[str, str | None],
) -> tuple[str | None, ...]:
    """Map logical axis names to physical mesh axes, ensuring no mesh axis appears twice."""
    if lspec is None:
        return ()
    used_axes: set[str] = set()
    pspec: list[str | None] = []
    for ax in lspec:
        phys = axis_rules.get(ax) if ax is not None else None
        if phys is not None and phys not in used_axes:
            used_axes.add(phys)
            pspec.append(phys)
        else:
            pspec.append(None)
    return tuple(pspec)


def compute_shard_shape(
    full_shape: tuple[int, ...],
    spec: tuple[Any, ...],
    mesh_axes: dict[str, int],
) -> tuple[tuple[int, ...], str]:
    """Compute the per-device shard shape and diagnose why a tensor stayed replicated."""
    if not full_shape:
        return (), "scalar"
    shard = list(full_shape)
    reasons: list[str] = []
    for dim_idx, axis_entry in enumerate(spec):
        if dim_idx >= len(shard) or axis_entry is None:
            continue
        if isinstance(axis_entry, (tuple, list)):
            deg = 1
            for sub_ax in axis_entry:
                deg *= _resolve_axis_degree(str(sub_ax), mesh_axes)
            axis_label = "*".join(str(x) for x in axis_entry)
        else:
            deg = _resolve_axis_degree(str(axis_entry), mesh_axes)
            axis_label = str(axis_entry)
        if deg > 1:
            if shard[dim_idx] % deg == 0:
                shard[dim_idx] //= deg
            else:
                reasons.append(f"dim{dim_idx}={shard[dim_idx]} not divisible by {axis_label}={deg}")
    reason_str = "; ".join(reasons)
    return tuple(shard), reason_str


@dataclasses.dataclass
class FakeDictKey:
    """Stdlib stand-in matching `jax.tree_util.DictKey` string & attribute semantics."""
    key: str

    def __str__(self) -> str:
        return f"[{self.key!r}]"


@dataclasses.dataclass
class FakeGetAttrKey:
    """Stdlib stand-in matching `jax.tree_util.GetAttrKey` string & attribute semantics."""
    name: str

    def __str__(self) -> str:
        return f".{self.name}"


@dataclasses.dataclass
class FakeSequenceKey:
    """Stdlib stand-in matching `jax.tree_util.SequenceKey` string & attribute semantics."""
    idx: int

    def __str__(self) -> str:
        return f"[{self.idx}]"


@dataclasses.dataclass
class LeafDescriptor:
    path: tuple[Any, ...]
    shape: tuple[int, ...]
    dtype_name: str
    itemsize: int
    explicit_spec: tuple[Any, ...] | None = None
    has_explicit_spec: bool = False
    raw_path_str: str | None = None


def leaf_from_dict(item: Mapping[str, Any]) -> LeafDescriptor:
    """Parse a JSON/dict leaf descriptor, preserving explicit `'spec'` if supplied."""
    raw_path = item.get("path", item.get("name"))
    if raw_path is None:
        raise ValueError(f"Leaf dict missing 'path' (or 'name'): {item!r}")
    if isinstance(raw_path, str):
        raw_path_str = raw_path
        path_tuple: tuple[Any, ...] = tuple(p for p in raw_path.split("/") if p != "")
    elif isinstance(raw_path, Sequence):
        raw_path_str = "/".join(str(p) for p in raw_path)
        path_tuple = tuple(raw_path)
    else:
        raw_path_str = str(raw_path)
        path_tuple = (raw_path,)

    raw_shape = item.get("shape")
    if raw_shape is None or not isinstance(raw_shape, Sequence) or isinstance(raw_shape, (str, bytes)):
        raise ValueError(f"Leaf dict missing valid 'shape' sequence: {item!r}")
    shape = tuple(int(d) for d in raw_shape)

    dtype_name, itemsize = dtype_to_name_and_itemsize(item.get("dtype", "float32"))
    if "itemsize" in item and item["itemsize"] is not None:
        itemsize = int(item["itemsize"])

    has_spec = "spec" in item
    explicit_spec = normalize_explicit_spec(item["spec"]) if has_spec else None
    return LeafDescriptor(
        path=path_tuple,
        shape=shape,
        dtype_name=dtype_name,
        itemsize=itemsize,
        explicit_spec=explicit_spec,
        has_explicit_spec=has_spec,
        raw_path_str=raw_path_str,
    )


def load_leaves_from_json(json_path: str | Path) -> list[LeafDescriptor]:
    """Load `LeafDescriptor` list from `--pytree-json` (`[...]` or `{'leaves': [...]}`)."""
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_leaves = payload
    elif isinstance(payload, dict) and isinstance(payload.get("leaves"), list):
        raw_leaves = payload["leaves"]
    else:
        raise ValueError(
            f"--pytree-json {str(json_path)!r} must contain a JSON list or an object with a 'leaves' list."
        )
    return [leaf_from_dict(entry) for entry in raw_leaves]


def _is_spec_leaf(obj: Any) -> bool:
    """Return True if `obj` is a PartitionSpec / Sharding / tuple-of-axis-names leaf rather than a PyTree container."""
    if obj is None:
        return True
    cls_name = type(obj).__name__
    if cls_name in (
        "PartitionSpec",
        "NamedSharding",
        "PositionalSharding",
        "SingleDeviceSharding",
        "GSPMDSharding",
    ) or hasattr(obj, "_partitions"):
        return True
    if isinstance(obj, (list, tuple)) and not hasattr(obj, "_fields"):
        if len(obj) == 0:
            return True
        return all(
            elem is None
            or isinstance(elem, str)
            or (
                isinstance(elem, (list, tuple))
                and not hasattr(elem, "_fields")
                and all(x is None or isinstance(x, str) for x in elem)
            )
            for elem in obj
        )
    return False


def _lookup_spec_along_path(specs_tree: Any, key_path: Sequence[Any]) -> Any:
    """Resolve the matching spec for `key_path` in `specs_tree`, supporting prefix broadcast."""
    cur = specs_tree
    for key in key_path:
        if _is_spec_leaf(cur):
            return cur
        if hasattr(key, "key") and isinstance(cur, Mapping):
            cur = cur[key.key]
        elif hasattr(key, "idx") and isinstance(cur, Sequence):
            cur = cur[key.idx]
        elif hasattr(key, "name") and hasattr(cur, key.name):
            cur = getattr(cur, key.name)
        elif isinstance(cur, Mapping) and _clean_key_name(key) in cur:
            cur = cur[_clean_key_name(key)]
        elif isinstance(cur, Sequence) and not isinstance(cur, (str, bytes)) and str(_clean_key_name(key)).isdigit():
            cur = cur[int(_clean_key_name(key))]
        elif hasattr(cur, _clean_key_name(key)):
            cur = getattr(cur, _clean_key_name(key))
        else:
            raise KeyError(
                f"Could not resolve path key {key!r} (clean={_clean_key_name(key)!r}) inside specs PyTree {type(cur).__name__}."
            )
    return cur


def _flatten_array_pytree(pytree: Any) -> list[tuple[tuple[Any, ...], Any]]:
    """Flatten a PyTree of arrays / ShapeDtypeStructs into `(path_tuple, leaf)` pairs."""
    try:
        import jax
        flat_with_path, _ = jax.tree_util.tree_flatten_with_path(pytree)
        return [(tuple(p), val) for p, val in flat_with_path]
    except ImportError:
        pass

    out: list[tuple[tuple[Any, ...], Any]] = []

    def _walk(node: Any, prefix: tuple[Any, ...]) -> None:
        if hasattr(node, "shape") and not isinstance(node, (dict, list, tuple)):
            out.append((prefix, node))
        elif isinstance(node, Mapping):
            for k, v in node.items():
                _walk(v, prefix + (FakeDictKey(str(k)),))
        elif isinstance(node, (list, tuple)) and not hasattr(node, "_fields"):
            for idx, v in enumerate(node):
                _walk(v, prefix + (FakeSequenceKey(idx),))
        elif hasattr(node, "_fields"):
            for fname in node._fields:
                _walk(getattr(node, fname), prefix + (FakeGetAttrKey(fname),))
        elif dataclasses.is_dataclass(node) and not isinstance(node, type):
            for f in dataclasses.fields(node):
                _walk(getattr(node, f.name), prefix + (FakeGetAttrKey(f.name),))
        else:
            out.append((prefix, node))

    _walk(pytree, ())
    return out


def _is_leaf_descriptor_item(item: Any) -> bool:
    return isinstance(item, LeafDescriptor) or (
        isinstance(item, Mapping) and "shape" in item and ("path" in item or "name" in item)
    )


def extract_leaves_from_factory_output(result: Any) -> list[LeafDescriptor]:
    """Convert a factory return value into `list[LeafDescriptor]`.

    Supported factory return formats:
      1. `Sequence[LeafDescriptor | dict]`
      2. `(pytree_of_arrays_or_shapes, pytree_of_specs)`
      3. `pytree_of_arrays_or_shapes`
    """
    if (
        isinstance(result, Sequence)
        and not isinstance(result, (str, bytes))
        and len(result) > 0
        and all(_is_leaf_descriptor_item(x) for x in result)
    ):
        return [x if isinstance(x, LeafDescriptor) else leaf_from_dict(x) for x in result]

    specs_tree: Any = None
    has_specs_tree = False
    params_tree = result

    if (
        isinstance(result, tuple)
        and not hasattr(result, "_fields")
        and len(result) == 2
        and not hasattr(result[0], "shape")
    ):
        # Distinguish `(params_pytree, specs_pytree)` from a 2-tuple of array subtrees:
        # in `(params_pytree, specs_pytree)`, leaves of `result[1]` are PartitionSpecs / shardings / None / tuples of axis names, not arrays with `.dtype`.
        cand_params, cand_specs = result
        if _is_spec_leaf(cand_specs) or isinstance(cand_specs, (Mapping, tuple, list)) or dataclasses.is_dataclass(cand_specs):
            # Probe one leaf of cand_params to check if cand_specs resolves to a spec rather than an array with .dtype
            flat_probe = _flatten_array_pytree(cand_params)
            if flat_probe:
                first_path, _ = flat_probe[0]
                try:
                    resolved = _lookup_spec_along_path(cand_specs, first_path)
                    if not hasattr(resolved, "dtype"):
                        params_tree = cand_params
                        specs_tree = cand_specs
                        has_specs_tree = True
                except Exception:
                    pass

    flat_leaves = _flatten_array_pytree(params_tree)
    descriptors: list[LeafDescriptor] = []
    for key_path, val in flat_leaves:
        if isinstance(val, LeafDescriptor):
            descriptors.append(val)
            continue
        if isinstance(val, Mapping) and "shape" in val:
            descriptors.append(leaf_from_dict(val))
            continue
        shape_attr = getattr(val, "shape", None)
        if shape_attr is None:
            raise TypeError(
                f"PyTree leaf at {join_clean_path(key_path)!r} ({type(val).__name__}) has no '.shape' attribute."
            )
        shape = tuple(int(d) for d in shape_attr)
        dtype_name, itemsize = dtype_to_name_and_itemsize(getattr(val, "dtype", "float32"))
        explicit_spec: tuple[Any, ...] | None = None
        if has_specs_tree:
            raw_spec = _lookup_spec_along_path(specs_tree, key_path)
            explicit_spec = normalize_explicit_spec(raw_spec)
        elif getattr(val, "sharding", None) is not None and hasattr(val.sharding, "spec"):
            explicit_spec = normalize_explicit_spec(val.sharding.spec)
            has_specs_tree_leaf = True
            descriptors.append(
                LeafDescriptor(
                    path=key_path,
                    shape=shape,
                    dtype_name=dtype_name,
                    itemsize=itemsize,
                    explicit_spec=explicit_spec,
                    has_explicit_spec=has_specs_tree_leaf,
                    raw_path_str="/".join(str(k) for k in key_path)
                    if all(isinstance(k, str) for k in key_path)
                    else None,
                )
            )
            continue

        raw_str = (
            "/".join(str(k) for k in key_path)
            if all(isinstance(k, str) for k in key_path)
            else None
        )
        descriptors.append(
            LeafDescriptor(
                path=key_path,
                shape=shape,
                dtype_name=dtype_name,
                itemsize=itemsize,
                explicit_spec=explicit_spec,
                has_explicit_spec=has_specs_tree,
                raw_path_str=raw_str,
            )
        )
    return descriptors


def build_synthetic_model_leaves(num_layers: int = 4) -> list[LeafDescriptor]:
    """Build a realistic Transformer parameter + Adam optimizer state leaf list with KeyEntry paths."""
    leaves: list[LeafDescriptor] = [
        LeafDescriptor(
            path=(FakeDictKey("params"), FakeDictKey("decoder"), FakeDictKey("tok_embeddings"), FakeDictKey("weight")),
            shape=(128256, 4096),
            dtype_name="float32",
            itemsize=4,
        )
    ]
    for i in range(num_layers):
        layer_k = FakeDictKey(f"layers_{i}")
        for proj, sh in (
            ("q_proj", (4096, 4096)),
            ("k_proj", (4096, 1024)),
            ("v_proj", (4096, 1024)),
            ("o_proj", (4096, 4096)),
        ):
            leaves.append(
                LeafDescriptor(
                    path=(FakeDictKey("params"), FakeDictKey("decoder"), layer_k, FakeDictKey("self_attn"), FakeDictKey(proj), FakeDictKey("kernel")),
                    shape=sh,
                    dtype_name="float32",
                    itemsize=4,
                )
            )
        for mlp_proj, sh in (
            ("gate_proj", (4096, 14336)),
            ("up_proj", (4096, 14336)),
            ("down_proj", (14336, 4096)),
        ):
            leaves.append(
                LeafDescriptor(
                    path=(FakeDictKey("params"), FakeDictKey("decoder"), layer_k, FakeDictKey("mlp"), FakeDictKey(mlp_proj), FakeDictKey("kernel")),
                    shape=sh,
                    dtype_name="float32",
                    itemsize=4,
                )
            )
        leaves.append(
            LeafDescriptor(
                path=(FakeDictKey("params"), FakeDictKey("decoder"), layer_k, FakeDictKey("input_layernorm"), FakeDictKey("weight")),
                shape=(4096,),
                dtype_name="float32",
                itemsize=4,
            )
        )
    # Add optimizer moment `mu` (bf16, GetAttrKey + SequenceKey + DictKey mix)
    param_leaves = list(leaves)
    for pl in param_leaves:
        leaves.append(
            LeafDescriptor(
                path=(FakeDictKey("opt_state"), FakeSequenceKey(0), FakeGetAttrKey("mu")) + pl.path[1:],
                shape=pl.shape,
                dtype_name="bfloat16",
                itemsize=2,
            )
        )
    return leaves


def audit_leaves(
    leaves: list[LeafDescriptor],
    mesh_axes: dict[str, int],
    axis_rules: dict[str, str | None],
    *,
    use_legacy_buggy_join: bool = False,
    max_unsharded_mib: float = 1.0,
    fail_on_replicated_mib: float | None = None,
    allow_replicated_regex: str | None = None,
) -> dict[str, Any]:
    """Audit sharding assignment across all leaves and flag unwhitelisted replicated tensors."""
    rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    dictkey_bracket_paths: list[str] = []

    custom_allow_re = re.compile(allow_replicated_regex) if allow_replicated_regex else None
    threshold_mib = (
        float(fail_on_replicated_mib)
        if fail_on_replicated_mib is not None
        else float(max_unsharded_mib)
    )
    fsdp_deg = _resolve_axis_degree("fsdp", mesh_axes)
    tp_deg = _resolve_axis_degree("tensor", mesh_axes)
    sharding_expected = (fsdp_deg * tp_deg) > 1

    for leaf in leaves:
        clean_name = join_clean_path(leaf.path)
        raw_path_candidate = (
            leaf.raw_path_str
            if leaf.raw_path_str is not None
            else (join_legacy_buggy_path(leaf.path) if use_legacy_buggy_join else clean_name)
        )
        bracket_bug = has_dictkey_bracket_bug(raw_path_candidate)
        if bracket_bug:
            dictkey_bracket_paths.append(raw_path_candidate)

        if leaf.has_explicit_spec:
            lspec = None
            pspec: tuple[Any, ...] = leaf.explicit_spec or ()
            joined_for_rule = raw_path_candidate
        else:
            joined_for_rule = (
                raw_path_candidate
                if (use_legacy_buggy_join or bracket_bug)
                else clean_name
            )
            lspec = infer_logical_spec(joined_for_rule, leaf.shape)
            pspec = map_logical_to_pspec(lspec, axis_rules)

        shard_shape, div_reason = compute_shard_shape(leaf.shape, pspec, mesh_axes)
        # If not divisible, XLA/trainer cannot shard along that axis -> falls back to replicated
        effective_pspec = () if div_reason else pspec
        if div_reason:
            shard_shape = leaf.shape

        numel = 1
        for d in leaf.shape:
            numel *= d
        shard_numel = 1
        for d in shard_shape:
            shard_numel *= d

        full_bytes = numel * leaf.itemsize
        dev_bytes = shard_numel * leaf.itemsize
        replicated = (dev_bytes == full_bytes) and len(leaf.shape) > 0

        reason = ""
        if replicated:
            if div_reason:
                reason = f"divisibility guard ({div_reason})"
            elif leaf.has_explicit_spec:
                if bracket_bug:
                    reason = f"str(DictKey) bracket bug in path with explicit {format_pspec(pspec)}"
                else:
                    reason = f"explicit spec {format_pspec(pspec)} leaves tensor replicated"
            elif lspec is None:
                clean_lspec = infer_logical_spec(clean_name, leaf.shape)
                if clean_lspec is not None and (use_legacy_buggy_join or bracket_bug):
                    reason = "str(DictKey) bracket bug broke suffix match"
                else:
                    reason = "no suffix rule matches (expected for 1D norm/bias)"
            else:
                reason = f"logical spec {lspec} mapped to None/size-1 axis"

        whitelisted = bool(DEFAULT_REPLICATED_WHITELIST_RE.search(clean_name))
        if custom_allow_re is not None and (
            custom_allow_re.search(clean_name) or custom_allow_re.search(raw_path_candidate)
        ):
            whitelisted = True
        if fail_on_replicated_mib is None and len(leaf.shape) <= 1 and (full_bytes / MIB) < 1.0:
            whitelisted = True

        row = {
            "name": clean_name,
            "lookup_key": joined_for_rule,
            "shape": leaf.shape,
            "dtype": leaf.dtype_name,
            "spec": format_pspec(effective_pspec if not leaf.has_explicit_spec else pspec),
            "explicit_spec": leaf.has_explicit_spec,
            "dictkey_bracket_bug": bracket_bug,
            "shard_shape": shard_shape,
            "full_bytes": full_bytes,
            "dev_bytes": dev_bytes,
            "replicated": replicated,
            "whitelisted": whitelisted,
            "reason": reason,
        }
        rows.append(row)
        if sharding_expected and replicated and not whitelisted and (full_bytes / MIB) >= threshold_mib:
            violations.append(row)

    total_full = sum(r["full_bytes"] for r in rows)
    total_dev = sum(r["dev_bytes"] for r in rows)
    total_rep = sum(r["full_bytes"] for r in rows if r["replicated"])
    n_rep = sum(1 for r in rows if r["replicated"])

    return {
        "mesh": mesh_axes,
        "use_legacy_buggy_join": use_legacy_buggy_join,
        "threshold_mib": threshold_mib,
        "sharding_expected": sharding_expected,
        "total_leaves": len(rows),
        "replicated_leaves": n_rep,
        "replicated_leaves_pct": 100.0 * n_rep / max(len(rows), 1),
        "total_logical_bytes": total_full,
        "total_logical_gib": total_full / GIB,
        "per_device_sharded_bytes": total_dev,
        "per_device_sharded_gib": total_dev / GIB,
        "replicated_bytes": total_rep,
        "replicated_pct": 100.0 * total_rep / max(total_full, 1),
        "compression_factor": total_full / max(total_dev, 1),
        "dictkey_bracket_count": len(dictkey_bracket_paths),
        "dictkey_bracket_paths": dictkey_bracket_paths[:20],
        "violations_count": len(violations),
        "violations": violations,
        "rows": rows,
        "passed": len(violations) == 0,
    }


def format_audit_report(rep: dict[str, Any]) -> str:
    """Format the sharding audit report as a Markdown table."""
    layer_rx = re.compile(r"layers_\d+")
    grouped: dict[tuple[str, str, str], dict[str, Any]] = collections.OrderedDict()
    for r in rep["rows"]:
        pat = layer_rx.sub("layers_*", r["name"])
        key = (pat, r["spec"], r["dtype"])
        g = grouped.setdefault(
            key,
            {
                "count": 0,
                "shape": r["shape"],
                "shard_shape": r["shard_shape"],
                "full_bytes": 0,
                "dev_bytes": 0,
                "replicated": r["replicated"],
                "reason": r["reason"],
            },
        )
        g["count"] += 1
        g["full_bytes"] += r["full_bytes"]
        g["dev_bytes"] += r["dev_bytes"]

    mode_str = "LEGACY BUG (`str(k)`)" if rep["use_legacy_buggy_join"] else "CLEAN (`_clean_key_name`)"
    lines = [
        f"## CPU Parameter & Optimizer Sharding Audit ({mode_str})",
        f"- **Mesh**: `{rep['mesh']}`",
        f"- **Total Logical State**: `{rep['total_logical_gib']:.3f} GiB` ({rep['total_logical_bytes'] / MIB:.1f} MiB)",
        f"- **Per-Device Sharded State**: `{rep['per_device_sharded_gib']:.3f} GiB` "
        f"({rep['per_device_sharded_bytes'] / MIB:.1f} MiB, **`{rep['compression_factor']:.2f}x`** reduction)",
        f"- **Replicated Bytes**: `{rep['replicated_pct']:.2f}%` ({rep['replicated_leaves']}/{rep['total_leaves']} leaves)",
        f"- **DictKey Bracket Paths**: `{rep.get('dictkey_bracket_count', 0)}`",
        f"- **Gate Status**: **{'PASS' if rep['passed'] else 'FAIL (Unsharded Large Tensors)'}** "
        f"(`{rep['violations_count']}` violations, threshold `{rep.get('threshold_mib', 1.0):.2f} MiB`)",
        "",
        "| Leaf Pattern | Shape | PartitionSpec | Per-Chip Shard | Count | Full (MiB) | Chip (MiB) | Status |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for (pat, spec, _dt), g in grouped.items():
        status = f"REPLICATED ({g['reason']})" if g["replicated"] else "SHARDED"
        lines.append(
            f"| `{pat}` | `{g['shape']}` | `{spec}` | `{g['shard_shape']}` | "
            f"{g['count']} | {g['full_bytes'] / MIB:.2f} | {g['dev_bytes'] / MIB:.2f} | {status} |"
        )
    return "\n".join(lines)


def run_self_test() -> int:
    """Verify `_clean_key_name`, explicit `PartitionSpec` auditing, `--pytree-json`, and `--factory`."""
    path_sample = (
        FakeDictKey("decoder"),
        FakeSequenceKey(0),
        FakeGetAttrKey("self_attn"),
        FakeDictKey("q_proj"),
        FakeDictKey("kernel"),
    )
    buggy = join_legacy_buggy_path(path_sample)
    clean = join_clean_path(path_sample)
    if buggy != "['decoder']/[0]/.self_attn/['q_proj']/['kernel']":
        raise AssertionError(f"Unexpected buggy path: {buggy}")
    if clean != "decoder/0/self_attn/q_proj/kernel":
        raise AssertionError(f"Unexpected clean path: {clean}")

    # Also test real jax.tree_util key classes if jax is installed
    try:
        configure_cpu_virtual_devices(8)
        import jax
        real_tree = {"decoder": {"layers_0": {"q_proj": {"kernel": 1}}}}
        paths_and_vals, _ = jax.tree_util.tree_flatten_with_path(real_tree)
        real_path = paths_and_vals[0][0]
        if join_clean_path(real_path) != "decoder/layers_0/q_proj/kernel":
            raise AssertionError(f"Failed on real jax.tree_util.DictKey: {join_clean_path(real_path)}")
        if join_legacy_buggy_path(real_path).endswith("kernel"):
            raise AssertionError("Real jax DictKey str() should end with \"['kernel']\", not 'kernel'")
    except ImportError:
        pass

    mesh = {"data": 4, "fsdp": 16, "tensor": 1}
    rules = dict(DEFAULT_AXIS_RULES)
    leaves = build_synthetic_model_leaves(num_layers=4)

    rep_buggy = audit_leaves(leaves, mesh, rules, use_legacy_buggy_join=True)
    rep_clean = audit_leaves(leaves, mesh, rules, use_legacy_buggy_join=False)

    if rep_buggy["replicated_pct"] < 99.9 or rep_buggy["passed"]:
        raise AssertionError(f"Expected legacy buggy join to replicate 100% and fail gate: {rep_buggy['replicated_pct']}")
    if not rep_clean["passed"] or rep_clean["replicated_pct"] > 1.0:
        raise AssertionError(f"Expected clean join to pass gate with <1% replicated: {rep_clean}")
    if rep_clean["compression_factor"] < 15.5:
        raise AssertionError(f"Expected ~16x FSDP compression factor, got {rep_clean['compression_factor']:.2f}x")

    # Verify explicit PartitionSpec preservation + DictKey bracket bug + allowlist regex via --pytree-json & --factory
    with tempfile.TemporaryDirectory(prefix="tpu_sharding_selftest_") as tmp:
        json_file = Path(tmp) / "leaves.json"
        # Note: 'custom_table' would NOT match infer_logical_spec, proving explicit spec=['fsdp', None] is honored directly;
        # meanwhile 'proj_unsharded' has explicit spec=[] (replicated) and triggers fail_on_replicated_mib unless allowlisted.
        json_file.write_text(
            json.dumps(
                {
                    "leaves": [
                        {"path": "params/custom_table", "shape": [32768, 1024], "dtype": "float32", "spec": ["fsdp", None]},
                        {"path": "['params']/['proj_unsharded']/['kernel']", "shape": [2048, 2048], "dtype": "float32", "spec": []},
                    ]
                }
            ),
            encoding="utf-8",
        )
        json_leaves = load_leaves_from_json(json_file)
        rep_json_fail = audit_leaves(
            json_leaves,
            mesh,
            rules,
            fail_on_replicated_mib=4.0,
        )
        if rep_json_fail["passed"] or rep_json_fail["dictkey_bracket_count"] != 1:
            raise AssertionError(f"Expected explicit empty spec + DictKey path to fail: {rep_json_fail}")
        if rep_json_fail["rows"][0]["replicated"] or rep_json_fail["rows"][0]["shard_shape"] != (2048, 1024):
            raise AssertionError(f"Explicit spec ['fsdp', None] was not honored: {rep_json_fail['rows'][0]}")

        rep_json_allow = audit_leaves(
            json_leaves,
            mesh,
            rules,
            fail_on_replicated_mib=4.0,
            allow_replicated_regex=r"proj_unsharded",
        )
        if not rep_json_allow["passed"]:
            raise AssertionError(f"Expected --allow-replicated-regex to whitelist leaf: {rep_json_allow}")

        factory_py = Path(tmp) / "sample_factory.py"
        factory_py.write_text(
            "class DummyShape:\n"
            "    def __init__(self, shape, dtype='float32'):\n"
            "        self.shape = shape\n"
            "        self.dtype = dtype\n\n"
            "def build_pair(hidden=1024):\n"
            "    params = {'enc': {'w': DummyShape((4096, hidden), 'bfloat16')}}\n"
            "    specs = {'enc': {'w': ('fsdp', None)}}\n"
            "    return params, specs\n",
            encoding="utf-8",
        )
        fn = load_callable_factory(f"{factory_py}:build_pair")
        fac_leaves = extract_leaves_from_factory_output(fn(hidden=2048))
        rep_fac = audit_leaves(fac_leaves, mesh, rules, fail_on_replicated_mib=1.0)
        if not rep_fac["passed"] or rep_fac["rows"][0]["shard_shape"] != (256, 2048):
            raise AssertionError(f"Factory (params, specs) audit failed: {rep_fac}")

    print(format_audit_report(rep_buggy))
    print("\n" + "=" * 80 + "\n")
    print(format_audit_report(rep_clean))
    print("\n[SELF-TEST PASS] CPU sharding auditor, explicit spec adapter, and `_clean_key_name` verified.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Zero-TPU CPU 64-Device Parameter & Optimizer Sharding Auditor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dp", type=int, default=4, help="Data-parallel mesh dimension.")
    ap.add_argument("--fsdp", type=int, default=16, help="FSDP mesh dimension.")
    ap.add_argument("--tensor", "--tp", dest="tensor", type=int, default=1, help="Tensor-parallel mesh dimension.")
    ap.add_argument("--layers", type=int, default=4, help="Number of Transformer layers to synthesize in demo mode.")
    ap.add_argument("--pytree-json", default="", help="Path to JSON leaf manifest ([...] or {'leaves': [...]}) with optional explicit 'spec'.")
    ap.add_argument("--factory", default="", help="Callable spec 'pkg.mod:fn' or 'path/to/file.py:fn' returning (params, specs), pytree, or leaf sequence.")
    ap.add_argument("--factory-kwargs", default="", help="JSON object of keyword arguments forwarded to --factory.")
    ap.add_argument("--legacy-bug-demo", action="store_true", help="Demonstrate the str(DictKey) 100%% replication bug.")
    ap.add_argument("--max-unsharded-mib", type=float, default=1.0, help="Fail-closed threshold (MiB) for unwhitelisted replicated leaves.")
    ap.add_argument("--fail-on-replicated-mib", type=float, default=None, help="Fail with exit code 1 if fsdp*tp > 1 and any non-allowlisted leaf >= threshold MiB is replicated.")
    ap.add_argument("--allow-replicated-regex", default="", help="Optional regex whitelisting intentionally replicated parameter paths.")
    ap.add_argument("--json-out", help="Optional path to save JSON audit output.")
    ap.add_argument("--self-test", action="store_true", help="Run built-in verification test.")
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if args.pytree_json and args.factory:
        ap.error("--pytree-json and --factory are mutually exclusive.")

    total_devices = max(1, int(args.dp) * int(args.fsdp) * int(args.tensor))
    configure_cpu_virtual_devices(total_devices)

    mesh = {"data": args.dp, "fsdp": args.fsdp, "tensor": args.tensor}
    rules = dict(DEFAULT_AXIS_RULES)

    if args.pytree_json:
        leaves = load_leaves_from_json(args.pytree_json)
    elif args.factory:
        fn = load_callable_factory(args.factory)
        f_kwargs = parse_factory_kwargs(args.factory_kwargs)
        leaves = extract_leaves_from_factory_output(fn(**f_kwargs))
    else:
        leaves = build_synthetic_model_leaves(num_layers=args.layers)

    rep = audit_leaves(
        leaves,
        mesh,
        rules,
        use_legacy_buggy_join=args.legacy_bug_demo,
        max_unsharded_mib=args.max_unsharded_mib,
        fail_on_replicated_mib=args.fail_on_replicated_mib,
        allow_replicated_regex=args.allow_replicated_regex or None,
    )
    print(format_audit_report(rep))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2)
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
