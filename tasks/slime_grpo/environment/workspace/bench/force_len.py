"""Judge-owned generate wrapper: every sample decodes exactly the number of tokens the frozen reference fixture
recorded for it (max_new_tokens = recorded length, ignore_eos), so reference and candidate rollouts do the same
amount of decode work. It calls the tree-under-test's own default generate(); nothing else changes."""
import json
import os

from slime.rollout.sglang_rollout import generate as _tree_generate

_LENGTHS = None
_PROFILED = False
_PROFILE_FIRST_INDEX = 64  # first sample of rollout 1 (16 prompts x 4 samples per rollout): skip the warmup rollout


def _maybe_profile(args, sample):
    """bench.py profile --side rollout: trace 200 decode steps on every sglang worker once rollout 1 starts
    (the profiler output goes to the run directory)."""
    global _PROFILED
    out = os.environ.get("SLIME_GRPO_BENCH_PROFILE_DIR")
    if not out or _PROFILED or sample.index < _PROFILE_FIRST_INDEX:
        return
    _PROFILED = True
    import requests
    router = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    workers = requests.get(f"{router}/workers", timeout=30).json().get("workers", [])
    for w in workers:
        url = w.get("url") if isinstance(w, dict) else w
        requests.post(f"{url}/start_profile", json={"output_dir": out, "num_steps": 200,
                                                    "activities": ["CPU", "GPU"]}, timeout=60)


def _lengths():
    global _LENGTHS
    if _LENGTHS is None:
        with open(os.environ["FORCE_LENGTHS"]) as f:
            _LENGTHS = {int(k): int(v) for k, v in json.load(f).items()}
    return _LENGTHS


async def generate(args, sample, sampling_params):
    n = _lengths().get(sample.index)
    if n is None:
        raise KeyError(f"sample index {sample.index} has no recorded length in the fixture")
    _maybe_profile(args, sample)
    sampling_params = dict(sampling_params)
    sampling_params["max_new_tokens"] = n + sample.response_length
    sampling_params["ignore_eos"] = True
    sampling_params.pop("stop", None)
    sampling_params.pop("stop_token_ids", None)
    return await _tree_generate(args, sample, sampling_params)
