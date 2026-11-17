"""Judge-owned rollout instrument (slime --custom-rollout-log-function-path), called in the rollout manager once a
rollout's samples are complete. It returns False, so slime's own logging follows unchanged.

Per call it appends one line to $SLIME_OPD_BENCH_DIR/rollout_clock.jsonl: the wall clock on entry (the rollout is
done) and on exit, after the judge's own work (the dump below and, on the first call, the layout query). The judge's
rollout time of rollout k is entry(k) - exit(k-1), so that work is never inside a timed window.

With SLIME_OPD_BENCH_DUMP=1 (rollout mode) it saves the rollout as $SLIME_OPD_BENCH_DIR/rollout_data/<id>.pt, in
slime's debug-dump format ({"rollout_id", "samples": [Sample.to_dict()]}), the input of the correctness gates.
On the first call it also asks every sglang engine behind the router for its server arguments (the parallel layout
that actually runs): $SLIME_OPD_BENCH_DIR/sglang_layout.json.
"""
import json
import os
import time

_FIRST = True


def _engines(args):
    import requests
    router = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    urls = []
    for path in ("/workers", "/list_workers"):
        try:
            r = requests.get(router + path, timeout=30)
            if r.ok:
                data = r.json()
                items = data.get("workers", data.get("urls", [])) if isinstance(data, dict) else data
                urls = [w.get("url") if isinstance(w, dict) else w for w in items]
                if urls:
                    break
        except Exception:  # noqa: BLE001 - try the other endpoint
            continue
    out = []
    for url in urls:
        info = None
        for path in ("/get_server_info", "/server_info"):
            try:
                r = requests.get(url + path, timeout=30)
                if r.ok:
                    info = r.json()
                    break
            except Exception:  # noqa: BLE001
                continue
        keep = ("tp_size", "dp_size", "pp_size", "ep_size", "base_gpu_id", "nnodes", "enable_dp_attention",
                "speculative_algorithm", "speculative_draft_model_path", "model_path")
        out.append({"url": url, **({k: (info or {}).get(k) for k in keep}), "answered": info is not None})
    return out


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    global _FIRST
    entry = time.time()
    d = os.environ.get("SLIME_OPD_BENCH_DIR")
    if not d:
        return False
    rec = {"rollout_id": rollout_id, "entry": entry, "reported_rollout_time": rollout_time,
           "samples": len(samples) if samples is not None else None}
    if _FIRST and not getattr(args, "debug_train_only", False):
        try:
            layout = {"engines": _engines(args), "rollout_num_gpus": getattr(args, "rollout_num_gpus", None),
                      "rollout_num_gpus_per_engine": getattr(args, "rollout_num_gpus_per_engine", None),
                      "colocate": getattr(args, "colocate", None)}
        except Exception as exc:  # noqa: BLE001 - recorded, judged by bench.py
            layout = {"error": repr(exc)}
        with open(os.path.join(d, "sglang_layout.json"), "w") as f:
            json.dump(layout, f, indent=1)
    _FIRST = False
    if os.environ.get("SLIME_OPD_BENCH_DUMP") == "1":
        import torch
        from slime.observability.rollout_data_utils import materialize_tensor_refs
        t0 = time.time()
        dumps = os.path.join(d, "rollout_data")
        os.makedirs(dumps, exist_ok=True)
        rows = [s.to_dict() for s in samples]
        for s in rows:
            s.pop("_queue_original_input", None)
        torch.save(materialize_tensor_refs({"rollout_id": rollout_id, "samples": rows}),
                   os.path.join(dumps, f"{rollout_id}.pt"))
        rec["dump_s"] = time.time() - t0
    rec["exit"] = time.time()
    with open(os.path.join(d, "rollout_clock.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return False
