"""Judge-owned teacher replay (bench/launch.sh MODE=teacher): the teacher alone, on fixed token sequences.

For every rollout id k in [start, start + num) it loads the frozen rollout dump <dumps>/<k>.pt (prompt + response
token ids of the student's samples), clears the teacher's earlier answer, and scores the samples through the tree
under test's reward path exactly as slime does in a rollout and in training:
  - slime.rollout.rm_hub.async_rm(args, sample) per sample (with --custom-rm-path: the custom reward function, which
    asks the teacher server at args.rm_url for the log-probabilities of the sample's tokens), all samples of the
    rollout concurrently (or --concurrency at a time), then
  - the custom reward post-processing function (--custom-reward-post-process-path, slime's batch builder hook), which
    turns the teacher's answers into sample.teacher_log_probs.
args is an argparse.Namespace with the fields these functions read in the recipe: rm_url, custom_rm_path,
custom_reward_post_process_path, rollout_temperature, reward_key (None), rm_type (None).

Outputs in --out:
  teacher_clock.jsonl     one line per rollout: wall clock before the first request and after the post-processing
  teacher_logprobs.npz    per response token, the teacher log-probability (values), and per sample its length (lens),
                          sample index (index) and rollout id (rollout_ids), in rollout and sample-index order
With --profile-dir: the teacher server's torch profiler over the scoring of rollout start + 1 (warmup rollout first).
"""
import argparse
import asyncio
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch


def load_samples(path):
    from slime.utils.types import Sample
    d = torch.load(path, weights_only=False)
    out = []
    for row in d["samples"]:
        row = dict(row)
        row["reward"] = None
        row["teacher_log_probs"] = None
        out.append(Sample.from_dict(row))
    return sorted(out, key=lambda s: s.index)


async def score(args, samples, concurrency):
    from slime.rollout.rm_hub import async_rm
    sem = asyncio.Semaphore(concurrency) if concurrency and concurrency > 0 else None

    async def one(s):
        if sem is None:
            return await async_rm(args, s)
        async with sem:
            return await async_rm(args, s)
    return await asyncio.gather(*(one(s) for s in samples))


def profile(url, out_dir, start):
    import requests
    base = url.rsplit("/", 1)[0]
    r = requests.post(f"{base}/start_profile", json={"output_dir": out_dir, "num_steps": 64,
                                                     "activities": ["CPU", "GPU"]}, timeout=120)
    print(f"start_profile before rollout {start + 1}: {r.status_code}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--num", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--custom-rm-path", required=True)
    ap.add_argument("--custom-reward-post-process-path", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=0)
    ap.add_argument("--profile-dir")
    a = ap.parse_args()

    from slime.utils.misc import load_function
    args = SimpleNamespace(rm_url=a.url, custom_rm_path=a.custom_rm_path,
                           custom_reward_post_process_path=a.custom_reward_post_process_path,
                           rollout_temperature=a.temperature, reward_key=None, rm_type=None)
    post = load_function(a.custom_reward_post_process_path)
    os.makedirs(a.out, exist_ok=True)
    values, lens, index, rids = [], [], [], []
    loop = asyncio.new_event_loop()
    for k in range(a.start, a.start + a.num):
        samples = load_samples(os.path.join(a.dumps, f"{k}.pt"))
        if a.profile_dir and k == a.start + 1:
            profile(a.url, a.profile_dir, a.start)
        t0 = time.time()
        rewards = loop.run_until_complete(score(args, samples, a.concurrency))
        for s, r in zip(samples, rewards):
            s.reward = r
        post(args, samples)
        t1 = time.time()
        with open(os.path.join(a.out, "teacher_clock.jsonl"), "a") as f:
            f.write(json.dumps({"rollout_id": k, "start": t0, "end": t1, "samples": len(samples),
                                "tokens": int(sum(len(s.tokens) for s in samples))}) + "\n")
        for s in samples:
            t = s.teacher_log_probs
            t = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t if t is not None else [])
            values.append(t.astype(np.float64))
            lens.append(len(t))
            index.append(int(s.index))
            rids.append(k)
        print(f"rollout {k}: {len(samples)} samples scored in {t1 - t0:.2f} s", flush=True)
    np.savez_compressed(os.path.join(a.out, "teacher_logprobs.npz"),
                        values=np.concatenate(values) if values else np.zeros(0), lens=np.array(lens),
                        index=np.array(index), rollout_ids=np.array(rids))


if __name__ == "__main__":
    main()
