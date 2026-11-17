"""Token forcing scorer of the student's rollout (after SpecPilot, arXiv:2611.02501 section 3.3, adapted to sampled
rollouts).

Teacher-forces the samples of rollout dumps through the STOCK student engine (the frozen reference sglang tree, TP2
engines like the rollout's, run by bench.py with the training environment) and writes, per response token, the stock
log-probability of the sampled token next to the log-probability the tree under test reported while sampling it
(scores.npz: stock, reported, lens, index, rollout_ids). With --topk K > 0 it also compares top-K log-probabilities
(not used by this recipe, which has no score centering: --topk 0).

usage: python -m bench.score --dumps DIR --ids 0 1 2 --model HF_DIR --out scores.npz [--topk 0]
"""
import argparse
import json
import os

import numpy as np
import torch


def load_samples(dump_dir, ids):
    out = []
    for rid in ids:
        d = torch.load(os.path.join(dump_dir, f"{rid}.pt"), weights_only=False)
        for s in d["samples"]:
            out.append((rid, s))
    return out


def as_array(x, dtype):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    return np.asarray(x, dtype=dtype)


def topk_compare(rep_ids, rep_lp, stock_top, k):
    """Per token: |reported ids & stock ids| / k; the sums of |reported - stock| and of reported - stock over the
    shared ids, and their count."""
    overlap = np.zeros(len(stock_top))
    absum, sgnsum, cnt = 0.0, 0.0, 0
    for j, entries in enumerate(stock_top):
        sto = {int(e[1]): float(e[0]) for e in (entries or [])[:k] if e is not None and e[0] is not None}
        rid, rlp = rep_ids[j], rep_lp[j]
        shared = [(float(v), sto[int(i)]) for i, v in zip(rid, rlp) if int(i) in sto]
        overlap[j] = len(shared) / k
        for a, b in shared:
            absum += abs(a - b)
            sgnsum += a - b
        cnt += len(shared)
    return overlap, absum, sgnsum, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    samples = load_samples(a.dumps, a.ids)
    import sglang as sgl

    ngpu = torch.cuda.device_count()
    engine = sgl.Engine(model_path=a.model, tp_size=a.tp, dp_size=max(1, ngpu // a.tp), skip_tokenizer_init=True,
                        mem_fraction_static=0.8, random_seed=0, disable_radix_cache=True, log_level="warning")
    stock, reported, lens, rids, idx = [], [], [], [], []
    tk = {"tokens": 0, "overlap_sum": 0.0, "abs_sum": 0.0, "signed_sum": 0.0, "abs_count": 0, "per_sample": {},
          "missing": 0}
    for b0 in range(0, len(samples), a.batch):
        chunk = samples[b0:b0 + a.batch]
        input_ids, starts = [], []
        for _, s in chunk:
            toks = list(s["tokens"])
            rl = int(s["response_length"])
            input_ids.append(toks)
            # sglang returns None for the token AT logprob_start_len: start one token earlier and drop that entry
            starts.append(len(toks) - rl - 1)
        extra = {"top_logprobs_num": a.topk} if a.topk > 0 else {}
        outs = engine.generate(input_ids=input_ids, sampling_params={"max_new_tokens": 1, "temperature": 1.0},
                               return_logprob=True, logprob_start_len=starts, **extra)
        for (rid, s), o in zip(chunk, outs):
            rl, sidx = int(s["response_length"]), int(s["index"])
            mi = o["meta_info"]
            lp = mi["input_token_logprobs"][1:]
            # entry j (after dropping the last prompt token) is log p(response token j | prompt + response[:j])
            vals = np.array([x[0] if x[0] is not None else np.nan for x in lp[:rl]], dtype=np.float64)
            if len(vals) != rl:
                raise RuntimeError(f"sample {sidx}: {len(vals)} scored tokens, expected {rl}")
            stock.append(vals)
            reported.append(np.asarray(s["rollout_log_probs"][:rl], dtype=np.float64))
            lens.append(rl)
            rids.append(rid)
            idx.append(sidx)
            if a.topk <= 0:
                continue
            top = (mi.get("input_top_logprobs") or [])[1:rl + 1]
            rep_ids = as_array(s.get("rollout_topk_token_ids"), np.int64)
            rep_lp = as_array(s.get("rollout_topk_log_probs"), np.float64)
            if rep_ids is None or rep_lp is None or len(top) != rl:
                tk["missing"] += 1
                tk["per_sample"][sidx] = None
                continue
            rep_ids, rep_lp = rep_ids[-rl:], rep_lp[-rl:]
            if rep_ids.shape != (rl, a.topk) or rep_lp.shape != (rl, a.topk):
                tk["missing"] += 1
                tk["per_sample"][sidx] = {"shape": [list(rep_ids.shape), list(rep_lp.shape)]}
                continue
            ov, absum, sgnsum, cnt = topk_compare(rep_ids, rep_lp, top, a.topk)
            tk["tokens"] += rl
            tk["overlap_sum"] += float(ov.sum())
            tk["abs_sum"] += absum
            tk["abs_count"] += cnt
            tk["signed_sum"] += sgnsum
            tk["per_sample"][sidx] = {"overlap": float(ov.mean()), "mean_abs": absum / max(cnt, 1),
                                      "mean_signed": sgnsum / max(cnt, 1)}
        print(f"scored {len(lens)}/{len(samples)} samples", flush=True)
    engine.shutdown()
    np.savez_compressed(a.out, stock=np.concatenate(stock), reported=np.concatenate(reported),
                        lens=np.array(lens), rollout_ids=np.array(rids), index=np.array(idx))
    tk["overlap"] = tk["overlap_sum"] / max(tk["tokens"], 1)
    tk["mean_abs"] = tk["abs_sum"] / max(tk["abs_count"], 1)
    tk["mean_signed"] = tk["signed_sum"] / max(tk["abs_count"], 1)
    if a.topk > 0:
        with open(os.path.splitext(a.out)[0] + ".topk.json", "w") as f:
            json.dump(tk, f, indent=1)
    print(f"scored {len(lens)} samples, {sum(lens)} tokens -> {a.out}")


if __name__ == "__main__":
    main()
