"""Gate 2b (interleaved-layout conformance): the pieces shared by the judge's entry (bagel_entry.py, inside the
training processes) and the bench (bench.py, the verdict).

What the gate measures. Upstream's attention mask of a packed sample (data/data_utils.py,
prepare_attention_mask_per_sample) depends on the sample's split layout: causal text, bidirectional image spans, noise
spans hidden from every later split. The recipe's data (data/configs/example.yaml under the frozen token limits) only
ever packs layouts where these rules coincide with simpler ones: a noise span is always the last split of its sample
and no two bidirectional spans are adjacent (the editing samples never fit; see docs/bagel_sft.md). An attention
implementation that is exact on those layouts can be wrong on the others and still pass gates 2 and 3. Gate 2b runs
the tree on a held-out data configuration of the same bagel_example data (bench/data_config_interleave.yaml: the
editing group mandatory, its images at most 512 px, so two- and three-image editing chains fit the recipe's limits)
for a few deterministic steps from the released checkpoint and compares the per-token flow-matching MSE of every
batch with the frozen reference's.

The noise is the judge's. The per-token MSE depends on two random draws: the VAE posterior sample (upstream's
AutoEncoder.encode, DiagonalGaussian: torch.randn_like) and the latent noise (Bagel.forward: torch.randn_like). A tree
may legitimately reorder its RNG consumption (VAE prefetch on a side stream, ...), so the gate does not rely on the
global RNG: while a judged call runs, every torch.randn_like / torch.randn / Tensor.normal_ request the call makes for
the noise is answered with a tensor the judge draws itself from a generator seeded with (kind, rank, ordinal of the
call): the k-th AutoEncoder.encode of a rank and the k-th Bagel.forward of a rank receive the same noise in the
reference and in the candidate, whatever else consumed the RNG in between. The timesteps are part of the batch (drawn
by the data pipeline) and the batches must be identical to the reference's (SHA-256 of every field).

The statistic. The batches' MSE tokens fall in pools, from the reference's own masks: N, the tokens of samples with
a noise span followed by another split; J, of samples with two adjacent bidirectional spans (and no such noise span);
B, all others (layouts the recipe data has). R_N, R_J and R_B are the relative L2 distances of the candidate's
per-token MSE from the reference's over each pool. Kernel and reduction-order changes move every pool alike; an
attention mask that is wrong only on interleaved layouts moves N or J alone. The gate fails when R_N or R_J exceeds
ratio_max x max(R_B, floor) (bench.json conformance). N and J are tested separately so that a fault confined to one
layout is not diluted by the tokens of the other."""
POOLS = ('n', 'j')
import hashlib
import math
import threading

LATENT, VAE = 'latent', 'vae'


# ---------------------------------------------------------------- inside the training processes (torch)

class Injector:
    """Answers the noise requests of the judged calls (thread-local, so a side thread that is not inside a judged
    call draws its own random numbers). install() wraps torch.randn_like, torch.randn and torch.Tensor.normal_ once,
    before the tree is imported."""

    def __init__(self, torch, rank):
        self.torch, self.rank = torch, rank
        self.local = threading.local()
        self.ordinal = {LATENT: 0, VAE: 0}
        self.log = []           # (kind, ordinal, numels of the injected tensors)
        self.last = {}          # kind -> numels injected in this thread's last call of that kind

    def install(self):
        torch, inj = self.torch, self
        orig_like, orig_randn, orig_normal = torch.randn_like, torch.randn, torch.Tensor.normal_

        def randn_like(t, *a, **k):
            z = inj.answer(t.numel(), k.get('dtype') or t.dtype, k.get('device') or t.device)
            return orig_like(t, *a, **k) if z is None else z.view(t.shape)

        def randn(*size, **k):
            if k.get('generator') is None and size:
                shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else size
                if all(isinstance(s, int) for s in shape):
                    z = inj.answer(math.prod(shape), k.get('dtype') or torch.get_default_dtype(),
                                   k.get('device') or 'cpu')
                    if z is not None:
                        return z.view(shape)
            return orig_randn(*size, **k)

        def normal_(t, mean=0.0, std=1.0, *a, **k):
            if k.get('generator') is None and not a:
                z = inj.answer(t.numel(), t.dtype, t.device)
                if z is not None:
                    with torch.no_grad():
                        t.copy_(z.view(t.shape) * std + mean)
                    return t
            return orig_normal(t, mean, std, *a, **k)
        torch.randn_like, torch.randn, torch.Tensor.normal_ = randn_like, randn, normal_

    def call(self, kind, fn, numel=None):
        """Run fn() as the next judged call of this kind: its noise requests (all of them for 'vae', those of `numel`
        elements for 'latent') get the judge's noise."""
        k = self.ordinal[kind]
        self.ordinal[kind] += 1
        state = {'kind': kind, 'ordinal': k, 'numel': numel, 'draws': []}
        prev = getattr(self.local, 'state', None)
        self.local.state = state
        try:
            return fn()
        finally:
            self.local.state = prev
            self.log.append((kind, k, state['draws']))
            self.last[kind] = state['draws']

    def answer(self, numel, dtype, device):
        state = getattr(self.local, 'state', None)
        if state is None or not dtype.is_floating_point or (state['numel'] is not None and numel != state['numel']):
            return None
        torch = self.torch
        g = torch.Generator(device=device)
        g.manual_seed(seed(state['kind'], self.rank, state['ordinal'], len(state['draws'])))
        state['draws'].append(numel)
        return torch.randn(numel, generator=g, device=device, dtype=torch.float32).to(dtype)


def seed(kind, rank, ordinal, draw):
    return int.from_bytes(hashlib.sha256(f'conform:{kind}:{rank}:{ordinal}:{draw}'.encode()).digest()[:7], 'big')


def interleaved(mask):
    """The pool of a sample from its dense mask (prepare_attention_mask_per_sample: 0 = attend, -inf = masked): 'n'
    when a row does not see an earlier key (a noise span followed by another split), else 'j' when two bidirectional
    spans are adjacent (a run of rows sharing their last visible key, followed directly by another), else '0' (a
    layout the recipe data has)."""
    import torch
    allowed = mask == 0
    n = allowed.shape[0]
    if n == 0:
        return '0'
    if (~allowed).tril(-1).any():
        return 'n'
    rows = torch.arange(n, device=mask.device)
    last = torch.where(allowed, rows, torch.full_like(rows, -1)).amax(1).tolist()
    i, previous_bidirectional_end = 0, None
    while i < n:
        end = last[i] + 1
        if end - 1 > i:                         # rows i..end-1 form a bidirectional span
            if previous_bidirectional_end == i:
                return 'j'
            previous_bidirectional_end, i = end, end
        else:
            i += 1
    return '0'


def pool_flags(masks, sample_lens, mse_indexes):
    """The pool ('n', 'j' or '0') of every MSE token of a batch, as one string."""
    flags, starts, o = [], [], 0
    for m, n in zip(masks, sample_lens):
        flags.append(interleaved(m))
        starts.append((o, o + n))
        o += n
    out, j = [], 0
    for i in mse_indexes:
        while not (starts[j][0] <= i < starts[j][1]):
            j += 1
        out.append(flags[j])
    return ''.join(out)


# ---------------------------------------------------------------- the verdict (plain Python)

def statistic(reference, candidate):
    """reference, candidate: [[{'mse': [...], 'pool': 'nj0...'} per forward] per rank] (pools from the reference).
    Returns r[pool] (relative L2 of the per-token MSE difference over the pool; None for an empty pool), tokens[pool]
    and per-batch values."""
    keys = POOLS + ('0',)
    num, den, tokens, batches = ({k: 0.0 for k in keys}, {k: 0.0 for k in keys}, {k: 0 for k in keys}, [])
    for rank, (ref_rank, cand_rank) in enumerate(zip(reference, candidate)):
        for k, (r, c) in enumerate(zip(ref_rank, cand_rank)):
            bn, bd = {p: 0.0 for p in keys}, {p: 0.0 for p in keys}
            for x, y, p in zip(r['mse'], c['mse'], r['pool']):
                bn[p] += (y - x) ** 2
                bd[p] += x * x
            for p in keys:
                num[p] += bn[p]
                den[p] += bd[p]
                tokens[p] += r['pool'].count(p)
            batches.append({'rank': rank, 'forward': k, **{f'tokens_{p}': r['pool'].count(p) for p in keys},
                            **{f'r_{p}': math.sqrt(bn[p] / bd[p]) if bd[p] else None for p in keys}})
    r = {p: (math.sqrt(num[p] / den[p]) if den[p] else None) for p in keys}
    return {'r': {('b' if p == '0' else p): v for p, v in r.items()},
            'tokens': {('b' if p == '0' else p): v for p, v in tokens.items()}, 'batches': batches}


def verdict(stat, ratio_max, floor):
    """'conforming' when every interleaved pool (N, J) with tokens has r <= ratio_max x max(r_b, floor), else
    'nonconforming'. Adds bound and ratio[pool]."""
    scale = max(stat['r']['b'] or 0.0, floor)
    bound = ratio_max * scale
    ratio = {p: (stat['r'][p] / scale if stat['r'][p] is not None else None) for p in POOLS}
    tested = [p for p in POOLS if stat['r'][p] is not None]
    ok = bool(tested) and all(stat['r'][p] <= bound for p in tested)
    return dict(stat, bound=bound, ratio_max=ratio_max, floor=floor, ratio=ratio, tested=tested,
                verdict='conforming' if ok else 'nonconforming')
