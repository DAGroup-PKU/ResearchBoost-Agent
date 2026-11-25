# nanochat: D24 base pretraining (2 GPUs, tiers N0 and N1)

Reference: [karpathy/nanochat](https://github.com/karpathy/nanochat) @ `92d63d4`, the D24 run of `runs/speedrun.sh`
(depth 24, SSSL windows, sequence 2048, 16 sequences per GPU, 262,144 tokens per update: upstream's 1,048,576 on 8 GPUs
scaled by 2/8, tensorwise FP8, data:param ratio 8) on 2 Hopper GPUs with at least 130 GiB each (H200). Accepted: tier N0 (the candidate's 10
deterministic updates reproduce every parameter of the reference bit for bit) and tier N1 (every exact item identical,
the probe-loss trajectory on one frozen reference seed within 3 x the reference's seed-to-seed sigma). The judge is
`environment/workspace/bench/` (`bench.py`, `nanochat_entry.py`, `traj.py`).

## Assets

`RESEARCHBOOST_ASSETS_ROOT/nanochat` must hold, before the first trial (no asset service; provision it on each host
with `environment/asset-prep/prepare_nanochat_assets.sh <image> <assets root>`):

- `data/base_data_climbmix/`: ClimbMix shards `shard_00000` ... `shard_00011` and the validation shard `shard_06542`
- `data/tokenizer/`: `tokenizer.pkl` and `token_bytes.pt`, trained on them by upstream's `scripts.tok_train`
- optionally `pack/`: a preparation pack (`pack.json` and `bench/records/ref.json`, `bench/data/probe.pt`,
  `bench/traj/ref_runs.json`, `bench/traj/frozen.json` copied from the workspace of an earlier full preparation on the
  same GPU model and the same assets). With a pack, preparation records the reference once and requires it to be
  bitwise identical to the pack's record; the reference trajectories, the full check and the controls are skipped.

Preparation (`environment/bootstrap.py`) records the reference twice and stops when the two records differ: a host that
cannot reproduce the reference bitwise cannot judge this task. Set `RESEARCHBOOST_NANOCHAT_CONTROLS=0` to skip the two
acceptance controls of a full preparation.

## Audit

Agents have internet access. The task statement forbids the published optimized forks (AI4AI-for-Anything Faster-*);
after a trial, search its agent logs for them (`grep -r -i -E 'AI4AI-for-Anything|Faster-NanoChat' <trial>/agent`)
and void a trial that accessed them.
