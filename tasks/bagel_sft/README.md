# bagel_sft: BAGEL-7B-MoT supervised fine-tuning (4 GPUs, tiers N0 and N1)

Reference: [ByteDance-Seed/Bagel](https://github.com/ByteDance-Seed/Bagel) @ `a2fa77d`, upstream's fine-tuning command
(TRAIN.md: the released BAGEL-7B-MoT checkpoint for the model and its EMA, 10240 packed tokens per GPU, lr 2e-5,
upstream's bagel_example data and data/configs/example.yaml) on 4 Hopper GPUs with at least 130 GiB each (H200),
FSDP sharded over the four (`--num_shard 4`; upstream's 8 GPUs keep the same per-GPU load). Accepted: tier N0 (the
candidate's 3 deterministic steps reproduce every shard of the parameters, gradients, AdamW moments and EMA bit for
bit) and tier N1 (every exact item identical, a paired one-seed trajectory whose CE, MSE, gradient-norm and
parameter-displacement endpoints are each within three times the reference's seed-to-seed sigma); both must also pass
gate 2b, a short deterministic run on held-out interleaved data (`bench/data_config_interleave.yaml`: multi-image
editing chains, which the recipe's data never packs) whose per-token MSE, with the judge's own noise, must move no
more on the interleaved samples than on the others (`bench/conform.py`). The judge is
`environment/workspace/bench/` (`bench.py`, `bagel_entry.py`, `conform.py`, `traj.py`).

## Assets

`RESEARCHBOOST_ASSETS_ROOT/bagel_sft` holds the following; the task's assets service downloads what is missing before
the trial starts (about 30 GB), and `python scripts/prepare_assets.py --task bagel_sft` does the same ahead of time
(`--model-dir <BAGEL-7B-MoT directory>` copies the checkpoint from a local directory instead):

- `BAGEL-7B-MoT/`: the released checkpoint (Hugging Face `ByteDance-Seed/BAGEL-7B-MoT` @ `5019f57d`): `ema.safetensors`
  (SHA-256 `0b41c438...`, checked at startup), `ae.safetensors`, the configuration and tokenizer files;
- `data/bagel_example/`: upstream's example data (`bagel_example.zip` from TRAIN.md, SHA-256 `5a2d18a8...`), unpacked:
  `t2i/`, `editing/seedxedit_multi/`, `editing/parquet_info/`, `vlm/images/`, `vlm/llava_ov_si.jsonl`.

Preparation (`environment/bootstrap.py`) records the reference twice and stops when the two records differ: a host that
cannot reproduce the reference bitwise cannot judge this task. Its gate-2b run is also made twice and must reproduce.

## Audit

Agents have internet access. The task statement forbids the published optimized forks (AI4AI-for-Anything Faster-*);
after a trial, search its agent logs for them (`grep -r -i -E 'AI4AI-for-Anything|Faster-BAGEL' <trial>/agent`) and
void a trial that accessed them.
