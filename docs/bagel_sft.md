# bagel_sft

Make BAGEL-7B-MoT supervised fine-tuning faster per optimizer step. The task statement is
[`tasks/bagel_sft/instruction.md`](../tasks/bagel_sft/instruction.md).

## Workload

Upstream [ByteDance-Seed/Bagel](https://github.com/ByteDance-Seed/Bagel) at `a2fa77d`, `train/pretrain_unified_navit.py`
with upstream's fine-tuning command (TRAIN.md): the released BAGEL-7B-MoT checkpoint (about 14.6B parameters: a Qwen2
mixture-of-transformers language model with an understanding and a generation expert, a SigLIP vision encoder, the
FLUX VAE frozen) initializes both the model and its EMA; packed sequences of 10240 expected tokens per GPU (at most
11520, samples up to 10240); learning rate 2e-5 with upstream's default 2000-step warmup; AdamW 0.9/0.95, eps 1e-15;
EMA 0.9999; clipping 1.0; one data worker; upstream's data configuration (`data/configs/example.yaml`). Upstream runs 8
GPUs; here 4 × H200 with the per-GPU load kept (a step trains on 4 × 10240 tokens instead of
8 × 10240) and FSDP `HYBRID_SHARD` over the four (`--num_shard 4`). The frozen arguments are `bench/recipe.py`. Timing:
5 warmup steps and 2 windows of 10 steps.

## Data and weights

`<assets root>/bagel_sft`, filled by the task's assets service when missing (or ahead of time with
`scripts/prepare_assets.py --task bagel_sft`; `environment/asset-prep/scripts/prepare_bagel_assets.py`):

| Path | Source | Check |
| --- | --- | --- |
| `BAGEL-7B-MoT/` | HF `ByteDance-Seed/BAGEL-7B-MoT` @ `5019f57d` | `ema.safetensors` SHA-256 at startup; every file a frozen input |
| `data/bagel_example/` | `bagel_example.zip` from upstream's TRAIN.md (SHA-256 `5a2d18a8...`): `t2i/` (10 parquet chunks), `editing/seedxedit_multi/` (10 chunks) and its index, `vlm/llava_ov_si.jsonl` (1000 rows) with `vlm/images/` | every file a frozen input |

Upstream's `data/dataset_info.py` names placeholder paths (`your_data_path/bagel_example/...`), and its editing index
`seedxedit_multi_nas.json`, which the archive ships as `editing/parquet_info/seedxedit_multi.json`. The judge's entry
points the three entries at the asset mount when the module is imported and writes the index with relocated keys into
the run's TMPDIR; the tree is not changed.

## Image

Built on site from `tasks/bagel_sft/environment/Dockerfile`, installed as upstream's README says:
`nvidia/cuda:12.4.1-devel-ubuntu22.04` (pinned by digest), Python 3.10 from Ubuntu, `/opt/bagel-env` with
upstream's `requirements.txt` (torch 2.5.1+cu124, torchvision 0.20.1, transformers 4.49.0, numpy 1.24.4, safetensors
0.4.5, decord 0.6.0, ...; `opencv_python` replaced by the headless build of the same version) and flash-attn 2.5.8
built from source for sm_90 (no prebuilt wheel for torch 2.5; about 12 minutes of the build). `pip check` passes.
`/opt/candidate-env` and `/submission/bagel_sft` (the reference on branch `opt`) as in the other tasks.

## Startup preparation

`environment/bootstrap.py` runs as root before the container reports ready:

1. Exactly 4 GPUs of compute capability 9.x with at least 130 GiB; the assets present.
2. The reference recorded twice in deterministic mode; the two records must be bitwise identical
   (`bench/records/ref.json`).
3. Reference trajectories on seeds 1–6, seed 1 again (it must reproduce every value), the frozen δ (`bench/traj/`);
   the reference's gate-2b run on the held-out interleaved data, twice (identical; `bench/conform/ref.json`).
4. Frozen inputs (`bench/`, the checkpoint, the data); the EMA weights must have the formulated SHA-256.
5. The unchanged tree's quick check (PASS, bitwise).
6. The reference timed three times (`bench.py baseline`), their mean and spread frozen as `task/timing/reference.json`.
7. The unchanged tree's full check (UNAVAILABLE, no_benefit) and the controls:

| Control | Edit (in `train/pretrain_unified_navit.py` unless named) | Mode | Expected |
| --- | --- | --- | --- |
| `beta2` | AdamW betas (0.9, 0.999) | `--quick` | INCORRECT (exact items: the hyper-parameters) |
| `ema_skip` | the EMA update skipped | `--skip-timing --final` | INCORRECT (statistical gate) |
| `ce_order` | CE normalized as `(ce / total).sum() * world` instead of `ce.sum() * world / total` | `--skip-timing --final` | PASS (bitwise: on 4 GPUs the factor 4 is exact) |
| `rms_div` | RMSNorm (`modeling/qwen2/modeling_qwen2.py`) as `x / sqrt(v + eps)` instead of `x * rsqrt(v + eps)` | `--skip-timing --final` | PASS (equivalent, statistical gate) |
| `noise_visible` | the attention masks in `modeling/bagel/bagel.py` with every earlier key of a sample visible (noise spans no longer hidden from later splits) | `--skip-timing --final` | INCORRECT at gate 2b (nonconforming); bitwise identical in gate 2 |
| `merge_full` | the attention masks in `modeling/bagel/bagel.py` with each bidirectional span also seeing the bidirectional span right after it | `--skip-timing --final` | INCORRECT at gate 2b (nonconforming); bitwise identical in gate 2 |

With a preparation pack (`tasks/bagel_sft/prep-pack/ws4/` in the repository, mounted read-only, else
`<assets root>/bagel_sft/pack/`: the reference record, trajectories and gate-2b run of an earlier full preparation on
the same GPU model and assets: `bench/records/ref.json`, `bench/traj/ref_runs.json`, `bench/traj/frozen.json`,
`bench/conform/ref.json` and a `pack.json`), steps 2–3 restore it and the reference is recorded once and run once on
the interleaved data; both must be bitwise identical to the pack's, else the full preparation runs. A pack without
`bench/conform/ref.json` (made before gate 2b) is not used.
Steps 4–6 follow; the full check and the controls stand from the pack's preparation.

## Judge notes

The judge follows the rules of the other tasks (timing plan, frozen reference timing, noise rules in `--final` only,
CPU binding, `time` without a frozen reference running the reference once) and the gates of the original
optimization work on this workload: exact items and a bitwise record of parameters, gradients, AdamW moments and EMA;
otherwise paired-seed trajectories with endpoints on CE, MSE, the gradient norm and the parameter displacements of
the ViT, the language model, the other parameters and the EMA (the displacements are what separates a learning-rate
change from noise; the losses alone do not). Differences from that work:

- 4 GPUs instead of 8, upstream's fine-tuning command instead of a smaller-batch recipe for 80 GB GPUs;
- trajectories of 150 steps (seeds: global seed 4396 + S, which also changes the data sampling, and the loaded
  weights perturbed by (1 + 2^-7 z) per parameter name), the endpoints over steps 50–100 and 100–150;
- 3 recorded steps (all within the learning-rate warmup);
- deterministic mode is strict: under `warn_only`, torch 2.5 keeps the non-deterministic backward of the
  memory-efficient SDPA kernel that upstream's language model calls per sample, and two records of the reference
  differ from step 0; strictly they are bitwise identical.

Measured on 2026-10-27 on 4 × H200: about 3 minutes from start to the first step (every rank builds the model and the
EMA in fp32 and loads the checkpoint into both); 2.38 s per step at production settings; a record run about 4
minutes; a 30-step trajectory about 5 minutes. The full preparation took 203 minutes (the reference timing frozen
at 2.3885 s/step, spread 0.7%; a 150-step trajectory about 15 minutes; the unchanged tree's full check 0.9996, no
benefit); `beta2`, `ema_skip` and `ce_order` as expected. `rms_div` was checked on the same machine from that
preparation's pack (startup 24 minutes): PASS, equivalent, on the first 3 seeds, every endpoint within its δ (the
largest offset 0.88 standard deviations of the reference's seeds, the ViT displacement over steps 50-100).

## Gate 2b: interleaved layouts

Why. Under the recipe's data configuration (upstream's `example.yaml`, 10240 / 11520 tokens) the editing samples are
never trained: a two-image edit (about 9.5k tokens at up to 1024 px) is drawn, buffered and then dropped because the
mandatory text-to-image and VLM samples are packed first; a three-image chain (about 19k tokens) exceeds the per-sample
limit. Every batch of gates 2 and 3 therefore holds only text-to-image (`[causal, noise]`, `[noise]`) and VLM
(`[full, causal, ...]`) layouts: a noise span is always the last split of its sample and no two bidirectional spans
are adjacent. An attention implementation whose mask is wrong on other layouts (noise spans visible to later splits,
adjacent image spans merged, ...) is bitwise identical there and passes gates 2 and 3. Measured on 640 recipe batches
(4 ranks x 160): 0 editing samples packed; planners with either fault: 0 wrong mask edges there, wrong on 155-353 of
400 batches of a configuration that packs editing chains.

What. `bench/data_config_interleave.yaml`: upstream's groups and data, the editing group mandatory and first with its
images at most 512 px (two- and three-image chains fit the recipe's limits), VLM mandatory with images at most 490 px
(upstream's loop needs CE tokens in every batch; both mandatory samples fit 11520), text-to-image as upstream.
`bench/bagel_entry.py --mode conform`: deterministic, `conformance.steps` steps from the released checkpoint (all in
the warmup: the weights move by about 1e-8 relative per step, far below bf16 resolution); per `Bagel.forward` the
per-token MSE (mean over the 64 latent channels) and CE; the batches' SHA-256; for the reference also each MSE
token's pool from its own dense masks (`bench/conform.py interleaved`: N = a row that cannot see an earlier key, i.e.
a noise span followed by another split; J = two adjacent bidirectional spans; B = the rest).

The noise. Upstream draws two random tensors per step that the MSE depends on: the VAE posterior sample
(`DiagonalGaussian`, inside `AutoEncoder.encode`) and the latent noise (inside `Bagel.forward`). Legitimate trees
reorder their RNG use (opus and sonnet encode the next batch's images on a side stream during the backward), so the
global RNG state at those calls is not the reference's. The judge therefore answers the noise requests itself:
`conform.Injector` wraps `torch.randn_like`, `torch.randn` and `Tensor.normal_` before the tree is imported; while a
judged call runs (the k-th `AutoEncoder.encode` or the k-th `Bagel.forward` of a rank; thread-local, so a prefetch
thread outside a judged call is not affected), every request of the noise's size is answered from a generator seeded
with (kind, rank, k, draw index). Everything else draws from the global RNG as before. The run records which draws it
injected; a candidate whose calls do not draw the same tensors (count and size) as the reference's is an ERROR
(`noise_not_pinned`), never compared with the wrong noise. The timesteps are drawn by the data pipeline and part of the
batch, whose SHA-256 must match the reference's (exact item).

The statistic. R_P = sqrt(sum over pool P of (mse_cand - mse_ref)^2 / sum of mse_ref^2), over all ranks and steps.
PASS when R_N and R_J are both <= `ratio_max` x max(R_B, `floor`). Kernel, fusion and reduction-order changes move all
pools alike (their ratio is about 1); a layout-dependent mask error moves N or J only. The two pools are tested
separately so that a fault confined to multi-turn chains is not diluted by the many single-turn edits with adjacent
image spans. `floor` covers a tree that is bitwise identical on B but not on N or J (a kernel used only for some
layouts). It runs for bitwise trees too: a mask fault confined to interleaved layouts is bitwise identical on gate 2.

Threshold, from data (2026-11-07, 4 Hopper GPUs of 141 GB, 16 steps = 64 batches; MSE tokens N 32,768, J 62,464, B 156,416). Ratio =
R / max(R_B, floor):

| Tree | Kind | R_B | N ratio | J ratio | Gate 2b | Check |
| --- | --- | --- | --- | --- | --- | --- |
| `ce_order` control | same function, bitwise | 0 | 0 | 0 | pass | PASS N0 |
| `ema_skip` control | EMA only (forward unchanged) | 0 | 0 | 0 | pass | INCORRECT (gate 3) |
| `rms_div` control | same function, other rounding | 0.0046 | 1.05 | 0.98 | pass | PASS N1 |
| reference + one extra CUDA `torch.randn` per step | RNG order changed | 0 | 0 | 0 | pass | PASS N1 |
| opus (accepted 2.885x; fused MoT, split-range FlashAttention, VAE ahead on a side stream, fused VAE) | legitimate | 0.0129 | 2.26 | 1.93 | pass | PASS N1 |
| sonnet (accepted 2.309x; FA3 per-split varlen with dense-mask fallback, VAE prefetch, sample migration) | legitimate | 0.0052 | 1.27 | 0.97 | pass | PASS N1 |
| haiku (accepted 1.602x; packed FlexAttention, compiled RMSNorm/RoPE/MLP) | legitimate | 0.0047 | 1.12 | 1.02 | pass | PASS N1 |
| astra (accepted 1.559x; exact spans to FlashAttention-2, single-replica FSDP, compiled ops) | legitimate | 0.0058 | 1.71 | 1.12 | pass | PASS N1 |
| `noise_visible` control | mask wrong on N layouts | 0.0042 | 20.1 | 0.95 | INCORRECT | INCORRECT |
| `merge_full` control | mask wrong on J (and N) layouts | 0.0042 | 83.7 | 15.4 | INCORRECT | INCORRECT |

`ratio_max` stays at 3, the stricter end: the largest legitimate value is 2.26 and the smallest fault 15.4, so every verdict above is the same at 3 and at 5. opus's 2.26 is not a mask
effect: its excess is in the editing samples of lowest MSE (median relative deviation per noise span 0.17 against
0.013-0.017 for the other trees; on the other three quarters about 2x the others), where the target is predicted from
the clean conditioning image, whose VAE latent its fused GroupNorm changes by 1.9% (upstream's own bf16 latent is 4.2%
from fp32, opus's 4.2%): numerics only editing samples see. A tree must also draw the noise where upstream does, else
the gate is an ERROR; every tree above did, the VAE prefetching ones included (their noise reached the judged calls).

Cost (measured, 4 Hopper GPUs of 141 GB): one gate-2b run 3.5-6 minutes (0.23-0.40 GPU-h) per check, check `--n1` and the final
evaluation; the reference's gate-2b run twice 8 minutes in the preparation; the two mask controls 9-12 minutes each
(they stop at gate 2b); gate 2b in the four older non-quick controls about 5 minutes each. The preparation grows by
about 50 minutes.

## Verifier

As for the other tasks (`tests/verify.py`): an invalid `switches.json` is a rejection, every `agent` process is
stopped, `bench.py check --final` runs as `agent` with the training environment's Python, the judge's output is kept
under `/logs/verifier`, and `check.json` becomes the reward as in the [README](../README.md#scoring).
