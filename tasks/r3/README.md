# r3: R3 Tree-RL of BAGEL-7B-MoT (4 GPUs, bitwise only)

Reference: [sen-ye/R3](https://github.com/sen-ye/R3) @ `761b0b0`, the GenEval variant of its tutorial, on 4 Hopper GPUs
with at least 130 GiB each (H200), one policy group of 4 ranks (upstream uses 8 GPUs; `--num_shard 4`,
decided 2026-11-05; the reference peaks at about 131 GiB per GPU). Only tier N0 is accepted: the candidate's 14 deterministic steps must reproduce the
reference record bit for bit (final parameters and Adam state as full tensors, learning rates, every reward request).
The judge is `environment/workspace/bench/` (`bench.py`, `r3_entry.py`, `reward_app.py`); the GenEval reward service
(yifan123/reward-server @ `8245af2`) is part of the judge and runs in its own environment in the image.

## Assets

`RESEARCHBOOST_ASSETS_ROOT/r3` must hold, before the first trial (no asset service; provision it on each host):

- `BAGEL-7B-MoT/`: `huggingface_hub.snapshot_download("ByteDance-Seed/BAGEL-7B-MoT", revision="5019f57d168e5816e8f3f701b17cc816bb7cf24b", local_dir=...)`
- `mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth`: the OpenMMLab checkpoint
  `mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth`
- `hf/hub/`: an HF hub cache with `timm/vit_large_patch14_clip_224.openai` (`*.json`, `open_clip_pytorch_model.bin`,
  `open_clip_model.safetensors`), the open_clip ViT-L-14 `openai` weights the reward service loads

Preparation (`environment/bootstrap.py`) records the reference twice and stops when the two records differ: a host that
cannot reproduce the reference bitwise cannot judge this task.

## Audit

Agents have internet access. The task statement forbids the published optimized forks (AI4AI-for-Anything Faster-*);
after a trial, search its agent logs for them (`grep -r -i -E 'AI4AI-for-Anything|Faster-(R3|BAGEL)' <trial>/agent`)
and void a trial that accessed them.
