# How To Run

[← Back to README](../README.md#how-to-run) · [Eval Protocol](eval-protocol.md)

## Quick start

On a Linux host with Docker, Docker Compose, the NVIDIA container toolkit and the task's GPUs:

```bash
git clone --branch bench --single-branch https://github.com/DAGroup-PKU/ResearchBoost-Agent.git RSI-Infra-Bench
cd RSI-Infra-Bench
pip install harbor==0.24.0
cp researchboost.env.example researchboost.env      # set RESEARCHBOOST_ASSETS_ROOT and RESEARCHBOOST_SCRATCH_ROOT
# provision the task's assets (see Assets below), then:
harbor run -p tasks -i llava --env-file researchboost.env -a <agent> -m <model> -n 1
harbor view ./jobs
```

`-i` selects tasks by directory name (`llava`, `yoco`, `nanochat`, `bagel_sft`, `r3`, `gaussian_splatting`,
`slime_grpo`, `slime_opd`, or a glob). `-a` is any Harbor agent; the task statement is the agent's whole prompt.

## Hardware

Hopper GPUs (compute capability 9.x) with at least 130 GiB each (H200 class); the counts are in the [Eval Protocol task table](eval-protocol.md#tasks-and-numerical-equivalence) and in
each `task.toml`. Startup refuses other GPU types and any other GPU count. The GPUs are the trial's for its whole
duration: the agent works on them and the judge's final check runs on them in the same container.

Run **one trial per host at a time**. The judges time the training loop; a timed window disturbed by other work makes
the verdict INVALID (to be judged again), and a task's scratch directory is shared by its trials. CPU-heavy work on
the same host (other trials, builds, large file walks) shifts timings by several percent.

## Images

**Images are built on site.** There are no prebuilt or registry images: every `harbor run` of a task builds its image
from `tasks/<task>/environment/Dockerfile` (fast when the builder still holds the layer cache). The CUDA base image is
pinned by digest, and so are the torch wheels the judge was formulated on (a constraints file applies to every
install). Everything else is upstream's own install with upstream's own pins, resolved at build time: versions of the
other packages may drift between builds. Each image lists what it installed in
`/opt/researchboost/installed-packages.txt`, the build fails if the training entrypoint does not import, and the judge
records the library versions of every run.

The builds need the CUDA base image, the Ubuntu archive, pypi.org, download.pytorch.org and GitHub (upstream code at
the pinned commits, some release wheels). The two slime tasks start from slime's official image, pinned by digest
(build argument `BASE_IMAGE`); the Dockerfile checks that it is the pinned build.

- **Mirrors** go in the build arguments `PIP_INDEX_URL`, `TORCH_INDEX_URL` and, where the Dockerfile has it,
  `GITHUB_MIRROR` (git clones only; every commit is still checked), through a run-level Compose override passed with
  `--extra-docker-compose`:

  ```yaml
  services:
    main:
      build:
        args:
          PIP_INDEX_URL: https://<your mirror>/simple/
  ```

  Behind a proxy, set `proxies` in the Docker client configuration (`~/.docker/config.json`).
- **Prebuild with the docker builder.** A trial's `docker compose build` reads the Docker daemon's own build cache. To
  build ahead of time (for example to retry a flaky network fetch until it succeeds), build with the same arguments
  and the default `docker` builder (`docker buildx build --builder default ...`): a build in a separate
  `docker-container` BuildKit builder fills a different cache, and the trial then builds every layer again.
- Some builds are long (Transformer Engine and other CUDA extensions compile in the image); `task.toml` allows one to
  two hours (`build_timeout_sec`).

## Assets

Data and weights are not in the images. They live in `<assets root>/<task>` on the host (`RESEARCHBOOST_ASSETS_ROOT`)
and are mounted read-only; every file is pinned by revision and checksum.

| Task | How to provision | Size |
| --- | --- | --- |
| `llava`, `yoco`, `bagel_sft` | automatic: each trial's Compose `assets` service downloads and verifies whatever is missing; `python scripts/prepare_assets.py --task <task>` does the same ahead of time (`--offline` only checks) | about 17 GB, 4.3 GB, 30 GB |
| `gaussian_splatting` | `python scripts/prepare_assets.py --task gaussian_splatting`, before the first trial | about 2 GB |
| `nanochat` | `tasks/nanochat/environment/asset-prep/prepare_nanochat_assets.sh <image> <assets root>` (uses the task image to train upstream's tokenizer), before the first trial | ClimbMix shards and the tokenizer |
| `slime_grpo`, `slime_opd` | `tasks/<task>/environment/asset-prep/prepare_<task>_assets.sh` (public files at pinned revisions, checked against `SHA256SUMS`, then the checkpoint converted for Megatron with the task image), before the first trial | Qwen3.5-9B (and Qwen3.5-27B for `slime_opd`) |
| `r3` | by hand, before the first trial: the BAGEL-7B-MoT snapshot, the Mask2Former GenEval detector and the open_clip ViT-L-14 weights, as listed in [tasks/r3/README.md](../tasks/r3/README.md) | — |

`RESEARCHBOOST_SCRATCH_ROOT/<task>` is the trial's `/tmp`; it is emptied when a trial starts.

## Expected durations

| Task | Startup preparation (full) | Final check of an unchanged candidate |
| --- | --- | --- |
| `llava` | about 1 h 54 min (2 GPUs) | about 6.5 min |
| `yoco` | about 1 h 30 min (2 GPUs) | about 5 min |
| `bagel_sft` | about 3 h 23 min, about 50 min more with its gate 2b (4 GPUs) | about 13.5 min |
| `gaussian_splatting` | about 100 min (1 GPU); about 14 min with a pack | — |
| `slime_grpo` | about 7 h 15 min (4 GPUs); about 2 h 30 min with a pack | about 75 min (three full-loop runs) |

The other tasks' durations are in their READMEs or `prepare.log`. A candidate that is not bitwise identical adds the
statistical gate's trajectories to its final check. `task.toml` allows **10 h for the agent** (`[agent] timeout_sec`),
6 to 8 h for startup (`yoco` 12 h, `slime_grpo` 16 h, `slime_opd` 20 h) and 4 h for the verifier (`yoco` 6 h,
`slime_grpo` 5 h, `slime_opd` 7 h).
