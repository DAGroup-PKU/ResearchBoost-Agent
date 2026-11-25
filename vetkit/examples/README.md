# 示例题

`tasks/nanochat/` 和 `researchboost.env.example` 原样取自 [ResearchBoost-Agent](https://github.com/DAGroup-PKU/ResearchBoost-Agent) 的 `bench` 分支，commit `ec38bcda2339be6f4527dc762d5a39988959b00a`（2026-11-23）。只有一处改动：`tasks/nanochat/tests/verify.py` 按规程补充 C10 加固了判卷的 `PATH`（只用 root-only 的目录和程序，用 `/proc` 找、杀 agent 的进程，以 agent 身份跑判卷和 `git` 时传干净的 `PATH` 和 `GIT_CONFIG_GLOBAL=/dev/null`，存档提交历史后再清一次 agent 的进程）；判卷工作区（`environment/`）没动，所以判卷哈希和参考包 key 不变。以后以 bench 分支为准；这里的副本只是为了让 `vetkit/` 不用另外克隆就能跑通一遍。

为什么选 nanochat：
- 全部公开：数据是 HF 上的 ClimbMix 分片，tokenizer 用任务镜像在本地训练，权重从头训练；不依赖任何私有镜像源。
- 2 张卡，判卷是标准形态：逐位记录（N0）、统计等价（N1）、分窗口计时，`bench.py` 有标准接口（`record / compare / time / check`），`vetkit/container/tasks/nanochat.py` 是它的参考包模块，通用模块 `_generic.py` 也是照它写的。
- 结构最简单，新题照它抄（规程 C 段的“选模板”）。

另外几种形态（带 preparation pack、配方走不到的分支单加一道门、RL 的 token forcing、按整次运行计时）看 bench 分支里的 `llava` / `yoco`、`bagel_sft`、`slime_grpo` / `slime_opd`、`gaussian_splatting`。

资产：`tasks/nanochat/README.md` 和 bench 分支 `docs/how-to-run.md` 的 Assets 一节（`environment/asset-prep/prepare_nanochat_assets.sh <镜像> <资产根目录>`）。
