# rsi-bench-vetkit

**简体中文** · [English](README.en.md)

把一个训练仓库做成 [RSI-Infra Bench](https://github.com/DAGroup-PKU/ResearchBoost-Agent/tree/bench)（ResearchBoost-Agent 的 `bench` 分支）的一道题，并验它该不该进 bench：规程、一个 Claude Code skill，和在自己机器上就能跑的判卷侧工具。

| 目录 | 是什么 |
|---|---|
| [`skills/rbb-vet/`](skills/rbb-vet/SKILL.md) | 验题 skill。规范是 [`references/protocol.md`](skills/rbb-vet/references/protocol.md)：A 案头、B 上限和基线强度、C 打包和验判卷、D pilot、E 收尾，每段的检查、阈值和打分卡，加《实践补充》里的通用规则。`references/guide.md` 是长版做法，`references/lessons.md` 是常见的坑；`scripts/` 里有实测峰值（`peaks.py`）、上限 T0 / T1（`bound.py`）、误拒率（`frr.py`）和打分卡（`scorecard.py`） |
| [`vetkit/`](vetkit/README.md) | 判卷侧工具，单机就能用：题目 linter（`onboard_check.py`）、泄漏扫描（`leak_guard.py`）、指定卡和 NUMA 绑核直接跑 Harbor trial（`run_trial.py`）、参考包的建包 / 验证 / 干净容器复判（`refpack.py` 和 `container/`）；一道示例题（nanochat，取自 bench 分支）；单元测试和只要 Docker 的端到端测试 |
| `tools/` | `scan_sensitive.py`：提交前的脱敏扫描。通用规则在脚本里；机构自己的专有词表放在仓库外（见脚本说明）。审过的误报写进 `tools/scan_allow.txt` 并注明原因 |

## 需要的环境

- Linux，root，Docker（Compose v2），NVIDIA container toolkit，Hopper GPU（H100 / H200 SXM；公开榜在 H200 级的卡上跑）。
- Python ≥ 3.11（或装了 `toml` 包），`pip install harbor==0.24.0`。
- 验题脚本（`skills/rbb-vet/scripts/`）要 numpy，`peaks.py` 要 torch。

自测：

```bash
python3 -m pytest skills/rbb-vet/scripts/test_vet.py
python3 vetkit/tests/test_tools.py && python3 vetkit/tests/test_container.py
bash vetkit/tests/e2e/run_e2e.sh          # 只要 Docker：纯 CPU 的玩具题上走一遍 lint → 建包 → 验证 → 判卷
```

## 安装 skill

```bash
git clone -b vetkit https://github.com/DAGroup-PKU/ResearchBoost-Agent.git rsi-bench-vetkit
ln -s "$PWD/rsi-bench-vetkit/skills/rbb-vet" ~/.claude/skills/rbb-vet   # 用链接：skill 会找 ../../vetkit
```

之后在 Claude Code 里说“把这个仓库验成 bench 的一道题”，或者输入 `/rbb-vet <题名> <仓库 URL> [commit]`。不用 Claude Code 时，把 `skills/rbb-vet/references/protocol.md` 当 prompt 交给别的 agent，或者人照着 `guide.md` 做。

## 快速开始

先读 [`vetkit/README.md`](vetkit/README.md)，用示例题 nanochat 走一遍：linter、泄漏扫描、nop 冒烟（reward 必须为 0），再建一个参考包、在另一组卡上验证、从一个跑完的 trial 复判。

## 规矩

- agent 能看到的东西里不写目标加速比、已有成绩、热点、开关名、优化思路、别的硬件（规程硬规则 1）。验题数据只放 `vetting/<题名>/`，不进题目、镜像，也不公开。
- 只用公开的代码、权重和数据，每个钉住 revision 和 checksum（硬规则 5）。
- 往这个仓库提交前跑 `python3 tools/scan_sensitive.py`，把结果贴进 PR 说明。
