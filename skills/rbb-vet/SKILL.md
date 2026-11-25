---
name: rbb-vet
description: Turn a training repository into one task of RSI-Infra Bench (the bench branch of ResearchBoost-Agent) and vet it on your own Hopper machine — A desk review, B bounds and baseline strength, C Harbor packaging and judge validation, D pilot runs, E wrap-up — ending in a go / no-go for the bench. Use when someone says /rbb-vet, 验题, 验一道新题, 把仓库做成 bench 的题, 新题进 bench, contribute a bench task, vet a task.
---

# rbb-vet：给一个仓库，验成一道 bench 题

调用：`/rbb-vet <题名> <仓库 URL> [commit]`。参数没给全也照常开始，缺的放进 A 段的问题里。

这份 skill 是通用版：一个贡献者在自己的 H 系列机器上就能按它验一道题，不依赖任何调度器、队列或通知服务。文中“如果你有……就……，没有就……”的地方，按你自己的环境选。

## 先读

1. **`references/protocol.md` 全文。** 这是规范：时间预算、硬规则、A–E 每段的检查和阈值、打分卡、《实践补充》。和本文件冲突时听它的。
2. `references/lessons.md`：常见的坑（计时、统计门、对照、镜像、机器）。动手前读。
3. `references/guide.md`：长版指南，有公式、Harbor 打包、变异和误拒率的写法、打分卡的用法，需要哪节读哪节。
4. 数据规矩：只用公开的代码、权重和数据，每个钉住 revision 和 checksum（规程硬规则 5）。如果你的机构对数据外传有规定（哪些数据不能离开哪些机器），把规定原文写进你派给任何子 agent 的 prompt，验题过程中不碰这类数据。

## 工具

本 skill 的 `scripts/`（单文件，拷到跑机上用；`python3 -m pytest scripts/test_vet.py` 自测，要 pytest 和 numpy）：
- `peaks.py`：从 torch trace 抽 GEMM 和注意力形状，实测峰值（GEMM、注意力、HBM、时钟、NCCL busbw）。第一次在新机器上用时，先用最小规模跑通再信它的数。
- `bound.py`：按 `stages.json` 算 T0 和 T1；被规则冻住的阶段用 `frozen_time`。
- `frr.py`：统计门对合法改写的误拒率（6 个种子、相关系数 ρ、3σ）。
- `scorecard.py`：读 `vetting.json`，逐项给 PASS / WARN / FAIL / 缺数据，阈值集中在文件开头。
- `example/`：stages、peaks、vetting 三个文件的格式样例（数是编的）。

仓库根目录的 `vetkit/`（判卷侧，说明见 `vetkit/README.md`；skill 是从仓库链接过去的，路径是 skill 目录的 `../../vetkit`）：
- `bin/onboard_check.py`：题目结构和判卷接口的 linter（C 段收尾必跑，退出码 2 表示有问题）。
- `bin/leak_guard.py`：扫 agent 能看到的所有文件（题面、镜像里 COPY 进去的、prep-pack），可选再扫构建好的镜像。
- `bin/run_trial.py`：直接跑 Harbor trial：指定卡、按 NUMA 绑核、主容器退出时及时停、密钥脱敏、每格一行汇总。
- `bin/refpack.py`：参考包：独占建包（完整准备一次）、在别的机器 / 卡上验证、从提交物在干净容器里重判。
- `examples/tasks/nanochat/`：一道已发布的题，当模板和冒烟用。

## 机器和规矩

- 机器：Linux，Hopper GPU（H100 / H200 SXM），root，Docker + Docker Compose + NVIDIA container toolkit，`pip install harbor==0.24.0`。nsys、ncu 有就用，没有或 ncu 读不了计数器（`ERR_NVGPUCTRPERM`）就跳过，报告里写一句。
- 公开榜跑在 H200 级的卡上（每卡 ≥ 130 GiB）：新题的 `gpu_types` 按这个写。只有 H100 时，A、B、C 可以先做，T0 用规程 B.4 的 H100 手册值、T1 用你的实测值；D 段的成绩要在目标卡上跑，在报告里写清楚。
- A 段先看环境：`nvidia-smi`（卡型、驱动、拓扑 `nvidia-smi topo -m`）、`df -h`、`docker info`、`harbor --version`、nsys、ncu、有没有 root、每张卡所在的 NUMA 节点。
- **用卡之前确认没人在用。** 有调度器或队列（Slurm、k8s、自己的作业系统），就经它拿卡，把整段计时也排成独占作业。没有，就手动：看 `nvidia-smi` 和 `docker ps`，在机器上留一个占用说明（例如 `/var/tmp/gpu-claims/<卡号>-<题名>.txt`，写谁、做什么、从什么时候到什么时候），用完删掉；别人的进程和容器一律不碰。
- **计时只在空闲时做**：B 段计时、建参考包独占整台机器；最终评测至少独占计时卡所在的 NUMA 半边（规程补充 D1）。计时期间这台机器上不 build、不下载、不扫网络盘、不跑别的 GPU 任务。
- 等长任务用后台命令加完成通知，不要 sleep 轮询。

## 工作目录、状态和停点

- 工作目录放在题目仓库外面。验题数据只放 `vetting/<题名>/`（打分卡、stages、peaks、REPORT、对照清单），不进 `tasks/`，不进镜像。
- `vetting/<题名>/status.json` 在开始时、每次换段时、每次停下时重写。如果你有一个会读它的编排器（例如一个负责提醒人的会话或脚本），它可以据此通知 owner；没有就只是一份记录：
  ```json
  {"task": "<题名>", "repo": "", "commit": "", "stage": "A|B|C|D|E",
   "state": "running|need_user|fail|over_budget|pilot_running|done",
   "stage_started": "2026-11-24T10:00:00+08:00", "budget_h": {"A": 2, "B": 12, "C": 24, "D": 12, "E": 2},
   "summary": "一句话：要 owner 定什么 / 什么 FAIL 了 / 结论", "report": "<REPORT.md 的路径>", "updated": "<ISO 时间>"}
  ```
  C 段大题把 `budget_h.C` 改成 48。D 段的 12 小时是 10 小时 agent 加终判。
- 停点只有规程说的三种：某项检查 FAIL、需要 owner 拍板、超出时间预算。停下时把 state 写成 `need_user`、`fail` 或 `over_budget`，summary 写一句话，然后在回复里写清问题，结束这一轮。
- **A 段一开始**就把所有要 owner 定的事一次问完：规程【输入】里没给的几项，加上 A 段要定的几项（命令、卡数和缩放、计时单位、精度下限按部分写、收哪几档、什么能改、每条歧义），每项附推荐值。D 段用哪些模型、跑在哪些卡上、要不要复本也在这一次里问。问完 state 写 `need_user`。

## B 段

照规程 B 和《实践补充》B1–B4：T_ref（≥ 3 个固定窗口、最慢的卡、空闲机器）、分阶段 profile、`stages.json`、`peaks.py` 实测峰值、`bound.py` 算 T0/T1、开关阶梯（≤ 5 个现成开关，记每一步）、A/A、完整 check 的耗时。B 段结束跑 `scorecard.py --stage B`。

## C 段

照规程 C 和《实践补充》C1–C10。从 `vetkit/examples/tasks/nanochat/` 或 ResearchBoost-Agent bench 分支里最像的那道题抄结构。收尾时依次跑：
```bash
python3 <repo>/vetkit/bin/onboard_check.py tasks/<题名>                 # 结构 + 判卷接口 + task.toml
python3 <repo>/vetkit/bin/leak_guard.py tasks/<题名> --deny vetting/<题名>/deny.txt   # 再人读一遍
python3 <repo>/vetkit/bin/run_trial.py --tasks-dir tasks --env-file researchboost.env \
    --trial <题名>:nop --gpu-ids 0,1 --cpus auto                       # nop 冒烟，reward 必须为 0
```
`deny.txt` 写这道题特有、不能出现在 agent 可见文件里的字符串（已有成绩、开关名、热点、别的硬件），一行一个正则。
判卷（`tests/test.sh`、`verify.py`）以 root 运行：只用 root-only 的 `PATH` 和绝对路径，进程用 `/proc` 找，以 agent 身份运行的东西也传干净的 `PATH`（补充 C10；`onboard_check.py` 会查按名字调用的地方，写法照 `vetkit/examples/tasks/nanochat/tests/verify.py`）。

多机或多卡位跑 D 段、或者题里有采样类统计门时，建一个参考包（规程补充 C6、lessons“参考包”）：
```bash
docker build -t rbb/<题名> tasks/<题名>/environment
python3 <repo>/vetkit/bin/refpack.py build  --task <题名> --tasks-dir tasks --image rbb/<题名> \
    --env-file researchboost.env --gpu-ids 0,1 --cpus auto --store packs
python3 <repo>/vetkit/bin/refpack.py verify --task <题名> --tasks-dir tasks --image rbb/<题名> \
    --env-file researchboost.env --gpu-ids 4,5 --cpus auto --store packs          # 另一组卡 / 另一台机器
```
建包独占整台机器；`refpack.py` 以 root 运行。拿 C 段的记录和包里的记录逐位比对。

## D 段

- 每格一个 `harbor run`，所有格子同时跑。按规程补充 D3 选 harness（effort 最高档；DeepSeek 用 Codex CLI）。
- **有队列或调度器**：每格提交成一个作业，要求整个 NUMA 半边（卡和核）；最终评测那段时间这半边不放别的作业。
- **没有**：一格占一个 NUMA 半边，手动起：
  ```bash
  python3 <repo>/vetkit/bin/run_trial.py --tasks-dir tasks --env-file researchboost.env \
      --secrets-env-file keys.env --trial <题名>:claude-code:anthropic/<模型> --gpu-ids 0,1,2,3 --cpus auto
  ```
  Harbor 在 agent 到点后立即在同一个容器里做最终评测：这时这半边上不能有别的东西（另一半可以继续跑 agent）。终判可疑时用 `refpack.py judge` 在独占整机上重判。
- 状态写 `pilot_running`。每个非 PASS 的格子按规程归因到 model / judge / infra；截止时遗留状态造成的 ERROR 按补充 C8 清理后重判。
- 算 h = ln(加速比) / ln(T_ref / T1)，看饱和线和区分度（补充 B4、D2）。

## E 段和交付

- 最好的那份提交用 final 重判 2 次，都要 PASS：
  ```bash
  python3 <repo>/vetkit/bin/refpack.py judge --task <题名> --tasks-dir tasks --image rbb/<题名> \
      --env-file researchboost.env --gpu-ids 0,1 --cpus auto --store packs --artifacts jobs/<job>/<trial>/artifacts --out judged/1
  ```
  记下判卷哈希、镜像 digest、参考包 key（`judge.json` 里有）和上游 commit（`bench.json` 的 `reference_commit`）。
- `vetting/<题名>/vetting.json` 按 `scripts/example/vetting.example.json` 的格式填（规程打分卡模板的超集；每段结束就填那一段，不要等到最后），跑 `scorecard.py` 得到逐项结果，再写 `vetting/<题名>/REPORT.md`，内容照规程“最后交给我”那几条。state 写 `done`，summary 写“进 / 不进 bench，一句理由”。
- 不 push，不开 PR，等 owner 看（规程 E）。owner 同意以后，交给 bench 维护者这些：题目分支和 commit（`tasks/<题名>` 干净、没有未提交的改动）、卡数、镜像里训练环境 python 的路径、verifier 超时、计时参考（哪个文件、哪个字段、C 段测的参考波动）、资产清单（每个文件的来源、revision、sha256）、参考包 key、A–E 的结论。`vetting/` 里的东西私下给，不放进公开的 PR。

## 时间

每段开始时在 status.json 记 `stage_started`，结束时把实际用的小时数写进 vetting.json 的 `hours`。某一段超出预算：state 写 `over_budget`，summary 写卡在哪里，然后停下来。
