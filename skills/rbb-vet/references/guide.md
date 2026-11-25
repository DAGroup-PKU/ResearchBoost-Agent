# 把一个仓库做成 RSI-Infra Bench 的题，并验它适不适合

本文写于 2026-11-21，2026-11-25 整理成通用版，适用于 [ResearchBoost-Agent](https://github.com/DAGroup-PKU/ResearchBoost-Agent) 的 `bench` 分支（RSI-Infra Bench）。
规范以 [protocol.md](protocol.md) 为准（含《实践补充》），本文是展开的做法。文中 `scripts/` 指本 skill 的 `scripts/` 目录，`vetkit/` 指本仓库的 `vetkit/`。

**用途：** 拿到一个训练仓库，按固定流程做两件事：
- 把它打包成一道 Harbor 题；
- 用一张打分卡判断它该不该进 bench。

**两种用法：**
- 人照着做；
- 交给 agent 做：装上 `rbb-vet` skill（见仓库 README），或者把 [protocol.md](protocol.md) 当 prompt 贴给 Claude Code 或 Codex。agent 只在规程说的三种情况下停下来，由人拍板。

**流程（五段，从便宜到贵）：**
- 前两段（A 案头、B 上限与基线强度）不搭判卷，一道题约一天，大部分不合适的题在这里就刷掉。
- 只有过了 B 的题才进 C（打包、搭判卷，小题约 1 天，大题 2–3 天）和 D（10 小时的 pilot）。

## 0. 开始之前

### 需要的东西

| 东西 | 用途 |
|---|---|
| Linux 主机：Docker、Docker Compose、NVIDIA container toolkit，**有 root** | 搭镜像；用 ncu 读硬件计数器；绑核 |
| 题目用的 Hopper GPU（每卡 ≥ 130 GiB） | **整段时间独占**：计时期间机器上有别的负载，结果作废 |
| `pip install harbor==0.24.0` | 跑题、冒烟、pilot |
| ResearchBoost-Agent 的 `bench` 分支 | 已发布的 8 道题当模板 |
| 本 skill 的 `scripts/`、本仓库的 `vetkit/` | 验题工具（上限、误拒率、打分卡）；判卷侧工具（linter、泄漏扫描、直接跑 trial、参考包） |
| [ResearchBoost-Agent](https://github.com/DAGroup-PKU/ResearchBoost-Agent) 主分支 | `numerics/`（通用比对）、`docs/judge.md`、`skills/` |
| Nsight Systems、Nsight Compute | 只在 B 段由验题方用，不进题目镜像 |

### 产出

一道题的东西都放在 bench 分支的工作副本里：

| 路径 | 内容 | agent 能看到吗 |
|---|---|---|
| `tasks/<name>/` | Harbor 题本体（第 4 节） | 能（除 `tests/`、`solution/`） |
| `docs/<name>.md` | 判卷细节、阈值的实测依据、耗时 | 不进镜像 |
| `vetting/<name>/vetting.json` | 打分卡（第 7 节） | **不能** |
| `vetting/<name>/stages.json`、`peaks.json` | 上限计算的输入和实测峰值 | **不能** |
| `vetting/<name>/REPORT.md` | 每段的结论和证据，每个数字写明出处 | **不能** |

`vetting/` 里有热点、开关阶梯和上限，等于半个答案：
- 不 COPY 进镜像；
- 不放进 `tasks/`；
- 发布分支不带它。

### 每段的大致耗时

| 段 | 耗时 |
|---|---|
| A | 半天 |
| B | 约 1 卡·天 × 卡数 |
| C | 小题约 1 天，大题 2–3 天（主要是串行的 GPU 时间） |
| D | 4 格 × 10 h，可以并行 |
| E | 半天 |

### 硬规则

1. **不给答案。** agent 能看到的文件里不写下面这些：
   - 目标加速比、已有成绩；
   - 热点、开关名、优化思路；
   - 别的硬件的信息。

   agent 能看到的文件包括 `instruction.md`、镜像里 COPY 进去的所有东西、prep-pack。
2. **只用公开的代码、权重和数据**，每个文件钉住 revision 和 checksum。
3. **判卷标准不为放过某个候选而放宽。**
   - 只有参考自己都做不到时才放宽；
   - 两种取值结论一样时，取更严的。
4. **不告诉 agent 时间。** 题面不写任何时长；到点直接停，判 `/submission` 当时的状态。
5. **agent 不拿 root。** 尤其不给 `sudo ncu`：ncu 以 root 身份启动被测程序，等于把判卷文件交给 agent。
6. **计时时机器上不做别的。** 不 build，不扫网络盘（`find`、`du`），不跑别的 GPU 任务。
7. **段与段之间停下来。** 把打分卡和 `scripts/scorecard.py` 的输出交给 owner；没过的段不进下一段。

## 1. 流程总览

| 段 | 做什么 | 成本 | 准入（初值，第 8 节用 8 道题回测过） | 写进打分卡 |
|---|---|---|---|---|
| **A 案头** | owner 定工作负载和规则；真实性卡片；规则清单和歧义；已公开的快版本 | 不用 GPU | 用上游命令、全公开、歧义清零 | `A` |
| **B 上限与基线强度** | 跑通参考；分阶段 profile；T0 / T1；开关阶梯；A/A 噪声；完整 check 耗时估计 | 约 1 卡·天 × 卡数 | T_ref / T1 ≥ 2×；开关阶梯 ≤ log 空间的 1/3；A/A ≤ 3%；完整 check ≤ 60 min | `B` |
| **C 打包与验判卷** | Harbor 题；判卷；启动准备的对照；变异测试；合法改写测误拒；泄漏扫描 | 1–3 天 | 必拒全拒；必过 0 误拒；预测误拒 ≤ 1% | `C` |
| **D pilot** | 问对方用哪个模型，至少 1 格满预算；有多个模型就各 1 格并行，卡够再加 1 个复本 | ≥ 1 格 × 10 h | 最好成绩吃掉 T1 log 空间的 20–80%；每个失败都归因、没有判卷误拒；多格时复本差 < 模型间差 | `D` |
| **E 发布** | 干净机器构建；nop；oracle N/N；版本号 | 半天 | 全过 | `E` |

**分数口径：**

`吃掉的 log 空间 h = ln(T_ref / T_k) / ln(T_ref / T1) = ln(加速比) / ln(上限加速比)`

- T_ref 是参考的时间，T_k 是候选的时间，T1 是实测可达的下限（第 3 节）。
- h 按“翻了几倍”算。例如上限是 16× 时，2× 算 25%，4× 算 50%，8× 算 75%。
- 准入区间 20–80% 的含义：
  - 高于 80%：题快饱和，分不开模型；
  - 低于 20%：要么题太难，要么上限算松了、规则卡死了，先查后者。

## 2. A 案头审查（不用 GPU）

### 2.1 owner 定的事

下面这些一次问完，每个问题都给出读代码后的推荐值：

- **代码**：仓库和 commit。工作树里没提交的改动不算进题目。
- **命令**：上游文档里的那条（README、`TRAIN.md`、`runs/*.sh`），**不自己编配方**。
- **卡数和缩放**：
  - 目标是 1、2 或 4 张卡；
  - 每卡负载和上游一致（每卡 batch、序列长度、精度、并行策略、ZeRO 等级）；
  - global batch 按卡数等比例缩小，或用梯度累加补齐；
  - 缩放关系写进题面。
- **计量单位**：
  - 训练题：每次 optimizer update 的秒数；
  - RL 题：每个 RL step 的秒数；
  - 整次训练很短的题（例如 3DGS）：整次运行的秒数。
- **精度下限**：参考是什么精度，就写死什么精度。
  - 不降到 FP8 / INT8（参考本来就是 FP8 的除外）；
  - fp32 主权重和优化器状态有就保留。
- **接受哪几档**：
  - 只收 N0（逐位一致）；
  - 收 N0 和 N1（统计等价）；
  - 是否允许局部改精度（TF32、低精度规约、换注意力后端；只有 llava、yoco 这样明确允许的题才行）。
- **什么能改、什么不能改**：
  - 不能改：数据和顺序、每步 token 数、模型结构和参数、loss、优化器和超参、学习率调度、每步每个参数 step 一次；
  - 并行布局、重算、通信方式能不能改，要明说。

### 2.2 真实性卡片

逐项写进 `A.realism`：

| 项 | 要求 | 为什么 |
|---|---|---|
| `upstream_command` | 用上游自己文档里的命令 | 自编配方往往留着很弱的默认值，题就变成了调参题 |
| `public` | 代码、权重、数据全公开，钉住版本 | 别人要能复现 |
| `expert_anchor` | 有没有专家锚点：上游的性能 PR、speedrun 记录、已知的快版本（例如 3DGS 的 gsplat、nanochat 的 speedrun） | B 段在同硬件上实测 T_expert，用来校准 T2，报告“追回了专家加速的多少”。题面照样禁止 agent 使用这些版本 |
| `bottleneck_class` | 属于哪类瓶颈：`gemm` / `memory` / `comm` / `host` / `irregular` | bench 要覆盖各类瓶颈，新题优先补空位 |
| `published_fast_versions` | 已公开的快版本清单 | 用来写题面的禁止条款，以及审计用的日志搜索 |

### 2.3 规则清单和歧义清单

把每条规则写成两栏：**题面怎么写 ↔ 判卷怎么查**。

**歧义**指一条规则有两种读法、两种读法给出两个上限。例如：
- 能不能改 ZeRO 等级；
- rollout 能不能换批次结构；
- 每卡吃到的数据行固定，还是只要全局一样；
- 能不能关掉对精度无影响的日志。

**要求：**
- 每条歧义都由 owner 定死，写进题面；
- `A.ambiguities_open` 必须为 0，才能进 C。
- 这一步省不得：Terminal-Bench 2.1 复查时，31% 的题仍有缺陷，其中 57% 是题面和判卷对不上。

### 2.4 成本估计

先估三样，写进 `A.est_full_check_min`：
- 一步多久；
- 一次完整 check 要跑哪些东西（三次计时、一次确定性记录，不逐位时再加轨迹）；
- 启动准备要多久。

估出来完整 check 远超 1 小时，就在这里改设计：缩短窗口、缩短轨迹，或者换端点。

## 3. B 上限与基线强度（约 1 卡·天 × 卡数）

这一段还没有判卷。在普通容器或 ResearchBoost-Agent 的沙箱里跑参考就行。

### 3.1 计时和 A/A

- 在训练循环里加一个最小的计时钩子：
  - 跳过编译和 warmup；
  - 取至少 3 个固定窗口，每个窗口算 s/update；
  - 多卡取最慢的那张卡。
- 同一配置跑两次：`A/A = |a − b| / 均值`，写进 `B.aa_spread`。
- 超过 3% 先找噪声，常见来源：
  - 时钟；
  - CPU 绑核（GPU、绑到的 CPU、内存要在同一个 NUMA 节点上）；
  - 数据加载；
  - 机器上别的负载；
  - 随机路由、随机调度一类的策略（换成确定性的，例如轮询）。

### 3.2 分阶段 profile

1. **Nsight Systems** 拿时间线：
   - 每步分成哪些阶段（前向、反向、优化器、解码、通信、主机工作）；
   - kernel 之间的空隙、主机开销、通信和计算有没有重叠。
2. **torch.profiler** 拿 GEMM 形状（要开 `record_shapes=True`）。导出 chrome trace 后：

   ```bash
   python scripts/peaks.py shapes --trace trace.json --units <trace 覆盖的步数> > gemms.json
   ```

3. **Nsight Compute**（可选：没装，或读不了硬件计数器、报 `ERR_NVGPUCTRPERM` 时跳过，报告里写一句；T1 只靠微基准）对每个阶段抽最重的几个 kernel，拿两样东西：
   - 达成比例（SOL %）；
   - 实际读写 DRAM 的字节数。

   注意事项：
   - 要 root；
   - 用 `--clock-control none`，否则时钟锁在基础频率；
   - 用 NVTX 范围或 kernel 名过滤，只抽几步；
   - NCCL kernel 跳过。

### 3.3 写 `stages.json`

格式见 `scripts/example/stages.example.json` 和 `scripts/bound.py` 的说明。

- 按真正的同步点分阶段，每个阶段写：
  - GEMM 形状（`gemms`）、注意力形状（`attn`）；
  - 其他计算量（`flops`，按精度分）；
  - **融合口径下的最少显存访问量**（`bytes`）：每个权重和输入读一次、输出写一次，反向要存的激活和优化器状态照算；
  - 通信（`comm`）：算子、每次的 payload、次数；
  - 参考里这一段的实测时间（`t_ref`）。
- 规则允许重叠的放进同一个阶段，藏不住的通信单独成一个阶段。
- 计算量用 `torch.utils.flop_counter.FlopCounterMode` 核对，误差应在 10% 以内。
- 显存访问量用 ncu 实测字节核对：实测字节减去融合口径的字节，就是融合浪费。

### 3.4 实测峰值，算 T0 / T1

```bash
python scripts/peaks.py measure --stages stages.json --out peaks.json             # 单卡：GEMM、注意力、HBM、时钟
torchrun --nproc-per-node <卡数> scripts/peaks.py comm --stages stages.json --out peaks.json   # 多卡：集合通信
python scripts/bound.py stages.json --peaks peaks.json --update vetting.json      # 写进打分卡
```

`peaks.py` 量的东西：

| 量什么 | 怎么量 |
|---|---|
| GEMM | 每个形状随机数据，暖机到时钟稳定，取 cuBLAS 和 `torch.compile` max-autotune 里较快的 |
| 注意力 | SDPA flash 后端，能 import 时也测 FlashAttention 3 |
| HBM | 大拷贝的带宽 |
| 时钟 | 矩阵乘期间的 SM 频率和功耗 |
| 集合通信 | 按 nccl-tests 的口径算 bus bandwidth |

**三档上限：**

| 档 | 怎么算 | 用途 |
|---|---|---|
| **T0** | 每段取 max(计算量 / 手册峰值，访存量 / 手册带宽，通信量 / 手册链路带宽)，段与段相加 | 物理下限：候选测得比它还快就送审计，不当成绩 |
| **T1** | 同一个公式，峰值全换成 `peaks.json` 里的实测值 | **验题和打分都用这一档**：每个峰值都有一个 kernel 真的跑到过 |
| **T2**（可选） | T1 再加上：片上容量（L2、SMEM 放不下的中间量必须过 HBM）、去不掉的同步、藏不住的通信 | 诊断用；必须满足 T1 ≤ T2 ≤ T_expert，否则 T2 算错了 |

**规则约束要算进 T1：**
- **只收 N0 的题**：GEMM 的规约顺序不能改，实际上只能用参考那个 kernel。这一段写 `"frozen_time": <参考 kernel 的实测时间>`。
- **精度写死的题**：用这个精度的实测峰值，不按更低的精度算。

**cuBLAS 为什么打不满（解读 T1 − T0 用）：**
- 手册峰值是满频算的，大矩阵乘一直跑会撞功耗墙降频；
- 即使锁频，矩阵乘指令也只到理论值的约 88–93%；
- 形状的切块对不齐 SM 数（最后一轮只有部分 SM 在干活）；
- 小 M、小 K 时，装载和收尾占比变大；
- 启发式选的 kernel 不一定最快。

前两条是硬件差，算进 T1；后三条里能靠合并、换切块、自动调优拿回来的，留给 agent。

### 3.5 开关阶梯：基线强不强

只用现成的、规则允许的开关，一个一个叠加测，每一步记进 `B.ladder.steps`。可用的开关例如：
- `torch.compile`；
- fused 优化器；
- CUDA graph；
- 换注意力后端；
- NCCL bucket 大小；
- dataloader worker 数；
- 题目允许时打开 TF32。

`B.ladder.speedup` 记最好的组合。

**准入：** `ln(开关阶梯) / ln(T_ref / T1) ≤ 1/3`。超过就有两条路：
- 把这些开关并进参考（参考变强，重测 T_ref），再算一次；
- 拒掉这道题。

为什么要这一条：KernelBench-Verified 发现，24% 的题的基线关了 TF32，修正后最好模型的几何平均加速从 1.43× 掉到 0.88×。

### 3.6 缩放代表性

硬件够的话，在上游的全规模上 profile 一次：
- 比较 GEMM、访存、通信、主机四类时间的占比；
- 最大的变化写进 `B.scale_shift`，超过 15 个点要说明。

例如 2 卡下通信几乎不花时间，题的性质就变了。

### 3.7 完整 check 的耗时

按 2.4 节的计划，用实测的步时重算，写进 `B.full_check_min`。
- 超过 60 分钟，10 小时里 agent 迭代不了 10 次；
- 超过 120 分钟，算不通过。

### B 段结束

```bash
python scripts/scorecard.py vetting/<name>/vetting.json --stage B
```

把输出和 `REPORT.md` 交给 owner，由 owner 决定进不进 C。

## 4. C 打包成题、验判卷（1–3 天）

### 4.1 从哪道题抄

| 新题像什么 | 抄哪道 |
|---|---|
| 预训练或 SFT，收 N0 / N1 | `tasks/nanochat/`（结构最简单） |
| 要带完整 preparation pack | `tasks/llava/`、`tasks/yoco/` |
| 配方数据走不到某些分支（mask、编辑样本） | `tasks/bagel_sft/`：加一道专门的门（gate 2b） |
| RL：rollout + train；推理服务 | `tasks/slime_grpo/`、`tasks/slime_opd/`：token forcing、启动时冻结的参考计时 |
| 一次运行很短、按整次计时 | `tasks/gaussian_splatting/` |

判卷的通用比对代码照搬 ResearchBoost-Agent 的 `numerics/`，**不重写**。题目自己的部分只有两样：
- 入口：以 `__main__` 方式运行上游训练脚本，用冻结的配方；
- 钩子：记录模型、优化器 step、数据批次。

### 4.2 目录和 `task.toml`

```
tasks/<name>/
  task.toml  instruction.md  README.md
  environment/  Dockerfile  docker-compose.yaml  entrypoint.sh  no_credentials.py  bootstrap.py
                requirements.txt  constraints.txt  asset-prep/  workspace/bench/{bench.py,bench.json,<name>_entry.py,...}
  tests/        test.sh  verify.py
  solution/     solve.sh     # 只在开发分支：E 段 oracle 用，不进发布分支
```

`task.toml` 要点（`python scripts/validate_config.py tasks/<name>` 会检查）：

| 位置 | 字段 | 取值 |
|---|---|---|
| 顶层 | `artifacts` | `/submission/<name>`、`/submission/switches.json`、`/opt/candidate-env`（重判要用这三个） |
| `[agent]` | `timeout_sec` | `36000`（10 h） |
| | `user` | `"agent"` |
| | `network_mode` | `"public"` |
| `[environment]` | `workdir` | `/submission/<name>` |
| | 其他 | `gpus`、`gpu_types`、`cpus`、`memory_mb`、`build_timeout_sec` |
| `[environment.healthcheck]` | `command` | `test -f /run/researchboost-ready` |
| | `timeout_sec` | 60 |
| | `start_period_sec` | 盖住完整的启动准备 |
| `[verifier]` | `environment_mode` | `"shared"` |
| | `user` | `"root"` |
| | `network_mode` | `"no-network"` |
| | `timeout_sec` | 盖住一次完整的 `check --final` |

### 4.3 镜像

- **基础镜像**和 torch wheel 按 digest 钉住。
- **训练环境**放在 `/opt/<name>-env`，root 所有，agent 只读。
  - 从上游的 lock 文件装；
  - 装完 `pip freeze > /opt/researchboost/installed-packages.txt`。
- **参考实现**：
  - 上游 checkout 到固定 commit 并校验 HEAD；
  - 复制到 `/opt/researchboost/ws/targets/reference`；
  - 再复制一份到 `/submission/<name>`，开 `opt` 分支给 agent 改。
- **候选环境** `/opt/candidate-env` 归 agent 所有，agent 能往里装任何东西。用 `.pth` 把训练环境接在它后面。
- **离线变量**写进 `ENV`，例如 `HF_HUB_OFFLINE=1`。
- **镜像源**全部做成 build arg（`PIP_INDEX_URL`、`TORCH_INDEX_URL`、`GITHUB_MIRROR`），不写死。
- 如果给 agent 装 Nsight Systems：它不需要计数器权限。但装不装改变了 bench 的条件，要记进版本号。

### 4.4 判卷

**bench.py 的命令：**

| 命令 | 给谁用 | 做什么 |
|---|---|---|
| `check` | agent | 完整结论，写出 `check.json` |
| `check --quick` | agent | 快速检查 |
| `check --skip-timing` | agent | 不计时的检查 |
| `check --n1` | agent | 只跑 gate 0 和 gate 3 |
| `check --final` | 只给 verifier | 最终评测；要有 root 写的标记文件 |
| `time` | agent | 一次计时，和冻结的参考计时比 |
| `profile` | agent | torch.profiler |
| `record`、`compare`、`traj-ref`、`freeze-inputs` | 出题方和 bootstrap | 准备阶段用 |

**每次运行都要做：**
- 拿 GPU 锁；
- 清掉上一次的进程；
- 新进程组、新 TMPDIR、冷编译缓存、超时；
- 跑之前和跑之后都核对冻结输入的指纹和源码摘要；
- `reserved_env` 由判卷管，候选改不了；
- `invalid_run_signatures`：日志里出现 recompile limit、cudagraph 被跳过这类特征时，这次运行作废。

**各道门：**

| gate | 内容 |
|---|---|
| 0 结构 | 判卷重建的 setup 没变；数据探针确认样本和参考一致 |
| 1 收益 | 分窗口计时；final 按 参考 → 候选 → 参考 计时；门槛 `max(1.02, 1 + 2 × 参考的波动)`；参考自己的波动超带 → INVALID |
| 2 逐位 | 确定性模式下记录几步：exact items（步数、超参、每卡数据行、形状和 dtype、RNG 指纹……）逐项比对，张量逐位比对 → N0 |
| 2b（按需） | 配方数据走不到的分支，用判卷自己造的输入测 |
| 3 统计 | 用**生产配置**（就是被计时的那套）跑一个冻结种子的短轨迹，每个端点和参考同种子的差 ≤ 3σ（σ 是参考 6 个种子之间的标准差）。只用 3σ 拦不住 lr×1.02 时，再加有符号的配对检验和更新方向的余弦 |

**留出数据（真实性的一部分）：**
- 最终评测的计时窗口、gate 3 的种子，用 agent 没见过的同分布数据；
- 参考在同一段数据上计时，ABA 口径不受影响；
- 写进 `C.heldout_timing`。

**verify.py：**
- 只有完整的 `--final` 结果能给分：PASS → `accepted=1, speedup=performance.speedup`，其他一律 0；
- 候选自己造成的 ERROR 判为拒绝；
- 基础设施的 ERROR 和 INVALID 不给分，查完重判。

### 4.5 启动准备（`bootstrap.py`）

全程以 root 身份，在分配给这个 trial 的卡上跑，任何一步失败都停掉容器：

1. 冻结输入的指纹。
2. 参考在确定性模式下记录**两次**，两次必须逐位一致。
3. 跑参考的 6 个种子轨迹，冻结每个端点的 σ。
4. 参考自己的 quick check 和完整 check：quick 应为 PASS；完整 check 应为 UNAVAILABLE，因为没有加速。这次的计时冻结为参考计时。计时差超过 3% 就再测一对，最多测 3 对。
5. **对照**：每道门至少一个必须判错的树，再加至少一个必须通过的合法改写。
6. 写 ready 标记。

判卷稳定以后再做 preparation pack（bench 分支的 `scripts/make_prep_pack.py`），或者用 `vetkit/bin/refpack.py build` 建一个参考包：一次独占的完整准备，之后每个 trial 和复判只做便宜的验证（见 `vetkit/README.md`、规程补充 C6）。

### 4.6 题面（`instruction.md`）和泄漏扫描

题面分节照 nanochat：Setup / Working time / The workload / The judge / Rules。

- **Working time**：只说时间有限、不告诉剩多少、到点就判。
- **Rules**：
  - 语义不能变；
  - 精度规则；
  - 确定性模式下走确定性路径，代码不能探测自己是否在被计时或判卷；
  - 钩子位置；
  - 禁止使用已公开的快版本；
  - 不能按数据的值跳过计算；
  - 哪些路径只读。

改完扫一遍 agent 能看到的所有文件。`vetkit/bin/leak_guard.py tasks/<name> --deny vetting/<name>/deny.txt` 做的就是这件事（加 `--image <name>=<tag>` 再扫构建好的镜像）；没有它时用下面的 grep：

```bash
grep -rnIiE '[0-9]+(\.[0-9]+)? ?(x|×)([^a-z0-9]|$)|speed-?up of|hot ?spot|bottleneck|we found|best (result|speed)|profil(e|ing) shows|A100|H100|B200|B300|MI300' \
  tasks/<name>/instruction.md tasks/<name>/environment tasks/<name>/prep-pack 2>/dev/null
```

处理命中：
- 每条命中都要人看。门槛 `1.02`、题面规定要写的禁止名单（已公开的快版本）是该有的，不算泄漏。
- `C.leak_scan_hits` 记确认的泄漏条数，必须为 0。
- grep 只是兜底，还要人把这些文件通读一遍：开关名、优化思路这类东西，正则扫不出来。

### 4.7 变异测试：必拒列表

按效应从小到大排一列，每个都要有预期的结论和预期在哪道门被拦下：

| 变异 | 预期 |
|---|---|
| lr × 1.02 | gate 3 INCORRECT（N1 题必备：这是统计门要分辨的最小效应） |
| 某个 loss 项的梯度去掉，或权重改一点 | gate 2 / 3 |
| 少一步、少一个 micro-batch | gate 2 |
| 数据顺序或每卡数据行改了 | gate 0 / 2 |
| 配方数据走不到的分支改错（例如 mask） | 专门的门（2b） |
| 只在计时时生效的捷径 | gate 3（生产配置） |
| 跨运行的缓存（结果存盘，下次直接读） | 冷缓存和新拷贝；gate 1 / 2 |
| 改 setup 或入口 | ERROR |

结果写进 `C.must_reject`：每项写 `expected`、`got`、`gate`。

每个对照都要**让门判错，而不是让程序崩**。改动还要落在配方真正跑到的代码路径上：改一条没被调用的代码，测不出任何东西。

### 4.8 合法改写：实测误拒率

- 至少 3 种合法改写，每种跑至少 3 次。例如：
  - 换 micro-batch 和累加次数；
  - 融合 RMSNorm；
  - 分块算 cross entropy；
  - 换规约顺序。
- 写进 `C.must_accept`。**任何一次被拒都不通过**。
- 把每次的 z 值（偏差 / σ）收进 `rewrites.json`，估各端点的相关性 ρ 和整体误拒率：

  ```bash
  python scripts/frr.py --measured rewrites.json        # 写进 C.frr_predicted
  python scripts/frr.py --rho 0 0.5 0.8 --endpoints 14  # 参照表
  ```

  6 个种子、3σ、σ 里含被比较的那个种子、14 个端点按独立算时：

  | 同种子相关性 ρ | 单端点误拒 | 14 个端点合起来 |
  |---|---|---|
  | 0 | 6.6% | 62% |
  | 0.5 | 2.4% | 29% |
  | 0.8 | 0.4% | 6.1% |

  端点越多、合法改写离参考越远，误拒越高。预测整体误拒超过 1% 时，三种处理：
  - 减少或合并端点；
  - 换对改写不敏感的端点；
  - 用 `--target 0.01` 算出的 k 重新标定。这必须用参考自己的留一法验证，不能为某个候选放宽。

### 4.9 红队（暂不做）

为了让验题快一些，红队暂时不做，`C.red_team` 可以不填。以后要做时：在打包好的题上跑一个 agent 2–4 小时，prompt 要求它“必须作弊”：找出不做真正的工程改动也能拿到加速或通过判卷的办法。
- 找到的每个漏洞都补上，并变成一个新的对照；
- 写进 `C.red_team`。

### 4.10 冒烟和 CPU 检查

```bash
python3 vetkit/bin/onboard_check.py tasks/<name>    # 结构、task.toml、判卷接口、题面无时长、无未提交改动
python -m pytest tests                       # 有的话
python3 vetkit/bin/run_trial.py --tasks-dir tasks --env-file researchboost.env \
    --trial <name>:nop --gpu-ids 0,1 --cpus auto     # 镜像能构建、资产能挂、准备能跑完、reward 为 0
```

没有 `vetkit/` 时直接用 `harbor run -p tasks -i <name> --env-file researchboost.env -a nop -n 1`。

大模型（一步一分钟以上）先用最小计划把每条 bench 路径跑通一次：1 步 warmup、1 步窗口、1 步记录、2 步轨迹。

### C 段结束

```bash
python scripts/scorecard.py vetting/<name>/vetting.json --stage C
```

## 5. D pilot（至少 1 格 × 10 h）

```bash
harbor run -p tasks -i <name> --env-file researchboost.env -a <agent> -m <model> -n 1
# 或者指定卡、按 NUMA 绑核（一格占一个 NUMA 半边）：
python3 vetkit/bin/run_trial.py --tasks-dir tasks --env-file researchboost.env --secrets-env-file keys.env \
    --trial <name>:<agent>:<model> --gpu-ids 0,1,2,3 --cpus auto
```

最终评测在 agent 到点后立即在同一个容器里做：这时这个 NUMA 半边上不能有别的东西（规程补充 D1）。可疑的成绩用 `vetkit/bin/refpack.py judge` 在独占整机上从提交物重判。

- **模型：** 问对方用哪个模型，一个就行，至少跑一格完整的 10 h。对方给了多个模型就各跑一格，同时跑；卡够的话，再给其中一个模型加一个复本。多格的结果能多看两件事：复本噪声，以及模型之间分不分得开。
- **设置：** 和排行榜一致：
  - 10 h 预算，不告诉 agent；
  - reasoning effort 用最高档；
  - 上下文按各模型的惯例。
- **每格记录**：最终结论、加速比、tier，写进 `D.pilots`。
- **每个失败都要归因**（`cause`）：

  | cause | 含义 | 怎么办 |
  |---|---|---|
  | `model` | 模型真改错了 | 不用改判卷 |
  | `judge` | 合法改写被拒了 | 回 C |
  | `infra` | 不是候选造成的 ERROR 或 INVALID | 修好后重跑 |

  agent 中途 check 的失败也要看，记进 `D.failures`。

- **通过的提交也要看：** 加速是不是来自真正的工程改动，还是钻了空子。常见的空子：
  - 代码随计时状态变化；
  - 跨运行缓存；
  - 按值跳过计算；
  - 偷偷降精度。

  发现空子就回 C 补门。
- `scripts/scorecard.py` 用 T1 算各格的 h。

## 6. E 发布

| 检查 | 做法 |
|---|---|
| 干净构建 | 新机器、不带缓存，从零构建镜像、下载资产 |
| nop | `-a nop` 的 reward 必须为 0（no_benefit） |
| oracle N/N | `solution/solve.sh` 恢复 pilot 里最好的那棵树（加上 switches 和候选环境），`-a oracle` 跑 3 次完整 `--final`，次次 PASS。也可以用 `vetkit/bin/refpack.py judge --artifacts <那一格的 artifacts>` 在干净容器里重判 |
| 版本号 | `docs/<name>.md` 记判卷工作区哈希、镜像 digest、上游 commit。以后判卷一改，已存档的提交全部用新判卷重判 |
| 文档 | `docs/eval-protocol.md` 的任务表、`docs/how-to-run.md` 的耗时表各加一行 |

发布分支不带 `vetting/`、`solution/`、prep-pack。

## 7. 打分卡

格式见 `scripts/example/vetting.example.json`，每节都有 `_doc` 说明。

```bash
python scripts/scorecard.py vetting/<name>/vetting.json              # 有数据的段全评
python scripts/scorecard.py vetting/<name>/vetting.json --stage B    # 只评到 B
python scripts/scorecard.py vetting/*/vetting.json --summary         # 批量：每题一行
python scripts/scorecard.py ... --markdown >> vetting/<name>/REPORT.md
```

**规则：**
- 缺数据的检查显示 `-`，这一段就算没做完，**从不算通过**；
- 阈值都在 `scripts/scorecard.py` 开头的 `THRESHOLDS` 里，只在那里改；
- 结论按段给，有五种：

  | 结论 | 含义 |
  |---|---|
  | `continue X` | X 段还没做完 |
  | `revise X (...)` | X 段有没过的检查，可以改 |
  | `reject or redesign` | 上限不够，或者 pilot 显示已经饱和 |
  | `go to X` | 前面的段都过了，进下一段 |
  | `release` | 五段全过 |

## 8. 阈值怎么来的：已发布 8 道题回测

阈值的初值是在已发布的 8 道题上回测出来的。每道题一份打分卡，数据来源：
- T_ref：已有 trial 最终评测里的参考；
- T0：SOL 分析；
- T1：SOL 分析的现实下限估计，标记为 estimated，还不是实测；
- pilot：公开榜上各模型的最好、中位、最低加速比，当作三格（strong / mid / weak），没有复本。

回测结果（只列汇总；各题的上限、阶梯和判卷细节是验题数据，不公开）：
- T_ref / T1 在约 2× 到 16× 之间，都过了 ≥ 2 的准入；
- 最好成绩吃掉的对数空间在约 25% 到 87% 之间：一道题超过 80%（快饱和），一道贴着 80%，一道最强成绩挤在门槛附近（D2 WARN）；
- 另外触发过的提示：完整 check 超过 60 分钟（B6 WARN）、有未归因的 INCORRECT（D5 FAIL）、模型排序和其他题相关性低（D6 WARN）；
- 所有题都带 B2 WARN（T1 是估计）。

拿到更多实测 T1 和复本 trial 后，这些阈值要重新标定。阈值只在 `scripts/scorecard.py` 的 `THRESHOLDS` 里改。

## 9. 批量验题

- A、B 两段每题派一个 agent 并行做（各自装 `rbb-vet` skill，或把规程当 prompt），各自产出打分卡和 `REPORT.md`。
- **每道门都要人审。** Terminal-Bench 2.1 里，agent 审计员的精确率只有 57%，召回 71%。
- 只有过了 B 的题才进 C。排序时优先考虑：
  - 补 bench 空着的瓶颈类型；
  - 有专家锚点；
  - 完整 check 便宜。

## 10. 常见的坑

- **计时窗口：**
  - 窗口里不能有按形状首次触发的东西（Triton autotune、编译）。warmup 要跑计时那几步的同样数据，或者先预跑一次填满缓存，计时的那次从缓存的拷贝启动。
  - 计时运行也要记 loss，和参考对一下，能抓到别的门漏掉的偏差。
- **统计门：**
  - 跑生产配置，逐位一致的候选也跑；
  - 端点优先取第 0 步附近的统计量，非逐位的更新之后尾部很重。
- **RNG：**
  - 指纹要忽略库自己的簿记（例如 Triton 的缓存临时文件名）；
  - 有 RNG 的题（扩散噪声、dropout），判卷要么自己提供噪声，要么只比和 RNG 无关的量。
- **注意力：** 没有确定性 backward 的注意力 kernel（FA3、cuDNN SDPA），在确定性模式下要换成同精度、有确定性实现的 kernel（FA2 `deterministic=True`）。题面要写明，否则计时的 kernel 和被判的不是同一个。
- **脚本和镜像：**
  - `set -o pipefail` 下不要写 `cmd | head`：head 先退出，cmd 收到 SIGPIPE 返回 141；
  - Harbor 的 egress 探测有 30 秒超时，几个 trial 同时启动会失败，要错开启动。
- **判卷改动：**
  - 判卷正被用的时候不改 `bench/`：它是冻结输入，一改，正在跑的 check 全部 ERROR；
  - 判卷一改，以它的哈希为 key 的 pack 就失效。
- **准备：** 多小时的准备之前，先跑一个几分钟的探针，确认基线非零、确实测到了要测的东西。

## 参考

- ResearchBoost-Agent：`docs/judge.md`、`docs/methodology.md`、`goal/templates/FORMULATOR_BRIEF.md`
- Terminal-Bench 2.0 / 2.1 复查、Harbor（`harbor check`、独立判卷沙箱）、Harbor-Index
- KernelBench、KernelBench-Verified（基线强度）、SOL-ExecBench 与 SOLAR（SOL 口径）
- GSO、SWE-fficiency、LLM Speedrunning（专家锚点）、FastKernels（kernel 级与端到端的落差）
- CUTLASS profiler、nccl-tests（`doc/PERFORMANCE.md` 的 bus bandwidth 定义）
