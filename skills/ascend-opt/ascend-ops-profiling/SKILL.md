---
name: ascend-ops-profiling
description: Collect and analyze Ascend operator profiles with msprof or msOpProf, diagnose runtime errors, and visualize msOpProf collections.
---

# ascend-ops-profiling

Choose msprof or msprof op based on the installed version and target, run a representative operator with matching arguments, and preserve collection outputs. Diagnose pipeline, memory and utilization using supported metrics; do not substitute kernel timings for the frozen task evaluator.

文中的采集和汇总脚本需要 [算子采集工具目录](../upstream.md#ascend-ops-profiling-workspace) 及其配套资源。从该目录运行相应的 `scripts/…` 命令。

Read [the technical guide](#上板性能采集与调优) for the procedure, examples and tool-specific fields. Execute documented skill procedures and their scripts as described; do not invoke a skill itself as a Bash command.

## Contents

- [When profiling encounters runtime errors](#when-profiling-encounters-runtime-errors)
- [上板性能采集与调优](#上板性能采集与调优)
- [msOpProf 采集与可视化](#msopprof-采集与可视化)
- [Ascend C 算子运行时错误调试](#ascend-c-算子运行时错误调试)

## When profiling encounters runtime errors

Read [runtime diagnostics](#ascend-c-算子运行时错误调试) to correlate the first ACLNN/CANN error with plog, environment, tiling or kernel lookup before retrying a collection. This is runtime triage, not an alternative numerical judge.

## 上板性能采集与调优

在真实 NPU 上采集算子性能数据，系统化解读指标文件，判定性能是否达标，定位瓶颈类型，并给出可操作的优化建议。

本技能保留 `msprof`、`msprof op` 两条独立路径。先按"外部选择优先"确定工具，只有 `msprof` 路径使用 10 组相邻预热与中位数的新计时协议；`msprof op` 保持自身的采集和归档口径。

### 高频问答速查（先查这里——本节答案即为权威结论，已与脚本实际行为对齐，无需再读源码验证）

| 问题 | 答案 |
|---|---|
| compare 模式下用例文件加载优先级？ | ① `<op>_perf_cases.jsonl`（标准命名）→ ② 任意 `*.jsonl` → ③ 任意 `*.json`（非 `.bak`）→ ④ `model.py` 的 `get_input_groups()`。同时存在时**选①**（详见下文「用例加载优先级」） |
| 怎么做标准可比较计时？ | `collect_perf_application.py`：一次 `msprof application` 覆盖完整 case 集，每例固定 10 组 `warmup → target`，取 target 中位数；大 case 集按会话上限自动拆分（详见「使用方式 1」与 [application_runner_contract.md](../upstream.md#ascend-ops-profiling-application-runner-contract-md)） |
| 怎么生成交互式 HTML 性能报告？ | 使用 [msOpProf 采集与可视化](#msopprof-采集与可视化)：已有含 `collection_manifest.json` 的采集目录用 `run_pipeline.py --mode visualize` 重渲染；有算子工程用其 `run_pipeline.py` 采集并生成报告。`msprof_profile_run.sh` 的 `PROF_GROUP_*` 归档不能直接用于此渲染链路。 |
| 怎么做加速比对比？ | `msprof_profile_run.sh --compare`（全量）/ `--quick`（1 轮快速），输出 `performance.json` + markdown 报告即完成，**不**走可视化流程 |
| quick / compare 如何保留原始 PROF？ | 推荐 `--keep-prof` 选项（quick 与 compare 模式均支持透传），作用是保留 msprof 原始 PROF 目录（默认在采集后清理），供后续深度分析或可视化工具导入；**不需要手工备份 PROF 目录，也不需要切换标准模式重新采集**。保留已有数据与补采缺失指标是不同需求，不能把重采作为保留目录的前提，也不能承诺标准模式可提供未经确认的逐指令流水数据。 |
| MC² / 多 rank（fork）算子用什么采集？ | **必须 `msprof`**，禁止 `msprof op`（详见 [msprof 采集指南](../upstream.md#ascend-ops-profiling-msprof-guide-md) 的「MC² 多 rank 算子采集」） |
| 采集结果如何判定瓶颈？ | 用 `msprof_perf_summary.py` 解析 `PROF_GROUP_*`，输出瓶颈类型与各 pipe 占比，判定规则见 [`references/msprof-guide.md`](../upstream.md#ascend-ops-profiling-msprof-guide-md)；需要 MBU/MFU/理论上限量化时用 `compute_efficiency.py`（[cost-model-contract.md](../upstream.md#ascend-ops-profiling-cost-model-contract-md)） |

#### 阅读纪律（防上下文爆炸，必须遵守）

- [`scripts/msprof_perf_summary.py`](../upstream.md#ascend-ops-profiling-scripts-msprof-perf-summary-py)（2400+ 行）与 [`scripts/msprof_profile_run.sh`](../upstream.md#ascend-ops-profiling-scripts-msprof-profile-run-sh) 是**可执行工具，不是文档**。需要确认某个行为时，只允许 `grep -n "<函数名/关键字>" scripts/msprof_perf_summary.py` 定位后**读命中处前后 ~30 行**；**禁止整文件通读**。
- `references/` 下各文档按文末「参考资源」表的"何时查阅"**按需加载**；回答本文件已写明的事实性问题时，不需要打开任何 reference 或 script。
- 本 SKILL.md 覆盖全部使用决策；若与脚本实际行为发现不一致，以脚本行为为准并在本文件勘误。

---

### 工具选择

选择依据是**外部输入与探测结果**，不得凭固定偏好选路。外部未指定时，**先探测工具可用性**（如 `which msopprof msprof`；必要时核对 `$ASCEND_HOME_PATH` / `$ASCEND_OPP_PATH` 的 CANN 安装完整性），再按探测结果分支：

1. 外部输入显式指定 `msprof` 或 `msprof op`：原样执行。调用方工作流或其他 skill 给出的 profiler 选择属于外部输入，本 skill 不解析其业务约束，也不覆盖其选择。
2. **只有 msopprof 可用**（`msprof` 不在）：走 `msprof op`。
3. **只有 msprof 可用**（`msopprof` 不在）：走 `msprof`。
4. **两者都可用**：遵循项目既有约定，并根据实际采集目标与支持情况选择；关键要求无法判断时再询问。
5. **两者都不可用**：停止采集并报告 **CANN / ASCEND_HOME 环境问题**（CANN 未安装或 `$ASCEND_HOME_PATH` / `$ASCEND_OPP_PATH` 未配置），不猜测选路。

补充规则：
- 已选定 `msprof` 后，环境不稳定导致 application 数据无法通过身份、频率、launch 映射或波动门禁：回落到 `msprof op`。必须记录回落原因，不得混合两条路径的数据做前后对比。
- 环境稳定性必须由实际探测与采集产物判断，不能仅凭工具是否安装推断。

**MC² / 多 rank（fork）算子是硬约束**：算子通过 `fork()` 创建多个子进程绑定不同 NPU 卡、子进程间经 SHMEM UDMA / BarrierAll / CrossCoreFlag 协同通信（如 alltoall_matmul、allgather_matmul、matmul_reducescatter）时，无论外部是否指定，都**必须使用 `msprof`、禁止 `msprof op`**（`msprof op` 对 fork 程序的采集行为未定义）。加载 [`references/msprof-guide.md`](../upstream.md#ascend-ops-profiling-msprof-guide-md) 的「MC² 多 rank 算子采集」章节，同时参考 Ascend C tiling and pipeline tuning guidance 的 `references/comm-compute/index.md`「性能采集方法」章节。

| 工具 | 流程文档 | 用途 |
|------|----------|------|
| **`collect_perf_application.py`** | [`references/application_runner_contract.md`](../upstream.md#ascend-ops-profiling-application-runner-contract-md) | `msprof application` 标准可比较计时：每例固定 10 组 `warmup → target`，取 target 中位数；内置 CANN 版本适配（`--aic-mode=task-based` 版本相关、不完整安装预检、单会话 relevant-launch 上限（默认 540）自动拆分与空导出重试） |
| **`msprof_profile_run.sh`** | [`references/msprof-guide.md`](../upstream.md#ascend-ops-profiling-msprof-guide-md) | `msprof` 深度指标采集、兼容对比测试和批量并行 |
| **`msprof_perf_summary.py`** | [`references/msprof-guide.md`](../upstream.md#ascend-ops-profiling-msprof-guide-md) | 统一解析入口：瓶颈分析、对比报告生成、批量汇总 |
| **`perf_summary.py`** | [`references/msprof-op-guide.md`](../upstream.md#ascend-ops-profiling-msprof-op-guide-md) | msprof op 归档（需 msopprof） |
| **`compute_efficiency.py`** | [`references/cost-model-contract.md`](../upstream.md#ascend-ops-profiling-cost-model-contract-md) | 实测耗时 → **MBU/MFU/瓶颈类型/理论上限 + 优化结论**；成本模型(FLOP/字节)可注册或 CLI 直传 |

---

### 使用方式

采集 CANNBotDSL 算子的性能摸底数据时，读取 [DSL 公开入口测量](../upstream.md#ascend-ops-profiling-cannbot-dsl-md)，按该方法区分编译、公开入口与设备耗时，并保留完整设备任务证据。

#### 1. `msprof` 标准可比较计时

一次 `msprof application` 覆盖完整 case 集。每个 case 固定执行 10 组相邻的 `warmup → target`，只统计每组 target 的完整 kernel sequence 总耗时，最终取 10 个 target 总耗时的中位数。不得改为均值、丢弃前若干次或追加样本择优。

warmup 固定按以下优先级选择：

1. 默认使用现成 ops：普通目标使用 `ArgMax`，目标属于 ArgMax 家族时使用 `Max`。
2. 只有 ops probe 失败时，才构建并使用内置 `OpsProfilingWarmup` 包。
3. ops probe 和内置包构建都失败时，采集失败。

先把内置包源码落地到目标工程，作为无 ops 环境时的构建后备；此步骤不默认编包：

```bash
python3 scripts/install_compiled_warmup.py --project-dir /path/to/operator-project
```

工程内持久 runner 有现成模板（[`references/application_runner_contract.md`](../upstream.md#ascend-ops-profiling-application-runner-contract-md)「现成 runner 模板」）：**有 torch/框架接口**用 `assets/app_runner/torch/torch_app_runner.py`；**只有 aclnn 接口**（算子已注册并编出 vendor 包、无框架绑定）用 `assets/app_runner/aclnn/` 纯 C++ 模板——warmup 直接走 aclnnArgMax/aclnnMax，不依赖 torch。两类模板只改标记的 ADAPTER 段，命令行、execution manifest 与 warmup 协议完全一致。

再按 [`references/application_runner_contract.md`](../upstream.md#ascend-ops-profiling-application-runner-contract-md) 实现工程内持久 runner，执行：

```bash
python3 scripts/collect_perf_application.py \
  --cases ./cases.csv \
  --runner-file ./run_cases.py \
  --identity-manifest ./perf-identity.json \
  --target-op Exp \
  --ops-warmup-probe-cmd 'python3 ./run_cases.py --probe-warmup-op {warmup_op}' \
  --warmup-build-cmd "bash ./ops_profiling_warmup_ext/build.sh --soc=ascend950" \
  --compiled-warmup-binary ./ops_profiling_warmup_ext/ops_profiling_warmup/_C.so \
  --application-cmd 'python3 ./run_cases.py --cases ./cases.csv --case-ids {case_ids} --samples {samples} --execution-manifest {execution_manifest} --warmup-mode {warmup_mode} --warmup-op {warmup_op}' \
  --aic-metrics PipeUtilization \
  --out-dir ./perf/baseline
```

采集和复用门禁见 [`references/msprof-guide.md`](../upstream.md#ascend-ops-profiling-msprof-guide-md) 与 runner 契约。

#### 2. `msprof` 深度瓶颈分析（标准采集模式）

对单个可执行文件采集 7 组 aic-metrics + sample-based：

```bash
bash scripts/msprof_profile_run.sh --warm-up=3 --output=./msprof_output -- \
    ./matmul_tutorial_mxfp4_pingpong 8448 4096 4096

# 解析结果
python3 scripts/msprof_perf_summary.py ./msprof_output/PROF_GROUP_* <ops_dir>
```

该入口用于补充诊断。其多指标结果不能替代第 1 节的标准 `total_us`，也不能混入 baseline/candidate 的可比较计时。

若源码含循环内跨核同步或重复目标写入，按 [`references/msprof-guide.md`](../upstream.md#ascend-ops-profiling-msprof-guide-md-f6664b48) 联合查看逐核负载、同步频次和任务依赖；把同步开销与负载不均衡作为待验证假设，不用聚合 busy 比例直接换算某个屏障的耗时。

#### 3. 对比测试模式（kernel-level 加速比）

对算子目录下的 `model.py` vs `model_new_ascendc.py` 做对比测试：

```bash
bash scripts/msprof_profile_run.sh --compare --output-dir=./output/GELU --warm-up=3 --device=0
```

输出：`performance.json` + `performance.log` + `perf_report.md`

#### 4. 快速模式（1 轮采集，只获取 kernel 时间）

对算子目录做快速对比测试，只跑 1 轮 msprof，不采集 7 个 aic-metrics：

```bash
bash scripts/msprof_profile_run.sh --quick --output-dir=./output/GELU --warm-up=3 --device=0
```

输出：`performance.json` + `performance.log` + `perf_report.md`

#### 5. 批量并行模式（多 NPU）

扫描目录下所有算子子目录，多 NPU 并行执行对比测试：

```bash
bash scripts/msprof_profile_run.sh --batch --base-dir=./output_performance --max-jobs=7 --device-start=1
```

输出：各子目录 `performance.json` + `batch_performance.log` + `batch_report.md` + `batch_summary.json`

---

### 输入目录结构（对比模式）

**传统模式（端到端自动开发）：**
```
{output_dir}/
├── model.py              # 参考 PyTorch 实现
├── model_new_ascendc.py  # AscendC 实现
├── <op_name>.json        # 测试用例（JSON Lines）
└── kernel/               # AscendC kernel + whl 包
```

**JSONL 模式（ascend-kernel 工程结构）：**
```
csrc/ops/<op>/test/
├── <op>_perf_cases.jsonl   # JSONL 用例
├── model.py                # 参考实现
├── model_new_ascendc.py    # AscendC 实现
└── kernel/                 # kernel 工程
```

#### 用例加载优先级

1. **优先**：`<op>_perf_cases.jsonl`（标准 JSONL 命名）
2. **回退**：任意 `*.jsonl` 文件
3. **回退**：任意 `*.json` 文件（非 `.bak`，JSON Lines 格式）
4. **回退**：从 `model.py` 的 `get_input_groups()` 加载

---

### 输出格式（对比模式）

Markdown 报告包含：
- **对比表**：`Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比`
- **全量汇总**：用例数、平均加速比、自定义/标杆更优条数
- **按数据类型汇总**：分 dtype 的统计
- **简短分析**：整体趋势结论
- **深度瓶颈分析入口**：提供 msprof 深度分析命令

额外输出：
- `performance.json` — 结构化数据（含 geomean/mean/median/min/max 加速比）
- `performance.log` — 屏显日志

---

### 深度分析：msprof / msprof op

当性能不达标或需要根因分析时，使用深度分析流程。

**arch22（A2/A3）支持详细性能分析**，可按 [CANN 官方算子性能分析文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/devaids/opdev/optool/atlasopdev_16_0097.html) 解读采集结果。该链接对应 CANN 8.0.0 alpha001；实际参数、指标与功能以目标 CANN/工具版本为准。arch35 的 RegBase/VF/RVEC 指标不用于推断 arch22 瓶颈，缺失字段不按零值处理。

工具路由只使用上方「工具选择」一节，不得在深度分析阶段重新选路。选择 `msprof` 时加载 [`references/msprof-guide.md`](../upstream.md#ascend-ops-profiling-msprof-guide-md)；选择 `msprof op` 时加载 [`references/msprof-op-guide.md`](../upstream.md#ascend-ops-profiling-msprof-op-guide-md)。

#### 可视化报告（上板数据 → HTML）

当用户要求把上板采集数据可视化为交互式 HTML 报告（Details、Roofline、Timeline、Cache 热力图、Source 等页面）时，按 [msOpProf 采集与可视化](#msopprof-采集与可视化) 选择工作流：

- 有算子工程可上板 replay → 用其 `run_pipeline.py` 做最小非重复采集并生成报告；
- 已有含 `collection_manifest.json` 的采集目录 → 只用 visualize 模式重渲染，不得重复采集。

以下场景**不进入**可视化流程：

- 加速比对比（compare / quick / batch）：输出 `performance.json` + markdown 即完成；
- MC² / fork 多进程算子：子技能基于 `msprof op`，该场景对 `msprof op` 禁用；
- 纯瓶颈分析或 CSV 指标解读且用户未提及可视化。

注意两套链路输出不互通：本技能的 `PROF_GROUP_*` 归档不能直接喂给可视化渲染，可视化流程的采集目录结构由其自身 `collect.py` 产生。

#### 参考资源

| 文件 | 内容 | 何时查阅 |
|------|------|---------|
| [`references/msprof-guide.md`](../upstream.md#ascend-ops-profiling-msprof-guide-md) | `msprof`：标准可比较计时、深度采集 / 归档 / **主 Bound 判定** / **MC² 多 rank 采集**（文末章节） | 选用 `msprof` 时，或 MC² / fork 多进程算子 |
| [`references/msprof-op-guide.md`](../upstream.md#ascend-ops-profiling-msprof-op-guide-md) | `msprof op`：构建 / 采集 / 归档 / 判定 / 瓶颈 / 回归 | 外部选择或环境回落到 `msprof op` 时（不适用于 MC²） |
| [`references/application_runner_contract.md`](../upstream.md#ascend-ops-profiling-application-runner-contract-md) | application runner、manifest、launch 映射与中位数协议 | `msprof` 可比较计时 |
| [`references/csv_fields_reference.md`](../upstream.md#ascend-ops-profiling-csv-fields-reference-md) | CSV 字段定义与阈值 | 理解指标含义时 |
| [`references/optimization_quickref.md`](../upstream.md#ascend-ops-profiling-optimization-quickref-md) | 瓶颈类型与优化方法 | 定位瓶颈后 |
| [`references/cost-model-contract.md`](../upstream.md#ascend-ops-profiling-cost-model-contract-md) | 成本模型契约、MBU/MFU/天花板计算、峰值口径 | 需要量化 bound / 天花板 / 前置剪枝时 |
| [`references/compiled_warmup.md`](../upstream.md#ascend-ops-profiling-compiled-warmup-md) | 内置 warmup 包的构建与使用 | ops probe 失败需要构建后备 warmup 时 |

---

### 错误处理

- 外部显式指定的工具不可用：直接报告缺失，不擅自换工具。
- `msprof` 路径持续无法通过环境或数据门禁：保存失败证据，记录原因后回落到 `msprof op`。
- application launch 映射不完整、乱序或身份不一致：整份 collection 作废，不手工补值。

---

### 适用场景总览

| 场景 | 推荐命令 | 说明 |
|------|----------|------|
| 标准可比较性能采集 | `collect_perf_application.py` | `msprof application`，每例 10 组 `warmup → target`，取 target 中位数；大 case 集自动按会话上限拆分 |
| **MC² 多 rank 算子采集** | `msprof --aic-mode=task-based` | **必须用 msprof，禁止 msprof op**，详见 msprof-guide.md「MC² 多 rank 算子采集」章节 |
| 算子开发完成后的性能验收 | `msprof_profile_run.sh --compare` | 快速对比自定义算子 vs 标杆 |
| 性能问题定位 | `msprof_profile_run.sh` + `msprof_perf_summary.py` | 深度瓶颈分析 |
| 优化效果验证 | `msprof_profile_run.sh` + `msprof_perf_summary.py` | 对比优化前后的归档数据 |
| 算子生成阶段快速测试 | `msprof_profile_run.sh --quick` | 快速对比测试（1 轮采集，只获取时间） |
| Agent team 测试阶段 | `--quick` / `--compare` + 标准采集 | 先快速对比，不达标再深度分析 |
| 端到端自动开发 Phase 5 | `msprof_profile_run.sh --quick` | 集成到 tilelang2ascendc-ops-generator 流程（只获取加速比） |
| 进化优化前基线测试 | `msprof_profile_run.sh --quick` | 快速获取基线加速比 |
| 批量性能测试 | `msprof_profile_run.sh --batch` | 多 NPU 并行批量测试 |
| 上板数据可视化报告 | [msOpProf 采集与可视化](#msopprof-采集与可视化) | 交互式 HTML：Details/Roofline/Timeline/Cache/Source 等页面 |

## msOpProf 采集与可视化

Identify the operator executable, environment and existing collection manifest. Reuse completed collections, collect only supported requested metrics, and distinguish unavailable metrics from zeros. This guide documents complete collection, visualize-only, targeted features and resulting report artifacts.

文中的 `scripts/…` 命令需在 [可视化工具目录](../upstream.md#ascend-msopprof-visualization-workspace) 中运行。请保留完整目录和依赖，并用绝对路径指向采集数据。

### 职责

将算子工程或已有 msOpProf 采集目录转换为可验证的独立 HTML 性能报告。

本技能负责：

1. 识别 CANN、msprof、设备、可执行文件和应用工作目录；
2. 按请求模块规划最少且不重复的 profiler replay；
3. 对命令结果和产物进行语义校验，安全复用已完成采集；
4. 构造统一报告载荷；
5. 生成交互式 HTML，并明确区分“可用、缺失、不可支持”；
6. 输出机器可读的验证结果和性能计时。

不得推测或伪造 profiler 未提供的指标。缺少证据时，必须显示不可用原因，或在非定向报告中明确省略。

### 输入模式

#### 1. 完整采集并生成报告

需要：

- `--operator-path`：算子工程根目录；
- `--output`：采集和报告输出目录；
- 可自动发现或显式指定的 `--app`；
- 应用必需参数，使用可重复的 `--app-arg`；
- 可选的 `--kernel-name`。

```bash
python scripts/run_pipeline.py \
  --operator-path /path/to/operator \
  --output /path/to/output \
  --preset complete
```

#### 2. 只使用已有采集结果生成报告

已有目录必须包含 `collection_manifest.json`。

```bash
python scripts/run_pipeline.py \
  --mode visualize \
  --output /path/to/collection \
  --preset complete
```

也可直接运行：

```bash
python scripts/visualize.py \
  --input /path/to/collection \
  --output /path/to/report \
  --preset complete
```

#### 3. 定向生成模块

```bash
python scripts/visualize.py \
  --input /path/to/collection \
  --output /path/to/report \
  --feature roofline \
  --feature timeline \
  --feature cache
```

查看可用特性：

```bash
python scripts/visualize.py --list-features
python scripts/visualize.py --explain source
```

不得自行创造特性名称。

### 标准工作流

#### 步骤 1：判断执行模式

- 用户提供算子工程：默认 `full`；
- 用户只需要采集：使用 `collect`；
- 用户已有 `collection_manifest.json`：优先 `visualize`，不得重复采集；
- 只需要检查命令规划：先使用 `--dry-run`。

#### 步骤 2：解析环境

稳定主机可建立可复用环境档案：

```bash
python scripts/environment_context.py init \
  --profile /stable/path/msopprof_environment.json \
  --base-dir /path/to/operator \
  --source 'source /usr/local/Ascend/ascend-toolkit/set_env.sh'
```

环境档案只允许复用稳定的 CANN 环境变量、`msprof` 路径和 CLI 能力信息。它不得跳过算子相关的可执行文件、Kernel、调试符号、插桩和产物校验。

涉及环境复用或失效时，读取 [`references/environment-and-failure.md`](../upstream.md#ascend-msopprof-visualization-environment-and-failure-md)。

#### 步骤 3：执行短时预检

正式采集前默认执行一次 `BasicInfo` canary：

- `--launch-count=1`；
- 使用解析出的应用工作目录；
- 使用短时超时；
- 校验返回码和实际产物。

超时、非零返回或语义空产物均视为基础设施失败。立即中止，不得继续执行 Occupancy、Roofline、Timeline、Source 等昂贵采集。

所有 profiler 命令必须在独立进程组中运行。超时后先终止、再强制清理完整进程树，并保存日志和诊断信息。

#### 步骤 4：选择最小采集计划

根据请求页面反推采集块，并执行语义去重：

- `Roofline` 已包含标准 Default 指标时，不额外执行相同 Default replay；
- `MemoryDetail` 缺失时，可在 Roofline 产物通过内存模块语义校验后复用；
- Raw Data 优先复用 MemoryDetail，其次复用 Roofline；
- 单次精确 Kernel、单次 launch 的成功 preflight 可复用为 Discovery；
- Timeline、Source、Occupancy 等语义独立模块不得仅因 CLI 支持逗号组合而合并。

详细规则读取 [`references/collection-workflow.md`](../upstream.md#ascend-msopprof-visualization-collection-workflow-md)。

#### 步骤 5：语义校验产物

命令返回码为 0 不等于采集成功。必须校验：

- profiler 输出目录存在；
- 必需文件存在；
- 产物中包含可解析、非空且符合模块语义的数据；
- alias/reuse 的来源和原因已写入 manifest；
- 顶层 `_internal/run_state.json` 为 `completed`，才允许后续复用。

Source 需要精确构建树中的 `.debug_line`。`-g` 是编译参数，不是 msprof 参数。不得把二进制 printable strings 当作完整源码映射。

#### 步骤 6：独立构建报告模块

支持页面：

- `details`：基本信息、核占用率、计算、内存和建议；
- `roofline`：实测点、带宽线、算力上限和交互坐标；
- `timeline`：原生事件、lane、范围选择和切片详情；
- `cache`：Hit/Miss、Block、Cache family 和统计；
- `onchip-memory`：仅使用编译器 `memory_info.json`；
- `source`：源码文件、行、地址、指令和显式关联；
- `warp-stall`：仅使用 SIMT/PCSampling 证据；
- `instruction-timeline`：仅使用受支持的指令时间线产物；
- `raw-data`：原始 CSV 表格。

模块数据合同读取 [`references/visualization-contract.md`](../upstream.md#ascend-msopprof-visualization-visualization-contract-md)。

#### 步骤 7：生成完整特性报告

要在同一报告中展示所有支持页面，并对缺失模块保留诊断页，使用：

```bash
python scripts/visualize.py \
  --input /path/to/collection \
  --output /path/to/report \
  --feature details \
  --feature roofline \
  --feature timeline \
  --feature cache \
  --feature onchip-memory \
  --feature source \
  --feature warp-stall \
  --feature instruction-timeline \
  --feature raw-data \
  --unavailable-policy explain \
  --report-name msopprof_complete_report.html
```

报告较大时可用 `--compress-payload on` 将载荷以 gzip 内嵌（体积约降低一个数量级，需现代浏览器解压）；默认 `off` 为纯 JSON 内嵌，兼容性最好。两种模式渲染内容一致。

需要预估各 preset 的采集耗时（不重跑 profiler）时：

```bash
python scripts/estimate_runtime.py \
  --timing /path/to/output/_internal/timing_summary.json
```

#### 步骤 8：验证与交付

运行结构和真实数据验证：

```bash
python scripts/self_check.py \
  --collection /path/to/collection \
  --output /path/to/validation-output
```

交付时至少说明：

- HTML 报告路径；
- `report_index.json`；
- 可用模块；
- 不可用模块及原因；
- 运行耗时和峰值内存；
- 验证是否通过。

不得让用户打开 `templates/report_template.html`。应打开由 `--report-name` 指定的最终报告，并返回其绝对路径。

### 输出合同

默认输出包括：

```text
OUTPUT/
├── collection_manifest.json
├── feature_catalog.json
├── 00_discovery/
├── 01_details/
├── 02_roofline/
├── 03_timeline/
├── 04_source/
├── 05_warp_stall/
├── 06_instruction_timeline/
├── 07_memory_detail/
├── 08_raw_data/
├── 09_timeline_detail/
├── 10_kernel_scale/
├── 11_onchip_memory/
├── _internal/
├── report_payload.json
├── report_index.json
└── report.html
```

`report_index.json` 是机器可读交付回执。`collection_manifest.json` 是采集真源。

### 强制规则

1. 不得伪造缺失指标或从不等价数据推导模块。
2. `memory_info.json` 缺失时，不得用 MemoryDetail 计数伪造片上内存分配寿命和地址。
3. 不得复用 `running`、`aborted`、超时、非零返回或语义空的采集结果。
4. 采集目录不得具有 group/other write 权限；发现后应修正并记录。
5. `--app` 一旦指定即为权威；未指定时必须记录候选排序和选择理由。
6. 应用工作目录默认是 `--operator-path`，不是调用命令时的 shell 目录。
7. Timeline 和 Source 成本较高；只有请求其独有证据时才执行。
8. 报告页面必须使用同一 canonical payload，不得为不同页面重复解析并产生不一致数据。
9. 完整报告允许模块不可用，但必须提供证据和下一步动作。
10. 所有正式提交不得包含 `.pytest_cache`、`__pycache__`、`*.pyc`、运行日志或历史验证结果。

### 参考资料

按需读取，避免一次性加载全部内容：

- [`references/collection-workflow.md`](../upstream.md#ascend-msopprof-visualization-collection-workflow-md)：采集计划、参数和 replay 去重；
- [`references/environment-and-failure.md`](../upstream.md#ascend-msopprof-visualization-environment-and-failure-md)：环境档案、预检、超时和诊断；
- [`references/data-contract.md`](../upstream.md#ascend-msopprof-visualization-data-contract-md)：manifest、alias、可用性和输出真源；
- [`references/visualization-contract.md`](../upstream.md#ascend-msopprof-visualization-visualization-contract-md)：各可视化模块的数据与交互合同；
- [`references/validation-and-performance.md`](../upstream.md#ascend-msopprof-visualization-validation-and-performance-md)：验证、浏览器检查和性能门禁。

## Ascend C 算子运行时错误调试

系统化调试 Ascend C 算子运行时错误，包括错误码处理、环境检查。

### 快速诊断

```
运行时错误
    │
    ├─ 返回码非0？
    │   ├─ 161xxx → 参数错误 → 见高频错误速查
    │   ├─ 361xxx → Runtime错误 → 见 debug_workflow.md
    │   ├─ 507035 → 向量核异常 → 见 debug_workflow.md#507035-向量核异常
    │   └─ 561xxx → 内部错误 → 见高频错误速查 / debug_workflow.md
    │
    └─ 复杂场景 → 开启日志调试
```

### 高频错误速查

| 错误码 | 类型 | 排查方向 | 详细方案 |
|-------|------|---------|---------|
| 507035 | 向量核异常 | 检查 UB 端地址 32B 对齐（含 VEC 指令操作数）/ GM→UB padParams 违规 / UB 溢出 | [debug_workflow.md](../upstream.md#ascend-ops-profiling-ops-ascendc-runtime-debug-references-debug-workflow-md-507035-89f36489) |
| 161xxx | 参数错误 | 检查 dtype/shape/nullptr | [debug_workflow.md](../upstream.md#ascend-ops-profiling-ops-ascendc-runtime-debug-references-debug-workflow-md-161xxx-dd168bfc) |
| 561002 | Tiling错误 | 检查 TilingKey/TilingFunc | [debug_workflow.md](../upstream.md#ascend-ops-profiling-ops-ascendc-runtime-debug-references-debug-workflow-md-561002-tiling-2aaed635) |
| 561003 | Kernel未找到 | 检查算子安装/环境配置 | [debug_workflow.md](../upstream.md#ascend-ops-profiling-ops-ascendc-runtime-debug-references-debug-workflow-md-561003-kernel-3ad3f0f2) |

更多错误码见 [error_codes.md](../upstream.md#ascend-ops-profiling-ops-ascendc-runtime-debug-references-error-codes-md)

### 未知错误码处理

遇到速查表中未列出的错误码时，见 [debug_workflow.md](../upstream.md#ascend-ops-profiling-ops-ascendc-runtime-debug-references-debug-workflow-md-62264d4c)

### 调试工具速查

| 工具 | 用途 | 使用 |
|-----|------|------|
| `aclGetRecentErrMsg()` | 获取错误详情 | 返回码非0时调用 |
| 环境检查 | ascendc-env-check skill | 见 skill:ascendc-env-check |
| [`parse_plog.py`](../upstream.md#ascend-ops-profiling-parse-plog) | 日志解析 | `python3 /path/to/parse_plog.py <plog_file>` |
| `ASCEND_SLOG_PRINT_TO_STDOUT=1` | 日志打屏 | 实时查看日志 |

详细调试方法见 [debug_workflow.md](../upstream.md#ascend-ops-profiling-ops-ascendc-runtime-debug-references-debug-workflow-md-19cc05b6)

### 详细资源

- **[debug_workflow.md](../upstream.md#ascend-ops-profiling-ops-ascendc-runtime-debug-references-debug-workflow-md)**：详细调试流程（错误码处理、环境检查）
- **[kernel_binary_debug.md](../upstream.md#ascend-ops-profiling-ops-ascendc-runtime-debug-references-kernel-binary-debug-md)**：Kernel 二进制构建调试（编译缓存、SEL匹配、多版本冲突、opParaSize）
- **[error_codes.md](../upstream.md#ascend-ops-profiling-ops-ascendc-runtime-debug-references-error-codes-md)**：完整错误码表（基本状态码、内部异常状态码）
- [ascend-environment](../ascend-environment/SKILL.md) 提供设备和 CANN 环境检查流程。
