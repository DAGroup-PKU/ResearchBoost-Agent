---
name: ascendc-perf-optimize
description: Diagnose operator bottlenecks and choose Ascend C tiling and pipeline optimizations.
---

# ascendc-perf-optimize

Use profiling evidence to choose card-level, inter-core and single-core tuning. Read metrics and the ping-pong example when the hypothesis matches.

Read and apply the skill workflow; do not invoke the skill itself as a Bash command. Commands inside a selected workflow can be run in their documented environment. Use the task's numerical and performance verdicts to validate any change.

## Contents

- [算子公共性能优化流水分析](#算子公共性能优化流水分析)
- [采集与指标口径（msprof 统一）](#采集与指标口径msprof-统一)
- [技术卡片 mem-02：Ping-Pong 预取下一 tile](#技术卡片-mem-02ping-pong-预取下一-tile)

## 算子公共性能优化流水分析

> **⚠️ 重要**：本技能定义了 **4 步分层流水**优化流程（Step 1 → Step 2 → Step 3 → Step 4），而非简化的 3 步流程。Step 2（卡间流水）、Step 3（核间流水）、Step 4（单核流水）是三个独立的流水分析步骤，各有明确的适用条件、输入输出要求。请勿将它们合并为单一的"流水分析"步骤。

本技能的四步流程和通用建模不按代际拆分。资源参数取自目标平台；知识卡片中的架构、API 和实测范围按各自标注使用。**RegBase / Reg / MicroAPI / VF 相关实现仅适用于 arch35（A5 / DAV_3510）**，不适用于 arch22（A2/A3 / DAV_2201）。需要核对代际相关知识时查 [适用范围说明](../upstream.md#ascendc-perf-optimize-architecture-routing-md)。

### 优化流程总览

| 步骤 | 名称 | 适用条件 | 关注点 |
|------|------|---------|--------|
| **Step 1** | Tiling 理论建模 | 所有算子 | 确定理想 tiling data |
| **Step 2** | 卡间流水优化 | 通信类算子 | 通算演算、卡间通信瓶颈。**MC² 算子须先采集 TilingData 和隔离测试数据（见 `comm-compute/index.md` Step 0/2），缺失时返回数据采集阶段补采** |
| **Step 3** | 核间流水优化 | 多核间同步算子 | 核间并行效率、同步开销 |
| **Step 4** | 单核流水优化 | 所有算子 | 核内流水、bound 诊断 |

**关键特征**：
- ✅ **4 步分层流水**，不是 3 步
- ✅ Step 2/3 有明确的**适用条件**，非适用算子跳过
- ✅ 每步都有独立的**仿真图分析 + profiling 报告 + 优化策略**输出
- ✅ Step 2/3/4 都会输出对 Step 1 tiling 策略的**修正建议**

### 优化流程

```
给定：算子类型、计算流程（kernel 代码/伪代码）、profiling 数据、仿真图（可选）

Step 1 — Tiling 理论建模 → 输出理想 tiling data
Step 2 — 通信类算子？ → 通算演算优化策略（卡间流水分析 + tiling 修正）
Step 3 — 多核间同步算子？ → 核间流水优化策略（核间流水分析 + tiling 修正）
Step 4 — 单核流水优化策略（核内流水分析 + tiling 修正）
```

每步的"流水"分析包含**仿真图解读**和 **profiling 数据**分析报告。Step 2/3/4 的策略输出均包含对 Step 1 tiling 策略的修正建议。

---

### Step 1 — Tiling 理论建模

**输入**：精确 SoC / DAV 分组、目标资源与来源、算子类型、Shape、dtype、计算流程

**过程**：根据算子 pattern 路由到对应的 Tiling 理论模型目录（详见 `references/tiling/`，入口为 [`references/tiling/index.md`](../upstream.md#ascendc-perf-optimize-tiling-index-md)），输出理想 tiling data。

**MatMul 默认分析**：按 [MatMul 入口](../upstream.md#ascendc-perf-optimize-tiling-matmul-index-md) 读取通用理论与目标架构参考，结合源码、Shape、dtype 和已有资料，给出几个可能的优化方向及简要依据即可。下列输出清单用于组织分析，不要求默认分析补齐完整 tiling data 或候选集；缺少实测证据时说明假设。只有用户明确要求执行实测，或接受已说明范围与预计耗时的实测方案后，才进入实测求解器，不因本步骤或后续流水分析需要验证而自动启动。

**输出**：
- [ ] 卡间切分方案（切分维度、通信内算子涉及）
- [ ] 多核切分方案（切分维度、单核任务量、核数）
- [ ] 单核切分方案：
  - Cube/融合类：L1 split（baseM/baseN/baseK、L1 ping-pong）+ L0 split（mL0/nL0/kL0）
  - Vec 类：UB split（block_size、repeat）
- [ ] Buffer 规划（各 buffer 用途与大小，区分 L1/L0/UB 层级）
- [ ] 融合算子内部计算单元清单；每个内嵌 MM 单独记录 `M/N/K`、dtype、layout 和 `mm_route`
- [ ] 内嵌 MM 的 L1/L0/输出流水：`L1 → L0(A2/B2) → Mmad(Cube) → Fixpipe/DataCopyCO12DstParams`、双缓冲和事件配对
- [ ] 分支场景覆盖（dtype、shape 大小、对齐）

---

#### 内嵌 AscendC MatMul 路由（Cube/融合类强制）

融合算子不能只按外层名称归类。只要计算流程中出现 `mm`/`matmul`/`mmad`，就把它作为独立 MatMul 单元完成 Step 1 建模，并将其 tiling、流水和候选路线写入报告

候选路线按以下顺序评估：

1. DAV_3510 先完成 Blaze 路线判定与评估（`blaze_native`/`blaze_custom`/`blaze_migration`/`unsupported`），命中 Blaze 路线后转 [Blaze 算子开发资料](../upstream.md#ascendc-blaze-development)（迁移路线为 [Blaze 迁移资料](../upstream.md#ascendc-blaze-migration)）执行。
2. Blaze 不适用、判定为 `unsupported` 或瓶颈仍在 MM 内部时，评估保留 Cube MMAD 的低阶流水重建：显式规划 `A2/B2/CO1` LocalTensor，使用 `LoadData2DParamsV2` 控制 L1 到 L0，使用 `MmadParams` + `Mmad` 执行 Cube 计算，并用 `Fixpipe`/`DataCopyCO12DstParams` 和 `MTE1`/`MTE2`/`MTE3` 事件控制流水。
3. 低阶流水重建受限时，再评估现有高阶 MatMul API 的 tiling、数据流、ping-pong、`unitflag`、`k-tail` 和 `Fixpipe` 配置。
4. AIV 替换是独立的计算引擎替换路线，只能在 Blaze、低阶 Cube、Cube 高阶和 AIV 的同 case、同环境、同协议实测对照后进入候选。低 MAC 利用率、较小 `K` 或“外层是融合 backward”不是充分证据。

“低阶 Cube MM 重建”与“把 MM 改成 AIV 小算子组合”必须使用不同的 `route_id` 和收益假设；`register-resident-vector-chain` 只用于已确认的向量计算链，不得用来证明 MM 应迁移到 AIV。

---

### Step 2 — 通算演算优化策略（卡间流水）

**适用条件**：通信类算子（如 AllReduce、AllGather、ReduceScatter 等）——含无计算融合的纯集合通信算子。

非通信类算子**跳过**此步骤。

> **边界澄清**：不要按"是否为 MC² 通算融合"来判定 Step 2 适用性——MC² 与否只决定下方"前置强制检查"和输出清单中的 MC² 必选项是否生效；纯集合通信算子（无内嵌计算单元）同样执行本步骤，只是无需 MC² 采集项，按通信掩盖与常规卡间流水优化推进。

**输入**：Step 1 的 tiling data + 计算流程 + 仿真图 + profiling 数据

> **⚠️ MC² 算子前置强制检查**：若算子为 MC² 通算融合算子（存在 fork 多 rank + CrossCoreFlag 同步），在进入流水配平分析前**必须**完成以下两项数据采集，缺失任何一项须返回数据采集阶段补采，不得在缺少数据的情况下凭经验给出切分方案：
>
> 1. **TilingData 采集**：从 host 代码打印获取 baseM/baseN/baseK/usedCoreNum。若 host 代码未打印，须在源码副本中启用 PrintTilingData 后重新运行。详见 `comm-compute/index.md` Step 0 和 `comm-compute/pipeline_balancing.md`「TilingData 依赖」章节。
> 2. **隔离测试**：在 tileCnt=1 下分别注释通信/计算 process，实测 T_comm 和 T_compute，计算 R 值判定 Bound 类型。详见 `comm-compute/bound_diagnosis.md`。

**过程**：加载 `references/comm-compute/`，按「远端卡数据经卡间链路（HCCS/NVLINK）到达本卡 UB，与本卡计算的流水重叠」进行通算演算，识别通信瓶颈。

**报告结构（四项必须逐项出现在报告中；场景数据不全时基于已知 profiling 结论推演、缺失项列为待补采，不得以"仅给出方向"省略任何一项）**：
1. **输入与适用性**：完整列出 Step 2 的输入——Step 1 的 tiling data、计算流程、仿真图与 profiling 数据；声明适用边界：通信类算子（AllReduce/AllGather/ReduceScatter 等，含纯集合通信与 MC² 融合）必做本步，**非通信类算子跳过此步骤**。
2. **通算演算（必须实际推演）**：以 profiling 结论为输入，演算「远端卡数据经卡间链路（HCCS/NVLINK）到达本卡 UB」与本卡计算的逐 chunk 重叠关系——通信段/计算段/等待段如何交替、瓶颈在哪一段（隔离测试 T_comm/T_compute/R 值、流水配平、膨胀分析、性能指标核算），不得只列策略清单。
3. **产出**：卡间流水仿真图分析（分块流水时间轴推演：收 chunk k+1 与算 chunk k 的重叠）、通信/profiling 数据报告、卡间流水优化策略。
4. **对 Step 1 tiling 策略的修正建议**：给出具体修正项与取值方向（如长块/短块各自独立 tiling data、块大小与 tileCnt 的调整方向及其约束公式），不做纯方向性描述。

**输出**：
- [ ] 卡间流水仿真图分析
- [ ] 通信/profiling 数据报告
- [ ] **TilingData 采集**（MC² 算子必选）：baseM/baseN/baseK/usedCoreNum（含长块/短块各自的 tiling data）
- [ ] **隔离测试与 Bound 判定**（MC² 算子必选）：T_comm/T_compute/R 值
- [ ] 卡间流水优化策略
- [ ] 对 Step 1 tiling 策略的修正建议
- [ ] **加速比与掩盖率**（MC² 算子必选）：基于隔离测试的 T_comm / T_compute 和各方案实测耗时，按 [`references/comm-compute/index.md`](../upstream.md#ascendc-perf-optimize-comm-compute-index-md) 的性能指标公式核算

> 加载 `references/comm-compute/`，按 Bound 判定 → 流水配平 → 膨胀分析 → 性能指标核算流程输出卡间流水优化策略与 Tiling 修正建议。

---

### Step 3 — 核间流水优化策略

**适用条件**：涉及多核间同步的算子（如跨核同步、核间数据依赖等）。

非多核同步算子**跳过**此步骤。

**输入**：Step 1 的 tiling data + 计算流程 + 仿真图 + profiling 数据

**过程**：加载 `references/inter-core-pipeline/`，分析多核间的流水并行效率。

当循环中反复出现跨核汇合，或多个核向数据决定的同一目标写入时，先按[核间依赖与同步分析](../upstream.md#ascendc-perf-optimize-inter-core-pipeline-index-md)建立生产者、可见性和归约顺序约束，再估算分块、分组归约等候选的同步次数、额外流量与空间代价。候选方案须满足原计算语义；不能仅凭聚合 profiling 推断某次同步的独立耗时。

**输出**：
- [ ] 核间流水仿真图分析
- [ ] 核间 profiling 数据报告
- [ ] 核间流水优化策略
- [ ] 对 Step 1 tiling 策略的修正建议

> 无跨核同步或共享目标写入时跳过本步骤；资料不足时列出缺失证据和可验证假设，不将猜测写成已确认瓶颈。

---

### Step 4 — 单核流水优化策略

**适用条件**：所有算子。

**输入**：Step 1 的 tiling data + 计算流程 + 仿真图 + profiling 数据 + 前几步的策略输出

**过程**：加载 `references/single-core-pipeline/`，按其两级结构展开——`index.md` 路由判定 bound → 对应诊断页（vec/scalar/memory/no-bound.md）看细分特征 → `techniques/` 取技术卡片实施。采集与指标口径统一见 `metrics.md`（msprof）。

**Bound 判定链（报告必须先完整枚举四类，再写排除过程）**：

| bound 类型 | 判据（仿真图/profiling） | 诊断页 |
|---|---|---|
| Scalar Bound / 小 case | scalar 耗时占比高，或计算量小 | scalar.md |
| 访存 Bound | MTE2/访存单元利用率高，带宽为主要限制 | memory.md |
| Vec Bound | Vector 单元利用率高，向量指令为主要限制（访存带宽充足、MTE 相对空闲） | vec.md |
| 无 Bound | 各单元利用率均不高（流水并行不足） | no-bound.md |

枚举后逐类写明排除依据（如「因访存带宽充足排除访存 Bound，命中 Vec Bound」），最后落到命中类型并输出对应诊断页策略；MC² 前置检查与下钻规则见 `single-core-pipeline/index.md`。

**输出**：
- [ ] 核内流水仿真图分析
- [ ] 核内 profiling 数据报告（bound 诊断结论）
- [ ] 核内流水优化策略
- [ ] 对 Step 1 tiling 策略的修正建议
- [ ] 最终优化方案汇总

---

### ⚠️ 常见误解澄清

#### 误解1：将流程简化为 3 步
❌ **错误理解**：Tiling 建模 → 流水分析 → Tiling 回修  
✅ **正确理解**：Tiling 建模（Step 1）→ 卡间流水（Step 2，条件）→ 核间流水（Step 3，条件）→ 单核流水（Step 4）

**说明**：Step 2/3/4 不是一个"流水分析"步骤的不同维度，而是三个独立的步骤，各有明确的适用条件和输出要求。

#### 误解2：将 Step 2/3/4 合并为"流水分析"
❌ **错误**：Step 2/3/4 是同一个"流水分析"步骤的卡间、核间、核内三个维度  
✅ **正确**：Step 2/3/4 是三个独立的分析步骤，按条件选择性执行，各自产生独立的优化策略输出

**说明**：
- Step 2 仅适用于通信类算子（如 AllReduce）
- Step 3 仅适用于多核间同步算子
- Step 4 适用于所有算子
- 非适用算子会跳过对应步骤

#### 误解3：混淆卡间通信和核内搬运的概念
❌ **错误**：认为卡间流水只关注计算，不关注数据搬运  
✅ **正确**：Step 2 卡间流水关注卡间数据通信与计算的重叠（通算演算）

**说明**：
- 卡间通信涉及的 DDR→UB 是指**远端卡的数据**通过卡间链路（HCCS/NVLINK）传输到**本卡 UB**
- 这与 Step 4 核内流水中的 DDR→UB（**本卡** HBM/DDR → 本卡 L1 → 本卡 UB）是不同的数据路径
- Step 2 关注：远端数据通过卡间链路到达本卡，与本卡计算的流水重叠（通算演算优化）
- Step 4 关注：本卡内部的数据搬运（本卡 GM→L1→UB）与本卡计算的流水重叠

#### 误解4：所有算子都执行全部 4 个步骤
❌ **错误**：每个算子都要经过 Step 1 → Step 2 → Step 3 → Step 4  
✅ **正确**：Step 1 和 Step 4 是必选，Step 2/3 根据算子类型选择性执行

**典型流程示例**：
- **普通计算算子**（如 MatMul）：Step 1 → Skip Step 2 → Skip Step 3 → Step 4
- **通信类算子**（如 AllReduce）：Step 1 → Step 2 → Skip Step 3 → Step 4
- **多核同步算子**：Step 1 → Skip Step 2 → Step 3 → Step 4

## 采集与指标口径（msprof 统一）

> 通用诊断适用于 arch22 / arch35，具体指标与卡片按目标能力选用；RegBase / Reg / MicroAPI / VF / RVEC 相关内容仅适用于 arch35（A5），示例阈值不作为跨代默认值。

> 本页是 single-core-pipeline 的**统一采集与读图口径**。诊断页（vec/scalar/memory/no-bound）与技术卡片库（`techniques/`）出现的所有指标、泳道、判定阈值，一律以本页定义为准。
> 采集工具统一为 **msprof**（上板 profiling）+ **CANN Simulator**（仿真 trace），不使用任何外部私有采集工具。

---

### 1. 采集

两条采集路径，按目标支持范围及有无 NPU 选择；本库 `ops-simulator` 路线仅用于其支持的 950 目标，A2/A3 缺板时记录动态证据缺口，不用 A5 仿真替代：

| 场景 | 工具 | 技能入口 | 产物 |
| --- | --- | --- | --- |
| 上板实测 | msprof（逐 case `aiv_time`） | Ascend operator profiling tools guidance | `kernel_details.csv`（含 aiv_time/各 ratio）、`op_summary.csv` |
| 无硬件仿真 | CANN Simulator | `/ops-simulator` | 仿真流水 trace（chrome trace json）、指令时间线 |

- **逐 case 采集**：性能优化工作流对 `{cases_csv}` 中**每个 case** 单独采一次 `aiv_time`，禁止只挑代表性用例；执行协议取调用方与 `ops-profiling`；CANN-Bench A5 的采集流程仅适用于该目标/场景，不能替代 A2/A3 或普通任务协议。
- **仿真 trace** 用于在无板环境看流水形态（空泡、泳道占用）；**最终性能结论以上板 msprof `aiv_time` 前后对比为准**，仿真只作定性初判。

---

### 2. 指标词汇表

| 指标 | 定义 | 从哪读 |
| --- | --- | --- |
| `aiv_time` | AI Vector core 单 case 执行耗时（us） | msprof `kernel_details.csv`，优化前后比这个 |
| `aiv_vec_ratio` | Vector（计算）指令耗时占 aiv_time 比例 | msprof；判 VEC bound 主指标 |
| `aiv_mte2_ratio` | MTE2（GM→UB 载入）耗时占比 | msprof；判访存 bound |
| `aiv_scalar_ratio` | 标量指令耗时占比 | msprof；判 scalar bound |
| 各单元 util | SCALAR/MTE2/VECTOR/MTE3 泳道忙碌时长 / 总跨度 | 仿真 trace 或 msprof ratio |
| lane 利用率 | 一条向量指令实际点亮 lane 数 / 满 lane（VL） | 仿真 trace 逐指令 mask 宽度推算 |
| 指令发射率 (IPC) | 每周期发射指令数 | 仿真 trace 定性；无绝对达标线，只做前后对比 |
| `aiv_mte3_wait_ratio` | 写回泳道处于等待的时间占比 | msprof `ResourceConflictRatio.csv`；> 0.4 多为消费阶段紧跟异步计算（nobound-02） |
| 重叠因子 | 各泳道忙碌时长之和 / 实际耗时（见 §2.1） | msprof `PipeUtilization.csv`；理想上界 `Σ / max`，越接近说明流水重叠越充分 |

#### 2.1 重叠因子：判"流水并行够不够"的入口指标

只用 `PipeUtilization.csv` 一张表即可，**上板随时能算，不需要仿真**：

```
重叠因子 = Σ(aiv_vec_time + aiv_scalar_time + aiv_mte2_time + aiv_mte3_time) / aiv_time
         = Σ(各 ratio)                                    # 两式等价，ratio 本就是 time/aiv_time
理想上界 = Σ / max(各 ratio)                               # 全并行时墙钟收敛到最长那条泳道
达成率   = 重叠因子 / 理想上界
```

实测算例（A5，56 核均值，某 RMSNorm+RoPE+KVCache 融合算子优化前）：

| 泳道 | 耗时 | ratio |
| --- | --- | --- |
| MTE2 | 29.52 us | 0.513 |
| VEC | 27.73 us | 0.483 |
| SCALAR | 27.57 us | 0.480 |
| MTE3 | 9.15 us | 0.159 |
| 合计 | **93.97 us** | **1.634** |

`aiv_time = 57.52 us`（Task Duration 60.08 us）。重叠因子 1.634、上界 `1.634/0.513 = 3.186`、**达成率 51%**——四条泳道忙了 93.97us 却只压缩到 57.52us，全并行本应收敛到最长的 29.52us。**没有单一泳道超过 0.52**，所以不是任何一种 bound，是编排问题 → 走 `no-bound.md`。

> 达成率没有硬阈值，但**接近 100% 说明已经排满、再调顺序无收益**；50% 量级说明一半时间在互相等，值得下钻。下钻手段见 `techniques/nobound-02`。

---

> **没有"绝对达标 IPC/util"**。判定永远是**优化前 vs 优化后同一指标对比**：`aiv_time` 降即正收益。指标占比阈值（如 Cast >20%、`aiv_vec_ratio` >80%）是经验判定线，阈值附近仍以 `aiv_time` 实测为准。

---

### 3. chrome trace 读图（5 条泳道）

A5（arch3510）RegBase 范式的仿真/msprof trace 按 5 条泳道观察。盯 **VECTOR（计算主力）** 泳道：越密（区间首尾相连、无空隙）说明计算单元越忙；空泡（bubble）越多说明被等待拖累，再看是被 MTE2/MTE3（访存）还是 SCALAR/SCALARLDST（标量准备）挡住。

| 泳道 | 含义 | 看什么 |
| --- | --- | --- |
| SCALAR | 标量计算 / 分支 / 地址准备 | 忙 → scalar 准备挡住计算（scalar bound 信号） |
| SCALARLDST | 标量 load/store、mask/setup | 与 SCALAR 一起忙 → 标量 round-trip、V_S/S_V 同步 |
| MTE2 | GM→UB 载入 | 忙且接近跨度 → 访存 bound；VF 路径几乎只搬 TilingData |
| VECTOR | 向量执行（计算主力） | 密 → 计算 bound；空泡 → 被依赖/寄存器/访存拖累 |
| MTE3 | UB→GM 写回 | 与 MTE2 成对频繁 → store→load 往返、未融合 |

**要点观察**：
- **DoubleBuffer 是否生效**：MTE2 与 VECTOR 交替出现 = 搬入与计算重叠（健康）；MTE2 全在前、VECTOR 全在后 = 串行（DB 未生效）。
- **依赖链空泡**：VECTOR 出现周期性空泡且此时 MTE2/MTE3 也不忙 → 依赖链长 / 寄存器压力（非访存）。
- **寄存器 spill**：VF 计算段内出现成对的 spill→reload 指令（preg 溢出，int64 等实际占用翻倍时常见），打断计算流。

> RegBase（RVEC）范式靠向量 load/store 在 UB↔向量寄存器间搬数，**MTE2/MTE3 几乎只搬 TilingData**。因此 VF 路径的瓶颈判断**不要拿 MTE2/MTE3 当 VF 瓶颈**——先看 VECTOR 空泡与 SCALAR 准备。

---

### 4. 概念对照表（从外部 VF 案例移植时翻译）

部分外部 VF 素材使用一套 RegBase 细粒度泳道/指标命名（RVEC 单元名 + 仿真 cycle/IPC）。移植其策略到本库卡片时，**一律翻译成下表右列的 msprof/trace 口径**，卡片正文只用右列词汇，不引任何外部工具命令、报告文件名或 golden 校验开关。

| RegBase 细粒度概念（仅作翻译参照） | 本库（msprof/trace）等价 | 说明 |
| --- | --- | --- |
| RVECEX（RVEC 执行单元）泳道 | VECTOR 泳道 / `aiv_vec_ratio` | 向量执行占比、计算主力 |
| RVECSU（RVEC setup 单元）忙 | SCALAR/SCALARLDST 泳道忙 + V_S/S_V 同步 | 标量地址/mask/准备开销 |
| RVECLD / RVECST（RVEC 载入/存储） | MTE2（载入）/ MTE3（写回）泳道 | 向量访存往返（VF 内为 UB↔寄存器） |
| preg spill/reload 指令成对出现 | trace 现 preg spill/reload | 寄存器溢出，int64 用双寄存器时实际 preg ×2 |
| 仿真 `total_cycle` / `IPC` | msprof `aiv_time`(us) / aicore cycle | 性能基准量，优化前后对比 |
| 外部工具的 golden 正确性校验 | 本工作流 `func_cases` 精度回归 | 正确性由算子级精度门（Step 0.5 / 阶段 2c）保证 |

> 说明：右列才是本库口径。凡卡片"何时用"引用泳道，用 VECTOR/MTE2/MTE3/SCALAR + `aiv_*_ratio`；凡"性能数据"给数字，用 `aiv_time`/aicore cycle；正确性一律指向 `func_cases` 精度回归，不引外部 golden 工具。

## 技术卡片 mem-02：Ping-Pong 预取下一 tile

- **适用代际**：通用方法（arch22 / arch35）；具体 API、指标和阈值按目标核验，示例或实测仅代表所标平台。
- **适用 bound**：访存 / 流水
- **一句话**：当前 tile 在计算时，向另一块 L1/UB buffer 发起下一 tile 的搬入；把每轮“先等数据再算”改成稳态重叠。

### 何时用（诊断特征）

- 每轮都在计算前等待 `MTE2` 或 L1 buffer 就绪，计算和 copy 时间线首尾相接而很少重叠。
- 连续 tile 之间没有数据依赖，且当前 tile 计算期间不会再写下一 tile 的目标 buffer。
- 单 buffer 的 tile 已能让计算和搬运各自达到合理粒度，UB/L1 仍容得下第二份输入。

### 原理

第一块预热后，tile `t` 的计算与 tile `t+1` 的搬入可并行。两个物理 buffer 轮换所有权：copy 只写“空闲”的 buffer，compute 只读“已就绪”的 buffer。关键不在多开一个 buffer，而在把下一轮的 copy 发射提到当前轮 compute 之前，并严格维护 MTE 与消费方的事件配对。

### 怎么改（before / after）

```cpp
// 伪代码：Wait*/Mark* 表示与实际 TQue/TBuf 和 event ID 对应的所有权、同步操作。
// before: 每轮搬入完成后才开始计算
for (uint32_t t = 0; t < tileNum; ++t) {
    CopyIn(buf, t);
    WaitCopyDone(buf);
    Compute(buf);
}

// after: 预热 tile 0；计算 t 时搬入 t+1 到另一 buffer
CopyIn(buf[0], 0);
MarkCopyDone(0);
for (uint32_t t = 0; t < tileNum; ++t) {
    uint32_t cur = t & 1;
    uint32_t next = cur ^ 1;
    if (t + 1 < tileNum) {
        WaitBufferReusable(next);
        CopyIn(buf[next], t + 1);       // 尽早发射，和下方 Compute(cur) 重叠
        MarkCopyDone(next);
    }
    WaitCopyDone(cur);
    Compute(buf[cur]);
    MarkBufferReusable(cur);
}
```

Tiling 侧同时验证 `2 * inputTileBytes + workspaceBytes + outputTileBytes`。若以 queue 实现，`InitBuffer` 的深度和 host UB 预算必须使用同一份 `bufferNum`，不能只改 kernel。

### 性能数据

用 msprof 时间线检查稳态区间：`MTE2` 应覆盖 Vector/Cube 的一部分而非整体串行。比较 `aiv_time` 与 MTE 等待空洞；首尾 prologue/epilogue 不应被误当作无法消除的稳态开销。

### 红线（Stop rules）

- tile 太小、copy 比 compute 长很多或反之时，双缓冲未必抵消额外 UB/L1 占用；先用单 buffer 基线测量。
- 最后一轮不能预取不存在的 tile；复用 buffer 前必须等原消费者完成，事件类型和 event ID 必须成对。
- 不要把仍需被当前计算读取的数据覆盖到 next buffer；跨阶段共享临时结果时先做 mem-04 的生命周期分析。
- 内存预算不足时减 tile 或回退单缓冲，不可为了开双缓冲挤掉对齐/padding 所需空间。

### 关联

- 源（5 次）：[ops-nn `eafe7968`](https://gitcode.com/cann/ops-nn/commit/eafe7968)（arch35 Conv2D：预热首块并在循环内预取下一 `cin` block 到另一 L1 buffer）、[ops-nn `162dd73b9`](https://gitcode.com/cann/ops-nn/commit/162dd73b9)（arch35 BatchNorm：BlockSplitR 流水）、[ops-transformer `a2ee2d928`](https://gitcode.com/cann/ops-transformer/commit/a2ee2d928)（arch35 QKV：tile 流水重排）、[ops-transformer `de8e0d03b`](https://gitcode.com/cann/ops-transformer/commit/de8e0d03b)（arch35 MegaMoE：动态 wave pipeline）、[ops-transformer `cad87fee0`](https://gitcode.com/cann/ops-transformer/commit/cad87fee0)（arch35 MegaMoE：按 UB 预算调整 pipeline buffer/batch）。
- 相邻卡片：mem-03（多 buffer 可同时服务于按行批搬）、mem-04（跨阶段临时量的复用）。
