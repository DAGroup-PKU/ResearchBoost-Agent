---
name: ascendc-mc2-best-practice
description: Design or tune Ascend C communication-compute fusion, including MC2 route selection and synchronization.
---

# ascendc-mc2-best-practice

Identify the chip, operator and communication route first; use the capability declaration and pipeline/profiling references for supported combinations and evidence.

路线表中的示例工程及 `references/…` 目录可在 [MC2 资源目录](../upstream.md#ascendc-mc2-best-practice-workspace) 中查找。

Read and apply the skill workflow; do not invoke the skill itself as a Bash command. Commands inside a selected workflow can be run in their documented environment. Use the task's numerical and performance verdicts to validate any change.

## Contents

- [Ascend C MC2 通算融合算子开发最佳实践](#ascend-c-mc2-通算融合算子开发最佳实践)
- [MC2 能力声明（路线登记表）](#mc2-能力声明路线登记表)
- [通算并行调优（tileCnt / headMSize）](#通算并行调优tilecnt--headmsize)
- [性能采集（msprof + L2 cache flush）参考](#性能采集msprof--l2-cache-flush参考)

## Ascend C MC2 通算融合算子开发最佳实践

MC2（Matrix Computation & Communication）= 多卡间集合通信 + 单卡内 Blaze Cube 计算 + 通算两层流水掩盖通信开销。提供 MC2 通算融合算子开发的技术参考：通信与计算的 API 选择、时序约束、验收标准。

**知识分层（依赖方向单向，禁止反向引用）**：

| 层 | 技能 | 事实源范围 | 本 skill 的使用方式 |
|----|------|-----------|-------------------|
| 底层（通用 API） | `ascendc-api-best-practices` | Ascend C API 签名、硬件限制、平台差异（DataCopy/Cast/HardEvent/CrossCore flag/Hcomm/HCCL host 等） | 一律引用其 references，本 skill **不复述** API 细节 |
| 底层（Blaze 模板） | [Blaze 算子开发资料](../upstream.md#ascendc-blaze-development) | Blaze/tensor_api 通用模板选型、Tiling 算法选择 | 同上 |
| **本层（领域知识）** | 本 skill | MC2 特有：路线决策、apace/blaze-shmem/MoE 框架契约、通算流水编排、质量门禁 | — |

> 下层技能不引用本 skill；本 skill 文档中凡 API 级事实（签名、限制、硬件行为）均指向下层锚点，正文只保留 MC2 场景应用与编排规则。

本 skill 覆盖四种编码底座，每个底座支持特定的通信路径、算子类型与芯片组合（确切组合以 [`capability-declaration.md`](#mc2-能力声明路线登记表) 路径登记表为准）：

| 底座 | 支持的通信路径 | 支持的算子类型 | 支持芯片 | 知识目录 |
|------|---------|---------|------|------|
| **blaze-shmem** | AIV+URMA | collective-comm（AllToAll+Matmul、AllReduce+Matmul、TP/SP 融合） | dav-3510（Ascend 950PR/950DT） | [`references/foundations/blaze-shmem/`](../upstream.md#ascendc-mc2-best-practice-foundations-blaze-shmem) |
| **apace** | AIV+URMA | collective-comm（AllToAll/AllGather + QuantMatmul 融合、compute-first 类；AllReduce 类需先扩展通信组件，见 [`paradigm-mapping.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-operator-design-paradigm-mapping-md)） | dav-3510（Ascend 950PR/950DT） | [`references/foundations/apace/`](../upstream.md#ascendc-mc2-best-practice-foundations-apace) |
| **ascendc-api** | AIV+UBMEM | moe（MoE Dispatch/Combine、专家并行 EP） | dav-3510（Ascend 950PR/950DT）+ dav-2201（Atlas A2/A3 系列）双平台 | [`references/foundations/ascendc-api/`](../upstream.md#ascendc-mc2-best-practice-foundations-ascendc-api) |
| **hccl-matmul** | HCCL 高阶 | collective-comm（AllGather+Matmul、AllReduce+Matmul 等，aclnn 注册） | dav-2201（Atlas A2/A3 系列） | [`references/foundations/hccl-matmul/`](../upstream.md#ascendc-mc2-best-practice-foundations-hccl-matmul) |

> 四种底座抽象层级不同（blaze-shmem 是库+模板手工组装，apace 是模板框架，ascendc-api 是直接使用ascendc API，hccl-matmul 是官方 HCCL 高阶 + `AscendC::Matmul` 高阶、仅注册），但在"选什么写代码"这个决策点上是并列选项。apace 模板库虽基于 Ascend C 基础 API 构建，但与 ascendc-api / hccl-matmul 路线约束集、工程边界、可修改范围完全不同——详见 §1/§2/§3/§4 各路线特有约束。

### 何时使用 / 不适用

**使用信号**（任一即可）：
- 场景：多卡协同的通算融合算子（AllToAll+Matmul、AllReduce+Matmul、MoE Dispatch/Combine、多卡 EP/TP/SP 融合 Kernel）
- 关键词："MC2"、"SHMEM"、"APACE"、"apace"、"通算融合"、"多卡通信直调"、"UDMA"、"URMA"、"AllToAll+Matmul"、"CollectiveComm"、"MoE Dispatch"、"MoE Combine"、"专家并行"、"EP"、"mega_moe"、"MTE通信"、"MTE window"、"token 路由分发"、"910B"、"HCCL"、"aclnn 通算融合"
- 代码：现有工程同时出现通信 API（`shmem.h`/`aclshmem*` 或 `collective_comm_api.h`/`CollectiveComm`）与 `blaze/gemm/` 模板（blaze-shmem/apace 路线），或出现 `winContext`/`mc2Context`/`HcclOpParam` 与 MTE 通信窗口结构（MTE通信 → ascendc-api 路线），或出现 `Hccl<HCCL_SERVER_TYPE_AICPU>` 与 `AscendC::Matmul`（HCCL 高阶 → hccl-matmul 路线）

**不适用**（走其他 skill）：
- 纯单卡 Matmul（无跨卡通信）→ [Blaze 算子开发资料](../upstream.md#ascendc-blaze-development)（950）或 [Matmul API](../upstream.md#ascendc-api-best-practices-api-matmul-md)（910B）
- Vector 类逐元素/归约算子（无 Cube、无跨卡通信）
- 通用 Ascend C API 用法查询 → `ascendc-api-best-practices`
- 非 3510 架构的 AIV+URMA 通算融合（MTE通信除外，支持 dav-2201 + dav-3510 双平台）

> 确切的支持组合（chip × 算子类型 × 调用形态 × 通信路径 × 编程抽象）以 [`references/capability-declaration.md`](#mc2-能力声明路线登记表) 路线登记表为准；表内同时登记**明确不支持**的组合及原因，命中否定行时直接答复用户不可用 + 替代建议。

### 决策树：先判断算子类型

```
用户要做通算融合算子
│
├─ 目标芯片 910B / dav-2201 / A2，且是集合通信+Matmul（AllGather/AllReduce/ReduceScatter/AlltoAll）？
│   → HCCL 高阶路径（仅注册 / aclnn，禁止 <<<>>> 直调）→ hccl-matmul 路线（下方§4）
│
├─ AllToAll+Matmul / AllReduce+Matmul / TP/SP 通算融合？（950）
│   → AIV+URMA 路径（仅 dav-3510 / Ascend 950）
│      ├─ 提到"apace"/"CollectiveComm"或代码在 ops-transformer/apace/ 下？→ apace 路线（下方§2）
│      └─ 否则 → blaze-shmem 路线（下方§1）
│
├─ MoE Dispatch / MoE Combine / mega_moe / 专家并行 EP？
│   → MTE通信 → ascendc-api 路线（dav-2201 + dav-3510 双平台，下方§3）
│      ├─ 先阅读已有实现？→ references/foundations/ascendc-api/moe-dispatch-combine/reading/
│      └─ 生成新算子或改造？→ references/foundations/ascendc-api/moe-dispatch-combine/samples/
│
└─ 不确定？
    → 看通信路径：SHMEM API = AIV+URMA 路径（仅 dav-3510）；winContext/mc2Context = MTE通信（dav-2201 + dav-3510 双平台）；
       Hccl<HCCL_SERVER_TYPE_AICPU> / InitV2 / aclnn*Matmul = hccl-matmul（dav-2201 注册）
```

选定路线后：

1. 查 [`capability-declaration.md`](#mc2-能力声明路线登记表)，确认 chip × 算子类型 × 调用形态组合命中 `supported` 行（命中否定行或无行 → 答复用户不可用 + 原因）
2. 进入需求分析（下节），输出 REQUIREMENTS.md
3. 按 §1/§2/§3/§4 进入对应路线的 references

### 需求分析（全路线共享）

> **MC2 特有需求维度**：除算子名/数学定义/dtype/shape 外，需求收集时还需明确通信路径（AIV+URMA / MTE通信 / HCCL 高阶）与编程抽象底座（blaze-shmem / apace / ascendc-api / hccl-matmul）。

| 文档 | 何时读 |
|------|--------|
| [`capability-declaration.md`](#mc2-能力声明路线登记表) | 确认路线可行性、多行命中时选编程抽象、定位参考实现 |
| [`requirement-analysis/grill-protocol.md`](../upstream.md#ascendc-mc2-best-practice-requirement-analysis-grill-protocol-md) | 需求拷问 8 维度判据（生成 REQUIREMENTS.md 前逐项拷问，先拷问再生成） |
| [`requirement-analysis/template.md`](../upstream.md#ascendc-mc2-best-practice-requirement-analysis-template-md) | 生成需求文档时（含 MC2 特有章节：组网规模、通信设计、开发就绪判断） |
| [`requirement-analysis/comm-path-decision.md`](../upstream.md#ascendc-mc2-best-practice-requirement-analysis-comm-path-decision-md) | 用户询问通信路径选型时（HCCL 高阶 / AIV+URMA / MTE通信 / CCU） |
| [`requirement-analysis/classification.md`](../upstream.md#ascendc-mc2-best-practice-requirement-analysis-classification-md) | 确定算子类型、查已有算子与可信源清单 |
| [`requirement-analysis/quality-checklist.md`](../upstream.md#ascendc-mc2-best-practice-requirement-analysis-quality-checklist-md) | 需求文档生成后自检（14 项 + 开发就绪闸门） |
| [`operators/collective-comm.md`](../upstream.md#ascendc-mc2-best-practice-operators-collective-comm-md) | 设计集合通信类算子时（通信语义、轴切分、通算流水的跨路线共性） |
| [`codebase-analysis.md`](../upstream.md#ascendc-mc2-best-practice-codebase-analysis-md) | Brownfield 从代码推断路线坐标（Stub；Greenfield 不读） |

### 红线（全路线共性，与底座无关）

| # | 红线 | 一句话理由 | 详见 |
|---|------|-----------|------|
| ① | 架构白名单以路线登记表为准 | 未验证组合（如 dav-2201 × AIV+URMA）禁止使用 | [`capability-declaration.md`](#mc2-能力声明路线登记表) |
| ② | AIV+URMA 路径性能采集必须刷 L2 cache | SHMEM B 矩阵驻留会污染下一轮带宽；**hccl-matmul（AICPU）无需 flush** | [`shared/profiling_mc2.md`](#性能采集msprof--l2-cache-flush参考) |

> 以下两条是 **AIV+URMA 路径下 blaze-shmem / apace 底座的选择性约束**，不是全路线共性。集合通信类通算融合在 **注册 / aclnn** 形态下走 §4 hccl-matmul 路线，此时 HCCL 高阶 API 与 `AscendC::Matmul` 高阶 API 恰是必选路径：
> - 禁止 HCCL 高阶 API（`Hccl::*`）—— HCCL 集合通信库依赖框架注入上下文，AIV+URMA 直调场景拿不到（详见 [`blaze-shmem/comm_shmem.md`](../upstream.md#ascendc-mc2-best-practice-foundations-blaze-shmem-comm-shmem-md) §5，7 类 18 个 API 清单）
> - Matmul 走 Blaze 模板，禁止 `AscendC::Matmul` 高阶 API —— 同理（详见 [`blaze-shmem/matmul_blaze.md`](../upstream.md#ascendc-mc2-best-practice-foundations-blaze-shmem-matmul-blaze-md) / [`apace/fundamentals/compute.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-fundamentals-compute-md)）

各底座特有约束与逐项审查清单见 §1/§2/§3/§4。

---

### §1 blaze-shmem 路线

SHMEM 通信库（cann/shmem，与 HCOMM 无关）驱动 AIV+URMA 跨卡搬运，Blaze 计算模板负责 Cube 计算，二者手工组装为独立 CMake 工程（自带 `include/` 全套）。通信侧 host `aclshmem_*`、device `aclshmemx_udma_*` + `aclshmem_barrier_all`；计算侧 `Blaze::Gemm::Block::BlockMmad` + `Blaze::Gemm::Kernel::*`。

#### 红线 / 约束

| # | 约束 | 说明 | 详见 |
|---|------|------|------|
| R1 | 架构白名单 | CMakeLists.txt 中 `npu-arch` = dav-3510 | 全路线共性红线 ① |
| R2 | 禁止 HCCL 高阶 API（`Hccl::*`） | HCCL 集合通信库依赖框架注入上下文，AIV+URMA 直调场景拿不到 | [`comm_shmem.md`](../upstream.md#ascendc-mc2-best-practice-foundations-blaze-shmem-comm-shmem-md) §5（7 类 18 个 API 清单） |
| R3 | 禁止 `AscendC::Matmul` 高阶 API | 高阶 API 不支持 AIV+URMA 直调场景 | [`matmul_blaze.md`](../upstream.md#ascendc-mc2-best-practice-foundations-blaze-shmem-matmul-blaze-md) |
| R4 | 通信走 SHMEM | 头文件含 `shmem.h`，device 侧用 `aclshmemx_udma_*`/`aclshmem_barrier_all` | [`comm_shmem.md`](../upstream.md#ascendc-mc2-best-practice-foundations-blaze-shmem-comm-shmem-md) |
| R5 | Matmul 走 Blaze | 头文件含 `blaze/gemm/block/block_mmad*.h` | [`matmul_blaze.md`](../upstream.md#ascendc-mc2-best-practice-foundations-blaze-shmem-matmul-blaze-md) |
| R6 | L2 flush 证据 | 代码中含 L2 flush kernel 调用或等效实现 | 全路线共性红线 ② |
| R7 | 流程门禁完整 | `docs/` 下 DESIGN/PLAN/WALKTHROUGH/REVIEW.md 齐全；环境检查通过 | — |

**逐项审查清单**：[`review-checklist.md`](../upstream.md#ascendc-mc2-best-practice-foundations-blaze-shmem-review-checklist-md)（含常见 FAIL 原因与修复方向，违反红线项 = FAIL）。

#### References

| 文档 | 何时读 |
|------|--------|
| [`references/foundations/blaze-shmem/workflow_integration.md`](../upstream.md#ascendc-mc2-best-practice-foundations-blaze-shmem-workflow-integration-md) | 设计 MC2 算子前，看 MC2 场景的技术要点和门禁 |
| [`references/foundations/blaze-shmem/mc2_architecture.md`](../upstream.md#ascendc-mc2-best-practice-foundations-blaze-shmem-mc2-architecture-md) | 第一次设计 MC2 算子，建立 AIV/AIC 分工 + 4-buffer 流水 + M 轴切分心智模型 |
| [`references/foundations/blaze-shmem/comm_shmem.md`](../upstream.md#ascendc-mc2-best-practice-foundations-blaze-shmem-comm-shmem-md) | 写/改通信层，查 SHMEM API、UDMA 用法、禁止 HCCL 清单（§5）、扩展其他通信原语 |
| [`references/foundations/blaze-shmem/matmul_blaze.md`](../upstream.md#ascendc-mc2-best-practice-foundations-blaze-shmem-matmul-blaze-md) | 写/改计算层，选 Blaze 模板、改 DispatchPolicy、处理 Scale |
| [`references/shared/profiling_mc2.md`](#性能采集msprof--l2-cache-flush参考) | 性能采集与调优时：msprof task-based 采集 + L2 flush + 多卡数据后处理 |
| [`references/shared/pipeline_tuning.md`](#通算并行调优tilecnt--headmsize) | 精度调试阶段用 tileCnt=1 做串行基线；性能调优阶段扫描 tileCnt 找最优值 |
| [`references/foundations/blaze-shmem/codebase_map.md`](../upstream.md#ascendc-mc2-best-practice-foundations-blaze-shmem-codebase-map-md) | 工程搭建时，定位"哪些文件改/不改"与改造食谱 |
| `references/foundations/blaze-shmem/all_to_all_matmul/` | 编译验证过的基底工程，blaze-shmem 路线所有 MC2 算子的起手模板 |

---

### §2 apace 路线

APACE（Ascend PArallel Communication-compute Engine）是通算融合算子的架构底座，提供可复用的 block 层接口、kernel 层参考实现和 tiling 算法。APACE 自建通信基础 API（基于 HCOMM 通信基础库构建，与 HCCL 集合通信库无关）。apace 路线在 `kernel/<op>/` 下新建算子，复用稳定共享层 `block/` `tiling/`。

> **apace 路线接口契约基准**：[cann/ops-transformer](https://gitcode.com/cann/ops-transformer) `mc2/common/op_kernel/apace/` 子树，以 pin 的已验证快照为结构基准。样例代码可使用 [fetch_apace.sh](../upstream.md#ascendc-mc2-best-practice-scripts-fetch-apace-sh) 及其依赖，从 [MC2 工具目录](../upstream.md#ascendc-mc2-best-practice-workspace) 运行该脚本（支持 `--ref`/`APACE_PIN_REF` 锚定 commit）；⚠️ master 结构会演进（如目录迁移、新增算子），拉取 master 后必须与快照 diff 校验，文档引用失效时更新文档——**已知漂移登记（pin 之后 10 个提交：block/→core/ 迁移、ReduceScatter UBMEM 官方模板、hcomm dcci 等）见 [`references/foundations/apace/fundamentals/communication.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-fundamentals-communication-md) §8，核对 master 前必读**。
>
> **工程实现默认事实源**：CANN 内置 apace 框架（路径随 CANN 打包形态实测定位，Step 1 登记；两种已验证形态：`opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/apace/`（cann-9.2.0）、`vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/apace/`（cann-9.1.0）），**直调独立工程默认 CMake 直引该路径，禁止整包复制**。

#### 按请求目的路由

| 请求目的 | 执行 |
|---|---|
| 要求完整开发一个 MC2 通算融合算子 | Step 1 → Step 2 → Step 3 → Step 4（plugin 场景以 [`workflow_integration.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-workflow-integration-md) 的 7 步映射为主要消费文档） |
| 只要求设计/方案分析并输出设计文档 | Step 2 → Step 3 |
| 咨询、解释、评审、排障、能力查询 | 只读相关 references |

#### 计算执行原则

通信与计算的流水编排、AIV-AIC 协同必须在 device 侧单次 launch 的融合 Kernel（MIX）中完成。host 侧只负责：数据准备与 buffer 分配、Tiling 计算、多 rank 启动与通信建链（rootInfo 交换、HCCL/Win 资源分配、跨 rank barrier）、Kernel launch（含 dtype dispatch）和结果搬运。不得将算子语义中的任何计算或通信步骤（如归约、数据搬移、状态同步）放到 host 侧执行；CPU golden 等精度验证基建不属于算子语义，不在此限。

#### 方法论速览（五阶段流程 + 六个决策点）

任何 apace 算子开发遵循：**语义分析 → 路线决策 → 设计合同 → 实现组装 → 验证** 五阶段；设计合同必须对六个决策点逐项定值：

| 决策点 | 问题 | 一句话判据 |
|---|---|---|
| D1 数据流方向 | 通信在前还是计算在前？ | 通信的输入是原始输入（前）还是计算产物（后） |
| D2 切分轴与轮次 T | 沿哪个轴切、切几轮？ | 轴由数据分布语义定（非通信原语）；T 权衡同步开销 vs 流水粒度 |
| D3 数据通路 | 连续 GM 还是 FragmentTensor？ | 通信后数据在 GM 上连续还是离散 |
| D4 同步合同 | 谁通知谁、几次、在哪等？ | flag idx/次数严格配对；barrier totalJobs 按分核映射；SyncAll 在守卫外 |
| D5 累加合同 | 部分和在哪累加？ | L0C 单次 fixpipe / AtomicAdd / DEFERRED_SYNC / staging+增量归约四选一 |
| D6 入口合同 | 几个入口、怎么分发？ | 入口数 = dtype 组合数；MIX_AIC_1_1；tiling 按值 ABI |

完整方法论（每阶段的产出/门禁/回退点、决策点取值示例、跨阶段纪律）：[`operator-design/dev-methodology.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-operator-design-dev-methodology-md)。新算子先在该文档 §5 找 D1-D6 取值最接近的实现作骨架参考。

#### 四步流程

1. **Step 1: Project Setup** — 建立算子工程骨架、校验 CANN 内置 apace 事实源与环境（dav-3510、多卡、HCCL）、登记只读参考点。不核对接口事实、不判定路线、不创建实现文件。→ [`workflow/step1-project-setup.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-workflow-step1-project-setup-md)
2. **Step 2: 设计前核对（轻量）** — 只读核对 apace 框架的接口事实（matmul 链路/通信接口/入口 ABI/官方覆盖性）；事实**默认内联记录于 DESIGN.md §0.3**（无需独立调查报告）。不做方案推荐、不匹配场景、不选择路线。→ [`workflow/step2-investigation.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-workflow-step2-investigation-md)
3. **Step 3: Design** — 依据需求和调查事实完成路线决策，生成定稿的 DESIGN 和可执行路线的 PLAN。只产出设计文档，不复制文件、不编写实现、不执行构建。→ [`workflow/step3-design.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-workflow-step3-design-md)
4. **Step 4: Implementation** — 按定稿的 DESIGN/PLAN 完成实现、多卡验证与性能采集；实现层问题在项目内修复，不改设计、不改接口、不扩大支持域。→ [`workflow/step4-implementation.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-workflow-step4-implementation-md)

> **plugin 消费流映射**：ops-direct-invoke plugin 执行 7 步流程（环境检查→设计→串讲→开发→审查→修复→验收→汇报），与本四步模型的对应为：plugin Step 1 ≈ 本 Step 1；plugin Step 2/2.5 ≈ 本 Step 2+3；plugin Step 3-7 ≈ 本 Step 4。**plugin 场景下以 [`workflow_integration.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-workflow-integration-md)（7 步映射）为主要消费文档**，四步文档为模型参考。

#### 路线模型

```text
implementation_route: apace_native | apace_custom | unsupported
selected_scenario: <仅 apace_custom 填写>
```

按官方 README 的两种使用方式决策：

- **`apace_native`**：官方 `apace/kernel` 已有算子可直接调用或参考复用——不读取场景注册表；
- **`apace_custom`**：官方 kernel 未覆盖，但可基于 `apace/block` 接口组合构建（含 compute-first 等自研编排）——需求语义命中场景注册表语义判据时**默认查阅**对应场景指导（准入条件只含需求侧语义判据，不含实现侧决策）；
- **`unsupported`**：`apace/block` 接口层也无法支撑的组合，或场景语义零命中/多命中。

> **场景文档查阅不受官方覆盖性判定反向阻断**：本仓/用户工程中已存在某场景的生产实现时（如 compute-first 类算子已生成过），阅读该场景文档不需要"官方未覆盖"前提——场景文档同时是该类算子的改造/复用手册。

官方接口未覆盖不是绕过 apace 接口层、退回裸 Ascend C 全自建的授权——那是 ascendc-api 底座的路线选择，不是 apace 路线内的 fallback。

> **compute-first（ReduceScatter 类语义）算子生成阅读顺序**：[`fusion.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-fundamentals-fusion-md) §6.2（方法论：编排模式/事件语义/tiling 组织）→ [`scenarios/compute-first-reduce-scatter/design.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-scenarios-compute-first-reduce-scatter-design-md)（设计合同）→ [`scenarios/compute-first-reduce-scatter/development.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-scenarios-compute-first-reduce-scatter-development-md)（实现模板与验收）。方法论原则（事实源唯一/设计三拷问/契约驱动/失败前置/验收分层/示例边界）见 [`architecture.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-fundamentals-architecture-md) §1.5。

#### 红线 / 约束（全局 R 系列 + 场景约束）

> 设计期与开发期必须逐条满足；完整定义、操作化检查方法与常见 FAIL 原因以 [`review-checklist.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-review-checklist-md) 为准。违反任意适用项 = FAIL。

**全局红线（所有 apace 算子必须满足）**：

| # | 约束 | 一句话理由 | 详见 |
|---|------|-----------|------|
| R1 | 禁止 `__schedmode__(1)` / `core_ratio(1,1)` | AIC/AIV 串行调度 → 死锁（507015）；核配比由 `KERNEL_TYPE_MIX_AIC_1_1` 保证 | architecture §10 ① |
| R2 | 每个入口含 `KERNEL_TYPE_MIX_AIC_1_1` | 核配比 1:1 唯一保证 | architecture §10 ① |
| R3 | 入口变体覆盖 dtype 合同全部组合 + host 运行期 dispatch | 硬编码单入口 → 异 dtype 字节流被错误模板解释，精度系统性失败 | operator-anatomy §5.3/§7.2 |
| R4 | `block/` `tiling/` 零修改 | 共享层漂移 = 幽灵 bug 温床 | architecture §10 ③ |
| R5 | CrossCore flag idx 配对 | AIV WaitFlag idx == AIC SetFlag idx | fusion §3 |
| R6 | CommContext 与引擎匹配 | UDMA 有 `__gm__ CommContext*`；HCCL windows 无 | communication |
| R7 | 禁止 `AscendC::Matmul` 高阶 API | 高阶 API 不支持直调场景 | compute |
| R8 | 禁止 `Hccl::*` 高阶 API | 依赖框架注入上下文，直调拿不到 | comm_shmem §5 |
| R11 | host 前置校验在 fork/建链前拒绝非法输入 | 非法输入进 kernel = 难查的死锁/精度问题（整除/对齐/核数/Win 容量等；compute-first 完整 9 项见场景文档） | development-guide §3.5 |
| R12 | UB 静态通信区物理隔离 | 混用重叠 → 通信数据被踩踏 → 死锁；**`TPipe` 与 `MakeMemPtr` 必须二选一，禁止混用** → 507015 | communication 陷阱 #9；operator-anatomy §4.3 |
| R13 | 通信对象 `totalJobs=rankSize` 多核并行 | 退化 totalJobs=1 → 通信时间放大 R 倍（已证伪臆造约束）；TeamBarrier totalJobs 按分核映射取值（官方前 R 核 = rankSize；compute-first 后 R 核 = 1 + 显式 CrossDevice） | communication §2.2 |
| R14 | Win 数据/元数据分离（硬红线）+ 单轮 PUT 大小（风险提示） | 覆盖元数据 → "假通过"；单轮 PUT 无官方硬上限（512KB 系 bring-up 经验，同平台生产实现 12.6MB 稳定），大单轮按 case 复核 | communication 陷阱 #12/#13 |
| R15 | 投产级性能验证门槛 | 真实大 shape × R=2/4 双档 × 三路径对标；仅基线 = 未达标 | host-and-testing |
| R20 | perf 模式 L2 flush 实接线 | 只分配 buffer 不调 kernel = 死代码 = MTE2 带宽虚高 | host-and-testing §4 |

**场景约束（按算子语义特征自动适用——含 compute-first 编排即适用 R9/R10/R16/R21、含归约模块即适用 R17-R19，不依赖场景注册表命中；完整定义见场景文档）**：

| # | 适用场景 | 约束 | 详见 |
|---|---------|------|------|
| R9 | compute-first | CrossCore flag Set/Wait 严格配对 + flagId ∈ [0, FLAG_ID_MAX)；计数器 0-15 衡量未消费积压（不构成 T 上限；T 上限按 case 实测），不设数值拒绝 | fusion §6.2.3 |
| R10 | compute-first | T 派生优先 `T \| mSeg` 无尾块；单尾块 ≤ 头块（16 对齐）PUT 直传合法；多尾块/尾>头走策略 A padding | fusion §6.2.7 |
| R16 | compute-first | mm 内核默认 FragmentTensor 消 R 循环；vendor R×T 子调用须论证 SCALAR 占比 | fusion §6.2.2 |
| R21 | compute-first | **localLast 编排禁止移除**：以"每轮 Set 双 flag"替代 → flagId 用量翻倍 + 丧失通信提前启动；`cFragAddrs_` 顺序写错才是 A/C 错位根因，非 localLast 本身 | fusion §6.2.2 |
| R17 | compute-first（含归约） | 归约 2D DataCopyPad 批量（blockCount=本批行数）；逐行归约 = 性能 FAIL | fusion §6.2.6 |
| R18 | compute-first（含 BF16→FP32 归约） | 归约独立 srcFP32 双缓冲，禁止 in-place 加宽 Cast | fusion §6.2.6 |
| R19 | compute-first（含归约） | 归约四类 HardEvent 同迭代配对 + 残留消费；Set 无配对 Wait = 挂死（507014） | fusion §6.2.6 |

#### 精度与验收纪律

以下纪律适用于所有 apace 通算融合算子（与方法论原则（architecture §1.5）、全局红线与场景约束互补，不重复展开）：

1. **以设备输入事实源计算 Golden。** 先冻结每卡输入/输出分布与切分轴，再生成测试输入；golden 切分轴写错则精度验证整体失效（golden 语义先行，方法论原则 2①）。
2. **golden 的 dtype 链必须完整可执行。** 从设备输入字节（FP8/MX 编码）解码 → 累加 dtype → 输出 dtype 的每步转换顺序、舍入与 clamp 都在 DESIGN 中写明；通信侧与计算侧的 dtype 约定不一致时标记冲突并停止，不用"golden 通过"掩盖。
3. **覆盖会改变失败边界的形状与规模组合。** 精度验证至少覆盖：rank 数端点、对齐与非对齐/尾块 shape、多 tile（T>1）与连续重复 launch；T=1 全部通过不能外推到 T>1——串行基线会掩盖多 tile 的 Win 区布局与 flag 配对问题。
4. **公开输出与诊断中间量分开判定。** 校验脚本分别报告用户可见输出与 Win 区/中间阶段数据的状态：输出通过不能掩盖中间量异常；DESIGN 已声明非公开的中间量异常也不能记为输出失败。

#### 失败案例记录

生产失败案例按"现象 → 根因 → 修复方向"补入 [`review-checklist.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-review-checklist-md) 常见 FAIL 原因表（失败模式前置，方法论原则 4）；跨算子可复用的归入 Skill，算子特有的留在项目记录。设备结论必须标注执行环境、设备节点、实际命令和返回码；编译/静态通过不能写成设备已验证，单 case 重跑通过不能抹掉原始挂起。

#### References

| 文档 | 何时读 |
|------|--------|
| [`workflow_integration.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-workflow-integration-md) | **plugin 7 步流程 apace 映射（主要消费文档）**：设计/开发/审查/修复/验收各阶段的 apace 技术要点与门禁 |
| [`review-checklist.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-review-checklist-md) | 逐项审查清单（全局红线 + 场景约束索引 + 操作化检查方法 + 常见 FAIL 原因） |
| [`troubleshooting/failure-navigation.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-troubleshooting-failure-navigation-md) | 按现象定位排查方向 |
| [`fundamentals/`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-fundamentals) | 架构、通信（含通信组件扩展指南/hcomm 未封装原语）、计算、融合组合模式基础知识 |
| [`operator-design/`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-operator-design) | **开发方法论**（[`dev-methodology.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-operator-design-dev-methodology-md)：五阶段流程 + D1-D6 决策点）+ DESIGN/PLAN 模板 + 算子解剖 + 开发指南 + **语义范式映射与架构差异**（需求存在注册形态既有实现时必读 [`paradigm-mapping.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-operator-design-paradigm-mapping-md)：算子族选型表、七维差异、可参考 vs 必须重写清单） |
| [`scenarios/`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-scenarios) | 自定义扩展场景注册表（PUT AllToAll / PUT AllGather 变体、GET 推导、compute-first 组合模式） |
| [`references/shared/pipeline_tuning.md`](#通算并行调优tilecnt--headmsize) | 通算并行调优：tileCnt 两阶段策略（两路线共享） |
| [`references/shared/profiling_mc2.md`](#性能采集msprof--l2-cache-flush参考) | 性能采集：msprof + L2 flush + 多卡后处理（两路线共享） |

---

### §3 ascendc-api 路线（MTE通信，MoE Dispatch/Combine）

裸 Ascend C API 全自建（不经任何通信库/模板库）。通信路径为 MTE通信 = AIV+UBMEM（AIV 触发 MTE 执行跨卡搬运），host 侧经 HCCL 分配 window 资源——此处 HCCL 仅做资源分配，非 HCCL 集合通信库高阶 API。支持 dav-2201（Atlas A2/A3 系列）+ dav-3510（Ascend 950PR/950DT）双平台，compat 层抹平两平台 window 地址结构差异（dav-2201/dav-3510 在 MoE SDK 工程代码中惯称 A3/A5，官方产品名以 Atlas A3 系列 / Ascend 950PR/950DT 为准）。注意区别于 apace 路线——apace 模板库虽基于 Ascend C API 构建（其通信基础 API 基于 HCOMM），但属独立编码底座。

> **跨路径设计原则**：阅读或改造 `dispatch`、`combine` 任一侧时，都要同步核对另一侧的接口和状态语义；`expandIdx`、`epRecvCounts`、`epSendCounts` 等中间量必须成对理解；host 侧只传总核数，kernel 侧负责决定每阶段实际使用多少核以及如何切分。

#### 红线 / 约束

##### ① 跨卡搬运走 MTE通信，禁止 HCCL 高阶通信原语做数据搬运

MoE Dispatch/Combine 通过 host 侧 `HcclAllocComResourceByTiling` 创建通信 window 资源，device 侧用 MTE + `DataCopyPad` 访问 window 地址做跨卡数据搬运。禁止用 `Hccl::AlltoAll()`/`Hccl::AlltoAllV()`/`Hccl::AllReduce()` 等高阶通信原语替代——它们是对整块数据的黑盒集合通信，无法在 Kernel 内按 token 粒度插入路由计算和状态协议逻辑。

##### ② window 地址必须走 compat 层，禁止直接硬编码平台结构体偏移

`winContext`（`mc2Context`）按平台解释成 `HcclA3OpResParam` 或 `HcclA5OpResParam`，字段布局不同（dav-2201 侧远端地址通过链表跳转，dav-3510 侧状态区在前 1MB）。必须通过 compat 封装的 `GetBaseWindAddrByRankId()`、`GetBaseWindStateAddrByRankId()` 等统一接口访问，不要在主流程中手写平台分支。**注意**：`HcclA3OpResParam`/`HcclA5OpResParam` 是样例内的**精简定义**（A3/A5 为工程惯称命名），仅用于工程内自建 context，不适用于解析真实 HCCL 返回的 context（真实解析须用 SDK 完整 `HcclOpResParam`，见 `mte-address-access.md` 重要警告）。

##### ③ 共享 GM/状态区走 DataCopyPad，禁止 GetValue/SetValue 直接访问

共享 GM、`workspaceGM`、状态区、window 数据区的默认读写路径是 `DataCopyPad` 或等价 GM 路径。`SetValue()`/`GetValue()` 直接访问共享区域会导致跨核数据观察不稳定。`SyncAll` 只保证执行同步，**不保证** Data Cache 与 GM 一致性。极少数场景如必须使用 `SetValue`/`GetValue`，必须额外补齐 `DataCacheCleanAndInvalid` 并重新验证核间可见性。

##### ④ 状态协议：先数据后状态、每核只写自己槽位、消费后清理

- 发布顺序：先写 count/offset/附带信息，再发布 ready 状态
- 等待侧只轮询自己负责的状态段
- 每核只写自己的共享槽位，禁止多核直接累加同一地址
- 状态消费完成后才清理本核负责段，不要沿用单核整块 reset 习惯

**Reviewer 速查**：

| # | 检查项 | 方法 |
|---|--------|------|
| M1 | 无 HCCL 高阶通信原语 | `grep -rn "Hccl::AlltoAll\|Hccl::AlltoAllV\|Hccl::AllReduce\|Hccl::AllGather\|Hccl::ReduceScatter" operators/{op}/` 应为空 |
| M2 | window 地址走 compat 层 | kernel 中使用 compat helper（`GetBaseWindAddrByRankId` 等），不直接硬编码 `HcclA3OpResParam`/`HcclA5OpResParam` 字段偏移 |
| M3 | 共享 GM 走 DataCopyPad | `grep -rn "SetValue\|GetValue" operators/{op}/` 检查是否用于共享 GM/状态区，若有需确认有 `DataCacheCleanAndInvalid` |
| M4 | 状态协议正确 | 检查发布顺序（先数据后状态）、每核只写自己槽位、消费后清理 |

#### References

**先判断任务类型**：当任务同时包含"先阅读现有实现"与"再进行改造"两部分时，路径顺序为：先完成入口 A 的阅读路径，再进入入口 B 的规格补齐。

##### 入口 A：阅读已有实现

适用场景：先建立现有 dispatch、combine 或 mega_moe / 融合实现的理解框架。

- 通用方法：[`references/foundations/ascendc-api/moe-dispatch-combine/reading/common-reading-method.md`](../upstream.md#ascendc-mc2-best-practice-foundations-ascendc-api-moe-dispatch-combine-reading-common-reading-method-md)
- dispatch 专用：[`references/foundations/ascendc-api/moe-dispatch-combine/reading/dispatch-reading-guide.md`](../upstream.md#ascendc-mc2-best-practice-foundations-ascendc-api-moe-dispatch-combine-reading-dispatch-reading-guide-md)
- combine 专用：[`references/foundations/ascendc-api/moe-dispatch-combine/reading/combine-reading-guide.md`](../upstream.md#ascendc-mc2-best-practice-foundations-ascendc-api-moe-dispatch-combine-reading-combine-reading-guide-md)
- mega_moe / 融合实现：[`references/foundations/ascendc-api/moe-dispatch-combine/reading/mega-moe-reading-guide.md`](../upstream.md#ascendc-mc2-best-practice-foundations-ascendc-api-moe-dispatch-combine-reading-mega-moe-reading-guide-md)

进入具体 guide 前，先判断是否属于融合实现（输入侧沿用 dispatch 风格、输出侧直接产出 combine 最终输出、主入口同时包含路由/发送/expert 计算/聚合/回写、通信协议不能拆成独立 dispatch+combine、关键内存布局同时承载多阶段共享状态）。命中任一特征时归入 mega-moe-reading-guide。

##### 入口 B：生成新算子或在样例工程基础上改造

适用场景：生成 moe dispatch、生成 moe combine，或在样例工程基础上修改 dispatch/combine。不再套用通用 `add.asc` 模板，默认拆成三部分推进：

1. **samples** / 规格补齐与样例工程：[`references/foundations/ascendc-api/moe-dispatch-combine/samples/index.md`](../upstream.md#ascendc-mc2-best-practice-foundations-ascendc-api-moe-dispatch-combine-samples-index-md) → `spec-template.md` → `dispatch-dataflow.md` 或 `combine-dataflow.md`；sample 内 helper 分层见 `sample-helper-map.md`
2. **api-rules** / MoE 特化 API 规则：[`references/foundations/ascendc-api/moe-dispatch-combine/api-rules/index.md`](../upstream.md#ascendc-mc2-best-practice-foundations-ascendc-api-moe-dispatch-combine-api-rules-index-md)（`DataCopyPad`、window 地址获取、同步与可见性、状态协议、dispatch/combine 接口契约）
3. **tiling-scheme** / MoE 特化分核方案：[`references/foundations/ascendc-api/moe-dispatch-combine/tiling-scheme/index.md`](../upstream.md#ascendc-mc2-best-practice-foundations-ascendc-api-moe-dispatch-combine-tiling-scheme-index-md) → `window-memory-layout.md` → `multi-core-formulas.md` → `split-core-design.md` → `double-buffer-protocol.md`

整体架构、阶段协作关系和设计动机背景见 [`references/foundations/ascendc-api/moe-dispatch-combine/reading/design-overview.md`](../upstream.md#ascendc-mc2-best-practice-foundations-ascendc-api-moe-dispatch-combine-reading-design-overview-md)，不替代三部分主路径中的规格、API 规则或分核设计文档。

##### 知识目录

| 目录 | 何时读 |
|------|--------|
| `references/foundations/ascendc-api/moe-dispatch-combine/reading/` | 阅读已有 dispatch/combine/mega_moe 实现时 |
| `references/foundations/ascendc-api/moe-dispatch-combine/samples/` | 规格补齐、工程组织参考、编译链路、接口语义和文件落点 |
| `references/foundations/ascendc-api/moe-dispatch-combine/api-rules/` | MoE 特有的 window 地址获取、DataCopyPad 规则、同步可见性、状态协议、接口契约 |
| `references/foundations/ascendc-api/moe-dispatch-combine/tiling-scheme/` | window 物理布局、工作量公式、各阶段分核方案、双缓冲轮转协议 |

---

### §4 hccl-matmul 路线（910B / HCCL 高阶 + AscendC::Matmul，仅注册）

910B（A2，`dav-2201`）上的集合通信类通算融合走官方路径：通信用 HCCL in-kernel 高阶 API（`Hccl<HCCL_SERVER_TYPE_AICPU>`，AICPU 引擎），计算用 `AscendC::Matmul`，工程形态为 **aclnn 单算子注册**（`build.sh` → `.run` + `libcust_opapi.so`）。通算融合**不支持 `<<<>>>` 直调与 GE 入图**（官方通算融合指南）。本期仅覆盖 A2，不涉及 A3（910_93）/950，不支持 quant。

> **与其他路线的边界**：910B 硬件不支持 SHMEM/UDMA，Blaze 为 950 专属；§1/§2 的"禁止 HCCL / 禁止 `AscendC::Matmul`"只约束 AIV+URMA 直调，本路线正好相反。MoE 的 MTE 窗口搬运仍走 §3，不要用 `Hccl::AlltoAll` 替代 token 级路由。

#### 红线 / 约束

| # | 约束 | 说明 | 详见 |
|---|------|------|------|
| R1 | 架构=910B(A2) | `ascend910b` / `dav-2201`；无 A3/910_93/950 | 全路线共性红线 ① |
| R2 | 通信走 HCCL 高阶 V2 | `Hccl<HCCL_SERVER_TYPE_AICPU>` + `InitV2`/`SetCcTilingV2`；禁 SHMEM/HCOMM/窗口手动 MTE/RAC | [`comm_hccl.md`](../upstream.md#ascendc-mc2-best-practice-foundations-hccl-matmul-comm-hccl-md) |
| R3 | Matmul 走 `AscendC::Matmul` | 头 `lib/matmul/matmul_intf.h`；禁 Blaze；不调 `SetLocalWorkspace` | [`matmul_fusion.md`](../upstream.md#ascendc-mc2-best-practice-foundations-hccl-matmul-matmul-fusion-md) |
| R4 | AIC-only + Finalize 前跨核同步 | Prepare/Wait/Finalize 在 `ASCEND_IS_AIC` 内；Finalize 前 `CrossCoreSetFlag`/`CrossCoreWaitFlag` | [`mc2_architecture.md`](../upstream.md#ascendc-mc2-best-practice-foundations-hccl-matmul-mc2-architecture-md) |
| R5 | 仅注册 / aclnn | 禁止 `<<<>>>` 直调；host 须 `MC2().HcclGroup("group")` | [`op_architecture.md`](../upstream.md#ascendc-mc2-best-practice-foundations-hccl-matmul-op-architecture-md) |
| R6 | 无 quant | 不引入 `SetQuant*` / `qbmm_mx` / int8 通信路径 | — |
| R7 | PTA 不在本路线重做 | 算子可跑通后走 `torch-ascendc-op-extension` **路线 B**（aclnn 注册） | 该 skill `routes/aclnn-registry.md` |

**逐项审查清单**：[`review-checklist.md`](../upstream.md#ascendc-mc2-best-practice-foundations-hccl-matmul-review-checklist-md)（违反红线项 = FAIL）。

精度判据须与融合方向匹配：先通后算（AllGather+MM）可用逐元素 `np.isclose` + 误差元素比例；先算后通（MM+ReduceScatter/AllReduce）须用整体相对误差 `‖y_npu−y_fp32‖₂/‖y_fp32‖₂ ≤ 4ε`，scatter 类须逐 rank 校验。详见 [`workflow_integration.md`](../upstream.md#ascendc-mc2-best-practice-foundations-hccl-matmul-workflow-integration-md) §1.4.1。

#### References

| 文档 | 何时读 |
|------|--------|
| [`workflow_integration.md`](../upstream.md#ascendc-mc2-best-practice-foundations-hccl-matmul-workflow-integration-md) | 设计/开发/验收各阶段的 910B MC2 动作与门禁 |
| [`mc2_architecture.md`](../upstream.md#ascendc-mc2-best-practice-foundations-hccl-matmul-mc2-architecture-md) | 第一次接触：AIC-only、prepare→Wait 两层流水、M 轴切分 |
| [`comm_hccl.md`](../upstream.md#ascendc-mc2-best-practice-foundations-hccl-matmul-comm-hccl-md) | 写/审通信层：HCCL 高阶 API、V2 生命周期、窗口优化 |
| [`matmul_fusion.md`](../upstream.md#ascendc-mc2-best-practice-foundations-hccl-matmul-matmul-fusion-md) | 写/审计算层与 HCCL 的耦合（API 签名一律引用 [Matmul API](../upstream.md#ascendc-api-best-practices-api-matmul-md)） |
| [`op_architecture.md`](../upstream.md#ascendc-mc2-best-practice-foundations-hccl-matmul-op-architecture-md) | aclnn 注册模型与可复用 `mc2_common` |
| [`codebase_map.md`](../upstream.md#ascendc-mc2-best-practice-foundations-hccl-matmul-codebase-map-md) | 从蓝本 `cp -r` 起手按 `[REUSE]/[MODIFY]` 定点改造 |
| [`pipeline_tuning.md`](../upstream.md#ascendc-mc2-best-practice-foundations-hccl-matmul-pipeline-tuning-md) | 本路线 tileCnt 约束与公式化 tiling 系数；两阶段策略复用 [`shared/pipeline_tuning.md`](#通算并行调优tilecnt--headmsize) |
| [`references/shared/profiling_mc2.md`](#性能采集msprof--l2-cache-flush参考) | msprof task-based + 多卡 last-N 取 max（本路线 **无需** 950 的 L2 flush） |
| `references/foundations/hccl-matmul/all_gather_matmul/` | 已验证参考实现（AllGather+Matmul，910B/HCCL/AICPU，aclnn）——`cp -r` 起手；**勿改蓝本** |

## MC2 能力声明（路线登记表）

> 本文件登记 MC2 通算融合算子的**已验证实现路线**。每行 = 决策栈上一条一致路线：L0（chip × op_type × 调用形态）→ L1 通信路径 → L2 编程抽象。
> Architect 在需求分析阶段查询本表：确认路线可用性 → 路由到知识目录 → 定位参考实现。本表也登记**明确不支持**的组合（否定行），让否决有据可查。

### 决策栈速览

| 层 | 回答的问题 | 本表对应列 |
|----|-----------|-----------|
| L0 需求层 | 做什么、在哪跑、怎么被调用 | `chip` / `op_type` / `调用形态` |
| L1 通信路径层 | 数据怎么跨卡搬（引擎 + 协议） | `通信路径` |
| L2 编程抽象层 | 用什么写（技术底座：通信库+模板组装 / 模板库 / 裸 API） | `编程抽象` |
| L3 工程组织层 | 代码怎么摆 | 由 L2 决定，见对应知识目录 |
| L4 流水编排层 | 通信与计算怎么重叠 | 见对应知识目录文档 |

> 关键关系：L1 与 L2 是**多对多**——UDMA 上可用 blaze-shmem 或 apace 底座；apace 底座可跑 UDMA（直调）或 HCCL windows（注册）。因此路线必须整行登记，不能由轴自由组合推导。

### 已验证 / 规划路线

| chip | op_type | 调用形态 | 通信路径 | 编程抽象 | status | reference_impl | 知识目录 |
|------|---------|---------|---------|---------|--------|----------------|---------|
| dav-3510 | collective-comm | 直调 | AIV+URMA | blaze-shmem | supported | `references/foundations/blaze-shmem/all_to_all_matmul/` | `references/foundations/blaze-shmem/` |
| dav-3510 | collective-comm | 直调 | AIV+URMA | apace | supported | ops-transformer `mc2/common/op_kernel/apace/kernel/all_to_all_quant_matmul/`、`.../all_gather_quant_matmul/`（通信在前 PUT）；compute-first ReduceScatter 语义参考 `mc2/matmul_reduce_scatter/`、`mc2/quant_reduce_scatter/`（⚠️ 均为注册形态实现、未用 apace，仅作语义/golden 参考） | `references/foundations/apace/` |
| dav-3510 | moe | 直调 | MTE通信（AIV+UBMEM） | ascendc-api | supported | `references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/` | `references/foundations/ascendc-api/moe-dispatch-combine/` |
| dav-2201 | moe | 直调 | MTE通信（AIV+UBMEM） | ascendc-api | supported | 同上（compat 层抹平 dav-2201/dav-3510 window 结构差异） | 同上 |
| dav-2201 | collective-comm | 注册 | HCCL 高阶 | hccl-matmul | supported | `references/foundations/hccl-matmul/all_gather_matmul/` | `references/foundations/hccl-matmul/` |
| dav-3510 | moe | 直调 | AIV+URMA | apace | planned | ops-transformer `mc2/mega_moe/` | — |

> 注：MTE通信路径下 HCCL 仅用于 host 侧 window 资源分配（`HcclAllocComResourceByTiling`），不涉及 `Hccl::*` 高阶集合通信 API。UDMA 即 URMA 协议的同义称呼。

### 否定行（明确不支持）

| 组合 | status | 原因 |
|------|--------|------|
| 直调 × HCCL 高阶集合通信（`Hccl::*`）× AscendC::Matmul 高阶 API | unsupported | HCCL 集合通信库（Ascend C 高阶 API 一部分，基于 HCOMM 构建）依赖框架注入的通信上下文，Kernel 直调拿不到；官方通算融合仅支持 acnn 单算子调用（详见 [`foundations/blaze-shmem/comm_shmem.md`](../upstream.md#ascendc-mc2-best-practice-foundations-blaze-shmem-comm-shmem-md) §5）。910B 集合通信+Matmul 应改走本表 `dav-2201 × collective-comm × 注册 × HCCL 高阶 × hccl-matmul` 行（详见 [`foundations/hccl-matmul/`](../upstream.md#ascendc-mc2-best-practice-foundations-hccl-matmul)） |
| 直调 × HCCL windows（`GetHcclContext`） | unsupported | 无 `__global__` 入口，不支持 Kernel 直调；仅限注册场景（apace 非直调变体，见 [`foundations/apace/fundamentals/architecture.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-fundamentals-architecture-md) §10 ④） |
| 直调 × CCU | unsupported | 同上，无 `__global__` 入口 |
| dav-2201 × UDMA（blaze-shmem 或 apace） | unsupported | UDMA 路径仅 dav-3510 已验证，其他架构行为未验证，禁止使用 |

未出现在任何行中的组合一律视为 unsupported（如 moe × blaze-shmem：无参考实现）。

### 字段说明

| 字段 | 含义 | 取值 |
|------|------|------|
| `chip` | 芯片编译宏架构 | `dav-3510`（Ascend 950PR/950DT）、`dav-2201`（Atlas A2/A3 系列）等 |
| `op_type` | 算子类型 | `collective-comm`（集合通信类）、`moe`（MOE 类），分类速查见 [`requirement-analysis/classification.md`](../upstream.md#ascendc-mc2-best-practice-requirement-analysis-classification-md) |
| `调用形态` | 工程调用方式 | `直调`（`<<<>>>`）、`注册`（op_host + op_kernel 完整工程） |
| `通信路径` | 跨卡数据搬运方式（通信引擎 + 协议的组合） | `AIV+URMA`、`MTE通信`（AIV+UBMEM）、`CCU`（CCU+URMA，尚无直调参考工程）、`HCCL 高阶`，选项详见 [`requirement-analysis/comm-path-decision.md`](../upstream.md#ascendc-mc2-best-practice-requirement-analysis-comm-path-decision-md) |
| `编程抽象` | 代码编写所基于的技术底座，值即 `references/foundations/` 下的底座目录名 | `blaze-shmem`（SHMEM 通信库 + Blaze 计算模板，手工组装）、`apace`（APACE 模板库，通信+计算+工程组织全包）、`ascendc-api`（裸 Ascend C API 全自建，MTE通信场景含 compat 层）、`hccl-matmul`（HCCL 高阶 + `AscendC::Matmul` 高阶，仅注册） |
| `status` | 可用性状态 | `supported`（有参考实现可开发）、`planned`（已规划无实现）、`unsupported`（不可用） |
| `reference_impl` | 参考实现路径 | skill 内相对路径或 ops-transformer 仓路径 |
| `知识目录` | 该路线的领域知识所在目录 | `references/foundations/` 下的目录（底座目录或其下路线目录），Architect 据此路由 |

### 查询规则

1. 按 L0 三列（`chip` / `op_type` / `调用形态`）过滤候选行
2. `supported` 优先于 `planned`；`planned` → 告知用户该组合尚无参考实现、风险自担；仅命中否定行或无行命中 → 给出确定答复（不可用 + 原因 + 替代建议）
3. 多行命中时按用户信号选编程抽象（见下表）
4. 按 `知识目录` 列路由到对应 references 目录；按 `reference_impl` 定位起手工程

#### 编程抽象选择信号（多行命中时）

| 信号 | 编程抽象 |
|------|---------|
| 用户提到"apace"、"APACE"、"通算融合框架"、"CollectiveComm"、"CRTP"、"四段式 API" | apace |
| 代码在 `ops-transformer` 仓 `mc2/common/op_kernel/apace/` 下 | apace |
| 用户提到"SHMEM"、"shmem"、"aclshmem" | blaze-shmem |
| 代码自带独立 CMake 工程 + `aclshmem_*` API | blaze-shmem |
| 用户未明确，但需要快速原型验证 | blaze-shmem（基底工程自包含） |
| 用户提到"910B"、"HCCL 高阶"、"aclnn 通算融合" | hccl-matmul |
| 代码含 `Hccl<HCCL_SERVER_TYPE_AICPU>` + `AscendC::Matmul` / `aclnn*Matmul` | hccl-matmul |
| Brownfield 模式从代码推断 | 见 [`codebase-analysis.md`](../upstream.md#ascendc-mc2-best-practice-codebase-analysis-md) |

> 例：moe × dav-3510 × 直调 命中两行——`supported`（MTE通信）优先，仅当用户明确要求 apace/AIV+URMA 路线时选 `planned` 行并告知风险。

### 扩展操作

| 扩展场景 | 操作 | 影响范围 |
|---------|------|---------|
| 新芯片验证通过 | 加/改行（`chip` 列） | 仅本表 |
| 新通信路径（如 CCU 直调落地） | 路径列加取值 + 加行 | 本表 + comm-path-decision.md |
| 新编程抽象（新技术底座） | 抽象列加取值 + 加行 + 新建 `references/foundations/{新底座}/` | 本表 + 一个知识目录 |
| 新算子类型 | `op_type` 加取值 + 加行（跨通信路径设计模式放 `references/operators/{type}.md`） | 本表 + operators/ |
| 注册形态路线验证（如 apace × HCCL windows × 注册） | 加行（调用形态=注册） | 仅本表 |

### 需求分析阶段相关文档

| 文档 | 用途 |
|------|------|
| [`requirement-analysis/template.md`](../upstream.md#ascendc-mc2-best-practice-requirement-analysis-template-md) | REQUIREMENTS.md 模板（MC2 特有章节） |
| [`requirement-analysis/comm-path-decision.md`](../upstream.md#ascendc-mc2-best-practice-requirement-analysis-comm-path-decision-md) | 通信路径选项（决策栈 L1 层） |
| [`requirement-analysis/classification.md`](../upstream.md#ascendc-mc2-best-practice-requirement-analysis-classification-md) | MC2 算子分类速查表和可信源清单 |
| [`requirement-analysis/grill-protocol.md`](../upstream.md#ascendc-mc2-best-practice-requirement-analysis-grill-protocol-md) | 需求拷问 8 维度判据 |
| [`requirement-analysis/quality-checklist.md`](../upstream.md#ascendc-mc2-best-practice-requirement-analysis-quality-checklist-md) | 需求文档自检 14 项清单 + 开发就绪闸门 |
| [`operators/collective-comm.md`](../upstream.md#ascendc-mc2-best-practice-operators-collective-comm-md) | 集合通信类算子跨编程抽象共性设计模式 |
| [`codebase-analysis.md`](../upstream.md#ascendc-mc2-best-practice-codebase-analysis-md) | Brownfield：从代码推断路线坐标（Stub） |

## 通算并行调优（tileCnt / headMSize）

本文档承载 MC2 skill 的"通算并行调优子能力"。核心结论：**`tileCnt` 是通算流水的深度旋钮**，在 CANNBot 流程中分两阶段调优——Step 2-4 用 `tileCnt=1` 做串行基线（简化精度/审查调试），Step 6 性能验收阶段扫描 `tileCnt` 找最优值。

> 参考工程默认 `headMSize=512`（即 `tileCnt = M/512`）只是经验值，**不是最优值**。不同 M/K/dtype 组合下最优 `tileCnt` 不同，必须以 `msprof --ai-core=on --aic-mode=task-based` 实测为准。

---

### 1. tileCnt 是什么

```cpp
// include/tiling/all_to_all_matmul_tiling_data.h
struct AllToAllCommTilingData {
    uint32_t tileCnt;      // M 轴切分块数 = 通算流水的深度
    uint32_t bufferSize;   // SHMEM buffer 轮转数（独立于 tileCnt，默认 4）
};

// src/all_to_all_matmul.cpp host 侧
uint32_t headMSize = 512;                       // 每块的 M 行数
uint32_t tileCnt = (m - tailMSize) / headMSize; // M 轴一共切多少块
tilingData.commTilingData.tileCnt = tileCnt;
```

`tileCnt` 与 `headMSize` 是反向关系：`headMSize = M / tileCnt`（假设不处理 tail）。

- AIV 把 M 轴切成 `tileCnt` 块，每块 `headMSize` 行，逐块 Put；
- AIC 在通信 buffer 上逐块做 Blaze Matmul；
- 通过 `CrossCoreSetFlag<0x2, PIPE_MTE3>(i)` / `CrossCoreWaitFlag<0x2, PIPE_MTE2>(i)` 让 AIV 的 Put(i+1) 与 AIC 的 MMAD(i) 并行——这就是通算并行。

> **路线差异**：上述 `PIPE_MTE2` 编排适用于 blaze-shmem 路线和 apace PUT 模式（AIC 侧等通信数据，gate MTE2 装载路径）。
> apace GET 模式（契约级推导，官网无样例）使用不同的配对：
> - GET 模式 AIV WaitFlag：**无模板形态** `CrossCoreWaitFlag(i)`（默认 mode 0 + PIPE_S，阻塞标量流——AIV Wait 后紧随标量流 Commit 下发，模板形态不阻塞标量流会引入竞态，见 `ascendc-api-best-practices` skill `references/api-crosscore-sync.md` §4.1）
> - GET 模式 AIC SetFlag：`PIPE_FIX`（而非 `PIPE_MTE3`）
> 详见 [`../foundations/apace/fusion.md`](../upstream.md#ascendc-mc2-best-practice-foundations-apace-fundamentals-fusion-md) §3（GET/PUT flag 编排模式）与 `ascendc-api-best-practices` skill `references/api-crosscore-sync.md`（API 签名与平台生效性）。

---

### 2. tileCnt 与通算并行度

```
tileCnt=1:
  AIV:  Put(0) ━━━━━━━━━━━━━━━━━
  AIC:            wait ━━ MMAD(0)
  → 通信延迟完全暴露（AIC 等 AIV Put 完才能开始算）

tileCnt=2:
  AIV:  Put(0) ━━ Put(1) ━━━━━━━
  AIC:     wait ━━ MMAD(0) ━━ MMAD(1)
  → Put(1) 与 MMAD(0) 并行，1 级流水掩盖

tileCnt=N (N≥2):
  AIV:  Put(0) ━ Put(1) ━ Put(2) ━ ... ━ Put(N-1) ━
  AIC:     wait ━ MMAD(0) ━ MMAD(1) ━ ... ━ MMAD(N-1)
  → N-1 级流水掩盖；稳态下通信延迟被 MMAD 完全吸收
```

**关键洞察**：
- `tileCnt=1` 不是"没有通算融合"——AIV/AIC 仍然在两个核上同时启动 kernel，但只有一个数据块，没有掩盖空间；
- `tileCnt>1` 才有真正的通算并行（Put 与 MMAD 重叠）；
- `tileCnt` 越大，掩盖级数越多，但 `headMSize = M/tileCnt` 越小，UDMA 单次 Put 数据量越小，带宽利用率下降。

---

### 3. CANNBot 两阶段调优策略

#### 阶段 A：`tileCnt=1` 串行基线（Step 2 设计 → Step 4 审查）

**DESIGN.md 中配置**：
```cpp
// host 侧
uint32_t headMSize = m;   // 整个 M 一块
uint32_t tileCnt = 1;
```

**为什么从 `tileCnt=1` 开始**：
- 算法正确性独立于 `tileCnt`——精度问题用 `tileCnt=1` 排查最简单（无流水干扰）；
- AIV/AIC 同步链路最短，CrossCore Flag 出错时直接 deadlock 而非"数据错位"；
- Reviewer 在 Step 4 审查时可以专注于算法正确性 + 三大约束，不必同时纠结通算性能；
- Blaze `tilingEngine.GetTilingData(headMSize=m, n, ka, ...)` 仍然会推导内部 `baseM/baseN/baseK`（典型 baseM=128/256），单卡 Matmul 正确性独立于通信切分。

**门禁**：精度 verify_result.py PASS + REVIEW.md 判定 PASS / PASS WITH NOTES。

#### 阶段 B：扫描 `tileCnt` 找最优（Step 6 性能验收）

**扫描候选值**：`tileCnt ∈ {1, 2, 4, 8, 16, 32}`（建议为 2 的幂，便于均匀切分和调试）

> 注意：`bufferSize` 独立于 `tileCnt`，默认 4。`tileCnt` 决定通信块总数，`bufferSize` 决定 SHMEM 中同时存在的 buffer 数。`bufferId = mLoopIdx & (bufferSize - 1)` 的位掩码约束作用于 `bufferSize`（要求 `bufferSize` 为 2 的幂），不约束 `tileCnt`。

**每个 `tileCnt` 的测试流程**：

> 示例调用的 [msprof_perf_summary.py](../upstream.md#ascendc-mc2-best-practice-ops-ops-profiling-scripts-msprof-perf-summary-py) 未随本技能打包。先获取脚本所在的完整检出及依赖，将 `CANNBOT` 设为该检出的根目录；以下命令读取其中的脚本。任务结果仍由任务的评测程序判定。

```bash
PROJ="$(pwd)"

# 1. 改 host 侧 headMSize 计算（src/{op_name}.cpp）
# 例如 tileCnt=8: headMSize = m / 8

# 2. 重新编译
cmake --build build -j

# 3. 跑精度验证（确保 tileCnt 改动未破坏正确性）
bash run.sh

# 4. msprof task-based 采集（无 warm-up；L2 flush 由 perf 主循环内部保证）
msprof --ai-core=on --aic-mode=task-based \
    --output="${PROJ}/docs/perf/tileCnt_8" \
    --application="${PROJ}/build/{op_name} 2048 8192 3584 4 perf"

# 5. 多卡后处理：使用算子 profiling 资料的 msprof_perf_summary.py
python3 ${CANNBOT}/ops/ops-profiling/scripts/msprof_perf_summary.py "${PROJ}/docs/perf/tileCnt_8" ops/{op_name}
```

**选最优的标准**（优先级从高到低）：
1. **整体 Task Duration（4 卡 max）最小**——直接性能指标；
2. **cube ratio 接近理论上限**（MC2 算子 40-70%）——AIC 被充分喂满；
3. **AIV/AIC time 接近**——通信延迟被计算掩盖（详见 `profiling_mc2.md §5.2`）。

---

### 4. headMSize 的约束

切换 `tileCnt` 时，`headMSize = M / tileCnt` 必须满足：

| 约束 | 说明 | 违反后果 |
|------|------|---------|
| `headMSize > 0` | `tileCnt ≤ M` | 编译期 OK，运行期 tile 为空 |
| `M % tileCnt == 0` | 否则要显式处理 tail | host 侧 `tileCnt = (m - tailMSize) / headMSize` 当前不支持非零 tail |
| `headMSize ≥ Blaze baseM`（典型 128/256） | Blaze `BlockMmad` 需要最小块 | `GetTilingData` 报错或推导失败 |
| `headMSize × kPerRank × sizeof(dtype)` 落在 UDMA 带宽高效区间 | 数十 KB ~ 512 KB（带宽高效区间）；单轮 PUT 可靠性上限无官方硬边界（512KB 系 bring-up 期经验，同平台官方实现单次 12.6MB 稳定——bring-up 期可按 ≤512KB 起步，放大后按 case 复核） | UDMA 带宽利用率下降，Put 开销暴露；单轮放大后未复核时存在通信可靠性风险 |
| `bufferSize × headMSize × kPerRank × rankSize × sizeof(dtype) ≤ SHMEM_SPACE_SIZE` | SHMEM 空间预算（默认 1 GB） | `aclshmem_align` 失败 |

**512 不是最优值**——参考工程固定 `headMSize=512` 只是为了在 `M=2048, K=8192, rankSize=4` 的典型 shape 下取得一个合理起点。其他 shape/dtype 下最优值不同：
- M 大、K 小时，`tileCnt` 增大（`headMSize` 减小）让通信粒度更细，掩盖级数更多；
- K 大时，单次 MMAD 已足够长，`tileCnt` 可减小（`headMSize` 增大）减少通信次数。

**按 rank 数分档的标定值**（Ascend 950PR/950DT（dav-3510）平台实测标定，不跨平台复用）：

| rankSize | 推荐 headMSize | 理由 |
|---|---|---|
| ≤ 2 | 512 | 大 tile、少 SyncAll、通信效率高（分片大，同步次数敏感） |
| ≥ 4 | 128 | 小 tile、多 commTurn，`AllToAll(t) ∥ Reduce(t-1)` 流水掩盖效果好 |

> ⚠️ **flag 约束（按 flagId 模式区分）**：**计数式固定 flagId**（推荐，apace compute-first 默认）下 T 无数值上限要求——每 flagId 计数器 0-15 衡量**未消费积压**，紧邻配对（每轮 Set 后立即 Wait）下积压 ≈1-2 不触顶，生产验证 T=16 稳定；**轮次索引式 flagId**（flagId=tid）须满足轮次索引 < 16 且避开 SyncAll 保留区（`SyncAll<true>` 占 14，实际安全上界 13）。按所选模式核对 `headMSize ≥ ceil(mSeg / 安全上界)`，不为满足分档值突破 flag 约束。口径详见 apace 路线 `fusion.md` §3.3 / R9。

同时受 Blaze baseM 约束：headMSize 需为 baseM（减半后典型 128）的整数倍；SWAT baseM=256 且 headMSize 可被 128 整除时可手动减半 baseM 至 128（见 fusion.md §6.2.8）。

---

### 5. 每块 Matmul 的 Tiling：`tilingEngine.GetTilingData`

每块通信数据的 Matmul tiling 由 host 侧 `tilingEngine.GetTilingData(headMSize, n, ka, ...)` 自动推导：

```cpp
// src/all_to_all_matmul.cpp host 侧
QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN> tilingEngine;

// 只需一份 tiling（按 headMSize 切块）
tilingEngine.GetTilingData(headMSize, n, ka, false, true, tilingData.tileQbmmTilingData);
```

**推导关系**：

| 输入 | 输出（自动推导） |
|------|----------------|
| `headMSize` | Blaze 内部 `baseM`（≤ headMSize，对齐 16/128） |
| `n` | `baseN`、`nTailTile` 等 |
| `ka = k/rankSize` | `baseK`、`scaleKL1`、`stepK` 等 |
| — | `usedCoreNum`（实际启动的 AIC 核数） |

**切换 `tileCnt` 时必须重新调 `GetTilingData`**——`headMSize` 变了，所有 Blaze tiling 字段都会跟着变。

**`tileCnt=1` 时**：`headMSize=m`（全 M），`GetTilingData` 推导出的 `baseM` 仍然是 128/256 这种小值，Blaze 内部仍会做多 block 切分（在 AIC 内部把 `headMSize` 再切成多个 baseM 块），所以**单卡 MMAD 性能不会因为 `tileCnt=1` 而下降**——下降的只是通算掩盖效果。

---

### 6. 调优决策树

```
Step 2-4: tileCnt=1, 跑通精度 + 审查通过
    │
    ▼
Step 6: 扫描 tileCnt
    │
    ├── 候选值：1, 2, 4, 8, 16, 32（2 的幂）
    │
    ▼
每个 tileCnt 跑 msprof task-based 采集，记录整体 Task Duration（4 卡 max）
    │
    ├── Task Duration 随 tileCnt 增大单调下降 → 继续增大 tileCnt
    ├── Task Duration 先降后升 → 取最低点
    └── Task Duration 随 tileCnt 增大单调上升 → 保持 tileCnt=2 或 4
    │
    ▼
最优 tileCnt 写入 DESIGN.md "性能调优记录" 小节
    │
    ▼
Step 7: 汇报（含最优 tileCnt + 对应 Task Duration）
```

---

### 7. 常见误区

| 误区 | 真相 |
|------|------|
| `tileCnt` 越大越好 | 过大会让 `headMSize` 太小，Blaze `baseM` 推导失败 / UDMA Put 带宽利用率下降 |
| `tileCnt=1` 等于没有通算融合 | AIV/AIC 仍同时启动；只是通信延迟无掩盖 |
| 固定 `headMSize=512` 最优 | 仅是参考工程经验值；不同 shape/dtype 最优值不同 |
| `tileCnt` 影响精度 | 不影响——只改变通信-计算时序，不改变数据内容 |
| `tileCnt` 必须是 `bufferSize` 的整数倍 | 不必——`bufferId = mLoopIdx & (bufferSize - 1)` 自动轮转 |
| 改 `tileCnt` 后 Blaze tiling 不变 | 错——`GetTilingData` 必须重新调用，`baseM` 等会跟着 `headMSize` 变 |

---

### 8. 与其他参数的关系

| 参数 | 关系 |
|------|------|
| `headMSize` | `headMSize = M / tileCnt`（反向） |
| `bufferSize` | 独立——SHMEM 中同时存在的 buffer 轮转数（默认 4）。增大 `tileCnt` 不需要改 `bufferSize` |
| `baseM / baseN / baseK` | Blaze 内部 tiling，由 `GetTilingData(headMSize, n, ka, ...)` 推导 |
| `SHMEM_SPACE_SIZE` | 总空间 ≈ `bufferSize × headMSize × kPerRank × rankSize × sizeof(dtype)` + scale 段 |

---

### 9. 后续阅读

| 想了解 | 读 |
|--------|---|
| 性能采集流程（msprof task-based + L2 flush + 4 卡后处理） | `profiling_mc2.md` |
| MC2 通算流水架构 | `mc2_architecture.md` §4 "4-Buffer 流水" |
| Blaze tiling 字段含义 | `matmul_blaze.md` §4 "Tiling 数据流" |
| Step 6 验收门禁 | blaze-shmem：`../foundations/blaze-shmem/workflow_integration.md` §Step 6；apace：`../foundations/apace/workflow/step4-implementation.md` §4 精度与性能验收 |

## 性能采集（msprof + L2 cache flush）参考

本文档承载 MC2 skill 的"性能子能力"。MC2 算子性能采集的三个核心要点：

1. **必须刷 L2 cache**：每轮主 kernel 前调 `heavy_add_kernel`，否则前一轮的 B 矩阵驻留 L2 让带宽指标虚高；
2. **用 `msprof --ai-core=on --aic-mode=task-based` 采集**：以 task（kernel launch）为单位记录，**不用 `--warm-up`**；
3. **多卡数据后处理规则**：每张卡取**最后 5 次** main kernel 的 Task Duration 求平均，4 张卡的平均值取**最大值**作为整体性能。

> 性能采集流程见算子 profiling 资料。本文件只补充 MC2 场景的差异化要点。

---

### 1. 为什么 MC2 性能采集需要刷 L2

MC2 算子的 perf 模式跑多轮（默认 10 轮），每轮都会让主 kernel 把 B 矩阵从 GM 读到 L2。如果不在每轮之间清空 L2：

- 下一轮的 B 直接从 L2 命中，**MTE2 带宽指标虚高**；
- 通信开销（UDMA）相对计算的占比也会失真。

参考工程的解法：每轮主 kernel 前调用 `heavy_add_kernel` 扫一遍大于目标芯片 L2 容量的 bf16 buffer（参考工程示例值 256 MB），把 B 挤出 L2。

> **为什么不用 `--warm-up`？**  `msprof op --warm-up=N` 跳过前 N 次采集，依赖 NPU 自然运行把 L2 填满稳定状态——但 MC2 算子的 B 矩阵驻留 L2 恰恰是要消除的干扰。我们改用 `heavy_add_kernel` 主动挤出 L2，所以不需要 warm-up。

---

### 2. heavy_kernels.h 剖析

参考工程在 `include/kernel/heavy_kernels.h` 提供两个"重负载"kernel：

#### 2.1 heavy_exp_kernel（boost 预热）

```cpp
__global__ __aicore__ __vector__ void heavy_exp_kernel(
    GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t blockLength)
```

- 输入：bf16 buffer（参考工程示例值 10000×10000 = 200 MB）；
- 操作：`y = exp(x)`，全部 AIV 核并行（block 数按目标芯片实际 AIV 核数确定，参考工程示例值 56）；
- 用途：**去 host bound**——上板首次运行受 host 启动开销影响，先跑 100 次 exp 把 host 端的异步任务队列填满，让 NPU 进入稳定状态。

参考工程 `src/all_to_all_matmul.cpp` 注释掉了 boost 阶段（`heavy_exp_kernel` 调用被注释）；新算子按需启用。

#### 2.2 heavy_add_kernel（L2 cache flush）

```cpp
__global__ __aicore__ __vector__ void heavy_add_kernel(
    GM_ADDR x, int64_t totalLength, int64_t blockLength)
```

- 输入：bf16 flush buffer，大小按目标芯片实际 L2 容量动态确定、须大于 L2 容量（参考工程示例值 128×1024×1024 = 256 MB）；
- 操作：`x += 1`，全部 AIV 核并行（block 数按目标芯片实际 AIV 核数确定，参考工程示例值 56）；
- 用途：**刷 L2 cache**——flush buffer 大于目标芯片 L2 容量即可，扫描一遍把 B 矩阵挤出 L2（L2 容量等芯片规格以 npu-arch skill 为唯一知识源，本文档不复述数值）。

参考工程 `src/all_to_all_matmul.cpp` 的 perf 主循环（`mode == "perf"` 分支）：

```cpp
constexpr int PERF_LOOP_COUNT = 10;
for (int i = 0; i < PERF_LOOP_COUNT; ++i) {
    // 1. 刷 L2
    heavy_add_kernel<<<HEAVY_BLOCK_NUM, nullptr, stream>>>(
        cacheFlush, CACHE_FLUSH_ELEM_COUNT, cacheFlushBlockLen);
    // 2. 跑主 kernel
    AllToAllQuantMatmulKernelE4M3E4M3<<<tilingData.tileQbmmTilingData.usedCoreNum, nullptr, stream>>>(
        shmemSpace, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceOutput, tilingData);
    ACL_CHECK(aclrtSynchronizeStream(stream));
}
```

每轮的 `heavy_add_kernel` + 主 kernel 都会被 msprof 单独采集，产生独立的 task 记录。分析时只取主 kernel 的记录。

#### 2.3 HEAVY_BLOCK_NUM 的选择

```cpp
constexpr int64_t HEAVY_BLOCK_NUM = 56;  // 示例值，按目标芯片实际 AIV 核数设置
```

HEAVY_BLOCK_NUM 取目标芯片的**实际 AIV 核数**（AIV 核数属芯片规格，以 npu-arch skill 为唯一知识源，本文档不复述典型值；运行时经 `npu-smi info` 或平台规格接口查询），让 heavy_add_kernel 占满所有 AIV 核，最大化 L2 flush 效果。

---

### 3. perf mode 调用约定

参考工程的 host 程序通过第 5 个命令行参数区分 precision/perf 模式：

```bash
# precision 模式（默认）：单次 kernel + 精度比对
./build/all_to_all_matmul 2048 8192 3584 4

# perf 模式：10 轮 (cache_flush + main kernel)，供 msprof 采集
./build/all_to_all_matmul 2048 8192 3584 4 perf
```

`run.sh` 透传这个参数（第 5 个位置参数）：

```bash
bash run.sh 2048 8192 3584 4 perf
```

#### precision vs perf 流程对比

| 步骤 | precision 模式 | perf 模式 |
|------|---------------|-----------|
| 数据准备 | gen_data.py 生成 input_*.bin | 同 precision |
| kernel 调用 | 单次 `AllToAllQuantMatmulKernelE4M3E4M3` | 10 轮（cache_flush + main） |
| 输出 | npu_out.bin + shmem_*.bin | 无（perf 模式不写文件） |
| verify_result.py | 跑（比对精度） | 不跑（跳过） |
| 用途 | 开发期精度验证 | msprof task 采集 |

---

### 4. msprof 采集命令与数据提取

#### 4.1 标准采集命令

```bash
PROJ="$(pwd)"  # 算子工程根目录

msprof --ai-core=on --aic-mode=task-based \
    --output="${PROJ}/docs/perf/round_001" \
    --application="${PROJ}/build/all_to_all_matmul 2048 8192 3584 4 perf"
```

#### 4.2 关键参数

| 参数 | 说明 | MC2 场景 |
|------|------|---------|
| `--ai-core=on` | 开启 AI Core 数据采集 | 必设 |
| `--aic-mode=task-based` | 以 task（kernel launch）为单位采集 | 必设——每个 kernel 一条记录 |
| `--output=<dir>` | 输出目录 | 直接归档到 `docs/perf/round_NNN/` |
| `--application="<cmd>"` | 被采集程序（含参数，整体用引号包起来） | 第 5 参必传 `perf` |
| **不用** `--warm-up` | MC2 用 `heavy_add_kernel` 主动刷 L2 替代 | — |

#### 4.3 输出目录结构

msprof 在 `--output` 指定目录下生成 `PROF_{timestamp}_{pid}/` 子目录：

```
{output}/
└── PROF_{timestamp}_{pid}/
    └── mindstudio_profiler_output/
        ├── op_summary_{ts}_{pid}.csv   # 每个 kernel 的 task 记录（最关键）
        ├── op_stat_{ts}_{pid}.csv      # 按 Op Name 聚合的统计
        └── ...
```

**`op_summary_*.csv` 的关键字段**（实际 header 以本地 CANN 版本为准）：

| 字段 | 含义 |
|------|------|
| `Op Name` | kernel 名（如 `AllToAllQuantMatmulKernelE4M3E4M3`、`heavy_add_kernel`） |
| `Task Duration (us)` | 单次 task 的耗时（微秒） |
| `aiv_time (us)` / `aic_time (us)` | AIV / AIC 流水耗时 |
| `L2 Hit Rate` | L2 cache 命中率 |

#### 4.4 多卡数据后处理规则

**核心规则**（多卡并行，整体性能由最慢的卡决定）：

```
1. 每张卡（rank）的 op_summary_*.csv 中：
   - 过滤 Op Name == 主 kernel（如 AllToAllQuantMatmulKernelE4M3E4M3）的记录
   - 共 10 条（对应 PERF_LOOP_COUNT=10）
   - 取最后 5 条的 Task Duration (us) 求平均 → 该卡性能 P_i

2. 整体性能 = max(P_0, P_1, P_2, P_3)
```

**为什么取最后 5 条？** 前 5 轮可能受冷启动 / device 频率未稳定影响，后 5 轮进入稳态，取均值降低随机波动。

**为什么 4 卡取最大？** MC2 是多卡并行 + 跨卡同步（`aclshmem_barrier_all`），整体性能由最慢的卡（木桶效应）决定。取平均会高估性能。

#### 4.5 多卡后处理

先获取 [perf_summary.py](../upstream.md#ascendc-mc2-best-practice-ops-ops-profiling-scripts-perf-summary-py) 所在完整源目录及其依赖，将 `CANNBOT` 设为该检出的根目录，再进行多卡数据后处理：

```bash
# 使用算子 profiling 解析工具（生成统计摘要并归档）
python3 ${CANNBOT}/ops/ops-profiling/scripts/perf_summary.py "$ROUND_DIR" ops/{op_name}
```

> `perf_summary.py` 对 CSV 指标生成 min/avg/max 统计摘要（不做判定）。MC2 场景需特别关注的后处理规则（§4.4）：每卡取最后 5 次 main kernel 的 Task Duration 求平均，4 卡取最大值作为整体性能——该规则需基于其输出的 CSV/摘要手动计算。

也可使用 [extract_perf.py](../upstream.md#ascendc-mc2-best-practice-shared-scripts-extract-perf-py) 直接计算 §4.4 的 last-N / max-rank 指标。它要求每卡独立的 `PROF_*` 目录、各目录一个 `op_summary_*.csv`，并按完整 `Op Name` 过滤主 kernel；不替代公共统计摘要与归档工具。

```bash
python3 "${CANNBOT}/ops/ascendc-mc2-best-practice/references/shared/scripts/extract_perf.py" \
    "$ROUND_DIR" "AllToAllQuantMatmulKernelE4M3E4M3" 5
```

---

### 5. MC2 性能指标判定

#### 5.1 主导流水判定

MC2 算子是"通信+计算融合"，预期主导流水是 **AIC cube**：

```
cube_ratio 应 >40%（下限来自 MindStudio 官方文档；上限无官方阈值，内部经验约 70%）
```

若 `vec_ratio` 过高，说明：
- AIV 在做大量非通信工作（可能 fixpipe 没下放到 AIC）；
- 或通信 Put 太密集，AIV 占用过高（Put 数量 / 数据量失衡）。

#### 5.2 通信隐藏率

理想情况下，AIV 的通信 Put 与 AIC 的 MMAD 完全并行：

- AIV 的 `aiv_time(us)` 应该 ≈ AIC 的 `aic_time(us)`；
- 若 AIV time >> AIC time：通信是瓶颈，需增大 `headMSize`（单次 Put 数据量更大）；
- 若 AIC time >> AIV time：计算是瓶颈，需优化 Blaze Tiling（增大 baseK、scaleKL1）或减少 AIC 内部 rank 循环开销。

#### 5.3 核间负载均衡

所有 AIC 核都按相同节奏遍历所有 rank 累加，理论上各核负载相同。但 Blaze 的尾块调度可能让部分核提前结束。判定：

- `op_summary_*.csv` 各核 `aic_time(us)` 差异 <10%：达标；
- 差异 10-30%：警告，检查 `BlockMmad` 的 dispatch 是否均匀；
- 差异 >30%：严重，重审 Tiling 的 `mBaseTailSplitCnt` 等字段。

#### 5.4 卡间负载均衡（4 卡之间）

每张卡都应给出近似的 `rank_avg`（§4.4 的 P_i）。若某张卡明显慢（如 P_2 >> 其他）：

- 检查该卡的 NPU 健康（`npu-smi info` 看 Health 状态）；
- 检查 SHMEM 端口冲突（`tcp://127.0.0.1:8998` 多卡共用时是否丢包）；
- 检查该卡的 AIC 核数是否与其他卡一致（部分卡可能因故障降级）。

#### 5.5 理论耗时计算

MC2 算子的理论耗时按"通信+计算取最大"估：

```
理论耗时 = max(通信耗时, 计算耗时)
通信耗时 = rankSize * commMSize * kPerRank * sizeof(dtype) / UDMA 带宽
计算耗时 = 2 * M * N * K / Cube 算力
```

若 §4.4 的整体性能（4 卡 max） ≈ 理论耗时 → 通算流水接近最优；
若整体性能 >> 通信耗时 且 >> 计算耗时 → 同步开销过大，检查 CrossCore Flag 使用。

---

### 6. 优化方向速查

| 瓶颈信号 | 优化方向 |
|----------|---------|
| 整体 Task Duration >> 理论耗时 | 检查 `CrossCoreWaitFlag` 的 idx 一致性，避免无谓 wait |
| AIV ratio 高（vec_ratio > 30%） | 增大 `headMSize`，减少 Put 次数 |
| AIC cube ratio 低（<40%） | 调整 Blaze Tiling（增大 baseK、scaleKL1） |
| L2 命中率过高（>80%） | L2 flush 没起作用，检查 `heavy_add_kernel` 调用与 `CACHE_FLUSH_ELEM_COUNT` |
| 单核不均衡（核间） | 调整 Blaze Tiling 的尾块分裂策略（mBaseTailSplitCnt / nBaseTailSplitCnt） |
| 单卡不均衡（卡间） | §5.4——查 NPU 健康 / SHMEM 端口 / 核数一致性 |
| AIC 内 rank 循环开销大 | 增大 `headMSize`，减少 tile 数；或合并多次 mmadOp_ 调用 |

详细优化方法见 [单卡性能优化速查](../upstream.md#ascend-ops-profiling-optimization-quickref-md)。

---

### 7. 排错速查

| 现象 | 可能原因 | 排查方向 |
|------|---------|---------|
| `op_summary_*.csv` 无主 kernel 记录 | 跑的是 precision 模式 | 第 5 参传 `perf` |
| `op_summary_*.csv` 只有 heavy_add_kernel | 主 kernel 在 precision 路径 | 同上 |
| PROF_xxx 子目录数量 < rankNum | 部分 rank fork 失败 / 提前退出 | 查 host stderr，确认所有子进程都跑完 |
| 主 kernel 记录 < 5 条 | `PERF_LOOP_COUNT < 5` | 调大 `src/*.cpp` 中的 PERF_LOOP_COUNT |
| L2 命中率 >80% | L2 flush 失效 | 检查 `heavy_add_kernel` 的 totalLength 是否大于目标芯片 L2 容量 |
| perf 模式跑得极慢 | `aclrtSynchronizeStream` 在每轮都同步 | 参考工程已正确实现，新算子不要每轮 sync |
| 某张卡的 avg 明显高于其他卡 | NPU 健康 / SHMEM 端口 / 核数问题 | §5.4 |
| Task Duration 在后 5 轮仍波动大 | device 频率未稳定 / 数据依赖 | 增大 PERF_LOOP_COUNT 到 20，取最后 5 |
| 对照路径（torch/分步实现）trace 出现无关 kernel、计时失真 | 同步方式污染：用空转 kernel（如 `torch.exp` 循环）或全局 sync 做计时同步，会 flush 全局流水线 | 对照实验用 `torch.npu.current_stream().synchronize()` 仅同步当前流；验证 trace 中无对照组无关 kernel |

---

### 8. 后续阅读

| 想了解 | 读 |
|--------|---|
| msprof CSV 字段详解 | [CSV 字段参考](../upstream.md#ascend-ops-profiling-csv-fields-reference-md) |
| 单卡算子的性能优化方法 | [单卡性能优化速查](../upstream.md#ascend-ops-profiling-optimization-quickref-md) |
| 通算融合整体架构 | `mc2_architecture.md` |
| `tileCnt` 调优（通算并行旋钮） | `pipeline_tuning.md` |
| 参考工程改造食谱 | `codebase_map.md` |
