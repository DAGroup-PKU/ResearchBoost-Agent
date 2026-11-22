---
name: ascend-profiling-analysis
description: Analyze existing Ascend model profiling artifacts for host, compute, and communication bottlenecks, timeline anomalies, and performance breakdown by model component.
---

# ascend-profiling-analysis

Inspect the profiler exports appropriate to the actual version. Align steps and ranks before comparing trace intervals; distinguish host scheduling, NPU kernel execution and communication exposure. Follow the analysis stages below and retain evidence for each reported hypothesis.

Read [the technical guide](#昇腾npu-profiling性能分析技能集) for the procedure, examples and tool-specific fields. Execute documented skill procedures and their scripts as described; do not invoke a skill itself as a Bash command.

## Contents

- [昇腾NPU Profiling性能分析技能集](#昇腾npu-profiling性能分析技能集)
- [Ascend Profiling Anomaly Discovery Skill](#ascend-profiling-anomaly-discovery-skill)
- [模型组件性能拆解](references/model-perf-breakdown.md)

## 昇腾NPU Profiling性能分析技能集

### 功能概述

profiling-analysis 是一套完整的华为昇腾NPU性能分析技能集，用于分析Profiling生成的性能数据，自动识别系统性能瓶颈类型（计算、通信、下发），并提供深入分析能力和优化建议。

### 核心功能与执行流程

profiling-analysis 技能通过以下流程自动完成性能分析：

#### 1. 文件扫描与收集
- **输入**：目标文件夹路径
- **处理**：递归遍历目录，自动查找所有 `step_trace_time.csv` 文件
- **验证**：未找到文件时输出错误信息并终止

#### 2. 性能数据解析与计算
- **处理**：验证文件结构，提取计算（Computing）、通信（Communication）、空闲（Free）等核心指标
- **计算**：计算各阶段耗时占比（保留2位小数）
- **异常处理**：跳过读取失败、字段缺失或总耗时为0的文件

#### 3. 全局统计与瓶颈判定
- **统计**：计算所有文件的平均耗时占比，识别各项占比最高的文件
- **判定逻辑**（优先级从高到低）：
  - 下发问题（Hostbound）：空闲耗时占比 > 20%
  - 计算问题（Computing）：计算耗时占比 > 85%
  - 通信问题（Communication）：通信耗时占比 > 10%
  - 无明显瓶颈：各项占比均在正常范围

#### 4. 按瓶颈选择辅助分析
根据判定的瓶颈类型，按下方「子技能介绍」中的链接打开对应指南进行分析：
- **Hostbound** → 调用 `profiling-analysis-hostbound` 子技能
- **Computing** → 调用 `profiling-analysis-computing` 子技能，执行完整分析流程
- **Communication** → 调用 `profiling-analysis-communication` 子技能

#### 5. 分析报告生成
输出完整的性能分析报告，包括：
- 文件扫描结果
- 单文件性能分析数据
- 全局统计与最高占比文件
- 瓶颈判定结论
- 子技能深入分析结果

#### 执行流程图
```
┌─────────────────────────────────────────────────┐
│ 文件扫描与收集                                 │
│ 递归查找 step_trace_time.csv 文件               │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ 性能数据解析与计算                             │
│ 提取核心指标并计算占比                         │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ 全局统计与瓶颈判定                             │
│ 计算平均占比并识别瓶颈类型                     │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ 按瓶颈选择辅助分析                             │
│ 根据瓶颈类型选择对应指南                       │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ 分析报告生成                                   │
│ 输出完整性能分析结果                           │
└─────────────────────────────────────────────────┘
```

### 输入参数

| 参数名称   | 类型   | 是否必填 | 描述                                                         |
| ---------- | ------ | -------- | ------------------------------------------------------------ |
| input_path | string | 是       | 输入路径：支持单个Profiling文件路径，或包含多个Profiling文件的文件夹路径 |

### 使用示例

#### 通过AI工具调用（推荐）
```
调用 skill "ascend-profiling-analysis" 分析 "input_path" 中的性能数据
```

#### 参数说明
- `input_path`：替换为实际的Profiling数据文件夹路径
- 该调用将自动完成文件扫描、性能分析、瓶颈判定和辅助分析

#### 子技能独立调用（可选）
如果工具已经加载了下方列出的辅助技能，可以直接用这些名称调用：

```
# 分析计算瓶颈
调用 skill "profiling-analysis-computing" 分析 "input_path" 中的高耗时算子

# 分析通信瓶颈
调用 skill "profiling-analysis-communication" 分析 "input_path" 中的通信性能

# 分析下发瓶颈
调用 skill "profiling-analysis-hostbound" 分析 "input_path" 中的Host侧问题

# 算力瓶颈 Roofline 核对
调用 skill "profiling-analysis-roofline" 分析 "kernel_details.csv"（需提供 NPU 峰值算力与带宽）
```

### 子技能介绍

以下辅助指南覆盖不同类型的瓶颈。按分析结果打开对应指南，并在需要运行脚本时获取该指南的完整目录与依赖。

| 子技能名称 | 功能描述 | 适用场景 |
|---------|---------|---------|
| [profiling-analysis-computing](../upstream.md#ascend-profiling-analysis-computing) | 计算瓶颈分析，包含：<br>- 高耗时算子筛选（Top-N算子）<br>- 算子性能数据透视表分析<br>- 关键算子（如MatMul系列）形状解析与优化建议<br>- 对 aic_mac_ratio≥80% 算子做 Roofline 理论/实际核对与是否需优化判断 | 计算占比过高时，深入分析具体的高耗时算子及其性能特征 |
| [profiling-analysis-communication](../upstream.md#ascend-profiling-analysis-communication) | 通信瓶颈分析，分析集合通信性能 | 通信占比过高时，分析通信操作的性能 |
| [profiling-analysis-hostbound](../upstream.md#ascend-profiling-analysis-hostbound) | 下发瓶颈分析，分析Host侧性能问题 | 空闲占比过高时，分析Host侧的下发问题 |
| [profiling-analysis-roofline](../upstream.md#ascend-profiling-analysis-roofline) | 算力瓶颈 Roofline 核对：对 aic_mac_ratio≥80% 算子比较理论/实际耗时，并给出是否需优化建议 | 需要判断算力瓶颈算子是否打满峰值、是否值得做单 kernel 优化时 |

### 支持的数据格式

- `step_trace_time.csv`：包含计算、通信、空闲等时间占比信息
- `op_statistic.csv`：包含算子执行统计信息
- `kernel_details.csv`：包含算子内核详细执行信息
- `op_summary.csv`：包含算子执行摘要信息
- `analysis.db`：Profiling数据库文件

### 安装与依赖

- Python 3.8+
- 依赖库：pandas, numpy, matplotlib (可选)

### 官方文档

- [华为昇腾Profiling工具文档](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/perftools/profiler/index.html)
- [MindStudio Profiling使用指南](https://www.hiascend.com/document/detail/zh/mindstudio/60RC1/profiling/index.html)

## Ascend Profiling Anomaly Discovery Skill

Identify the actual rank, step and trace clock before analyzing bubbles and wait anchors. Use unions of overlapping stream intervals for wall time, avoid double-counting concurrent kernels, and report missing inputs as unavailable. The guide specifies step-to-structure analysis and report evidence.

### Purpose

Analyze Ascend NPU profiling data through three parallel pipelines:

1. **Structure breakdown**: step → structure (layer) → block / side → op → PMU judgement — answers *where the time goes*.
2. **Anomaly discovery**: step → device busy union → bubble detection → anomaly tags → soft attribution — answers *what looks unnatural and where hidden issues may lurk*.
3. **Model architecture analysis**: FIA timeline → pass boundaries → layer classification → per-layer sub-structure → communication pipeline → architecture summary — answers *what is this model and how does each component execute*. **Produces a separate Markdown report file.**

The core philosophy is **separation of concerns**: "anomaly exists" is a hard fact derived from device intervals; "why it exists" is a soft attribution that may require additional evidence. Even under weak profiling configurations (no stacks, no shapes, sparse host events), the skill must still reliably surface device idle bubbles and risk labels.

### Reference Files — When to Read

Read these before starting analysis:

| File | When to read | What it contains |
|---|---|---|
| [`references/kernel_data_guide.md`](../upstream.md#ascend-profiling-anomaly-kernel-data-guide-md) | **Always — read first** | Raw data column schemas for kernel_details.csv, op_summary, trace_view.json; the step → structure → block/side → op hierarchy; how to parse, filter, assign kernels at each level; multi-stream handling; per-level timing aggregation |
| [`references/rulebook.md`](../upstream.md#ascend-profiling-anomaly-rulebook-md) | Always | Anomaly thresholds, tagging rules, decision tables, soft attribution rules, AICPU classification, wait-anchor rules |
| [`references/architecture_report_template.md`](../upstream.md#ascend-profiling-anomaly-architecture-report-template-md) | **Always — read before producing the architecture report** | Full template for the standalone Markdown architecture report: required sections, formatting rules, analysis techniques for layer classification, communication overlap measurement, per-layer timing breakdowns |
| [`references/schema.json`](../upstream.md#ascend-profiling-anomaly-schema-json) | When producing structured JSON output | Full JSON schema for the `anomaly_discovery` output object |
| [`scripts/reference_host_gap_branch.py`](../upstream.md#ascend-profiling-anomaly-scripts-reference-host-gap-branch-py) | When writing analysis scripts | Reference Python implementation for interval merging, bubble metrics, soft attribution, wait-anchor scoring |

### Pipeline Overview

The full state machine:

```
INGEST → INVENTORY → FACT_EXTRACTION → CANDIDATE_STEP_DETECTION →
STEP_GROUPING → MACRO_STEP_RESOLUTION → SEGMENTATION →
CLOCK_ACCOUNTING → ANOMALY_DISCOVERY → PERF_JUDGEMENT →
RECOMMENDATION → RENDER → DONE
                                        ↘
                          ARCHITECTURE_ANALYSIS → ARCH_REPORT_RENDER → ARCH_REPORT_SAVE
```

`ANOMALY_DISCOVERY` sits after `CLOCK_ACCOUNTING` and before `PERF_JUDGEMENT`. It receives already-segmented steps and structures, and runs the bubble detection pipeline on top of them.

`ARCHITECTURE_ANALYSIS` runs in parallel with `PERF_JUDGEMENT`, using the same segmented data from `CLOCK_ACCOUNTING` plus FIA timeline analysis. It produces a **separate Markdown report file** saved alongside the profiling data.

---

### Data Hierarchy: Step → Structure → Block/Side → Op

Understanding how raw kernel data maps to each level is essential. Read [`references/kernel_data_guide.md`](../upstream.md#ascend-profiling-anomaly-kernel-data-guide-md) for the full column schemas and parsing details. Here is the conceptual overview:

#### Level 0: Raw Kernels

Each row in `kernel_details.csv` represents a single device kernel execution — one invocation of an AI Core task, an AI CPU task, or an HCCL communication task. Key fields: `Name`, `Task Type`, `Start Time(us)`, `Duration(us)`, `Wait Time(us)`, `Accelerator Core`, `Stream ID`, `Input Shapes`, `Output Shapes`.

#### Level 1: Step

A step is one training/inference iteration. Steps are identified by `ProfilerStep#N` user annotations or `Iteration#N` markers in `trace_view.json`. Each step defines a **service window** `[S_i, S_{i+1})`. All kernels whose start time falls within this window belong to step `i`.

At step level, compute:
- Total service time, device busy union, underfeed ratio
- Prelaunch gap, tail gap, internal bubbles
- Per-step anomaly tags and soft root-cause labels

#### Level 2: Structure (Layer)

Within a step, kernels form repeating **structures** — typically corresponding to model layers (e.g., transformer blocks, attention layers, MLP blocks). Segmentation identifies these by:

1. Repeating name-pattern sequences in the kernel timeline
2. Significant time gaps between kernel groups
3. User annotations marking layer boundaries (if present)

Each structure contains a contiguous span of kernels within the step window.

At structure level, compute:
- Structure wall time, device busy union within structure span
- Structure share of total step time
- Per-structure bubble metrics (is the bubble inside this structure or between structures?)

#### Level 3: Block / Side

Within each structure, kernels split into:

- **Block (main compute path)**: The dominant chain of AI Core kernels forming the forward or backward pass of this layer. These execute on the main compute stream.
- **Side (auxiliary ops)**: Everything else — small element-wise ops, communication (HCCL), AI CPU fallback ops, memory copies, synchronization events. These may execute on separate streams.

At block/side level, maintain **four timing perspectives** simultaneously:
- `wall_ms` — wall-clock time from first kernel start to last kernel end
- `busy_union_ms` — merged device-busy time (accounts for multi-stream overlap)
- `kernel_sum_ms` — arithmetic sum of all kernel durations (ignores overlap)
- `total_cost_ms` — sum of `duration + wait` for all kernels

Conclusions based on only one metric are incomplete. A block appearing heavy in `kernel_sum` but light in `wall` means high stream parallelism. A side appearing heavy in `total_cost` but light in `duration` means wait-anchor false hotspot risk.

#### Level 4: Op (individual operator)

The finest grain. Each op may produce one or many device kernels. Op-level analysis handles:
- Top ops by total cost vs. top ops by kernel duration (these rankings often differ)
- Wait-anchor detection: ops with `wait_ratio > 0.95` and tiny `duration` but high `total_cost`
- AICPU classification: ops running on AI CPU instead of AI Core, classified by `masked_ratio`
- Small-op initial judgement: whether small individual ops are real inefficiencies or noise

---

### Anomaly Discovery Pipeline (Detail)

#### Phase 1: BUILD_DEVICE_INTERVALS

For each step, collect all device kernel intervals from `kernel_details.csv`:

```
device_intervals = []
for each kernel row where Start_Time_us is within step window:
    s = max(step_start_us, row.Start_Time_us)
    e = min(step_end_us, row.Start_Time_us + row.Duration_us)
    if e > s:
        device_intervals.append(Interval(s, e))
```

Key rules:
- Clip kernel intervals to the step window boundary
- Apply communication dedup rules BEFORE interval statistics (see [`references/kernel_data_guide.md`](../upstream.md#ascend-profiling-anomaly-kernel-data-guide-md) section on comm dedup)
- Include AI_CORE, AI_CPU, and HCCL tasks — all count as "device busy"
- When multiple streams exist, intervals from all streams are collected into the same set

#### Phase 2: MERGE_INTERVALS

Sort intervals by start time, merge overlapping ones:

```
merged = merge(device_intervals)  # see scripts/reference_host_gap_branch.py
busy_union = sum of merged segment durations
```

This produces the merged busy segments from which all bubble metrics derive.

#### Phase 3: GAP_CLASSIFICATION

From the merged segments, compute per-step metrics:

- `service_ms` = step window duration
- `device_busy_union_ms` = sum of merged segment durations
- `underfeed_ms` = service − busy_union
- `underfeed_ratio` = underfeed / service
- `prelaunch_gap_ms` = start(first_merged_segment) − step_start
- `tail_gap_ms` = step_end − end(last_merged_segment)
- `internal_bubble_total_ms` = sum of gaps between consecutive merged segments
- `largest_internal_bubble_ms` = max gap between consecutive merged segments
- `bubble_count` = number of inter-segment gaps

#### Phase 4: HOST_EVIDENCE_COLLECTION

For each bubble window (gap between merged segments, or prelaunch/tail gap), scan the same time range in host events from `trace_view.json`:

Host event categories to collect:
- `cpu_op` / `python_function` / `user_annotation` — general host activity
- `AscendCL@*` — ACL runtime events
- `HostToDevice` / `torch_to_npu` / `aclrtMemcpy*` / `aclrtSynchronize*` — sync/copy markers
- `c10d` / `Hccl` / `hcom` / `StreamWaitEvent` / `Notify_Wait` — communication markers

Compute overlap ratios:
- `host_visible_coverage_ratio` = fraction of bubble covered by any host event
- `sync_marker_overlap_ratio` = fraction covered by sync/copy markers
- `comm_marker_overlap_ratio` = fraction covered by communication markers

#### Phase 5: SOFT_ATTRIBUTION

For each significant bubble, assign probability-level labels based on overlap ratios:

| Condition | Label |
|---|---|
| sync_overlap ≥ 0.20 | `possible_sync_or_h2d` |
| comm_overlap ≥ 0.20 | `possible_comm_wait` |
| host_coverage < 0.05 | `possible_untraced_host_blocking` |
| host_coverage ≥ 0.10 but no sync/comm dominance | `possible_host_launch_lag` |
| host_parallelism < 1.2 and none of above | `possible_python_serialization_or_lock` |
| nothing applies | `insufficient_evidence` |

Multiple labels can co-exist. These are explicitly NOT unique root causes.

#### Phase 6: ANOMALY_TAGGING

Apply the anomaly tags from [`references/rulebook.md`](../upstream.md#ascend-profiling-anomaly-rulebook-md) decision tables. Core tags:

**Bubble severity**: `DEVICE_IDLE_GAP_HEAVY`, `PRELAUNCH_GAP_HEAVY`, `TAIL_GAP_HEAVY`, `INTERNAL_BUBBLE_HEAVY`

**Risk tags**: `HOST_ORIGINATED_RISK`, `COMM_SYNC_RISK`, `WAIT_POLLUTION_RISK`, `WAIT_ANCHOR_FALSE_HOTSPOT`, `AICPU_EXPOSED_RISK`, `UNTRACED_HOST_BLOCKING_RISK`, `PARTIAL_CAPTURE_BOUNDARY`, `VARIABLE_SHAPE_SAME_TEMPLATE`

#### Phase 7: WAIT_ANCHOR_SCAN

At op level, scan for false hotspots:

```
wait_ratio = wait_us / (duration_us + wait_us)
if wait_ratio > 0.95 and duration_us < 10.0 and total_cost_rank <= 10:
    tag WAIT_ANCHOR_FALSE_HOTSPOT
```

These ops absorb idle wait time and appear expensive, but their kernel execution is negligible. Demote them in root-cause ranking.

#### Phase 8: GROUP_AGGREGATION

Aggregate step-level metrics by `step_group_id`:
- Compute avg, median, P90, P95 for each bubble metric
- `recurring_bubble_pattern` = true if ≥60% of steps in group have `bubble_count > 0`
- `dominant_idle_pattern` = whichever of prelaunch/internal_bubble/tail contributes most

#### Phase 9: PRODUCE_OUTPUT

Merge anomaly results with structure breakdown. The report MUST include:

##### Hidden Issue Discovery section
- Dominant step/group bubble metrics (service, busy_union, underfeed_ratio)
- **Raw kernel evidence table**: for each top bubble window, list the kernel(s) immediately before and after the gap — their names, task types, durations, streams — so the human expert can locate the exact spot in the timeline
- Top 5 bubble windows with timestamps, scope, and host evidence
- Bubble periodicity statistics across the step group
- Host evidence coverage assessment
- Soft root-cause labels with evidence chains
- Follow-up sampling recommendations

##### Bubble-first Summary (fixed 5-question template)
1. Are there significant device idle bubbles?
2. Which step type/group do they concentrate in?
3. Are they primarily prelaunch / tail / internal / inter-step?
4. Is there significant host-originated risk?
5. Is evidence sufficient for root cause? If not, say so explicitly.

##### Structure-Level Bubble Drill-Down
For the dominant step, break down bubble contributions per structure:
- Which structure(s) contain the largest internal bubbles?
- Are bubbles concentrated at structure boundaries (between layers) or within structures?
- For structures with large bubbles, what are the surrounding kernel names and task types?

---

### Model Architecture Analysis Pipeline (Separate Markdown Report)

This pipeline produces a **standalone Markdown report file** that documents the model architecture as reverse-engineered from profiling data. Read [`references/architecture_report_template.md`](../upstream.md#ascend-profiling-anomaly-architecture-report-template-md) for the full template, formatting rules, and analysis techniques.

The report is saved as `model_architecture_report_<profiling_dir_name>.md` in the profiling or output directory.

#### When to produce the architecture report

**Always.** Every profiling analysis MUST produce this report alongside the anomaly discovery output. The architecture report provides essential context that makes the anomaly findings interpretable.

#### Architecture Analysis Phases

##### Phase A1: FIA_TIMELINE_ANALYSIS

Use FusedInferAttentionScore (FIA) invocations as the primary structural marker:

1. Extract all FIA kernels from `kernel_details.csv` (match by name containing `FusedInferAttentionScore`)
2. Sort by `Start Time(us)`
3. Classify each FIA as prefill (duration > 10ms) or decode (duration < 1ms)
4. Determine pass count: `num_passes = total_FIA / FIA_per_pass`
5. Identify phase transitions by timestamp gaps between prefill and decode FIA clusters

##### Phase A2: PASS_BOUNDARY_DETECTION

For each forward pass, determine:
- FIA index range (e.g., #0–#94)
- Time span (absolute timestamps)
- Wall time from first kernel to last kernel
- Average prefill FIA duration
- Total kernel count

Cross-pass variation (FIA duration, wall time) should be noted — it may reveal KV cache growth or memory pressure effects.

##### Phase A3: LAYER_CLASSIFICATION

For each layer (delimited by consecutive FIA invocations), extract the kernel sequence and classify:

| Classifier kernel | Layer type |
|---|---|
| No MoE markers (no MoeGatingTopK, no DFC, no GroupedMatmul) | **Dense** |
| DispatchFFNCombine present | **MoE+DFC** |
| GroupedMatmul + alltoallv present (no DFC) | **MoE+GMM** |
| Sampling ops (ArgMax, rejection_*) present | **includes decode/sampling logic** |

Build a summary table: Layer Type | Layer Range | Count | Characteristics

##### Phase A4: CROSS_VERIFICATION

Count key ops per pass and verify they match the layer classification:
- FIA count should equal layer count
- DispatchFFNCombine count should match MoE+DFC layer count
- GroupedMatmul count should match MoE+GMM layers + decode layers
- MoeGatingTopK count should match all MoE layers (DFC + GMM + decode)

Discrepancies indicate classification errors — resolve before proceeding.

##### Phase A5: PER_LAYER_SUBSTRUCTURE

For EACH distinct layer type, analyze the kernel execution sequence:

1. Group kernels by functional role (attention, projection, TP comm, MoE routing, expert dispatch, expert FFN, post-MoE norm, next-layer prep, EP comm, sampling)
2. Measure wall time per functional group
3. Identify which stream each group runs on
4. Compute timing breakdown: Component | Wall time | Share of layer
5. Note anomalies specific to the layer type (warm-up overhead in layer 0, extra kernels in transition layer, AICPU ops that should be on AI_CORE)

Present as kernel sequence trees with timing annotations and stream labels. See the template for the exact tree notation format.

##### Phase A6: DECODE_ANALYSIS

Decode layers require separate analysis because they have fundamentally different cost profiles:
- FIA duration drops dramatically (prefill 28ms → decode 0.2ms)
- Communication (especially all-to-all for EP) often dominates
- Expert compute shifts from fused (DFC) to explicit (GroupedMatmul)

Produce a dominant costs table and explain why the cost profile differs from prefill.

##### Phase A7: COMMUNICATION_PIPELINE

Document the multi-stream overlap strategy:
1. Map each stream to its purpose (main compute, AI_CPU comm, HCCL, alltoall)
2. Measure overlap ratios between streams
3. Draw an ASCII pipeline diagram showing how communication hides behind compute
4. Compute what fraction of total communication is hidden (overlapped)
5. Explain the kernel_sum >> wall relationship

##### Phase A8: ARCH_REPORT_RENDER

Assemble all findings into the Markdown report following the template in [`references/architecture_report_template.md`](../upstream.md#ascend-profiling-anomaly-architecture-report-template-md). All 10 required sections must be present:

1. Configuration Context
2. Model Architecture Determination (evidence chain table)
3. Forward Pass Boundaries (per-pass table)
4. Layer Classification (type table)
5. Cross-Verification Table
6. Per-Layer Sub-Structure (kernel sequence trees + timing breakdowns for EACH layer type)
7. Decode Phase Analysis (dominant costs + prefill vs decode comparison)
8. Communication Pipeline Structure (stream table + ASCII pipeline diagram)
9. Layer-to-Layer Variation (comparative table)
10. Model Architecture Summary (ASCII model diagram + execution timeline)

##### Phase A9: ARCH_REPORT_SAVE

Save the report as `model_architecture_report_<profiling_dir_name>.md`. Inform the user of the file location.

---

### Critical Constraints

- **Never skip anomaly discovery because root cause is unclear.** Bubble facts are hard conclusions; root cause labels are soft conclusions. Both must always be reported.
- **Never call a high-wait tiny-duration op a real hotspot** without checking wait-anchor risk.
- **Never ignore local bubbles just because the step is device-bound overall.**
- **Never use only one timing metric** — always maintain dual clock accounting (wall_ms + busy_union_ms + kernel_sum_ms + total_cost_ms) at block/side level.
- **Always output `insufficient_evidence` or `possible_untraced_host_blocking`** when host evidence is sparse — never silently omit the anomaly section.
- **Always include raw kernel context** around reported bubbles — the kernel names, task types, durations, and stream IDs immediately before and after each bubble gap.
- **Use layered certainty language**: declarative for facts, `possible / probable / insufficient evidence` for root causes.
- **Always produce the model architecture Markdown report** as a separate file — never fold it into the anomaly output alone.
- **Never skip per-layer sub-structure analysis.** Every distinct layer type must have its own kernel sequence tree with timing breakdown and stream annotations.
- **Always include the evidence chain table** in the architecture report — the reader must see how layer count and pass structure were determined from raw FIA data.
- **Always cross-verify** op counts against layer classification before finalizing the architecture report. Discrepancies must be resolved or explicitly noted.

### Graceful Degradation

| Missing data | Impact | Action |
|---|---|---|
| `record_shapes=false` | Cannot detect shape variation | Bubble detection continues; tag `VARIABLE_SHAPE_SAME_TEMPLATE` skipped |
| `with_stack=false` | Soft attribution specificity degrades | Lower confidence; bubble detection unaffected |
| Sparse host events | Cannot narrow root-cause family | `UNTRACED_HOST_BLOCKING_RISK`, `requires_host_followup=true` |
| Capture boundary truncation | Edge gaps may be artifacts | `PARTIAL_CAPTURE_BOUNDARY` on boundary-adjacent gaps |
| No `communication.json` | Cannot assess comm wait | Skip comm overlap, note in evidence gaps. Architecture report omits comm pipeline bandwidth stats but still documents stream roles |
| No step markers | Cannot define step windows | Fall back to global capture span as single pseudo-step |
| `op_summary` only (no `kernel_details`) | Coarser granularity | Use op-level intervals instead; note in limitations. Architecture report uses op counts for layer classification but cannot produce per-layer kernel sequence trees |
| No FIA kernels detected | Cannot determine layer boundaries via FIA | Architecture report falls back to alternative structural markers (e.g., repeating kernel patterns, communication boundaries). Note reduced confidence in layer count |
| Single forward pass captured | Cannot cross-validate pass consistency | Architecture report documents single pass; notes that cross-pass variation analysis is unavailable |
| No decode FIA detected | Inference-only or prefill-only capture | Architecture report omits decode phase analysis section; notes capture scope limitation |

### Output Contract Summary

Every analysis must produce TWO outputs:

#### 1. Anomaly Discovery Output

The `anomaly_discovery` top-level object containing: `enabled`, `dominant_group_id`, `global_device_gap_analysis`, `step_group_anomalies`, `bubble_windows`, `wait_anchor_ops`, `soft_root_cause_summary`, `requires_host_followup`, `confidence`.

Each step result must include bubble metrics, anomaly tags, and soft root-cause labels.

For the full JSON schema, read [`references/schema.json`](../upstream.md#ascend-profiling-anomaly-schema-json).

#### 2. Model Architecture Report (Markdown file)

A standalone Markdown file saved as `model_architecture_report_<profiling_dir_name>.md` containing all 10 required sections from the architecture template. Read [`references/architecture_report_template.md`](../upstream.md#ascend-profiling-anomaly-architecture-report-template-md) for the full specification.

The report must include at minimum:
- Evidence chain proving layer count and pass structure
- Layer classification table with all distinct layer types
- Per-layer kernel sequence trees with timing breakdowns for EACH layer type
- Communication pipeline structure with stream overlap diagram
- Model architecture ASCII summary diagram and per-pass execution timeline

The architecture report is the primary deliverable for understanding model structure. It must be self-contained — a reader should be able to understand the full model execution without referring to the anomaly output.

### Recommendations

Each recommendation must include `scope` (global/step_group/structure/side/op), `followup_required`, `evidence_gap`, and `priority` (P0–P3).

Common follow-up patterns:
- High bubble + missing stacks → re-profile with `with_stack=true`
- Missing shapes + unstable grouping → `record_shapes=true`
- High host risk + low evidence → host-side sampling / thread view
- High wait pollution → check communication and sync paths
- Large inter-structure bubble → check host-side layer dispatch latency
