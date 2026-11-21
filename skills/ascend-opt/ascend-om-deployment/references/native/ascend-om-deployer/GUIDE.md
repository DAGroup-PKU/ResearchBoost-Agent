# Ascend OM Deployer

编排完整的 Ascend OM 部署任务：从源码分析或现成 ONNX 入手，经过环境准备、ONNX 到 OM 转换、接回原始推理流程、端到端验证，最终交付可运行的部署方案。

## 子 Skill 分工

主 Skill 只做编排和跨阶段决策，具体能力由两个子 Skill 承担：

| 子 Skill | 职责 |
| --- | --- |
| `ascend-om-pipeline-adapter` | 源码侧分析：真实推理入口、候选边界、输入输出契约、接回主链、fallback 路径 |
| `ascend-onnx-atc-pipeline` | 工具链执行：环境检查、ONNX 获取/优化、ATC 转换、OM 推理验证 |

每次调用子 Skill 时必须说明当前阶段和要消费的标准输出，根据子 Skill 返回结果决定推进、回退或重规划。

## Gotchas

- 不要跳过 Workflow 0 直接开始导出或转换——缺信息会导致后续全链返工
- 不要把 fallback 当初始策略；fallback 只能在 OM 尝试失败且重规划不成立后才出现
- 不要在验证时自行发明精度阈值——必须使用 [验证门控标准](references/validation-gates.md)
- 不要在未经用户确认的情况下修改用户工程源码（Workflow 6 接回）
- 不要忽略子 skill 缺失——进入对应 Workflow 前必须确认子 skill 可用
- 不要把"代码支持可变"等同于"交付必须可变"——先检查交付入口是否已固定某些轴
- 不要用零张量、随机张量或常量占位替代参与后续计算的真实输出

## 路径约定与子 Skill 校验

引用子 Skill 内脚本时：

```bash
export SKILL_DIR=/path/to/<skill-name>
"$SKILL_DIR/scripts/..."
```

**子 Skill 可用性校验**：进入 Workflow 1/4/6 前，确认 `ascend-om-pipeline-adapter` 存在且 SKILL.md 可读；进入 Workflow 2/3/5 前，确认 `ascend-onnx-atc-pipeline` 存在且 SKILL.md 可读。若子 Skill 缺失，停机并告知用户安装对应 skill，不得跳过或降级执行。

## 核心设计原则（摘要）

完整说明见 [references/core-principles.md](references/core-principles.md)。遇到需要判断的模糊场景时按需加载。

| 原则 | 一句话摘要 |
| --- | --- |
| 1. 大图优先 | 围绕本次交付场景寻找最大连续 tensor 计算边界；小图拆分必须有大图失败或重规划证据 |
| 2. 图侧主策略唯一 | 单个 OM / 单次 ATC 只选一种输入形态策略：static / dymbatch / dymhw / dymdims / dymshape |
| 3. 输出保真 | 替换点原始输出全集先记录；本次场景内下游会消费的输出必须由 OM 或原始实现真实提供 |
| 4. 接回 context 隔离 | 混用 .om 与其他 runtime 时必须做 context 隔离，防止互相踩踏 |
| 5. 候选状态机闭环 | 每个候选沿 pending→selected→in_progress→om_done→validated→integrated 推进，或以 failed_replan / fallback / excluded 闭环 |
| 6. 功能范围与批次闭环 | 首批不是任务结束；本次场景内未闭环主链候选必须继续推进，场景外候选记录为后续项 |
| 7. 验证门控标准 | 使用标准化判据，不得临时发明阈值。详见 [validation-gates.md](references/validation-gates.md) |

## 进度跟踪

复制此清单并跟踪进度：

```text
部署进度：
- [ ] Workflow 0：任务启动（锁定信息、决定起点）
- [ ] Workflow 1：源码分析与候选规划
- [ ] Workflow 2：环境准备
- [ ] Workflow 3：ONNX 获取与 OM 转换
- [ ] Workflow 4：边界重评估（按需触发）
- [ ] Workflow 5：子图级验证与批次门控
- [ ] Workflow 6：接回原始推理流程
- [ ] Workflow 7：端到端验证与交付
```

## Workflow 0：任务启动

进入任何 Workflow 前，先锁定以下信息（缺失则先补齐）：

| 信息 | 来源 |
| --- | --- |
| 模型源码或工程路径 | 用户提供 |
| 权重路径或下载来源 | 用户提供 |
| 目标 SoC | 用户提供 |
| CANN / Python / ONNX / ATC 环境 | 环境检测 |
| 用户真正关心的推理入口 / 本次初步适配场景 | 用户提供；未指定时从 README、官方 demo、benchmark 或 service API 中选择最高频、输入契约稳定且代表主要重计算的入口，并记录依据 |
| 是否需要接回原始工程 | 用户确认 |
| 是否已有可复用 ONNX | 用户确认 |
| 测试输入 / 本轮暂不支持记录 | 测试输入优先用户提供；未提供则从仓库样例或最小构造输入补齐；不在本轮范围的入口、功能分支或服务接口必须写入交付记录 |

锁定后决定起点：有源码 → 从 Workflow 1 开始；只有现成 ONNX → 完成 Workflow 2 后进入 Workflow 3。

## Workflow 1：源码分析与候选规划

**目标**：锁定真实推理入口、本次初步适配场景、候选全集、首批候选、后续批次计划和图侧策略。

调用 `ascend-om-pipeline-adapter`，消费其标准输出：
- `Model Shape Of Work`
- `Functional Coverage Contract`
- `Candidate Stage Inventory`
- `Source-Side Subgraph Contract`
- `Output Preservation Contract`
- `Pipeline Boundary Analysis`

本阶段按 **大图优先** 选择首批候选，优先围绕高频交付场景形成最大连续主链大图；按 **图侧主策略唯一** 确定每个候选的初始策略，按 **输出保真** 记录替换点输出全集。只做规划，不执行导出，不把首批候选视为候选闭环。

**输出**：`First-Batch OM Decision`——包含候选全集、首批选择及理由、未选候选的暂缓原因与下一步 OM 尝试条件、每个候选的图侧策略和动态控制参数处理方案、是否覆盖主链主要计算阶段、是否覆盖本次初步适配场景，以及本轮暂不支持功能的记录位置。

## Workflow 2：环境准备

**目标**：确认 CANN、Python、ONNX 工具链、ATC、推理工具可用。

调用 `ascend-onnx-atc-pipeline` 的环境检查和修复流程。

**门控**：必须拿到 `Environment Readiness Summary` 且 "是否可继续导图与 ATC" 为 "是"，否则不得进入 Workflow 3。

## Workflow 3：ONNX 获取与 OM 转换

**目标**：获取 ONNX、优化、ATC 转换产出 `.om`。

调用 `ascend-onnx-atc-pipeline`。

**进入前提**：
- 有源码场景：必须已有 Workflow 1 的 `Source-Side Subgraph Contract`、`Output Preservation Contract`、候选图侧策略和动态轴依据
- 仅 ONNX 场景：`ONNX IO Contract Report` 暂代源码侧契约；一旦进入接回或 fallback，必须回 Workflow 1 补齐

导出时优先复用项目已有导出入口。临时 wrapper 只能做输入整理、输出暴露和契约对齐，不得改业务语义。兼容输入分支若共享替换点、主要重计算或输出消费者，应合并为同一候选大图；若样例 shape 或样例控制值与 Workflow 1 的源码侧契约冲突，回 Workflow 1 补证或进入 Workflow 4 重评估边界。

**输出**：`ONNX IO Contract Report`、`ONNX Optimization Summary`、`ATC Attempt Summary`。未产出可用 `.om` 则按失败回跳处理。

## Workflow 4：边界重评估（按需触发）

**触发条件**：仅在导出/ATC/验证/接回暴露边界问题时进入，不在正常顺序流中执行。

**目标**：当导出/ATC/验证/接回暴露边界问题时，决定局部修复、合并前后图、拆小还是保留原始实现。

**触发场景**：大图导出或 ATC 失败集中在局部结构、切分导致过多碎图、状态语义或输出契约接回困难、子图验证边界不合理、端到端性能瓶颈来自切分边界。

调用 `ascend-om-pipeline-adapter`，消费 `Boundary Revision Record`。

**输出**：`Boundary Revision Record`——原候选、触发原因、保留/放弃/新增候选、修订结果和影响。

## Workflow 5：子图级验证与批次门控

**目标**：确认当前批次 `.om` 可稳定运行、ONNX vs OM 输出对齐。

调用 `ascend-onnx-atc-pipeline` 完成 OM 推理和精度对比，消费 `Minimal Validation Summary`。

按 [验证门控标准](references/validation-gates.md) 的精度门控表判定每个候选的通过/复查/不通过状态。先区分输入输出数量/shape 等契约错误和纯数值偏差：契约错误必须修复或重规划；契约一致但子图级数值不通过时不得直接终止，应完成偏差诊断，必要时作为“端到端复查候选”进入 Workflow 6/7，用真实业务结果决定是否可接受。按 **输出保真** 和 **候选状态机闭环** 判断每个候选进入接回、回 Workflow 3 还是回 Workflow 4。

子图验证后的流程判定：

- 契约不一致：不得接回，回 Workflow 3 修复 ONNX / ATC / 推理 mode / 输出 shape；若问题来自切分边界或输出保真，回 Workflow 4
- 契约一致且数值通过：进入 Workflow 6 接回，候选可继续向 `validated / integrated` 推进
- 契约一致但数值不通过、偏差有合理解释且不影响高置信度主输出或下游消费语义：标记为端到端复查候选，进入 Workflow 6/7；候选状态不得直接视为 `validated`
- 端到端复查可接受：候选进入 `validated / integrated`
- 端到端复查不可接受：回 Workflow 3 做精度修复，或回 Workflow 4 重规划边界；修复/重规划仍不可接受时，形成 fallback 或停止并报告用户
- 仅有现成 ONNX 且没有原工程或业务端到端入口时，不得自动套用端到端复查候选结论；必须要求用户提供业务验证入口/样本，或把偏差作为风险交付并等待用户确认

**输出**：
- `Validation Gate Decision`——哪些 `.om` 可直接接回、哪些作为端到端复查候选接回、哪些需修复或重规划
- `Next-Batch Decision`——当前批次状态、剩余候选、是否继续下一批

`Next-Batch Decision` 只登记后续候选和推进条件，不是暂停点。当前批次通过后仍需进入 Workflow 6/7。Workflow 7 后若本次场景内仍有未闭环主链候选，继续回到 Workflow 3/4 推进下一批；场景外候选写入未支持功能 / 后续候选记录，不阻塞本轮交付。

## Workflow 6：接回原始推理流程

**目标**：把 `.om` 接回原始工程，形成可运行主路径和 fallback 路径。

**用户确认门槛**：本步骤会修改用户工程代码。执行前必须向用户说明：将修改哪些文件、插入什么桥接代码、主路径与 fallback 策略。用户明确确认后方可动手。

调用 `ascend-om-pipeline-adapter`。

按 **接回 context 隔离** 实现混合 runtime 的 runner 模式。替换重计算部分，保留原始前处理、后处理、状态管理和调度逻辑。

必须确认：替换点已明确、桥接代码位置已明确、主路径与 fallback 已明确、混合 runtime context 恢复已实现并最小验证、各入口状态已更新。

**输出**：`OM Pipeline Integration Summary`、更新后的 `Functional Coverage Contract`、更新后的 `Output Preservation Contract`。

## Workflow 7：端到端验证与交付

**目标**：端到端验证并形成交付说明。

验证要求：
- 优先复用官方端到端入口（demo、benchmark、service API）
- 本次场景内主流程必须经过主路径验证、fallback 验证或用户确认排除
- 本轮暂不支持的官方入口、功能分支或服务接口必须写入交付记录，说明未支持原因、风险和后续触发条件
- 最终精度结论以端到端验证为准，子图级 ONNX vs OM 只作为定位材料
- 性能结论必须基于实测；若只有子图时延，只能写成“子图级性能参考”

**最终交付**：

| 类别 | 内容 |
| --- | --- |
| 可复现路径 | 执行命令和环境要求 |
| 产物 | `.om` 文件、日志、接回后的可运行推理流程 |
| 路径说明 | 主路径与 fallback、每步是 `.om` / 原始实现 / `torch_npu` |
| 混合 runtime | context 恢复方式和最小验证结论 |
| 功能状态 | 本次场景内每个入口的验证状态 |
| 未支持功能 | 后续候选、未支持原因、风险和触发条件 |
| 性能 | 端到端验证结果、当前瓶颈与推荐调优参数 |
| 风险 | 未完成项和已知风险 |

## 失败回跳规则

| 失败类型 | 回跳目标 |
| --- | --- |
| 环境不可用 | 留在 Workflow 2 |
| 导出、优化、ATC、动态策略、契约不一致或需精度修复的图内验证 | 回 Workflow 3 |
| 契约一致但子图数值不通过且可解释 | 作为端到端复查候选进入 Workflow 6/7 |
| 部署边界、状态语义、输出保真、切分过碎或应合并前后图 | 回 Workflow 4 |
| 桥接代码、输入整理、输出回填、dtype/shape/device 对齐 | 留在 Workflow 6 |
| 端到端验证暴露边界或性能瓶颈 | 按原因回 Workflow 3/4/6 |

## 停止条件

同时满足以下四条才可结束：

1. **功能闭环**：本次场景内主流程已主路径验证、fallback 验证或用户确认排除；暂不支持功能已写入交付记录
2. **候选闭环**：场景内功能分支和主链重计算候选均已达 `integrated / failed_replan / fallback / excluded`，fallback 有尝试失败、重规划不成立或非可图化证据；场景外候选已记录为后续候选或暂不支持
3. **契约闭环**：输出满足 **输出保真**；主路径/fallback/混合 runtime 已说明并验证，混合 runtime 的 context 恢复方式已完成最小验证
4. **交付闭环**：部署报告、端到端结论、性能参考已落盘；本次场景内无需继续拆分、合并或重规划的 OM

最终回复用户时：阐释清楚采用的 `.om`、主路径和 fallback、端到端验证结果，以及剩余候选或已知风险。
