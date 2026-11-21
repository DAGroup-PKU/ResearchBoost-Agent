# Ascend OM Pipeline Adapter

本 Skill 是 `ascend-om-deployer` 的子 Skill，真正负责源码侧分析与 OM 接回。

## 职责与对接

- **本 Skill 负责**：读取源码、产出源码侧契约（子图输入、输出保真、部署边界）、候选全集与首批候选推荐、接回原始流程时的替换点与桥接代码。
- **本 Skill 不负责**：最终 `shape / dtype / layout / dynamic strategy` 和 ATC 参数由 `ascend-onnx-atc-pipeline` 复核并落地；本 Skill 只给出图侧主输入形态策略初判。
- **信息缺失时**：图侧契约不清晰交回主 Skill 由 `ascend-onnx-atc-pipeline` 复核；源码侧契约不清晰本 Skill 必须继续补证。
- **候选规划原则**：候选全集、首批候选、后续批次条件和边界重规划由本 Skill 负责；不得用 `fallback` 替代主链重计算候选的 OM 尝试规划。

主 Skill 调用时应提供：当前阶段、模型源码路径、用户关心的推理入口或交付入口、已知状态对象、候选模块、候选 OM 范围，以及已生成 `.om` 的位置（若已进入接回）。信息缺失时本 Skill 从源码、README、官方 demo、benchmark、service API 或调用链中补齐。

标准输出契约（供主 Skill 和下游 Skill 消费）：

- `Model Shape Of Work`、`Functional Coverage Contract`、`Candidate Stage Inventory`、`Source-Side Subgraph Contract`、`Output Preservation Contract`、`Pipeline Boundary Analysis` — 首轮分析阶段产出
- `Boundary Revision Record` — 边界重评估阶段补充
- `OM Pipeline Integration Summary` — 接回阶段补充，并更新 `Functional Coverage Contract` 与 `Output Preservation Contract`

## Gotchas

- 不要跳过候选全集直接选首批——没有全集就无法说明暂缓原因和后续批次条件。
- 不要把主链重计算候选直接归为 `fallback`；必须先尝试 OM 或给出重规划证据。
- 不要用零张量、常量或仅形状占位替代会参与后续计算的真实输出。
- 不要因为导出困难就把大图拆成大量轻量碎图；先判断是否能局部修复或合并。
- 接回时务必恢复 runtime context——`torch_npu` 和 OM session 混合运行时遗漏 `acl.rt.set_context` 会导致后续算子静默出错。
- 非 tensor 控制值（构造参数、配置值）不要直接当作 OM 的图输入；必须先捕获语义和来源，判断能否固定。
- 源码侧契约不清晰时不要交回主 Skill——这是本 Skill 的职责，必须继续补证。

## Workflow 1: 功能范围与真实入口

> **前置条件**：主 Skill 已提供模型源码路径和推理入口信息。
> **完成条件**：产出 `Model Shape Of Work` 和 `Functional Coverage Contract`；所有已知入口已标注本轮/后续/暂不支持。

先确定本次初步适配场景。若用户未指定，优先选择官方 README、demo、benchmark 或 service API 中最高频、最能代表交付链路的入口作为本轮交付场景，同时枚举其他入口并记录是否本轮支持、后续候选或暂不支持。

必须回答：

- 用户真正调用的入口是什么；若未指定，哪个官方入口最适合作为首批交付场景。
- 该场景能固定哪些业务维度，例如 batch、输入尺寸、token / prompt / 候选数量、top-K、状态档位或帧步长。
- 源码 / README / 官方 demo 暴露了哪些基本推理入口，应归入哪些功能分支，本轮哪些属于初步适配场景，哪些记录为暂不支持或后续候选。
- 是否存在 `start / update / finalize` 或同类阶段式接口。
- 是否存在跨帧、跨 chunk、跨页、跨 token 的状态。
- 哪些代码属于控制面、状态管理、前处理、主链重计算、后处理和结果聚合。

本轮初步适配以已限定场景功能正常为目标；不在本轮场景内的入口不要求全部适配，但必须写入输出记录，说明未支持原因、后续触发条件和风险。如果某个入口仍声明属于本轮场景，则必须主路径验证、fallback 验证或用户确认排除。

## Workflow 2: 候选全集与首批候选

> **前置条件**：Workflow 1 已完成，`Functional Coverage Contract` 已产出。
> **完成条件**：产出 `Candidate Stage Inventory`；所有候选已标注状态、暂缓原因、所属功能分支、shape / 控制值变化来源、下一步 OM 尝试条件和图侧主策略。

推荐首批 OM 范围前，必须先产出 `Candidate Stage Inventory`。候选全集既服务首批决策，也服务后续批次推进；不能在首批结束后丢失，也不能只输出当前准备导出的首批列表。

术语约定：

- `候选重计算子图`：分析阶段识别出的计算热点单元，用于回答”哪些局部结构值得优先 OM 化”。
- `候选大图`：部署阶段准备尝试导出 OM 的合并候选，通常由一个或多个候选重计算子图、可共存输入分支和必要连接层组成。

候选枚举规则和动态策略选择的详细说明见 [references/CANDIDATE_RULES.md](references/CANDIDATE_RULES.md)。核心原则：

- 必须先枚举候选全集，再推荐首批；不得跳过枚举。
- 主链重计算候选不得未尝试就归为 `fallback`。
- 首批优先形成最大连续主链大图；相似候选优先合并。
- 每个候选必须给出图侧主策略初判（`static / dymbatch / dymhw / dymdims / dymshape`，详见 reference）。

## Workflow 3: 源码侧子图输入契约

> **前置条件**：Workflow 2 已完成，候选已进入本批或后续批次。
> **完成条件**：每个本批候选产出 `Source-Side Subgraph Contract`；输入来源、可选关系和非 tensor 控制值已记录。

对每个进入本批或后续批次的候选，必须形成 `Source-Side Subgraph Contract`。该契约回答的是“原始工程里这个替换点真实需要什么输入”，不是 ONNX 图导出后的输入清单。

必须回答：

- 候选子图的输入来自哪里，进入子图前已经做过哪些 resize、padding、bucket、embedding、mask 构造或语义整理。
- 哪些输入是必选，哪些输入是可选；可选输入之间是可共存、互斥，还是由控制面选择。
- 若多个输入形态进入同一替换点、共享主要重计算或共同影响同一下游输出，是否应合并为一个候选大图。
- 输入 shape、dtype、layout 的源码证据在哪里，shape 变化来自 batch、H/W、点数、mask 数、帧数、序列长度、状态缓存还是其他业务维度。
- 是否存在非 tensor 控制值、构造参数或初始化参数影响图内 tensor 分区、slice、token 排除、RoPE、cache、状态布局或循环展开；若存在，记录真实运行值、源码位置、来源类型和分区语义。
- 当前切分边界下输入是否稳定，是否适合继续导图，哪些问题需要图侧验证。

图导出后的 graph input / output、动态策略与 ATC 参数不在本 Skill 内最终定稿；但候选拆分阶段必须先完成足以支撑首批选择和暂缓理由的源码侧初判。

## Workflow 4: 输出保真契约

> **前置条件**：Workflow 3 已完成，子图输入契约已稳定。
> **完成条件**：每个本批候选产出 `Output Preservation Contract`；所有未导出/占位输出已证明不参与后续计算或已标记为阻塞项。

不得为了降低导出难度随意删减替换点输出。本次适配场景内会被下游消费的输出，必须由 OM 真实输出或保留原始实现提供；场景外输出若本轮不支持，也必须记录消费者、未支持原因和后续风险。

必须回答：

- 候选替换点的原始输出全集是什么。
- 每个原始输出被哪些下游路径消费，是否属于本次适配场景内的后续计算。
- 导出 / OM 输出清单与原始输出全集是否一致。
- 是否存在未导出输出、占位输出、伪造输出或合并输出。
- 占位只允许用于已证明不被目标功能范围消费的结构兼容字段；不得用零张量、随机张量、常量或仅形状占位替代会参与后续计算的真实输出。

## Workflow 5: 部署边界与重规划

> **前置条件**：Workflow 3-4 已完成，输入契约和输出保真契约已稳定。
> **完成条件**：产出 `Pipeline Boundary Analysis`；首批 OM 范围、保留原始实现部分和待图侧验证问题已明确。重规划时补充 `Boundary Revision Record`。

部署边界目标是完整推理流程可运行，不是为了导出而导出。优先找覆盖目标交付场景主链、重计算集中、输入契约清晰、能接回主链的大图；不要把简单连接、投影、轻量卷积切成大量独立小图。若相邻阶段之间主要是 tensor 直连、reshape、投影、轻量卷积或 dtype / layout 适配，应优先并入同一候选大图。

当控制面或状态管理影响正确性时，保留原始控制面与状态管理；其中调用的主链重计算候选仍要完成 OM 评估，不能把整个状态流程直接作为 fallback。

出现以下情况时，必须重新审视部署边界并产出 `Boundary Revision Record`：

- 大图 ONNX 导出失败，且问题集中在局部结构。
- 大图 ATC 失败，且失败位置显示当前切分边界不合理。
- 为了过导出或过 ATC 被迫拆出多个轻量碎图。
- 相邻小图之间主要只是简单连接、投影、`reshape` 或轻量卷积。
- 接回主链后发现输入契约、状态语义或推理入口不易对齐。
- 接回时发现原始输出被删减、占位或伪造，且这些输出可能被本次适配场景内任一入口消费。
- 只跑通单一路径 smoke，但本次适配场景内其他必要入口无法运行或未验证。
- 功能已跑通，但性能瓶颈主要来自图间搬运或过碎切分。

触发重规划不等于立刻拆小。必须先判断当前问题更适合局部修复、围绕固定交付场景合并前后图、重新切边界，还是保留原始实现。

## Workflow 6: OM 接回主链

> **前置条件**：`.om` 已由 `ascend-onnx-atc-pipeline` 生成成功。
> **完成条件**：产出 `OM Pipeline Integration Summary`；接回后最小验证通过或已明确失败回跳位置。

接回阶段不是只写结果摘要，必须按以下顺序完成，并最终产出 `OM Pipeline Integration Summary`。

1. 定位替换点
   - 明确原始推理流程中哪一段将被 `.om` 替换。
   - 明确替换前后的输入输出契约是否一致。
   - 明确替换点原始输出全集、OM 输出清单、未导出输出、占位输出及其下游消费者。
   - 若存在未导出或占位输出，必须证明它不参与本次适配场景内任何后续计算；否则接回阶段不算完成。
   - 明确该替换点前后的控制面、状态对象和后处理是否继续保留原始实现。
2. 确认运行位置
   - 明确每个 `.om` 的调用位置。
   - 明确剩余未 OM 化部分位于原始 Python、`torch_npu` 还是 CPU；CPU 仅作为最后兜底，不应成为主路径。
   - 明确主路径与 fallback 路径各自经过哪些阶段。
   - 明确本次适配场景内每个入口走主路径、fallback 路径、待 `ascend-om-deployer` 的 `Workflow 7` 验证还是用户确认排除；场景外入口记录为后续候选或本轮暂不支持。
   - 若同一进程内存在多个 `.om` session，或 `.om` 与 `torch_npu` / aclruntime / ais_bench 混合运行，接回实现必须采用可复用 runner 管理每次 OM 推理前后的 runtime context。
3. 接回实现
   - 优先替换原始代码中的模型推理部分，尽量不改原始前处理、后处理、状态管理和调度逻辑。
   - 若必须增加少量桥接代码，只允许做输入整理、输出回填、dtype / shape / device 对齐、runtime context 恢复和调度接线，不得顺手改业务语义。
   - 桥接代码不得伪造会参与后续计算的模型输出。
   - 多个 `.om` 或 `.om` 与 `torch_npu` 同进程混合运行时，桥接代码应收敛为 runner 模式；最小模式如下，按项目实际 `InferSession` 入参和输出格式调整即可：

   ```python
   class OMRunner:
       def __init__(self, model_path, device_id=0, mode="static"):
           origin_ctx, ret = acl.rt.get_context()
           self.session = InferSession(device_id=device_id, model_path=str(model_path))
           self.origin_ctx = origin_ctx if ret == 0 and origin_ctx else None
           if self.origin_ctx is not None:
               acl.rt.set_context(self.origin_ctx)
           self.mode = mode

       def __call__(self, inputs):
           self.session.set_context()
           try:
               return self.session.infer(inputs, mode=self.mode)
           finally:
               if self.origin_ctx is not None:
                   acl.rt.set_context(self.origin_ctx)
   ```

   - OM 与原流程之间用明确的 host / tensor 数据契约交接；OM 输出回到 `torch_npu` 流程前，先恢复原 runtime context，再构造目标 device tensor。
4. 接回后最小验证
   - 先确认接回后链路能跑通。
   - 再确认 `.om` 前后的张量语义没有接错。
   - 再确认输出保真契约未被破坏，且本次初步适配场景内入口已主路径验证、fallback 验证、用户确认排除或标记为待 `ascend-om-deployer` 的 `Workflow 7` 端到端验证。
   - 若当前 `.om` 是子图数值不通过但契约一致的 `e2e_review_candidate`，接回摘要必须记录子图偏差来源、风险输出、端到端复查入口和不可接受时的回跳位置。
   - 若接回后仍有本次适配场景内主链重计算候选未尝试，应记录下一批候选并交回主 Skill；是否立即推进由 `ascend-om-deployer` 在当前批次 `Workflow 7` 后决定。
5. 失败回跳
   - 若是桥接代码、输入整理、输出回填、dtype / shape 对齐或调度接线问题，优先留在当前接回阶段修正，并重新执行接回后最小验证。
   - 若是 `.om` 自身或图侧契约问题，回 `ascend-om-deployer` 的 `Workflow 3`。
   - 若是接回后暴露部署边界不合理，例如状态语义难对齐、图间搬运过重、切分过碎，则回 `ascend-om-deployer` 的 `Workflow 4`。

## 标准输出

各阶段输出模板见 [references/OUTPUT_TEMPLATES.md](references/OUTPUT_TEMPLATES.md)。这些模板是主 Skill 和 `ascend-onnx-atc-pipeline` 的契约输入，不是可选摘要；缺字段时应说明无法填写的原因和补证路径。

各阶段应产出的模板：

- **首轮分析阶段**：`Model Shape Of Work`、`Functional Coverage Contract`、`Candidate Stage Inventory`、`Source-Side Subgraph Contract`、`Output Preservation Contract`、`Pipeline Boundary Analysis`
- **边界重评估阶段**：追加 `Boundary Revision Record`
- **接回阶段**：追加 `OM Pipeline Integration Summary`，并更新 `Functional Coverage Contract` 与 `Output Preservation Contract`

生成主 Skill 的最终部署文档时，可参考：

- [references/DEPLOYMENT_TEMPLATE.md](references/DEPLOYMENT_TEMPLATE.md)
- [references/SOURCE_ANALYSIS.md](references/SOURCE_ANALYSIS.md)
