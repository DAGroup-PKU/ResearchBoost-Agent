# Ascend ONNX ATC Pipeline

这个子 Skill 负责环境准入、ONNX 图侧契约复核、ONNX 优化、ATC 转换和 OM 子图级验证，尽量把高重复、低自由度步骤收敛到脚本。

## 职责边界

- 本 Skill 消费主 Skill 和 `ascend-om-pipeline-adapter` 已锁定的候选范围、源码侧输入契约、输出保真要求和图侧主输入形态策略。
- 本 Skill 不负责自动理解复杂工程源码语义，也不负责决定“该导出哪个子图”；复杂工程的切图、wrapper 语义对齐和接回位置由主 Skill 与 `ascend-om-pipeline-adapter` 负责。
- 本 Skill 负责执行既定 ONNX 导出入口或消费现成 ONNX，检查图侧输入输出契约，执行 `optimize_onnx.py`，按统一顺序 ATC，并给出最小验证结论。
- 本 Skill 的脚本只作为本 Skill 的实现资源使用，不当作跨 Skill 公共脚本库直接复述。

## Gotchas

- 不要手工维护多份 ATC 命令，统一走 `pipeline_onnx_to_om.py`
- 不要直接把新导出的 ONNX、手工 slim 产物或临时修补产物送入 ATC；必须先过 `optimize_onnx.py` 产出 `optimization_summary.json`
- 不要在导出入口未明确时套用 `export_onnx.py`；复杂工程的 wrapper 由主 Skill 负责；默认 ONNX 导出路径使用 legacy `torch.onnx.export`，支持 `dynamo` 参数时显式设置 `dynamo=False`
- 不要为了规避动态参数而生成多个静态 OM；多静态 OM 只能是动态策略失败后的回退
- `static / dymbatch / dymhw / dymdims / dymshape` 是互斥的主输入形态策略，单次 ATC 只能选一种
- 环境检查通过后不要重复执行；只有环境被修改时才回到 Workflow 1
- 定向排障的 `--candidate` 失败不能直接作为 fallback 结论，必须交回主流程判断
- 不要把 ONNX vs OM 数值不通过直接当作流程终点；先确认输入输出数量和 shape 契约，契约一致时继续做偏差诊断和端到端复查判断

## 进度跟踪

复制此清单并跟踪当前阶段：

```text
Pipeline 进度：
- [ ] Workflow 1：环境准备
- [ ] Workflow 2：ONNX 获取 / 导出协作
- [ ] Workflow 3：ONNX 图侧输入输出契约
- [ ] Workflow 4：统一 ONNX 优化流程
- [ ] Workflow 5：统一 ATC 尝试顺序
- [ ] Workflow 6：失败处理与动态策略切换（按需）
```

## 与其他 Skill 的对接

主 Skill 调用本 Skill 前，源码 / 接回场景至少应锁定以下输入契约：

- 当前阶段：环境准备、ONNX 获取 / 导出协作、ONNX -> OM、子图级验证。
- 候选 OM 范围或候选子图名称。
- ONNX 导出入口，必要时补充可直接复用的 ONNX 路径。
- `Source-Side Subgraph Contract`：input names、输入来源与含义、shape / dtype / layout 源码证据、可选输入及其共存 / 互斥关系、影响图内数学的构造或控制参数。
- `Output Preservation Contract` 摘要：替换点原始输出全集、目标范围内被消费的输出、允许保留在原始实现中的输出。
- `Model Shape Of Work` 和候选的图侧主输入形态策略初判。
- 目标 SoC。

只有当前任务是“用户已提供现成 ONNX，且不要求接回原工程”时，才允许以 ONNX graph inputs / outputs 和 `ONNX IO Contract Report` 作为临时图侧契约继续推进；一旦进入接回、端到端主路径或 fallback 说明，仍需补齐源码侧契约和输出保真契约。

若源码 / 接回场景的信息还不完整，应先回到 `ascend-om-deployer` 的 `Workflow 1` 或 `Workflow 4` 补全，而不是直接猜测导图参数。

本 Skill 返回的标准输出是后续阶段的契约输入：

- `Environment Readiness Summary`：供主 Skill 判断是否可继续导图、优化、ATC 和验证。
- `ONNX IO Contract Report`：供主 Skill 判断图侧输入输出是否符合源码侧契约。
- `ONNX Optimization Summary`：供主 Skill 判断 `original / onnxslim / auto_optimizer` 哪些候选可进入 ATC。
- `ATC Attempt Summary`：供主 Skill 判断候选是否已产出 OM、需要修复还是需要重规划。
- `Minimal Validation Summary`：供主 Skill 判断候选是否可进入接回阶段。

独立性规则：

- 本 Skill 独立拥有 `scripts/` 和 `references/`。
- 其他 Skill 通过调用本 Skill 复用能力。

## Workflow 1: 环境准备

先检查环境，再决定是否修复：

```bash
export SKILL_DIR=/path/to/ascend-onnx-atc-pipeline
bash "$SKILL_DIR/scripts/check_env_enhanced.sh" || \
  source "$SKILL_DIR/scripts/setup_env.sh"
```

规则：

- 当前 Skill 优先维护和验证的环境组合是 `CANN >= 8.3.RC2 + Python 3.11 + ONNX==1.16.1`
- 更完整的 CANN / Python 对应关系见 `references/CANN_VERSIONS.md`
- 若机器上已有可用 CANN，则直接复用，不重复安装
- 若 CANN 缺失或版本过旧，默认优先准备 `8.3.RC2` 官方完整安装包集合，而不是只准备单个 `toolkit`；版本说明、分版本安装规则与安装入口见 `references/CANN_VERSIONS.md` 和 `scripts/install_cann.sh`
- 若 `ais_bench` / `aclruntime` 未预装，则 `scripts/repair_python_env.sh` 会按当前 Python 版本与 CPU 架构自动 `wget` 官方 wheel 并安装；若本机已装，则直接跳过
- 若权重未提前准备且用户或任务上下文未明确来源，先要求用户确认下载来源；若已给出本地文件、本地目录、URL 或 ModelScope 仓库/文件，则默认可用 `scripts/fetch_model_artifacts.py` 获取
- `check_env_enhanced.sh` 是进入后续 ONNX / ATC 流程前的唯一环境准入检查；若首轮检查失败，必须执行 `setup_env.sh`
- `setup_env.sh` 会在 `repair_python_env.sh` 结束后自动复检；复检失败则本阶段失败，不得继续导图、优化、ATC 或精度验证
- 一旦本阶段已经得到“是否可继续导图与 ATC: 是”，后续 Workflow 不再重复执行同一套环境检查；只有环境再次被修改时，才重新回到本阶段

本阶段必须向主 Skill 返回：

```text
=== Environment Readiness Summary ===
CANN:
Python:
ONNX:
onnxruntime:
onnxslim:
auto_optimizer:
ais_bench:
aclruntime:
是否可继续导图与 ATC:
```

只有当 `是否可继续导图与 ATC: 是` 时，主 Skill 才能进入后续 ONNX 获取、优化、ATC 和最小验证阶段。

## Workflow 2: ONNX 获取 / 导出协作

本 Skill 只承接以下三类输入：

- 项目中已有、可直接执行的导出入口
- 主 Skill 依据 `Source-Side Subgraph Contract` 编写好的临时 wrapper
- 用户已提供的现成 ONNX

核心规则：

- 导出入口未明确时，停止在此，不猜测导出参数
- 本 Skill 在导出阶段的职责是执行既定导出入口、检查导出产物的图侧输入输出契约，并继续执行优化、ATC 和最小验证
- 协作约束、`export_onnx.py` 的适用边界和 dynamo exporter 规则见 `references/ONNX_EXPORT_RULES.md`；进入导出或消费 wrapper 前必须按该文件检查，不能把复杂性、动态性或多输入直接作为 fallback 理由。若导出 wrapper 或 ONNX 中出现影响分区、slice、token / 候选数量、cache 布局或循环展开的常量，必须能追溯到源码侧契约；只能追溯到单个样例观察值时，停止 ATC 并回到主流程补证或重规划。

## Workflow 3: ONNX 图侧输入输出契约

如果当前阶段是 ONNX 获取 / 导出协作，本 Skill 只负责消费主 Skill 或项目导出入口生成的 ONNX，并检查图侧契约；源码侧切图分析和临时 wrapper 的输入输出契约仍由主 Skill 依据 `ascend-om-pipeline-adapter` 的结果落地。

先基于源码侧子图输入契约，看图的输入、输出、opset、动态维，再决定后续策略：

```bash
python3 -u "$SKILL_DIR/scripts/inspect_onnx_io_shapes.py" \
  --model /path/to/model.onnx
```

本阶段必须形成 `Pre-Optimization Contract Decision`，不能只把脚本输出当作通过结论。检查项至少包括：输入数量 / 顺序 / 名称映射、输入来源与含义、shape / dtype / layout、可选输入的共存或互斥关系、影响图内数学的常量来源、输出数量 / 名称 / 顺序、被下游消费的输出是否仍在图里，以及与 `Output Preservation Contract` 是否一致。只有结论为 `contract_ok` 时才进入 Workflow 4；若为 `needs_export_repair / needs_boundary_replan / onnx_only_risk`，应先修复导出、回主流程重规划，或把现成 ONNX 场景的风险交给用户确认。

如果输入已经固定，但 graph output 仍残留动态符号，可以继续做 probe；Workflow 3 只记录 probe 结果和 output shape 处理决策，不修改 ONNX：

```bash
python3 -u "$SKILL_DIR/scripts/inspect_onnx_io_shapes.py" \
  --model /path/to/model.onnx \
  --probe-outputs
```

如果输入是动态的，但已有能覆盖动态轴变化的代表性 profile，可以用 ORT 实跑辅助判断哪些输出维稳定、哪些输出维随输入变化。Workflow 3 不使用 `--patch-output-model`，只形成 `patch_with_optimizer / skip_output_shape_probe / needs_export_repair / needs_boundary_replan` 等决策。

```bash
python3 -u "$SKILL_DIR/scripts/inspect_onnx_io_shapes.py" \
  --model /path/to/model.onnx \
  --probe-outputs \
  --probe-shape-profile "input:1,3,1024,1024" \
  --probe-shape-profile "input:4,3,1024,1024"
```

输出 shape 是否可补全按证据判断，不按任务类型一刀切。若静态输入或代表性 profile 证明输出维度由模型结构、配置或 profile 决定且多次实跑稳定，可以补全为固定维；若输出维度由输入内容决定或证据不足，则不修改。真实动态维必须继续保留为动态。`dymdims` 的 profile 应覆盖计划使用的档位；`dymshape` 若没有可验证的动态轴覆盖证据，只能记录 ORT 实跑观察结果，不能据此改写通用输出 shape。
这里做的是图侧契约复核，不替代 `ascend-om-pipeline-adapter` 产出的源码侧子图输入契约。

权重和模型文件获取规则：

- `local-dir` / `local-file` 默认直接复用原始绝对路径
- 只有显式指定 `--local-policy copy` 时，才复制到 `--target-dir`
- `local-file` 若同时指定 `--output-name`，会物化到 `--target-dir`

## Workflow 4: 统一 ONNX 优化流程

默认统一到 `onnxslim` 命名，不再混用 `onnxsim` 文案。每个准备进入 ATC 的 ONNX 都必须先执行本节脚本并产出 `optimization_summary.json`；除非用户明确要求跳过优化，否则不得直接把新导出的 ONNX、手工 slim 产物或临时修补产物送入 ATC。

本节只消费 Workflow 3 已放行的 ONNX。Workflow 3 不物化 patched ONNX；若需要补全 graph output shape 标注，统一由 `optimize_onnx.py` 在优化前修改其 `original.onnx` 副本，并在 `optimization_summary.json` 里记录 `source_model`、动态 output patch 证据和 patch 后 IO。

Workflow 4 不负责重新推导或固定输入 shape。输入 shape 是否静态、分档动态还是 `dymshape`，应已经由 Workflow 3 的图侧契约和主流程的源码侧契约决定；ATC 侧的 `--input-shape` / `dynamic_*` 参数在 Workflow 5 使用。若 Workflow 3 已执行 probe 并判定需要补全 output shape，Workflow 4 使用同一组 profile 调用 `optimize_onnx.py` 完成修改；若 Workflow 3 判定无需 probe、无需 patch 或 patch 不安全，Workflow 4 必须传 `--skip-output-shape-probe`，避免优化脚本对静态输入的动态 graph output 自动 probe / patch。

```bash
python3 -u "$SKILL_DIR/scripts/optimize_onnx.py" \
  --model /path/to/model.onnx \
  --output-dir /path/to/artifacts
```

若 Workflow 3 判定由优化流程补全 output shape，可把候选动态策略的代表性 profile 传给优化脚本。脚本会先验证 profile 证据是否足以 patch 输出 shape；不足时不改图，只把 probe 与跳过原因写入 `optimization_summary.json`：

```bash
python3 -u "$SKILL_DIR/scripts/optimize_onnx.py" \
  --model /path/to/model.onnx \
  --output-dir /path/to/artifacts \
  --probe-shape-profile "input:1,3,1024,1024" \
  --probe-shape-profile "input:4,3,1024,1024"
```

若 Workflow 3 判定不应做 output shape probe / patch，应显式跳过：

```bash
python3 -u "$SKILL_DIR/scripts/optimize_onnx.py" \
  --model /path/to/model.onnx \
  --output-dir /path/to/artifacts \
  --skip-output-shape-probe
```

该脚本会生成并记录：

- `original.onnx`
- `onnxslim.onnx`
- `auto_optimizer.onnx`
- `optimization_summary.json`，作为后续 ATC 输入候选选择依据

校验原则：

- `onnxslim.onnx` 需要通过结构检查和 ONNX Runtime 加载检查
- `auto_optimizer.onnx` 需要通过结构检查和 ONNX Runtime 加载检查；若优化脚本对已知图结构兼容问题做了局部、等价、可回溯的修补，应记录修补结果并以复检通过为准；如果 `auto_optimizer` 失败，必须记录失败日志、失败原因和可回退候选；只有复检通过的候选，才允许进入 ATC
- 如果 `auto_optimizer` 明确返回 `no knowledges matched` / no-op 类结果，应记录为已尝试但无可用优化候选，不复制或伪造 `auto_optimizer.onnx`；后续按默认回退顺序使用 `onnxslim` 或 `original`
- 若优化后图侧契约与源码侧契约不一致，必须停止 ATC 并回到契约复核或边界重规划；不能为了过 ATC 静默改 shape、删输入或固定动态轴
- 对 `original / onnxslim / auto_optimizer` 每个候选都必须记录优化前后契约 diff：输入签名是否变化、输出签名是否变化、动态维是否被固定或扩大、graph output 是否悬空、ONNX Runtime 加载是否通过、是否发生修复动作、是否允许进入 ATC、禁止进入 ATC 的原因
- `optimization_summary.json` 存在不等于 Workflow 4 成功；至少一个候选 `success: true` 才能进入 Workflow 5。若所有候选都不可用，留在 Workflow 4 做导出修复、优化修复，或回 Workflow 3/主流程重规划边界

若当前候选子图不是“已有 ONNX，直接优化”的场景，而是“需要从复杂工程中导图”，本 Skill 只负责消费导图入口与导图产物，不替主 Skill 自动完成源码侧切图分析。

## Workflow 5: 统一 ATC 尝试顺序

不要手工维护多份 ATC 命令，统一走：

```bash
python3 -u "$SKILL_DIR/scripts/pipeline_onnx_to_om.py" \
  --artifacts-dir /path/to/artifacts \
  --output-dir /path/to/om_out \
  --base-name model_name \
  --soc-version <soc_version> \
  --input-shape "images:1,3,1024,1024"
```

默认尝试顺序：

1. `auto_optimizer.onnx`
2. `onnxslim.onnx`
3. `original.onnx`

每次尝试都会单独落日志。若前一种失败，再回退后一种。

说明：

- `pipeline_onnx_to_om.py` 当前负责执行“已判定好的 ATC 策略”，不是自动替主流程推导动态策略
- 图侧主输入形态策略仍应先由源码侧输入契约和图侧 IO 契约共同确定
- 脚本当前会做的是：参数透传、主策略互斥校验、候选 ONNX 回退和日志归档
- 只要 `optimization_summary.json` 中的 `auto_optimizer` 候选仍标记为可用，主流程就应先让它进入 ATC；这里的“可用”应以结构检查和 ONNX Runtime 加载检查通过为准，不能只因结构合法就优先进入 ATC
- 当前实际采用的候选以 `atc_selection.json` 为准；若与预期不一致，先检查 `optimization_summary.json`、ATC 日志和 `--candidate` 覆盖情况；如果 `optimization_summary.json` 不存在，必须回到第 4 节，不得继续 ATC

如果只是做定向排障，可用 `--candidate` 临时覆盖默认顺序；这属于调试入口，不应替代默认主流程。定向失败不能直接作为 fallback 结论，应把日志交回主流程判断修复、切换动态策略、重新导出或重规划。

## Workflow 6: 失败处理与动态策略切换

优先按以下顺序处理：

1. 复核源码侧子图输入契约与图侧契约是否一致
2. 检查图的输入输出 shape、dtype、layout
3. 对 ONNX vs OM 数值不通过做精度诊断，先排除输入输出数量/shape/mode/profile 不一致，再分析是否属于任务可接受的低置信度、排序、mask/padding 或混合精度扰动
4. 复核优化产物是否改变了图语义
5. 重新尝试不同动态策略或 ATC 精度策略
6. 必要时做局部、等价、可回溯的基础算子替换，再重新导出
7. 仍不成立时，回到部署边界重规划；只有重规划后仍无合理 OM 边界，才标记 fallback

局部实现改写必须记录改写文件、改写理由、替换前后的输入输出契约是否变化，以及重新导出后的验证结果。

精度诊断要求：

- 输出数量或 shape 不一致时，这是契约错误，不是可接受精度差异；必须修正 ONNX 导出、ATC 动态参数、推理 mode、输出 shape patch 或边界切分后重试
- 输出契约一致但数值超过阈值时，返回偏差分类：`contract_mismatch / numeric_passed / e2e_review_candidate / needs_precision_repair / needs_replan`
- 对 `e2e_review_candidate`，本 Skill 只给出子图级数值参考和风险，不直接判定最终业务精度；主 Skill 应接回后用官方 demo、benchmark、service API 或代表性输入做端到端复查
- 对 `needs_precision_repair`，优先尝试切换 ONNX 候选、关闭可疑优化、复核 `precision_mode` / 混合精度策略、保留敏感算子更高精度、修正 dtype/layout/preprocess/output mapping 或调整动态 profile/mode

动态策略要求：

- 主策略 `static / dymbatch / dymhw / dymdims / dymshape` 互斥，单次 ATC 只能选一种；切换策略必须作为新的 ATC 尝试记录
- 优先考虑静态 shape；选择动态 OM 必须说明原因和验证覆盖。若候选阶段已经判定当前功能分支需要动态 OM，且源码侧契约与图侧 IO 契约一致，应直接执行相应动态 ATC 尝试，不得复用前一功能分支的静态 OM 停止结论
- 不要为了规避动态参数而生成多个静态 OM
- 完整的动态策略选择规则、推荐顺序和协调规则见 `references/ATC_DYNAMIC_STRATEGIES.md`

## 标准输出

本 Skill 返回给主 Skill 的结果至少应包含：

```text
=== ONNX IO Contract Report ===
候选子图:
源码侧输入契约摘要:
输入清单:
输入 shape / dtype / layout:
输出 shape:
推荐图侧主输入形态策略:
可选输入组合是否符合源码侧契约:
构造/控制参数常量是否符合源码侧契约:
Output shape probe 结论: patch_with_optimizer / skip_output_shape_probe / needs_export_repair / needs_boundary_replan
Pre-Optimization Contract Decision: contract_ok / needs_export_repair / needs_boundary_replan / onnx_only_risk
允许进入优化:
若不允许进入优化，原因:

=== ONNX Optimization Summary ===
original:
onnxslim:
auto_optimizer:
优化前后契约 diff:
是否跳过 output shape probe:
允许进入 ATC 的候选:
禁止进入 ATC 的候选及原因:
最终保留候选:

=== ATC Attempt Summary ===
尝试顺序:
最终成功候选:
日志路径:
.om 路径:
候选图侧结论: om_done / needs_repair / needs_replan

=== Minimal Validation Summary ===
验证方式:
输入来源:
OM 推理模式:
对比 ONNX 候选:
输出数量 / shape:
输出数量是否一致:
输出 shape 是否一致:
契约门控结论: contract_ok / contract_mismatch
atol / rtol:
max_abs_diff:
cosine_similarity:
数值门控结论: passed / review / failed
偏差初判:
是否允许端到端复查接回:
候选验证结论: validated / e2e_review_candidate / needs_precision_repair / needs_repair / needs_replan
结果:
说明:
```

主 Skill 在不同阶段消费本 Skill 输出时，应按以下方式使用：

- 环境准备阶段：消费 `Environment Readiness Summary`
- ONNX 获取 / 导出协作阶段：消费 `ONNX IO Contract Report`
- ONNX -> OM 阶段：消费 `ONNX IO Contract Report`、`ONNX Optimization Summary`、`ATC Attempt Summary`
- 子图级验证阶段：额外消费 `Minimal Validation Summary`

`Minimal Validation Summary` 中的 `结果` 和 `说明` 应遵循：

- 这里描述的是“子图级数值参考结论”，不是最终业务精度结论
- 若任务属于检测、排序敏感、多候选输出或低置信度结果易扰动场景，应明确提示最终结论必须回到端到端官方样例或官方流程中确认
- 子图级数值失败但输出契约一致时，应说明可疑偏差来源、是否建议端到端复查接回、若端到端不可接受应优先尝试的修复手段

## 参考资料

按需读取，不要一次性全加载：

- `references/ONNX_EXPORT_RULES.md`
  用于确认 ONNX 获取 / 导出协作的约束规则和 `export_onnx.py` 的适用边界。
- `references/CANN_VERSIONS.md`
  用于确认 CANN / Python / ONNX 的默认基线和安装入口。
- `references/PARAMETERS.md`
  用于确认 ATC 常用参数和动态策略选择。
- `references/ATC_DYNAMIC_STRATEGIES.md`
  用于判断固定 shape、`dymshape`、`dynamic_dims`、`dynamic_batch_size`、`dynamic_image_size` 该怎么选。
- `references/AIPP_CONFIG.md`
  仅在图像前处理准备下沉到 AIPP 时再读。
- `references/INFERENCE.md`
  用于 `.om` 生成后的最小推理验证和精度对比。
- `references/FAQ.md`
  用于 ATC、输入契约、优化产物和部署边界相关的常见问题排查。
