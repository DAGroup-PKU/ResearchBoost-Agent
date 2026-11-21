# ONNX 获取 / 导出协作规则

命令示例中的 `$SKILL_DIR` 表示本 Skill 安装目录，例如 `/path/to/ascend-onnx-atc-pipeline`。

## 协作约束

- `export_onnx.py` 只适用于输入边界清晰的简单模型导出，不能替代复杂工程的子图切分与 wrapper 构造
- 若当前任务属于多输入、带状态、流式、服务式或多阶段系统，应先由主 Skill 和 `ascend-om-pipeline-adapter` 锁定源码侧子图输入契约，再决定导出入口；若契约表明多个可选输入可共存且共享替换点，应先消费合并候选的导出产物
- 对已进入目标范围的主链重计算候选，复杂性或动态性不能直接解释为 fallback；只要源码侧契约足以构造入口，应至少推进一次合理的 ONNX 优化与 ATC 尝试，失败后再回传修复或重规划证据
- 对来自本次适配场景内未完成功能分支的候选，只要主 Skill 已给出导出入口和图侧主输入形态策略，本 Skill 应直接执行对应 ONNX 优化与 ATC；动态策略已明确时不得停下来等待用户再次确认。场景外后续候选只有在主 Skill 明确纳入下一批适配场景后才执行
- 若导出 wrapper 或 ONNX 中出现影响分区、slice、token/候选数量、cache 布局或循环展开的常量，必须能追溯到源码侧契约中的配置、入口契约或运行时值；若只能追溯到单个样例观察值，应停止 ATC 并回到 `ascend-om-deployer` 补证或重规划
- 若导出入口本身还未明确，本 Skill 应停止在此，不猜测导出参数，也不直接套 `export_onnx.py`

## `export_onnx.py` 的定位

- 适用于整模型 checkpoint、TorchScript 或边界清晰的简单模块导出
- 不承诺覆盖复杂工程中的候选子图导出
- 复杂工程的 wrapper 生成与语义对齐仍由主 Skill 负责
- 默认不绑定具体 PyTorch 版本；但默认 ONNX 导出路径应使用 legacy `torch.onnx.export`
- 如果当前 `torch.onnx.export` 支持 `dynamo` 参数，导出入口必须默认显式设置 `dynamo=False`，避免新 exporter 默认引入 `onnxscript` 依赖或改变图导出行为
- 只有在 legacy exporter 无法覆盖当前模型，且已明确记录新增依赖与图侧差异时，才允许显式开启 dynamo exporter
