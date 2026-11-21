# ONNX -> OM 常见问题

本文件只记录当前 Skill 需要稳定记住的问题类型和处理方向，不保留旧项目流水账。

命令示例中的 `$SKILL_DIR` 表示本 Skill 安装目录，例如 `/path/to/ascend-onnx-atc-pipeline`。

## 1. ATC 失败后要不要直接放弃当前子图

不要。

至少先完成以下检查：

1. 输入形态是否选错
2. ONNX 输入输出契约是否被优化步骤改写
3. graph output 是否残留不合理动态 shape
4. 是否是局部实现不友好，而不是整张图都不能转
5. 当前部署边界是否过大或过碎

## 2. ONNX 已经能导出，为什么 ATC 还会失败

常见原因不是“图不存在”，而是“图不适合当前 ATC 输入条件”：

- 输入 shape 策略不对
- 中间 shape / dtype / layout 信息不完整
- 某些局部结构对导出友好，但对 ATC 不友好
- 优化后产物改变了输入契约
- 切图边界不合理

处理顺序：

1. 先看 `inspect_onnx_io_shapes.py`
2. 再看 `optimize_onnx.py` 生成的候选产物
3. 再按 `auto_optimizer -> onnxslim -> original` 逐个尝试
4. 最后才决定改边界或回退

## 3. 优化后的 ONNX 为什么还要重新检查输入输出

因为优化步骤可能改变输入形态或图签名。

典型风险：

- 可选输入被折叠
- 某些输出被清理或改名
- graph output 仍保留动态符号
- 输入已经固定，但输出 shape 标注没有同步固定

建议先执行：

```bash
python3 -u "$SKILL_DIR/scripts/inspect_onnx_io_shapes.py" \
  --model /path/to/model.onnx
```

## 4. 输入已固定或有动态 profile，但输出还是动态 shape，怎么办

先确认“真实运行输出是否固定”，不能只看 ONNX 文本。

如果需要判断真实输出是否固定，先做 probe，不在 Workflow 3 修改 ONNX：

```bash
python3 -u "$SKILL_DIR/scripts/inspect_onnx_io_shapes.py" \
  --model /path/to/model.onnx \
  --probe-outputs
```

如果输入是动态的，可以重复传入 `--probe-shape-profile` 覆盖目标动态轴：

```bash
python3 -u "$SKILL_DIR/scripts/inspect_onnx_io_shapes.py" \
  --model /path/to/model.onnx \
  --probe-outputs \
  --probe-shape-profile "input:1,3,1024,1024" \
  --probe-shape-profile "input:4,3,1024,1024"
```

Workflow 3 只形成结论，不物化 patched ONNX。脚本 probe 结果用于判断 graph output shape 是否可由 Workflow 4 的 `optimize_onnx.py` 在优化前补全。

输出 shape 是否可补全按证据判断，不按任务类型一刀切。固定输入 shape 只说明输入张量维度稳定，不自动证明输出 shape 与数据内容无关；若静态输入或代表性 profile 证明输出维度由模型结构、配置或 profile 决定且多次实跑稳定，可以补全为固定维，否则不修改。`dymdims` 应覆盖计划使用的档位；`dymshape` 若没有可验证的动态轴覆盖证据，只能记录观察结果，不能强行写死。

Workflow 3 如果已经执行 probe 并形成结论，Workflow 4 按结论执行：需要 patch 时用同一组 profile 调用 `optimize_onnx.py`，由优化脚本修改 `original.onnx` 副本；无需 patch 或 patch 不安全时调用 `optimize_onnx.py --skip-output-shape-probe`，避免优化脚本对静态输入的动态 graph output 自动 probe / patch。这一步仍只是补全 output shape 标注，不负责重新固定 input shape。

## 5. `onnxslim` 和 `auto_optimizer` 都必须成功吗

不是。

当前 Skill 的默认顺序是：

1. 先保留 `original.onnx`
2. 再生成 `onnxslim.onnx`
3. 再尝试 `auto_optimizer.onnx`
4. 最终按 `auto_optimizer -> onnxslim -> original` 做 ATC 回退

因此某一步失败，不代表整个流程失败。

但“命令执行成功”不等于“候选可用”。候选至少还应满足：

- ONNX checker 通过
- graph output 没有悬空
- ONNX Runtime 加载校验通过

如果候选连基础结构都不满足，应直接剔除，不要继续拿去 ATC。

## 5.1 `auto_optimizer` 为什么没有进入 ATC

当前规则以 `optimization_summary.json` 中的候选可用性为准：

1. `onnxslim` 和 `auto_optimizer` 都需要通过结构检查、graph output 悬空检查和 ONNX Runtime 加载校验
2. `auto_optimizer` 若被标记为 `success: true`，统一 ATC 脚本会按默认优先级先尝试它
3. `auto_optimizer` 若被标记为 `success: false`，统一 ATC 脚本会跳过它并回退到 `onnxslim` 或 `original`

也就是说：

- `auto_optimizer` 不能因为“结构合法但 ORT 未通过”而直接进入 ATC
- 只有复检通过的候选，才允许进入 ATC
- `auto_optimizer` 如果返回 `no knowledges matched` 这类 no-op 结果，只记录为已尝试且无可用优化候选，不复制 `onnxslim` 冒充 `auto_optimizer`；统一 ATC 脚本会回退到 `onnxslim` 或 `original`
- 真正生效的最终选择以 `atc_selection.json` 为准

如果某次实际执行中直接从 `onnxslim` 开始，优先检查：

1. 是否根本没走 `pipeline_onnx_to_om.py`
2. 是否 `optimization_summary.json` 已把 `auto_optimizer` 标成 `success: false`
3. 是否人为用 `--candidate onnxslim` 覆盖了默认顺序

## 5.2 `onnxslim` 产物如果 graph output 悬空怎么办

先看是不是优化前图里存在 `Identity` 别名关系。

例如某些多输出 encoder 或中间特征导出场景里，原图可能是：

- `vision_features -> Identity -> backbone_fpn_2`

而 `onnxslim` 后变成：

- 图里只剩 `backbone_fpn_2`
- 但 `graph.output` 还保留 `vision_features`

这种情况不是“动态 shape 没推出来”，而是 graph output 已经失效。应先修复输出别名或直接剔除该候选，不能继续带着坏图做 ATC。

## 6. 什么情况下应该重规划部署边界

出现以下任一情况就该重看边界：

- 大图导出失败，但问题集中在局部
- 大图 ATC 失败，且失败点说明切图位置不合理
- 为了过导出被迫拆出多个轻量碎图
- 相邻小图主要只是简单连接、投影、`reshape` 或轻量卷积
- 接回主链后发现输入契约和状态语义难以对齐
- 功能已跑通，但性能瓶颈来自图间搬运

## 7. 主路径应该怎么安排

默认原则：

- 优先 OM
- 再考虑 `torch_npu`
- CPU 只做最后兜底

但不要为了“全 OM”把链路切得过碎。主路径必须首先满足完整流程可运行、输入契约可对齐、状态语义不被破坏。
