# OM 推理与验证参考

本文件用于说明 `.om` 产物生成后，如何做基础推理验证与精度比对。

命令示例中的 `$SKILL_DIR` 表示本 Skill 安装目录，例如 `/path/to/ascend-onnx-atc-pipeline`。

## 1. Python 侧依赖

统一先跑环境总检查：

```bash
bash "$SKILL_DIR/scripts/check_env_enhanced.sh"
```

默认直接复用当前环境中已安装的 `ais_bench` / `aclruntime`。

若本机尚未安装，可先检查：

```bash
python3 -c "import aclruntime, ais_bench"
```

如果缺失，先直接跑统一修复脚本；它会按当前 Python 版本和 CPU 架构自动下载官方 wheel：

```bash
ASCEND_ONNX_ATC_PIPELINE_PYTHON=python3 \
bash "$SKILL_DIR/scripts/repair_python_env.sh"
```

如果你已经准备好了 wheel，也可以显式指定：

```bash
ASCEND_ONNX_ATC_PIPELINE_OM_WHEEL_DIR=/path/to/wheels \
ASCEND_ONNX_ATC_PIPELINE_PYTHON=python3 \
bash "$SKILL_DIR/scripts/repair_python_env.sh"
```

## 2. OM 产物最小验证

在接回完整业务前，先做子图级验证：

- `.om` 是否能加载
- 输入 shape / dtype 是否与导出约定一致
- 输出数量、名称、shape 是否符合预期

若已有输入 `.npy`，可先用 `ais_bench` 做最小验证。

## 3. Python 推理入口

脚本入口：

```bash
python3 -u "$SKILL_DIR/scripts/infer_om.py" --help
```

适合做：

- 单次加载 `.om`
- 读取输入 `.npy`
- 执行推理
- 保存输出 `.npy`

如果当前 `.om` 是动态模型，显式传入对应推理模式。推理 mode 必须与 ATC 导出的动态能力一致：

| ATC 导出侧参数 | ais_bench 推理 mode |
| --- | --- |
| `--dynamic_batch_size` | `dymbatch` |
| `--dynamic_image_size` | `dymhw` |
| `--dynamic_dims` | `dymdims` |
| 仅使用含 `-1` 的 `--input_shape`，且未指定 `dynamic_*` | `dymshape` |

例如 `dymshape`：

```bash
python3 -u "$SKILL_DIR/scripts/infer_om.py" \
  --model /path/to/model.om \
  --input /path/to/input.npy \
  --mode dymshape
```

例如 `dymhw`：

```bash
python3 -u "$SKILL_DIR/scripts/infer_om.py" \
  --model /path/to/model.om \
  --input /path/to/input.npy \
  --mode dymhw
```

例如 `dymdims`：

```bash
python3 -u "$SKILL_DIR/scripts/infer_om.py" \
  --model /path/to/model.om \
  --input /path/to/input.npy \
  --mode dymdims
```

例如 `dymbatch`：

```bash
python3 -u "$SKILL_DIR/scripts/infer_om.py" \
  --model /path/to/model.om \
  --input /path/to/input.npy \
  --mode dymbatch
```

如果 `dymshape` 推理时报输出内存不足、输出 buffer 不够或类似 custom size 问题，先确认输入契约正确，再调大：

```bash
--custom-sizes 200000
```

最小功能验证通常不需要 warmup；需要性能统计时再显式设置 `--warmup` 和 `--loop`。

## 3.1 子图级性能统计

`infer_om.py` 除了最小推理验证，也可用于子图级 OM 性能统计。

例如：

```bash
python3 -u "$SKILL_DIR/scripts/infer_om.py" \
  --model /path/to/model.om \
  --input /path/to/input.npy \
  --mode static \
  --warmup 5 \
  --loop 50
```

或动态模型：

```bash
python3 -u "$SKILL_DIR/scripts/infer_om.py" \
  --model /path/to/model.om \
  --input /path/to/input.npy \
  --mode dymshape \
  --warmup 5 \
  --loop 50
```

脚本会输出：

- 推理次数
- latency 的 min / max / mean / median / p99
- 基于平均时延估算的 throughput

约束：

- 这里得到的是“单个 `.om` 子图”的性能参考，不是整条业务链的端到端性能
- 若需要端到端性能，优先复用上游仓库的官方 benchmark、官方 demo 或真实业务入口计时
- 若最终报告同时包含子图级和端到端结果，必须显式区分两者，不能混写成同一组结论

## 4. 精度比对

先使用同一组真实输入，分别跑 ONNX Runtime 与 OM，再比较输出。

脚本入口：

```bash
python3 -u "$SKILL_DIR/scripts/compare_precision.py" --help
```

动态 `.om` 同样要显式指定模式，例如：

```bash
python3 -u "$SKILL_DIR/scripts/compare_precision.py" \
  --onnx /path/to/model.onnx \
  --om /path/to/model.om \
  --input /path/to/input.npy \
  --mode dymshape \
  --output /path/to/comparison.json \
  --save-diff /path/to/diff_arrays
```

如果使用 `dymhw`、`dymdims` 或 `dymbatch` 导出的 `.om`，这里也应改成对应 mode。

原则：

- 优先用真实业务输入，不要长期依赖随机输入
- 若是多输入模型，要保证每个输入都来自同一条真实样本
- 若图有动态分支，优先覆盖真实会走到的输入场景
- 比较结论先看输入输出数量、shape、dtype、layout 和推理 mode/profile 是否一致；这些不一致时是契约错误，不是可忽略精度差异
- 这一步只给出子图级数值参考，不直接等价于最终业务精度结论
- 检测、排序敏感、多候选输出或低置信度结果较多的任务，容易出现结果顺序扰动；这类任务不能只凭子图级 `OM vs ONNX` 对比下最终结论
- 最终是否可接受，应回到官方样例、官方 demo 或官方端到端流程里比较业务结果，再结合差异分析交由用户决定

`compare_precision.py` 在数值超过阈值时可能返回非 0。该退出码表示子图数值参考未通过，不等同于部署流程必须停止；应先检查日志中的 `ONNX outputs`、`OM outputs`、每个输出 shape 和差异指标。只有输出数量/shape 无法对齐、推理失败或输入契约不一致时才按硬失败处理；契约一致时继续按下节做偏差诊断。

## 4.1 子图数值不通过后的处理

如果 ONNX 与 OM 输出数量和 shape 一致，但 `max_abs_diff`、`cosine_similarity` 或 `allclose` 不达标，不要立即把候选判死。先完成以下判断：

1. 复核比较条件：输入文件、预处理、输入顺序、dtype/layout、ONNX 候选、OM 推理 mode、动态 profile、输出顺序都必须一致。
2. 分析任务语义：低置信度候选、近似相等分数的排序、NMS/top-K/argmax tie-break、padding/masked 区域、未被本轮功能消费的辅助输出、FP16 / 混合精度舍入、并行归约顺序，都可能造成子图数值差异但端到端结果等价。
3. 若偏差只影响低置信度、未消费、被 mask/padding 覆盖、后处理会丢弃或排序等价的结果，可把候选交回主 Skill 标记为 `e2e_review_candidate`，先接回主流程做端到端验证。
4. 若偏差影响高置信度主结果、类别/文本/结构化字段、状态缓存或下游会消费的特征，标记为 `needs_precision_repair`，先修复再接回。

常见修复手段包括：改用 `original.onnx` 或另一个优化候选，关闭可疑优化，复核 ATC `precision_mode` / 混合精度策略，对敏感算子保留更高精度，修正输入 dtype/layout/preprocess，修正输出 mapping，调整动态 profile/mode，或回到部署边界重新切分。

## 5. 何时可以认为一个 OM 基本可用

至少满足以下条件：

- `atc` 成功产出 `.om`
- `.om` 能被正确加载
- 输入输出契约与上游代码一致
- 与 ONNX 对比通过，或已证明子图数值偏差只属于端到端复查候选且端到端结果可接受
- 接回主链后在真实流程中能跑通

注意：

- “子图级比对通过”只说明该 `.om` 候选具备继续接回主链的基础可信度
- 最终业务侧精度、排序稳定性和阈值敏感性，必须看端到端结果
- 子图级性能统计结果只用于描述该 `.om` 候选在给定输入、推理模式和测试参数下的性能参考，不直接等价于整条业务链性能结论

## 6. 常见偏差来源

- 输入预处理与原始代码不一致
- 输入 dtype 或 layout 不一致
- 不同候选 ONNX 的输入输出签名不一致
- graph output shape 标注有误
- 优化候选本身无效，例如 graph output 悬空或 ORT 无法加载
- 接回主链时张量顺序或含义接错
- OM 推理 mode 或动态 profile 与 ATC 导出策略不一致
- FP16 / 混合精度、并行归约或算子融合导致低位误差
- NMS、top-K、argmax、排序或分数接近阈值导致输出顺序扰动
- padding、mask、cache 尾部或未消费辅助输出存在差异
- 量化、clip、round、resize、interpolate 等数值敏感算子策略不同
