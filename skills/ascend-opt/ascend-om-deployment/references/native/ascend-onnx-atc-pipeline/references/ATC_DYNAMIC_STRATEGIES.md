# ATC 动态策略选择

命令示例中的 `$SKILL_DIR` 表示本 Skill 安装目录，例如 `/path/to/ascend-onnx-atc-pipeline`。

## 1. 先判断，不要直接选参数

先回答三个问题：

1. 当前候选子图的真实输入契约是否已经稳定
2. 输入哪些维度固定，哪些维度会变
3. 变化是连续动态，还是可以用有限档位覆盖主要场景

只有这三个问题清楚后，才选 ATC 动态策略。

## 2. 常见策略

### 固定 shape

适用条件：

- 输入维度固定
- 或者业务已明确只交付单一输入档位
- 不适用于“真实输入不可固定，只是为了导出方便临时选一个样例 shape”的情况

示例：

```bash
python3 -u "$SKILL_DIR/scripts/pipeline_onnx_to_om.py" \
  --artifacts-dir /path/to/artifacts \
  --output-dir /path/to/om_out \
  --base-name model_name \
  --soc-version Ascend910B4 \
  --input-shape "images:1,3,1024,1024"
```

### `dynamic_dims`

适用条件：

- 输入不是连续动态
- 已知有限维度组合可以覆盖主要业务场景

例如 pointer 长度、少量固定分辨率桶、少量固定序列长度。

如果输入 shape 不可固定但可按有限组合覆盖主要场景，优先使用 `dynamic_dims`，不要拆成多个静态 OM。

### `dymshape`: 未指定 `dynamic_*` 的动态 `input_shape`

适用条件：

- 输入某些维度天然不固定
- 无法稳定压成少量固定档位
- 不能用 `dymbatch`、`dymhw` 或 `dymdims` 覆盖主要业务场景

ATC 导出侧不要写不存在的 `--dynamic_shape` 参数。若使用含 `-1` 的 `--input_shape`，且没有再指定 `dynamic_batch_size`、`dynamic_image_size` 或 `dynamic_dims`，这就是 `dymshape` 路线。

### `dynamic_batch_size`

适用条件：

- 主要变化只有 batch
- 通常仍需结合 `input_shape` 描述输入张量
- batch 不可固定但可枚举主要档位时，优先用该策略，不要生成多个 batch 静态 OM

### `dynamic_image_size`

适用条件：

- 官方图像类接口明确按该参数建模
- 实际变化就是图像高宽
- 通常仍需结合 `input_shape` 描述输入张量
- H/W 不可固定但可枚举主要分辨率档位时，优先用该策略，不要生成多个分辨率静态 OM

## 3. 互斥规则

以下主动态策略不要在同一条 ATC 命令里混用：

- `--dynamic_batch_size`
- `--dynamic_image_size`
- `--dynamic_dims`

当前 `pipeline_onnx_to_om.py` 已对这组互斥关系做了基本校验。

`--input_shape` 是基础输入描述，固定 shape 和动态 shape 都可能需要它。若使用含 `-1` 的 `--input_shape`，但没有附加任何 `dynamic_*` 参数，则按 `dymshape` 理解；若已显式指定某个 `dynamic_*` 参数，则以该专项动态策略为准。

## 4. 推荐尝试顺序

优先级不是固定不变的，但默认建议：

1. 目标功能范围内真实输入完全固定时，使用固定 shape
2. 只有 batch 变化且可分档覆盖主要场景时，优先 `dynamic_batch_size`
3. 主要为 H/W 变化且可分档覆盖主要场景时，优先 `dynamic_image_size`
4. 有限多个维度组合可覆盖主要场景时，优先 `dynamic_dims`
5. 确认无法固定且没有有效分档方式时，使用 `dymshape`，即使用含 `-1` 的 `input_shape` 且不再指定 `dynamic_*`

不要把“输入不可固定但可分档”的情况拆成很多个静态 OM。多个静态 OM 只作为动态分档或 `dymshape` 失败后的回退方案，并且需要记录失败证据和接回成本。

## 5. 失败后怎么处理

ATC 因动态策略失败时，优先按这个顺序检查：

1. 源码侧输入契约和当前图侧输入是否一致
2. ONNX 输入输出 shape 信息是否完整
3. 优化产物是否改写了输入形态
4. 当前策略是否和真实业务输入不匹配
5. 当前部署边界是否导致本来可固定的输入被迫动态化

如果问题根源在边界，而不是参数本身，应回到主流程重做部署边界。

## 6. 动态策略选择与 ATC 的协调规则

以下规则在 Workflow 6 失败处理时适用：

- `input_shape` 是基础输入描述，不是独立动态参数；固定 shape 使用全静态 `input_shape`，对应 `static` 主策略。
- 从优化角度优先考虑静态 shape：若源码侧部署契约、预处理或服务入口已经固定 batch 和输入尺寸，通常使用全静态 `input_shape`。如果为了覆盖本次适配场景的真实变化或降低分档复杂度选择动态 OM，必须说明原因和验证覆盖。
- 动态路线通常也需要结合 `input_shape` 描述输入；若 `input_shape` 含 `-1` 且未指定 `dynamic_batch_size`、`dynamic_image_size` 或 `dynamic_dims`，这一路线就是 `dymshape`，推理时使用 `dymshape`。
- 输入 shape 不可固定但可按主要场景分档时，可落地 `dynamic_batch_size`、`dynamic_image_size` 或 `dynamic_dims` 对应的动态分档 OM；如果分档成本高、档位不稳定或更适合统一运行入口，也可选择 `dymshape`，但必须记录选择依据、推理侧设置方式和验证 profile。
- 不要为了规避动态参数而直接生成很多个静态 OM；多个静态 OM 只能作为明确场景分档或动态策略失败后的回退，并记录原因和接回成本。
- 若候选阶段已经判定当前功能分支需要动态 OM，且源码侧契约与图侧 IO 契约一致，应直接执行相应动态 ATC 尝试；不要因前一功能分支使用静态 OM 而复用其停止结论。
- `static / dymbatch / dymhw / dymdims / dymshape` 是互斥的图侧主输入形态策略；单个 OM / 单次 ATC 只能选择一种。
- 若需要切换策略，必须作为新的 ATC 尝试记录，不能把多个主策略混进同一条命令。
