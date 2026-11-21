# ATC 参数速查

本文件只保留当前 Skill 最常用、最容易出错的参数。

命令示例中的 `$SKILL_DIR` 表示本 Skill 安装目录，例如 `/path/to/ascend-onnx-atc-pipeline`。

## 必填参数

### `--model`

- 含义：输入 ONNX 路径
- 例子：`--model=/path/to/model.onnx`

### `--framework`

- ONNX 固定使用：`5`

### `--output`

- 含义：输出前缀，不带 `.om`
- 例子：`--output=/path/to/out/model_name`

### `--soc_version`

- 含义：目标 SoC 型号
- 例子：`Ascend910A`、`Ascend910B4`

## 输入形态相关

### `--input_shape`

固定 shape 常用。

```bash
--input_shape="images:1,3,1024,1024"
```

多输入：

```bash
--input_shape="image:1,3,1024,1024;tokens:1,256"
```

动态路线通常也需要 `--input_shape`。如果在动态维上写 `-1`，且没有再指定 `dynamic_batch_size`、`dynamic_image_size` 或 `dynamic_dims`，这就是 `dymshape` 路线。

### `--dynamic_dims`

适合已有固定档位集合的场景。

## 动态策略互斥规则

以下参数代表不同的主动态策略：

- `--dynamic_batch_size`
- `--dynamic_image_size`
- `--dynamic_dims`

单次 ATC 不要把这些主策略混在同一条命令里。

`--input_shape` 是基础输入描述：全静态时表示固定 shape；含 `-1` 且不搭配上述 `dynamic_*` 参数时表示 `dymshape` 路线；若已指定某个 `dynamic_*` 参数，则以对应专项动态策略为准。

推荐做法：

- 先明确当前子图到底是固定 shape、分档 shape 还是动态 shape
- 一次只尝试一种主策略
- 失败后再切到下一种，而不是把多种策略同时塞给 `atc`

## 常用辅助参数

### `--input_format`

在输入格式需要显式声明时使用。

### `--precision_mode`

仅在需要明确精度策略时填写。

### `--log`

建议在排障阶段显式加：

```bash
--log=info
```

## 动态策略选择原则

- 输入完全固定：使用固定 shape
- 只有 batch 变化且可分档覆盖主要场景：优先 `dymbatch`
- 主要为 H/W 变化且可分档覆盖主要场景：优先 `dymhw`
- 有限多个维度组合可覆盖主要场景：优先 `dymdims`
- 确认无法固定且无法有效分档：走 `dymshape`，即使用带 `-1` 的 `input_shape` 且不附加 `dynamic_*` 参数
- 不要把输入不可固定但可分档的候选拆成很多个静态 OM；多个静态 OM 只作为动态策略失败后的回退方案

不要脱离真实业务输入去盲目追求“全动态”。

## 推荐入口

不要手工散写多份命令，优先走：

```bash
python3 -u "$SKILL_DIR/scripts/pipeline_onnx_to_om.py" --help
```
