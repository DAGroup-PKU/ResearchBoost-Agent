# AIPP 配置参考

AIPP 用于把部分图像预处理放到昇腾侧执行。它不是默认必用项，仅在图像输入路径明确、且前处理适合下沉时再考虑。

## 什么时候考虑 AIPP

适合：

- 输入是标准图像张量或原始图像数据
- 前处理主要是颜色空间转换、归一化、裁剪、缩放
- 希望减少 CPU 侧预处理开销

不适合：

- 原始工程前处理包含复杂条件分支
- 预处理语义和业务逻辑强绑定
- 现阶段主要目标是先跑通主链

## 常见能力

AIPP 常用于处理：

- YUV / RGB / BGR 格式转换
- 均值方差归一化
- 裁剪
- 缩放
- 数据格式调整

## 最小示例

```protobuf
aipp_op {
    aipp_mode: static
    input_format: RGB888_U8
    src_image_size_w: 224
    src_image_size_h: 224
    csc_switch: false
    mean_chn_0: 123.68
    mean_chn_1: 116.78
    mean_chn_2: 103.94
    var_reci_chn_0: 0.017
    var_reci_chn_1: 0.017
    var_reci_chn_2: 0.017
}
```

## 常见输入格式

- `RGB888_U8`
- `BGR888_U8`
- `YUV420SP_U8`
- `YUV422SP_U8`

## 常看参数

### 输入格式

- `input_format`
- `src_image_size_w`
- `src_image_size_h`

### 颜色空间转换

- `csc_switch`
- `matrix_r0c0` 到 `matrix_r2c2`
- `input_bias_0` 到 `input_bias_2`

### 归一化

- `mean_chn_0` 到 `mean_chn_3`
- `min_chn_0` 到 `min_chn_3`
- `var_reci_chn_0` 到 `var_reci_chn_3`

### 裁剪与缩放

- `crop`
- `load_start_pos_w`
- `load_start_pos_h`
- `crop_size_w`
- `crop_size_h`
- `resize`
- `resize_output_w`
- `resize_output_h`

## 与 ATC 结合

```bash
atc \
  --model=/path/to/model.onnx \
  --framework=5 \
  --output=/path/to/model_aipp \
  --soc_version=<soc_version> \
  --insert_op_conf=/path/to/aipp.cfg
```

## 使用原则

- 先确认原始代码前处理语义，再决定是否下沉到 AIPP
- 不要为了“更多算子在 NPU 上跑”而破坏原始前处理语义
- 若前处理较复杂，优先保留原始代码实现
