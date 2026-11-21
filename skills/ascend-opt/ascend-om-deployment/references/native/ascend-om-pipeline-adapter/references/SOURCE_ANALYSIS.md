# 源码分析与证据链速查

命令示例中的 `$PROJECT_DIR` 表示用户工程根目录。

```bash
export PROJECT_DIR=/path/to/project
```

## 1. 查真实推理入口

优先查用户真正调用的入口，而不是只盯着 `model.forward()`。

```bash
rg -n "main\\(|infer|predict|serve|session|start|update|finalize|track|pipeline|runner" "$PROJECT_DIR"
```

如果是服务式或多阶段工程，再看：

```bash
rg -n "grpc|fastapi|flask|stream|chunk|request|response|worker|handler" "$PROJECT_DIR"
```

## 2. 查状态对象和跨步缓存

重点看跨帧、跨页、跨 token、跨 chunk 的缓存和会话对象。

```bash
rg -n "state|cache|memory|history|session|context|kv_cache|tracker|buffer|prev|past" "$PROJECT_DIR"
```

要确认的问题：

- 状态是在 Python 控制面维护，还是在模块内部维护
- 哪些状态会进入候选子图
- 哪些状态只参与调度或聚合，应保留原始实现

## 3. 查候选子图输入来源

先找模块调用点，再向上回溯输入来源。

```bash
rg -n "backbone|encoder|decoder|head|fusion|prompt|embed|memory|sampler|neck" "$PROJECT_DIR"
rg -n "\\.forward\\(|model\\(|module\\(|predictor\\(" "$PROJECT_DIR"
```

对候选子图至少要补齐：

- 输入清单
- 输入来源
- 输入含义
- 可选输入
- 进入子图前已经做过哪些处理

## 4. 查进入子图前的处理

很多 shape 不是在子图入口才第一次确定，而是上游已经做过整理。

```bash
rg -n "resize|interpolate|pad|padding|bucket|crop|normalize|tokenize|embedding|mask|bbox|point|prompt" "$PROJECT_DIR"
```

重点确认：

- 是否已经做过 `resize / padding / bucket`
- 是否已经把文本、几何、mask、prompt 等输入编码成张量
- 候选子图看到的是原始输入，还是上游整理后的中间张量

## 5. 查现有 ONNX 导出线索

如果仓库里已经有导图入口，优先复用。

```bash
rg -n "torch\\.onnx\\.export|onnx\\.export|input_names|output_names|dynamic_axes|opset" "$PROJECT_DIR"
```

## 6. 证据链输出最低要求

不要只给结论，至少保留以下证据：

- 真实推理入口所在文件和函数
- 候选子图调用点
- 输入来源的关键上游代码
- 进入子图前处理的关键代码
- 状态对象定义或更新位置

建议输出格式：

```text
=== Source-Side Subgraph Contract ===
输入清单:
输入来源:
输入含义:
进入子图前的处理:
可选输入:
输入证据:
```

```text
=== Pipeline Boundary Analysis ===
推理入口:
控制面:
状态对象:
候选重计算子图:
候选大图:
前处理 / 后处理边界:
推荐首批 OM 范围:
保留原始实现的部分:
```
