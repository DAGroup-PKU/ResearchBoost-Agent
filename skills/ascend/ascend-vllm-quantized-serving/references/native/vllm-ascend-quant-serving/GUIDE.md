# 量化模型 vllm-ascend 服务化闭环

本技能只解决一件事：**把一份已生成的量化权重，正确地在 vllm-ascend 上跑起来并确认可用**。

## 适用场景

- 量化产物已就绪（本地目录或 ModelScope 下载），需要拉起推理服务
- 拉起失败或行为异常，需要按"产物 → 参数 → 环境"三层排查
- 需要一套最小验收动作确认服务可用后再交给下游

## 执行规则

- 统一使用占位符：`${QUANT_PATH}`（量化权重目录）、`${TP}`（tensor 并行数）、`${PORT}`、`${SERVED_NAME}`；禁止写死绝对路径。
- **量化产物的生成不在本技能范围**：需要做量化时，转交 `msmodelslim-quick-quant` 等官方 MindStudio 技能，或参考 msmodelslim 仓 `example/` 与 `lab_practice/` 下对应模型的已验证配置。
- 集群化/PD 分离部署不在本技能范围，本技能覆盖单机单实例拉起。
- 310P/Atlas 300I Duo 与 A2/A3 的量化能力集不同（见排错参考），本技能默认面向 A2/A3。
- 回答保持"先体检、再决策、后拉起"的顺序，不要跳步。

## 第 0 步：产物体检（决定后面一切）

先判定产物来源，二者的启动方式**不同**：

| 判定依据 | 来源 | 启动参数 |
|---|---|---|
| 目录含 `quant_model_description.json`（msModelSlim ascendv1 saver 输出） | **msModelSlim** | 必须加 `--quantization ascend` |
| `config.json` 内含 `quantization_config`（compressed-tensors 格式） | **LLM-Compressor** | **不加** `--quantization` 参数，自动识别 |

体检清单（任一不满足先停下修产物，不要带病拉起）：

1. `config.json`、tokenizer 文件、safetensors 分片齐全；
2. msModelSlim 产物：`quant_model_description.json` 存在且非空；
3. `config.json` 的 `model_type` 属于 vllm-ascend 已适配的模型族（未适配的 model_type 会在权重命名重映射阶段失败，见排错参考）；
4. 确认量化类型与目标硬件匹配（例如 MXFP 系列对硬件代际有要求；310P 仅支持其专属能力集）。

## 第 1 步：环境与版本配套

```bash
# 确认 vllm 与 vllm-ascend 已安装且版本配套（二者需按官方配套表成对安装）
pip show vllm vllm-ascend | grep -E "Name|Version"

# 若产物由 msModelSlim 生成，确认量化时使用的 msmodelslim 版本
# 与 vllm-ascend 推荐配套版本一致（当前稳定分支为 26.0.0）
```

版本不配套是"能拉起但输出异常"的第一嫌疑人：量化侧与推理侧对权重打包格式的约定随版本演进，跨大版本混用时优先重新对齐版本再排查其他问题。

## 第 2 步：拉起

### 在线服务

```bash
# msModelSlim 产物：
vllm serve ${QUANT_PATH} \
    --served-model-name ${SERVED_NAME} \
    --tensor-parallel-size ${TP} \
    --max-model-len 4096 \
    --port ${PORT} \
    --trust-remote-code \
    --quantization ascend

# LLM-Compressor / compressed-tensors 产物：去掉 --quantization ascend，其余相同
```

### 离线推理（快速确认产物可加载）

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="${QUANT_PATH}",
    max_model_len=4096,
    tensor_parallel_size=${TP},
    trust_remote_code=True,
    quantization="ascend",   # 仅 msModelSlim 产物需要；compressed-tensors 产物删除此行
)
out = llm.generate(["你好，请自我介绍。"], SamplingParams(temperature=0.6, top_p=0.95))
print(out[0].outputs[0].text)
```

## 第 3 步：冒烟与最小验收

服务拉起后按序执行（可用 `scripts/smoke_test.sh` 一键跑完）：

1. **连通性**：`GET /v1/models` 返回 `${SERVED_NAME}`；
2. **生成质量抽检**：中文问答、英文问答、代码片段各 1 条，肉眼确认无乱码、无复读；
3. **量化确实生效**：对比 NPU 显存占用与同模型 BF16 拉起时的差值（W8A8 权重显存约为 BF16 的一半量级）；
4. 若需正式精度/性能验收（数据集评测、TTFT/TPOT 基线对比），转交量化验收类技能或评测工具，不在本技能展开。

## 常见报错速查

| 现象 | 第一嫌疑 | 处置 |
|---|---|---|
| 启动即报权重/张量相关错误，日志出现权重名匹配失败 | `model_type` 未适配或命名重映射缺失 | 见 `references/troubleshooting.md` §1 |
| `bias` dtype 相关报错（多见于 310P） | 硬件专属量化格式限制 | 见 §2 |
| `aclnnQuantMatmulWeightNz` 类算子报错 | 权重打包格式与算子期望不符（常伴随版本不配套） | 见 §3 |
| 能拉起，输出乱码/复读 | 量化产物损坏或版本不配套 | 见 §4 |
| 加了 `--quantization ascend` 后反而失败 | 产物其实是 compressed-tensors 格式 | 回到第 0 步重新判定来源 |

## 参考

- vllm-ascend 官方量化指南（feature_guide/quantization）
- msModelSlim 仓库：`example/`（量化脚本）、`lab_practice/`（各模型已验证 YAML 配置）
- 本目录 `references/troubleshooting.md`：报错处置详表
- 本目录 `scripts/smoke_test.sh`：冒烟脚本
