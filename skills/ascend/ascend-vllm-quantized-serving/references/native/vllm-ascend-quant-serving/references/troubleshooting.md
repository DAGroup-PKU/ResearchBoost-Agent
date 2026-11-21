# 量化模型拉起报错处置详表

> 按"产物 → 参数 → 环境"三层归因。每节给出：现象、根因、处置步骤、佐证方式。
> 报错样例源自 vllm-ascend 社区 issue 的高频模式；具体错误码以现场日志为准。

## §1 权重命名重映射失败（model_type 未适配）

**现象**：启动阶段抛出权重名找不到 / KeyError / AttributeError（如访问不存在的模块属性），
日志中可见加载器在按量化描述文件逐层映射权重时中断。

**根因**：vllm-ascend 加载 msModelSlim 产物时，依据 `quant_model_description.json` 做逐层
映射，并通过内置的 model_type → packed modules 映射表抹平命名差异。若目标模型的
`model_type` 不在映射表中，或该模型结构较新（映射表未跟上），重映射即失败。

**处置**：
1. 打开 `${QUANT_PATH}/config.json` 确认 `model_type` 字段；
2. 在 vllm-ascend 源码 `vllm_ascend/quantization/modelslim_config.py` 中检索该
   model_type 是否收录于 `packed_modules_model_mapping`；
3. 未收录：优先在 vllm-ascend 仓提 issue 请求适配（附 model_type 与结构说明），
   或参考已收录相近结构自行补映射后从源码安装验证；
4. 已收录仍失败：对比 `quant_model_description.json` 中的层名与模型实际层名，
   检查量化时是否使用了过旧/过新的 msmodelslim 版本。

## §2 310P 上 bias dtype / 专属格式类报错

**现象**：A2/A3 上可正常拉起的 W8A8 产物，在 310P/Atlas 300I Duo 上启动报
bias 数据类型不符（社区常见样例为要求 INT32）或直接不支持。

**根因**：310P 的量化走独立实现路径（含稀疏压缩等专属格式），能力集与 A2/A3 不同；
面向 A2/A3 量化的产物不保证在 310P 可用。

**处置**：
1. 确认目标硬件；310P 需使用面向 310P 的量化配置重新生成产物；
2. 310P 支持范围与已知问题以 vllm-ascend 仓 `310p` 标签下的 issue 为准；
3. 无法重新量化时，回退 BF16/FP16 拉起并记录差异，勿强行改产物文件。

## §3 aclnnQuantMatmul 类算子报错

**现象**：推理首个请求或启动自检时报 aclnn 量化算子错误（如 WeightNz 相关错误码）。

**根因**：权重打包/排布格式与当前 CANN 算子库期望不符，通常由三方版本不配套引起：
msmodelslim（量化侧）、vllm-ascend（加载侧）、CANN（算子侧）。

**处置**：
1. 三件套版本对齐到官方配套组合（msmodelslim 当前稳定分支 26.0.0）；
2. 用配套版本**重新量化**（不要只升级推理侧）；
3. 仍复现则带三件套版本号与完整报错栈到 vllm-ascend 仓提 issue。

## §4 可拉起但输出乱码 / 复读 / 明显降智

**排查顺序**：
1. **参数层**：msModelSlim 产物是否漏加 `--quantization ascend`？
   compressed-tensors 产物是否误加了该参数？（两种错配都可能表现为静默降质）
2. **产物层**：safetensors 分片是否完整（对比生成端与部署端的文件数与大小）；
   量化时校准集是否与目标任务域差异过大；
3. **版本层**：按 §3 对齐三件套版本；
4. **基线对照**：同机拉起原始 BF16 权重跑同一组 prompt——
   BF16 正常而量化异常 → 产物问题；两者都异常 → 环境/框架问题。

## §5 快速自检命令

```bash
# 产物体检
ls ${QUANT_PATH} | head -30
python3 -c "import json;print(json.load(open('${QUANT_PATH}/config.json')).get('model_type'))"
test -s ${QUANT_PATH}/quant_model_description.json && echo "ModelSlim 产物" || echo "非 ModelSlim 产物（查 config.json 的 quantization_config）"

# 服务体检
curl -s http://127.0.0.1:${PORT}/v1/models | python3 -m json.tool
```
