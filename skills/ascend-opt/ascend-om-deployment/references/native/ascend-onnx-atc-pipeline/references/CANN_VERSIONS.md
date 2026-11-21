# CANN 版本与环境基线

本文件只保留当前 Skill 需要的环境判断规则，不记录特定项目案例。

## 推荐组合

当前 Skill 维护的推荐环境组合是：

- `CANN >= 8.3.RC2`
- `Python 3.11`
- `NumPy 1.26.4`
- `ONNX == 1.16.1`

若本机尚未安装 CANN，当前 Skill 默认优先准备 `8.3.RC2`，并按该版本对应的完整安装方式准备本地安装包集合，而不是只准备单个 `toolkit` 包。

这套组合用于当前 Skill 的环境检查、依赖修复和脚本提示；若用户环境与之不同，应先按真实 CANN 版本核对 Python 兼容关系，再决定是否沿用现有环境。
其中当前脚本默认把 `NumPy 1.26.4`、`ONNX==1.16.1` 和 `onnxruntime==1.15.1` 视为 ONNX -> OM 流程的维护基线；固定 `ONNX==1.16.1` 主要是为了降低 `auto_optimizer` 兼容风险并保持环境可复现。

## CANN / Python 对应关系

这里记录的是本 Skill 维护时采用的环境判断规则，不把某一组版本写成“唯一正确答案”：

| CANN 区间 | Python 建议 | 说明 |
| --- | --- | --- |
| `>= 8.3.RC2` | `3.11` | 当前 Skill 默认维护和优先验证的组合 |
| `8.1.x - 8.3.RC1` | 以对应 CANN 官方要求为准，常见为 `3.10` 或更低 | 旧环境优先复用，不要为追求统一而直接升级 |

使用原则：

- 若用户机器上已有可用 CANN，先查该版本对应的 Python，再决定是否复用
- 若准备新建环境且没有额外约束，优先采用本 Skill 当前推荐组合
- 不要把 `Python 3.11` 写成所有 CANN 版本下的唯一允许环境
- 也不要把旧 CANN 时代的 Python 建议直接外推到新版本 CANN
- 但一旦进入当前 Skill 默认维护的 ONNX -> OM 流程，就应尽量把 Python 侧依赖收敛到脚本维护的基线

## 环境路径

常见安装路径有两类：

- `/usr/local/Ascend/ascend-toolkit`
- `/usr/local/Ascend/cann`

环境脚本会优先自动探测，不要求用户手工先判断。

## 官方安装方式变化

当前 Skill 需要区分两个安装时期，不能再把所有版本都按“装一个 toolkit 就够了”处理。

### 8.3.RC2 及其同代拆包方式

参考 `8.3.RC1/8.3.RC2` 同代安装文档，主安装包仍是分拆式：

- `Toolkit` 开发套件包
- `NNAE` 深度学习引擎包
- `NNRT` 离线推理引擎包
- 与芯片/产品匹配的 `Kernels` 算子包
- `NNAL` 神经网络加速库为可选项

这一代的环境变量路径也仍是分拆的：

- `Toolkit`: `${HOME}/Ascend/ascend-toolkit/set_env.sh`
- `NNAE`: `${HOME}/Ascend/nnae/set_env.sh`
- `NNRT`: `${HOME}/Ascend/nnrt/set_env.sh`

对当前 Skill 而言，若要避免后续 ONNX -> OM、推理或依赖侧缺件，默认完整安装规则是：

- `toolkit`
- `nnae`
- `nnrt`
- 与目标 SoC/产品匹配的 `kernels`

`nnal` 仍按官方文档视为可选加速库，不纳入默认完整安装。

### 8.5.x 及更新版本的安装方式

参考 `8.5.0` 官方安装文档，主路径已经变化为：

- `Toolkit` 开发套件包
- 与芯片/产品匹配的 `ops` 算子包
- `NNAL` 神经网络加速库为可选项

官方文档明确写到：`Toolkit` 与 `ops` 需要同时安装。

这一代默认环境变量路径也变成：

- `${HOME}/Ascend/cann/set_env.sh`

对当前 Skill 而言，8.5.x 及以上的默认完整安装规则是：

- `toolkit`
- 与目标 SoC/产品匹配的 `ops`

`nnal` 仍按官方文档视为可选加速库，不纳入默认完整安装。

## 官方安装包命名规则

当前 Skill 不在主流程里硬编码大量下载链接，但默认认可官方 OBS 的命名规律。

`toolkit` 样例：

```text
https://ascend-repo.obs.myhuaweicloud.com/CANN/CANN%20<VERSION>/Ascend-cann-toolkit_<VERSION>_linux-aarch64.run
```

同版本的其他组件通常沿用相同目录与版本命名，只是包名前缀不同，例如：

- 8.5.x 的 `Ascend-cann-910b-ops_<VERSION>_linux-aarch64.run`
- 8.3.x 的 `Ascend-cann-kernels-910b_<VERSION>_linux-aarch64.run`
- 8.3.x 的 `Ascend-cann-nnae_<VERSION>_linux-aarch64.run`
- 8.3.x 的 `Ascend-cann-nnrt_<VERSION>_linux-aarch64.run`
- 8.5.x 的 `Ascend-cann-nnal_<VERSION>_linux-aarch64.run`

使用原则：

- 若用户未指定版本，默认优先准备 `8.3.RC2`
- 若机器上已有可用 CANN，优先复用，不因默认版本而强制覆盖
- 若用户明确指定更高版本，应先核对该版本对应的 Python 与工具链兼容性，再继续
- 文档里的命名规则用于帮助定位官方安装包，不把某一条 OBS 域名写成唯一来源

## 使用原则

- 若机器上已有可用 CANN，则优先复用，不重复安装
- 若 CANN 缺失、版本过旧或 `atc` 不可用，再进入安装流程
- 不要在主 Skill 中默认强制重装 CANN

## Python 版本说明

- 当前 Skill 默认以 `Python 3.11` 作为推荐环境，是因为它与当前维护的 `CANN >= 8.3.RC2` 基线配套
- 若用户机器上已有匹配其 CANN 版本的其他 Python 环境，可在确认兼容后继续使用
- 先看 CANN 版本，再决定 Python 版本；不要反过来先锁死 Python
- 8.3.RC1/RC2 同代文档给出的 Python 支持范围更保守，文档可见区间是 `3.7.x` 到 `3.11.4`
- 8.5.0 文档给出的 Python 支持范围更宽，文档可见区间是 `3.7.x` 到 `3.13.x`

## 何时需要重新核对环境

出现以下情况时，应重新检查 CANN / Python / ONNX 版本配套关系：

- `atc` 不在 `PATH`
- `atc` 已在 `PATH`，但执行即报环境错误
- ONNX 优化工具无法导入
- `auto_optimizer` 无法运行
- 同一张图在不同机器上结果不一致

## 推荐检查命令

```bash
export SKILL_DIR=/path/to/ascend-onnx-atc-pipeline
bash "$SKILL_DIR/scripts/check_env_enhanced.sh"
source "$SKILL_DIR/scripts/setup_env.sh"
```

若 `ais_bench` / `aclruntime` 未安装，`repair_python_env.sh` 会根据当前 Python 版本和 CPU 架构自动下载官方 wheel；若本机已安装，则直接跳过。统一入口：

```bash
ASCEND_ONNX_ATC_PIPELINE_PYTHON=python3 \
bash "$SKILL_DIR/scripts/repair_python_env.sh"
```

如果你已经准备好了 wheel，也仍可显式指定：

```bash
ASCEND_ONNX_ATC_PIPELINE_OM_WHEEL_DIR=/path/to/wheels \
ASCEND_ONNX_ATC_PIPELINE_PYTHON=python3 \
bash "$SKILL_DIR/scripts/repair_python_env.sh"
```

## 安装入口

仅在需要时使用：

```bash
bash "$SKILL_DIR/scripts/install_cann.sh" --help
```

若用户未指定安装包，脚本默认按当前 Skill 的推荐基线优先准备 `8.3.RC2`；若已提供本地 `.run` 包目录，则默认按“完整安装”规则从目录中解析并安装所需组件集合。

当前 `install_cann.sh` 的默认模式不是“只装 toolkit”，而是按版本分流：

- `8.5.x+`: `toolkit + 对应产品 ops`
- `8.3.RC2` 及其同代拆包体系: `toolkit + nnae + nnrt + 对应产品 kernels`

若只想安装单个包，才显式使用 `--mode single --package ...`。
