---
name: ascendc-api-best-practices
description: Resolve target-version Ascend C APIs for buffers, copy alignment, vector operations and synchronization.
---

# ascendc-api-best-practices

Choose the API by target SoC and installed CANN headers. The quick reference covers API selection; the buffer guide explains queue and temporary storage.

Read and apply the skill workflow; do not invoke the skill itself as a Bash command. Commands inside a selected workflow can be run in their documented environment. Use the task's numerical and performance verdicts to validate any change.

## Contents

- [Ascend C API 最佳实践](#ascend-c-api-最佳实践)
- [API 快速参考](#api-快速参考)
- [UB 缓冲区管理指南](#ub-缓冲区管理指南)

## Ascend C API 最佳实践

---

### API 类别索引

| API 类别 | 涵盖 API | 核心文档 | 典型场景 |
|---------|---------|---------|---------|
| **算术运算** | Add, Sub, Mul, Div, Adds, Muls | [api-arithmetic.md](../upstream.md#ascendc-api-best-practices-api-arithmetic-md) | Softmax, LayerNorm, 广播优化 |
| **跨代际迁移（Subnormal 与超越函数）** | Exp, Ln, Sqrt, Rsqrt, Div, Reciprocal | [api-cross-gen-migration.md](../upstream.md#ascendc-api-best-practices-api-cross-gen-migration-md) | DAV_2201→DAV_3510 迁移精度对齐、高精度/高性能模式选择 |
| **归约操作** | ReduceMax, ReduceSum | [api-reduce.md](../upstream.md#ascendc-api-best-practices-api-reduce-md), [api-reduce-pattern.md](../upstream.md#ascendc-api-best-practices-api-reduce-pattern-md) | Softmax, LayerNorm, ReduceMean |
| **归并排序** | Sort, Concat, MrgSort, Extract | [api-mrgsort.md](../upstream.md#ascendc-api-best-practices-api-mrgsort-md) | Sort, ArgSort, TopK |
| **数据搬运** | DataCopy, DataCopyPad | [api-datacopy.md](../upstream.md#ascendc-api-best-practices-api-datacopy-md) | 非对齐处理、多维搬运、gap 地址公式、UB→UB |
| **布局变换搬运** | Nd2NzParams, Nz2NdParamsFull, NdDmaLoopInfo | [api-nd2nz-nz2nd-nddma.md](../upstream.md#ascendc-api-best-practices-api-nd2nz-nz2nd-nddma-md) | ND2NZ / NZ2ND、Slice 切片、**NDDMA（仅 DAV_3510）** |
| **标量访问 Get/SetValue** | GlobalTensor/LocalTensor GetValue, SetValue | [api-scalar-access-patterns.md](../upstream.md#ascendc-api-best-practices-api-scalar-access-patterns-md) | 按访问用途、执行频率、目标架构判断是否替换及替换路径 |
| **LoadData / Cube 加载** | LoadData2D, LoadData2DV2, LoadData2DMx, LoadData3D | [api-loaddata.md](../upstream.md#ascendc-api-best-practices-api-loaddata-md) | L1 → L0 加载、Cube GEMM、MX 块量化格式、img2col |
| **L0C 回写 / Fixpipe** | DataCopyCO12DstParams, FixpipeParamsV220 | [api-cross-gen-fixpipe.md](../upstream.md#ascendc-api-best-practices-api-cross-gen-fixpipe-md) | L0C → GM/UB 回写、unitFlag、跨代参数差异 |
| **1C:2V 双 AIV 分发（Fixpipe L0C→UB）** | FixpipeParamsArch3510, dualDstCtl, subBlockId, McgShfMode | [api-fixpipe-1c2v.md](../upstream.md#ascendc-api-best-practices-api-fixpipe-1c2v-md) | 单 AIC 写两个 AIV 的 UB、SplitM/SplitN 选型、intra-block 握手 |
| **Transpose / 重排** | TransDataTo5HD, Gather | [api-transpose.md](../upstream.md#ascendc-api-best-practices-api-transpose-md) | 小通道 transpose、permute |
| **数据过滤/解交织** | GatherMask | [api-gathermask.md](../upstream.md#ascendc-api-best-practices-api-gathermask-md) | RoPE 奇偶拆分、列分量分离、非均匀间隔自定义 mask |
| **Matmul 高阶 API（Cube）** | MatmulImpl, MatmulConfig, IterateAll | [api-matmul.md](../upstream.md#ascendc-api-best-practices-api-matmul-md) | MatMul, BatchMatMul, MatMulBias；**限 Atlas A2/A3 系列（DAV_2201）平台** |
| **GMM 高阶 API** | GMMBaseParams, GMMArray, per-token dequant 管线 | [api-gmm.md](../upstream.md#ascendc-api-best-practices-api-gmm-md) | GroupedMatmul, A8W8/A4W4 量化反量化；**限 Atlas A2/A3 系列（DAV_2201）平台** |
| **Buffer 管理** | TBuf, TQue | [api-buffer.md](#ub-缓冲区管理指南) | Double Buffer、内存规划 |
| **精度转换** | Cast | [api-precision.md](../upstream.md#ascendc-api-best-practices-api-precision-md) | FP16/FP32 混合精度 |
| **流水线同步** | EnQue, DeQue, SetFlag | [api-pipeline.md](../upstream.md#ascendc-api-best-practices-api-pipeline-md) | 多级流水线、事件同步 |
| **Compare 256B对齐** | Compare | [api-restrictions.md](../upstream.md#ascendc-api-best-practices-api-restrictions-md-21-compare-api-256-bd2ae303) | Padding 策略 |
| **repeatTime 限制** | repeatTimes ≤ 255 | [api-repeat-limits.md](../upstream.md#ascendc-api-best-practices-api-repeat-limits-md) | 分批处理 |
| **API 限制** | - | [api-restrictions.md](../upstream.md#ascendc-api-best-practices-api-restrictions-md) | 禁用 API、编译期限制 |
| **Host Runtime** | aclrtSetDevice, aclrtGetDeviceInfo | [api-host-runtime.md](../upstream.md#ascendc-api-best-practices-api-host-runtime-md) | 设备初始化、核数获取 |
| **点对点通信** | Hcomm, ReadNbi, WriteNbi, Drain | [api-hcomm.md](../upstream.md#ascendc-api-best-practices-api-hcomm-md) | 跨卡点对点搬运、URMA 队列语义 |
| **跨核同步** | CrossCoreSetFlag, CrossCoreWaitFlag, SyncAll | [api-crosscore-sync.md](../upstream.md#ascendc-api-best-practices-api-crosscore-sync-md) | AIC↔AIV 通知、flagId 硬件规则 |
| **HCCL Host** | HcclCommInitRootInfoConfig, HcclChannelAcquire, HcclEngineCtxCreate | [api-hccl-host.md](../upstream.md#ascendc-api-best-practices-api-hccl-host-md) | 通信域/channel/engine ctx 管理 |
| **DMA 原子操作** | SetAtomicAdd, SetAtomicMax, DisableDmaAtomic | [api-atomic.md](../upstream.md#ascendc-api-best-practices-api-atomic-md) | 多核/多 rank 部分和累加、split-K 累加 |

---

### 场景索引

| 使用场景 | 相关文档 | 关键技巧 |
|---------|---------|---------|
| **Softmax/LayerNorm** | [api-reduce.md](../upstream.md#ascendc-api-best-practices-api-reduce-md), [api-reduce-pattern.md](../upstream.md#ascendc-api-best-practices-api-reduce-pattern-md), [api-arithmetic.md](../upstream.md#ascendc-api-best-practices-api-arithmetic-md) | 标量操作、广播优化、Buffer 复用 |
| **逐行处理（AR 模板）** | [api-arithmetic.md](../upstream.md#ascendc-api-best-practices-api-arithmetic-md) | Adds/Muls、节省 UB |
| **MatMul/BatchMatMul（Ascend C 高阶 API，Atlas A2/A3 系列）** | [api-matmul.md](../upstream.md#ascendc-api-best-practices-api-matmul-md) | MatmulConfig 选择、enUnitFlag、IterateAll、TilingHeader stack 加载 |
| **GroupedMatmul / GMM（Ascend C 高阶 API，Atlas A2/A3 系列）** | [api-gmm.md](../upstream.md#ascendc-api-best-practices-api-gmm-md) | GMMArray 分组索引、AIC/AIV 分职、per-token dequant 管线 |
| **Transpose / 重排** | [api-transpose.md](../upstream.md#ascendc-api-best-practices-api-transpose-md) | 2维度 转置性能 |
| **多行广播（ARA 模板）** | [api-arithmetic.md](../upstream.md#ascendc-api-best-practices-api-arithmetic-md) | BinaryRepeatParams.src1RepStride=0、分批处理 |
| **半精度加减法（FP16/BF16 Add/Sub）** | [api-arithmetic.md](../upstream.md#ascendc-api-best-practices-api-arithmetic-md), [api-precision.md](../upstream.md#ascendc-api-best-practices-api-precision-md) | 默认升精度（除非 spec 明确同量级）、in-place 复用 |
| **非对齐数据** | [api-datacopy.md](../upstream.md#ascendc-api-best-practices-api-datacopy-md) | DataCopyPad、32 字节对齐 |
| **Cube 布局转换（ND→NZ / NZ→ND）** | [api-nd2nz-nz2nd-nddma.md](../upstream.md#ascendc-api-best-practices-api-nd2nz-nz2nd-nddma-md) | `dstNzC0Stride` 单位是 C0_SIZE、`srcDValue` 是行 pitch |
| **1~5 维通用 stride / pad（NDDMA）** | [api-nd2nz-nz2nd-nddma.md](../upstream.md#ascendc-api-best-practices-api-nd2nz-nz2nd-nddma-md) | **仅 DAV_3510 可用**；搬运前必须 `NdDmaDci()`；`MultiCopy*` 是旧名 |
| **stride 单位查询 / 搬运接口选型** | [api-quickref.md](#api-快速参考) | 跨接口 stride 统一速查表、完整 DMA 选型决策树 |
| **UB 内搬运（DataCopy vs Copy）** | [api-datacopy.md](../upstream.md#ascendc-api-best-practices-api-datacopy-md) | 两者参数模型不同；`Copy` 的 count 模式仅 DAV_3510 |
| **标量访问与批量访问选择** | [api-scalar-access-patterns.md](../upstream.md#ascendc-api-best-practices-api-scalar-access-patterns-md) | 判断访问用途、执行频率及目标架构，查阅适用条件与替代方案 |
| **混合精度** | [api-precision.md](../upstream.md#ascendc-api-best-practices-api-precision-md) | FP16 输入 FP32 计算 |
| **Subnormal 精度丢失（DAV_3510 跨代迁移）** | [api-cross-gen-migration.md](../upstream.md#ascendc-api-best-practices-api-cross-gen-migration-md) | FTZ 语义、PRECISION_1ULP_FTZ_FALSE、eps 规避 |
| **Fixpipe 跨代参数（L0C 回写）** | [api-cross-gen-fixpipe.md](../upstream.md#ascendc-api-best-practices-api-cross-gen-fixpipe-md) | DAV_2201→DAV_3510 回写参数切换：FixpipeParamsArch3510 字段与单位差异 |
| **流水线优化** | [api-pipeline.md](../upstream.md#ascendc-api-best-practices-api-pipeline-md), [api-buffer.md](#ub-缓冲区管理指南) | Double Buffer、事件同步 |
| **性能调优** | [api-buffer.md](#ub-缓冲区管理指南), [api-repeat-limits.md](../upstream.md#ascendc-api-best-practices-api-repeat-limits-md) | Double Buffer、repeatTimes 优化 |
| **遇到 API 限制** | [api-restrictions.md](../upstream.md#ascendc-api-best-practices-api-restrictions-md) | 替代方案、避坑指南 |
| **通算融合（MC2）** | [api-hcomm.md](../upstream.md#ascendc-api-best-practices-api-hcomm-md), [api-crosscore-sync.md](../upstream.md#ascendc-api-best-practices-api-crosscore-sync-md) | Hcomm/CrossCore 原语层 API；**编排层（核分工、flag 流水、UB 隔离、通信并行度）请参考对应框架/领域技能文档** |
| **多卡通信** | [api-hccl-host.md](../upstream.md#ascendc-api-best-practices-api-hccl-host-md) | HCCL 建链、engine ctx 下发、资源生命周期 |
| **RoPE 奇偶拆分** | [api-gathermask.md](../upstream.md#ascendc-api-best-practices-api-gathermask-md) | pattern 1/2、Normal/Counter 均可、零拷贝偏移；**限 Atlas A2/A3 系列（DAV_2201）平台** |
| **Interleaved 列分离** | [api-gathermask.md](../upstream.md#ascendc-api-best-practices-api-gathermask-md) | 官方：Normal 模式、stride=8；实测：tile≥2048、外层循环；**限 Atlas A2/A3 系列（DAV_2201）平台** |
| **非均匀间隔自定义 mask** | [api-gathermask.md](../upstream.md#ascendc-api-best-practices-api-gathermask-md) | 用户自定义 LocalTensor mask、类型匹配 |
| **1 AIC 喂 2 AIV（Cube:Vector=1:2）** | [api-fixpipe-1c2v.md](../upstream.md#ascendc-api-best-practices-api-fixpipe-1c2v-md), [api-crosscore-sync.md](../upstream.md#ascendc-api-best-practices-api-crosscore-sync-md) | 四种分发策略选型、每 lane 半 M、AIC 发两槽 / AIV 只等 base |
| **Fixpipe 结果错位 / 参数填错** | [api-fixpipe-1c2v.md](../upstream.md#ascendc-api-best-practices-api-fixpipe-1c2v-md) | §6.1 断言速查表、`srcStride = Align(baseM,16)`、真机无运行时校验 |
---

### 快速参考

完整的 API 参数速查表：[api-quickref.md](#api-快速参考)

---

### ⛔️ API 黑名单

**禁止在生产代码中使用**：

| API | 禁止原因 | 替代方案 | 文档 |
|-----|---------|---------|------|
| `GlobalTensor::SetValue()` | 效率极低 | `DataCopyPad` | [api-datacopy.md](../upstream.md#ascendc-api-best-practices-api-datacopy-md) |
| `GlobalTensor::GetValue()` | 效率极低 | `DataCopyPad` | [api-datacopy.md](../upstream.md#ascendc-api-best-practices-api-datacopy-md) |
| `LocalTensor` 主循环/嵌套循环单点 `GetValue/SetValue` | PIPE_S 标量通路，吞吐率差数量级 | vector 通路（按目标架构选广播/批量方案） | [api-scalar-access-patterns.md](../upstream.md#ascendc-api-best-practices-api-scalar-access-patterns-md)（判定标准与替代方案索引） |

**限制使用的 API**：

| API | 限制条件 | 说明 | 文档 |
|-----|---------|------|------|
| `DataCopy(GM↔UB)` | 仅当搬运数据**严格 32 字节对齐**时允许使用 | 非对齐场景必须使用 `DataCopyPad` | [api-datacopy.md](../upstream.md#ascendc-api-best-practices-api-datacopy-md) |

**仅允许调试时使用**：
```cpp
// ✅ 调试：单点验证
AscendC::printf("debug: xGm[0]=%f\n", xGm.GetValue(0));
```

## API 快速参考

Ascend C API 使用核心决策索引。遇到具体问题时，按需查阅详细文档。

---

### 目录

1. [核心原则速查](#核心原则速查)
2. [决策树：我应该用什么API？](#决策树我应该用什么api)
3. [按场景查阅详细文档](#按场景查阅详细文档)

---

### 核心原则速查

#### 1. DataCopy vs DataCopyPad

**原则：优先使用 DataCopyPad，参数统一用 Ext 版本**

| 场景 | API | 参数 | 原因 |
|-----|-----|------|------|
| 所有 GM ↔ UB 搬运 | `DataCopyPad` | `DataCopyExtParams` + `DataCopyPadExtParams<T>` | 统一处理对齐/非对齐，参数范围更大 |
| 确定数据严格 32 字节对齐 | `DataCopy` | `DataCopyParams` 或 `DataCopyExtParams` | 简单场景可用 |

**详细用法**：[api-datacopy.md](../upstream.md#ascendc-api-best-practices-api-datacopy-md)

#### 2. Cast RoundMode 选择

| 转换方向 | RoundMode | 原因 |
|---------|-----------|------|
| half → float | `CAST_NONE` | 低→高精度，无损失 |
| float → half | `CAST_ROUND` | 高→低精度，需舍入 |
| half → int32_t | `CAST_ROUND` | 量化场景 |
| int32_t → float | `CAST_NONE` | 整数→浮点 |

**详细用法**：[api-precision.md](../upstream.md#ascendc-api-best-practices-api-precision-md)

#### 3. TBuf vs TQue 选择

| 场景 | 类型 | 说明 |
|------|------|------|
| MTE2/MTE3 搬运缓冲区 | `TQue<VECIN/VECOUT>` | `InitBuffer(que, num, size)` |
| 纯 Vector 计算缓冲区 | `TBuf<VECCALC>` | `InitBuffer(buf, size)` |

**详细用法**：[api-buffer.md](#ub-缓冲区管理指南)

#### 4. 流水线同步

**原则**：必须用 EnQue/DeQue 同步 MTE 和 Vector

**核心模式**：
```
CopyIn → EnQue → DeQue → Compute → EnQue → DeQue → CopyOut
```

**详细用法**：[api-pipeline.md](../upstream.md#ascendc-api-best-practices-api-pipeline-md)

#### 5. Vector API repeatTime 限制

**核心限制**：repeatTime 为 uint8_t 时，最大值 255

**处理方法**：Host 侧限制 R_max 或 Kernel 侧分批处理

**详细用法**：[api-repeat-limits.md](../upstream.md#ascendc-api-best-practices-api-repeat-limits-md)

#### 6. Reduce API 选择

| 场景 | 接口 | 说明 |
|-----|------|------|
| 逐行独立 Reduce | Level 2: `ReduceMax(dst, src, tmp, count)` | 无对齐要求，count 传 rLength |
| 跨行批量 Reduce | Pattern: `ReduceMax<T, Pattern::AR>(...)` | 需 32 字节对齐 |

**详细用法**：[api-reduce.md](../upstream.md#ascendc-api-best-practices-api-reduce-md) | [Pattern 接口详解](../upstream.md#ascendc-api-best-practices-api-reduce-pattern-md)

#### 7. GatherMask 解交织

**三种用法**：

| 场景 | 模式 | 关键参数 |
|------|------|---------|
| RoPE 奇偶拆分、按步长取元素 | 内置 pattern 1/2, Normal/Counter 均可 | `{1, 1, 8, 0}` |
| (N,4)→4 列分量分离 | 内置 pattern 3/4/5/6 | `src0RepeatStride=8`, `repeatTimes=alignedLen/16` |
| 非均匀间隔、复杂 mask | 用户自定义 `LocalTensor` mask | Counter/Normal 均可，mask 类型匹配 dst |

⛔ **Critical**：`src0RepeatStride=8` 不可写成 1（repeat 重叠 224B → 精度全 fail）。**详细用法**：[api-gathermask.md](../upstream.md#ascendc-api-best-practices-api-gathermask-md)

#### 8. 跨接口 stride 统一速查

> **定位**：本表是**跨接口对照入口**——把同一个词「stride / gap」在不同接口下的单位与语义并排放，专门用于识别跨接口误用。各家族文档内的「参数单位速查」是**本族字段全集**（含本表未覆盖的字段与取值范围），两者分工不同、互为补充。
> **口径**：默认按 **DAV_2201（Atlas A2/A3，如 910B3）**；平台受限的行已单独标注（NDDMA、`LoadData2DParamsV2`、`LoadData2DMxParams`）。

| 接口 | 参数 | 含义 | 单位 | 连续值 |
|---|---|---|---|---:|
| 普通 `DataCopyParams` | `srcGap` / `srcStride` | 前块尾 → 后块头 gap | **32B** | `0` |
| 普通 `DataCopyParams` | `dstGap` / `dstStride` | 前块尾 → 后块头 gap | **32B** | `0` |
| GM→UB `DataCopyPad` | `srcStride` | GM block gap | **byte** | `0` |
| GM→UB `DataCopyPad` | `dstStride` | UB block gap | **32B** | `0` |
| UB→GM `DataCopyPad` | `srcStride` | UB block gap | **32B** | `0` |
| UB→GM `DataCopyPad` | `dstStride` | GM block gap | **byte** | `0` |
| `LoadData2D`（V1） | `srcStride` | fractal **头 → 头** pitch | **512B** | **`1`** |
| `LoadData2D`（V1） | `dstGap` | fractal **尾 → 头** gap | **512B** | `0` |
| `LoadData2DParamsV2`（**仅 950**） | `srcStride` / `dstStride` | K 方向相邻分形起始地址间隔 | **512B** | — |
| `LoadData2DMxParams`（**仅 950**） | `srcStride` / `dstStride` | X 方向相邻分形起始地址间隔 | **32B** | — |
| ND2NZ | `srcDValue` | ND 行 pitch | element | `dValue` |
| ND2NZ | `srcNdMatrixStride` / `dstNzMatrixStride` | 相邻矩阵起点距离 | element | — |
| ND2NZ | `dstNzC0Stride` / `dstNzNStride` | NZ 内 C0 / Z 排布偏移 | **C0_SIZE（32B）** | — |
| NZ2ND | `srcNdMatrixStride` | 源相邻 NZ 矩阵偏移 | **256（16×16）个元素** | — |
| NZ2ND | `srcNStride` | 源 NZ 内相邻 Z 排布偏移 | **16 个元素** | — |
| NZ2ND | `dstDStride` / `dstNdMatrixStride` | ND 行 pitch / 相邻 ND 矩阵偏移 | element | `dValue` |
| NDDMA（**仅 950**） | `loopSrcStride` / `loopDstStride` | 元素 → 下个元素地址增量 | element | `1` |
| NDDMA broadcast（**仅 950**） | `loopSrcStride` | 重用同一位置 | element | **`0`** |
| L0C 回写 / Fixpipe | `dstStride` | 随输出格式变化 | **32B 或 element** | 视模式 |
| L0C 回写 / Fixpipe | `srcStride` | 相邻 Z 排布偏移 | **C0_SIZE（16×sizeof(T)）** | — |

**两条记忆规则**：

1. **同一个词，单位随接口变**：`DataCopyPad` 按操作数位置分（GM = byte、UB = 32B DataBlock）；`DataCopyParams` 两侧都按 32B 且语义是 tail → head；`LoadData2D`（V1）是 512B 且语义是 head → head。
2. 布局变换族**不是「只有 ND2NZ 的 C0 字段例外」**：`dstNzC0Stride` / `dstNzNStride` 用 C0_SIZE（32B），NZ2ND 的 `srcNdMatrixStride` / `srcNStride` 分别用 **256 / 16 个元素**，其余 stride 类字段才是 element。

⛔ **`stride = 0` 在不同接口含义相反**：普通 `DataCopyParams` 的 `0` = 无 gap = **连续**；NDDMA 的 `0` = 地址不前进 = **broadcast**。不要跨接口套用。

**详细用法**：[api-datacopy.md](../upstream.md#ascendc-api-best-practices-api-datacopy-md)、[api-loaddata.md](../upstream.md#ascendc-api-best-practices-api-loaddata-md)、[api-nd2nz-nz2nd-nddma.md](../upstream.md#ascendc-api-best-practices-api-nd2nz-nz2nd-nddma-md)、[api-cross-gen-fixpipe.md](../upstream.md#ascendc-api-best-practices-api-cross-gen-fixpipe-md)

---

### 决策树：我应该用什么API？

#### Q1: 需要搬运数据？

先判断**通路**，再判断对齐 / 连续性：

```
数据要搬运
├─ GM ↔ UB
│   ├─ 对齐状态
│   │   ├─ 确定 32B 对齐 → 连续？
│   │   │        ├─ 是 → DataCopy(..., count)
│   │   │        └─ 否 → DataCopyParams（2D stride 能描述 → 一次 DMA；
│   │   │                 不能 → 合并连续维 + 少量外层循环，仍复杂则 Slice
│   │   │                 [api-nd2nz-nz2nd-nddma.md] / 自行展开）
│   │   ├─ 确定不对齐 → DataCopyPad
│   │   └─ 不确定       → DataCopyPad（推荐）
├─ Cube / L0
│   ├─ GM → L1（布局转换）→ ND2NZ DataCopy
│   ├─ L1 → L0A / L0B     → LoadData
│   └─ L0C → GM / L1      → Fixpipe / DataCopyCO12DstParams
│                            ⛔ 不要用普通 GM↔UB DataCopy 理解
├─ 跨卡                    → Hcomm ReadNbi / WriteNbi（见 Q6）
└─ 1~5 维通用 stride / pad → NDDMA
                             ⚠️ 仅 950（DAV_3510）可用，A2/A3 不可用
```

详见 [api-datacopy.md](../upstream.md#ascendc-api-best-practices-api-datacopy-md)、[api-nd2nz-nz2nd-nddma.md](../upstream.md#ascendc-api-best-practices-api-nd2nz-nz2nd-nddma-md)、[api-loaddata.md](../upstream.md#ascendc-api-best-practices-api-loaddata-md)、[api-cross-gen-fixpipe.md](../upstream.md#ascendc-api-best-practices-api-cross-gen-fixpipe-md)

#### Q2: 需要精度转换？

```
half → float → CAST_NONE
float → half → CAST_ROUND
其他 → 查阅 api-precision.md
```

#### Q3: 需要分配 UB 缓冲区？

```
涉及 MTE 搬运 → TQue + InitBuffer(que, num, size)
纯 Vector 计算 → TBuf + InitBuffer(buf, size)
```

#### Q4: 遇到数据错误/随机值？

```
1. 检查是否缺少 EnQue/DeQue → api-pipeline.md
2. 检查 DataCopyPad 参数 → api-datacopy.md
3. 检查 Reduce API 的 tmpBuffer 类型 → api-reduce.md
 4. 检查多行处理时的 rowOffset 计算 → api-reduce.md
 5. 检查 repeatTime 是否溢出 → api-repeat-limits.md
```

#### Q5: 需要混合精度计算（FP16 输入，FP32 中间计算）？

```
查阅 api-precision.md 的混合精度模式
```

#### Q6: 需要跨卡数据搬运？

```
拉远端到本地 → Hcomm ReadNbi，详见 api-hcomm.md
推本地到远端 → Hcomm WriteNbi，详见 api-hcomm.md
需要远端通知 → WriteWithNotifyNbi
```

#### Q7: AIC 和 AIV 需要跨核协同？

```
AIC 通知 AIV：AIC CrossCoreSetFlag<0x2, PIPE_FIX>(id) → AIV CrossCoreWaitFlag<0x2, PIPE_S>(id)
AIV 回压 AIC：AIV CrossCoreSetFlag<0x2, PIPE_MTE3>(id) → AIC CrossCoreWaitFlag<0x2, PIPE_M>(id)
注意：A3/910b（Atlas A2/A3 系列，DAV_2201）上 modeId/pipe 模板参数不生效；950（Ascend 950PR/950DT，DAV_3510）上模式 0/1/2 不支持 PIPE_S
详见 api-crosscore-sync.md
```

#### Q8: 依赖所有 block 都完成通信后才能继续？

```
Drain 只保证本 block 的 channel 任务完成
需要"所有 block 完成"语义时加 SyncAll<true>() 块间同步
详见 api-hcomm.md、api-crosscore-sync.md
```

#### Q9: Host 侧如何建链并下发通信上下文？

```
HcclCommInitRootInfoConfig → HcclChannelDescInit + HcclChannelAcquire
→ HcclEngineCtxCreate + HcclEngineCtxCopy → 跨 rank 带外 barrier → launch
详见 api-hccl-host.md
```

#### Q10: 多核/多 rank 写入同一 GM 地址？

```
是 → 需要原子操作
  累加语义（ReduceScatter/AllReduce/split-K）→ SetAtomicAdd<T>()
  取最大值（attention 归约）→ SetAtomicMax<T>()
  取最小值 → SetAtomicMin<T>()
  ⛔ FP8 不支持原子操作，需先转 BF16/FP32
  ⛔ 用完必须 DisableDmaAtomic() 关闭
否 → 不需要（默认状态）
详见 api-atomic.md
```

---

### 按场景查阅详细文档

| 场景 | 文档 | 核心内容 |
|-----|------|---------|
| 数据搬运（GM↔UB） | [api-datacopy.md](../upstream.md#ascendc-api-best-practices-api-datacopy-md) | DataCopyPad 参数、stride 计算、非对齐处理、gap 地址公式、UB→UB |
| 布局变换搬运（ND2NZ / NZ2ND / NDDMA / Slice） | [api-nd2nz-nz2nd-nddma.md](../upstream.md#ascendc-api-best-practices-api-nd2nz-nz2nd-nddma-md) | Cube 通路概览、Nd2NzParams 单位、**NDDMA 平台门禁** |
| LoadData（L1→L0） | [api-loaddata.md](../upstream.md#ascendc-api-best-practices-api-loaddata-md) | LoadData2D V1/V2/MX 字段单位、C0 元素数、LoadData3D |
| L0C 回写 / Fixpipe | [api-cross-gen-fixpipe.md](../upstream.md#ascendc-api-best-practices-api-cross-gen-fixpipe-md) | DataCopyCO12DstParams、FixpipeParamsV220、Arch3510 跨代差异 |
| 精度转换/混合精度 | [api-precision.md](../upstream.md#ascendc-api-best-practices-api-precision-md) | Cast RoundMode、FP16 混合精度模式 |
| UB 缓冲区管理 | [api-buffer.md](#ub-缓冲区管理指南) | TBuf/TQue 选择、Double Buffer、批量搬运 |
| 流水线同步 | [api-pipeline.md](../upstream.md#ascendc-api-best-practices-api-pipeline-md) | EnQue/DeQue 同步机制、时序图 |
| repeatTime 限制 | [api-repeat-limits.md](../upstream.md#ascendc-api-best-practices-api-repeat-limits-md) | 分批处理 |
| 点对点通信 | [api-hcomm.md](../upstream.md#ascendc-api-best-practices-api-hcomm-md) | ReadNbi/WriteNbi/Drain、URMA 队列语义 |
| 跨核同步 | [api-crosscore-sync.md](../upstream.md#ascendc-api-best-practices-api-crosscore-sync-md) | CrossCoreSetFlag/WaitFlag、SyncAll、flagId 硬件规则 |
| HCCL Host | [api-hccl-host.md](../upstream.md#ascendc-api-best-practices-api-hccl-host-md) | 建链、engine ctx、资源生命周期 |
| DMA 原子操作 | [api-atomic.md](../upstream.md#ascendc-api-best-practices-api-atomic-md) | SetAtomicAdd/DisableDmaAtomic、多 rank 累加前置条件 |

## UB 缓冲区管理指南

TBuf/TQue 选择、Double Buffer 流水线并行、批量搬运模式。

---

### 目录

1. [TBuf vs TQue 选择](#tbuf-vs-tque-选择)
2. [TQue 详解](#tque-详解)
3. [TBuf 详解](#tbuf-详解)
4. [Double Buffer 流水线并行](#double-buffer-流水线并行)
5. [批量搬运 + 逐行计算模式](#批量搬运--逐行计算模式)

---

### TBuf vs TQue 选择

| 场景 | 推荐类型 | 说明 |
|-----|---------|------|
| MTE2/MTE3 搬运缓冲区（标准 GM↔UB 搬运流水） | `TQue<VECIN/VECOUT>` | 需要与 Vector 并行，需要 EnQue/DeQue |
| 纯 Vector 计算缓冲区 | `TBuf<VECCALC>` | 不涉及 MTE 搬运，用 `Get<T>()` 获取 |
| Double Buffer | `TQue` + `InitBuffer(que, 2, size)` | 在 InitBuffer 中设置 num=2 开启 |
| 手动 UB 管理（无 TPipe） | `LocalTensor` + 字节偏移 | 多阶段流水算子常见模式，见下文 |

---

### UB 管理的两种模式（TPipe vs 手动偏移）

UB 并非只有 TPipe 一种管理方式。两种模式均合法，按场景选择：

| 模式 | 要点 | 适用 |
|------|------|------|
| TPipe（TQue/TBuf） | 声明式分配、自动管理生命周期与事件 | 标准单算子搬运/计算流水 |
| 手动 UB 偏移 | `LocalTensor`/`__ubuf__` 指针 + 字节偏移自行规划布局，配合手动管理的 SetFlag/WaitFlag 事件流水 | 需要精细控制 UB 布局的场景：跨阶段长生命周期 buffer、per-tile 高频调用（避免 TPipe 重复 InitBuffer 耗尽 UB）、多 buffer 复杂流水编排 |

**混用红线（静态区隔离）**：kernel 内存在**裸静态区域**（用 `MakeMemPtr<Location::UB>`/固定偏移排布的通信工作区等）时，TPipe 动态分配的 buffer 必须与其**物理隔离**——否则 TPipe 可能把 buffer 分配到静态区上，踩踏数据 → 死锁/精度错。两种隔离做法：

1. **guard TBuf**：进入 TPipe 分配前，先 `InitBuffer` 一个占位 TBuf（大小 = 静态区总字节），TPipe 后续分配自然落在其后
2. **全手动偏移**：所有 buffer 统一手动偏移规划，静态区与计算区各自预留段，预算合并核算

手动偏移模式的纪律：偏移单一来源（一处计算多处传递）、对齐到 32B（或按 DMA 效率 64B）、UB 总预算含 pitch padding 核算、per-tile 循环外一次性完成布局。

---

### TQue 详解

#### 模板参数

```cpp
template <TPosition pos, int32_t depth, auto mask = 0> class TQue;
```

| 参数 | 说明 |
|------|------|
| `pos` | 队列逻辑位置：`VECIN`, `VECOUT`, `A1`, `A2`, `B1`, `B2`, `CO1`, `CO2` |
| `depth` | 队列深度，表示可连续 EnQue/DeQue 的次数 |
| `mask` | 数据格式转换（ND↔NZ）或编译期优化参数 |

#### depth 参数关键说明

| depth 值 | 适用场景 | 说明 |
|---------|---------|------|
| `depth=1` | **默认推荐**，非 Tensor 原地操作 | 编译器有特殊优化，性能更好 |
| `depth=0` | **Tensor 原地操作** | 需要设置 |
| `depth=2` | 连续 2 次 EnQue 场景 | 与 InitBuffer 的 num 参数独立 |

**注意**：`depth` 与 Double Buffer 无关。Double Buffer 由 `InitBuffer` 的 `num` 参数控制。

```cpp
// ✅ 非连续入队（普通场景）：depth=1 即可
AscendC::TQue<AscendC::TPosition::VECIN, 1> que;
pipe->InitBuffer(que, 1, size);
auto tensor = que.AllocTensor<T>();
que.EnQue(tensor);
tensor = que.DeQue<T>();
que.FreeTensor(tensor);
```

#### Double Buffer 配置

**Double Buffer 是在 `InitBuffer` 的 `num` 参数中设置，与模板参数 `depth` 无关。**

| InitBuffer 参数 | 作用 | 说明 |
|----------------|------|------|
| `InitBuffer(que, num, size)` | `num` 控制 Double Buffer | `num=1`=单 Buffer，`num=2`=开启 Double Buffer |
| 模板参数 `depth` | 队列深度 | 表示可连续 EnQue 的次数 |

```cpp
// ✅ 开启 Double Buffer：在 InitBuffer 中设置 num=2
AscendC::TQue<AscendC::TPosition::VECIN, 1> que;  // 模板 depth=1 即可
pipe->InitBuffer(que, 2, size);  // num=2 开启 Double Buffer

// ✅ 关闭 Double Buffer
AscendC::TQue<AscendC::TPosition::VECIN, 1> que;
pipe->InitBuffer(que, 1, size);  // num=1 单 Buffer
```

#### TQue Buffer 数量限制

| 产品系列 | eventID 数量 | 最大 TQue 数量 |
|---------|-------------|---------------|
| Atlas 训练系列 | 4 | 4 |
| Atlas 推理系列 AI Core | 8 | 8 |
| Atlas 推理系列 Vector Core | 8 | 8 |
| Atlas A2/A3 系列 | 8 | 8 |

**注意**：
- 不开启 Double Buffer（num=1）：最多可申请 8 个 TQue
- 开启 Double Buffer（num=2）：每个 TQue 占用 2 个 buffer，最多只能申请 4 个 TQue
- **eventID 是 TQue 与手动 `SetFlag/WaitFlag(HardEvent)` 共享的资源**：kernel 内另用手动事件（如 MTE2_V/V_MTE3 配对）时，可用于 TQue 的预算要相应扣减，混用时合并计算配额

```cpp
// 开启 Double Buffer 时，最多只能申请 4 个 TQue
pipe->InitBuffer(que0, 2, size);  // ✅
pipe->InitBuffer(que1, 2, size);  // ✅
pipe->InitBuffer(que2, 2, size);  // ✅
pipe->InitBuffer(que3, 2, size);  // ✅
pipe->InitBuffer(que4, 2, size);  // ❌ 超过限制
```

#### TQue 正确用法

```cpp
// TQue：需要队列管理（MTE 搬运相关）
// 模板 depth=1 即可，Double Buffer 在 InitBuffer 的 num 参数中设置
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
pipe->InitBuffer(inQueueX, 2, bufferSize);  // num=2 开启 Double Buffer

AscendC::LocalTensor<half> x = inQueueX.AllocTensor<half>();
AscendC::DataCopyPad(x, xGm, {1, (uint32_t)(size * sizeof(half)), 0, 0, 0}, {false, 0, 0, static_cast<half>(0)});
inQueueX.EnQue(x);
// ...
AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
inQueueX.FreeTensor(xLocal);
```

---

### TBuf 详解

#### 特性

| 特性 | 说明 |
|------|------|
| 内存用途 | 只能参与计算，无法执行 EnQue/DeQue |
| 内存分配 | 每次 InitBuffer 只分配一块内存 |
| Tensor 释放 | 无需手动释放 |

```cpp
// TBuf：纯计算缓冲区
AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
pipe->InitBuffer(workBuf, bufferSize);

// ✅ 使用 Get<T>() 获取 Tensor，无需释放
AscendC::LocalTensor<float> work = workBuf.Get<float>();
// ... 计算逻辑 ...
// 无需 FreeTensor
```

---

### Double Buffer 流水线并行

#### 核心认知

**Double Buffer 不是"用2块内存计算"，而是"用2块内存做搬入/搬出，使 MTE2/MTE3 与 Vector 计算并行"。**

本质：**内存搬运与计算并行，掩盖搬运延迟**。

#### 硬件原理

- **MTE2**：搬运工，GM → UB
- **Vector**：加工员，计算
- **MTE3**：搬运工，UB → GM

#### 时间线对比

**无 Double Buffer（串行）**：
```
Row 0: [MTE2][Vector][MTE3]
Row 1:                      [MTE2][Vector][MTE3]
```

**有 Double Buffer（并行）**：
```
Row 0: [MTE2-B0][Vector-B0][MTE3-B0]
Row 1:          [MTE2-B1][Vector-B1][MTE3-B1]
                  ↑ MTE2与Vector并行！
```

#### 实现原则

| Buffer 类型 | InitBuffer num | 说明 |
|------------|----------------|------|
| `TQue<VECIN>` (MTE2 搬运) | **2** | num=2 开启 Double Buffer，与 Vector 并行 |
| `TQue<VECOUT>` (MTE3 搬运) | **2** | num=2 开启 Double Buffer，与 Vector 并行 |
| `TBuf<VECCALC>` (纯计算) | - | TBuf 不涉及 MTE 搬运 |

#### 正确用法

```cpp
// 1. Init: num=2 开启 Double Buffer
pipe->InitBuffer(inQueueX,  2, tileSize * sizeof(T));
pipe->InitBuffer(outQueueY, 2, tileSize * sizeof(T));
pipe->InitBuffer(workBuf, workSize * sizeof(T));

// 2. Process: 单循环结构，TQue 自动轮转
for (int i = 0; i < totalTiles; i++) {
    CopyIn(i);   // MTE2 异步搬运
    Compute(i);  // Vector 计算
    CopyOut(i);  // MTE3 异步搬出
}

// 3. CopyIn
void CopyIn(int i) {
    LocalTensor<T> x = inQueueX.AllocTensor<T>();
    DataCopyPad(x, xGm[i * tileSize], {1, (uint32_t)(tileSize * sizeof(T)), 0, 0, 0}, {false, 0, 0, static_cast<T>(0)});
    inQueueX.EnQue(x);
}

// 4. Compute
void Compute(int i) {
    LocalTensor<T> x = inQueueX.DeQue<T>();
    LocalTensor<T> y = outQueueY.AllocTensor<T>();
    Add(y, x, constTensor, tileSize);
    outQueueY.EnQue(y);
    inQueueX.FreeTensor(x);
}

// 5. CopyOut
void CopyOut(int i) {
    LocalTensor<T> y = outQueueY.DeQue<T>();
    DataCopyPad(yGm[i * tileSize], y, {1, (uint32_t)(tileSize * sizeof(T)), 0, 0, 0});
    outQueueY.FreeTensor(y);
}
```

#### 为什么能并行？

| 操作 | 特性 |
|------|------|
| `DataCopy` | 异步 DMA，立即返回 |
| `EnQue` | 非阻塞，标记就绪 |
| `DeQue` | 阻塞，等待就绪 |

#### 常见误区

| 误区 | 正确理解 |
|------|---------|
| 需要手动拆成 Ping/Pong 两套代码 | 单循环 + `InitBuffer(que, 2, size)` 自动管理 |
| depth 模板参数控制 Double Buffer | Double Buffer 由 `InitBuffer` 的 `num` 参数控制 |
| depth 越大越好 | 模板 depth 通常设为 1，性价比最高 |
| 所有 buffer 都要 num=2 | 只有涉及 MTE 搬运的才需要 Double Buffer |

---

### 批量搬运 + 逐行计算模式

#### 适用场景

处理多行数据时，批量搬运减少 MTE2/MTE3 调用次数，充分利用带宽。

#### 模式结构

```
CopyInBatch(N行) → 逐行计算(N次) → CopyOutBatch(N行)
```

#### 代码模板

```cpp
__aicore__ inline void ProcessBatch()
{
    uint32_t totalRowsToProcess = endRow - startRow;
    if (totalRowsToProcess == 0) return;
    
    for (uint32_t tile = 0; tile < tilesPerCore; tile++) {
        uint32_t startLocalRow = tile * tileRows;
        
        // 边界检查：防止 uint32_t 下溢
        if (startLocalRow >= totalRowsToProcess) break;
        
        uint32_t remaining = totalRowsToProcess - startLocalRow;
        uint32_t rowsThisTile = (remaining < tileRows) ? remaining : tileRows;
        
        CopyInBatch(startLocalRow, rowsThisTile);
        ComputeBatch(rowsThisTile);
        CopyOutBatch(startLocalRow, rowsThisTile);
    }
}
```

#### Host 侧 Tiling 计算

```cpp
// UB 容量示例值 192KB（Atlas A2/A3 系列）——生产代码按目标芯片实际 UB 容量动态获取
// （device 侧 GetCoreMemSize(UB)），勿硬编码；芯片规格以 npu-arch skill 为唯一知识源
constexpr uint64_t UB_SIZE = 192 * 1024;  // 示例值
constexpr uint32_t MAX_BLOCK_COUNT = 4095;  // DataCopyPad blockCount 限制

// bytesPerTileRow: double buffer (in*2 + out*2)
uint32_t bytesPerTileRow = paddedColsT * typeSizeBytes * 4;

// tileRows
uint32_t tileRows = (UB_SIZE - overheadBytes) / bytesPerTileRow;
tileRows = std::max(1u, std::min(tileRows, MAX_BLOCK_COUNT));
```

#### 注意事项

1. **tileRows 限制**：DataCopyPad 的 `blockCount` 最大 4095
2. **尾核处理**：`startLocalRow >= totalRowsToProcess` 时提前退出
3. **stride 计算**：UB 侧 stride 单位是 32 字节块，GM 侧是字节

---

### 多 stage 共享 L1 / L0 Buffer 的常量一致性

#### 适用场景

mix kernel（`__mix__(N, M)`）或多 stage 算子中，同一对 L1 / L0 Buffer 经常被多个 Compute stage 函数共享访问做轮转（例如 GEMM 算子中两个连续 Mmad 计算共享同一对 L1 输入 buffer）。

#### 必须一致的常量

各 stage 函数内使用以下常量，**必须与 InitBuffer 的实际分配字节数一致**：

- 单 slot 元素数（`slotElems`）
- 单 slot 字节数（`slotBytes`）
- per-slot stride / 槽偏移基数

#### 典型踩坑

```cpp
// InitBuffer 时分配：
buf.InitBuffer(matAL1_, 64 * 1024 * PRELOAD_NUM);   // 每 slot 64KB

// ComputeStage1 中（正确）：
const uint32_t slotElems = 64 * 1024 / sizeof(DATA_T);   // 与 InitBuffer 一致
auto a = matAL1_.Get<DATA_T>()[loopSlot * slotElems];

// ComputeStage2 中（错误！）：
const uint32_t slotElems = 128 * 1024 / sizeof(DATA_T);  // ❌ 误写 128KB
auto b = matAL1_.Get<DATA_T>()[loopSlot * slotElems];    // task=0 偏移=0 蒙混，task=1+ 读越界脏数据
```

#### 症状

- 任务 task=0 输出正常（偏移=0，即使常量错也不越界，读到的还是合法分配区）
- 任务 task=1+ 输出 NaN / inf（偏移到 buffer 末尾外的脏数据）
- "偶数 task PASS / 奇数 task FAIL" 或 "首个 task PASS / 后续 task 全爆炸" 型周期性错误

#### 工程约束

把所有 per-slot 常量提到单一头文件或单一 constexpr 定义，所有 stage 引用同一定义：

```cpp
// constants.h
constexpr uint32_t L1_BUF_A_SLOT_BYTES = 64 * 1024;
constexpr uint32_t L1_BUF_B_SLOT_BYTES = 64 * 1024;
constexpr uint32_t L1_BUF_A_SLOT_ELEMS = L1_BUF_A_SLOT_BYTES / sizeof(DATA_T);
```

避免在每个 stage 函数内各自声明 `const uint32_t slotElems = ...;`。
