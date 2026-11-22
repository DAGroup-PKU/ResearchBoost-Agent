---
name: ascendc-tiling-design
description: Design Ascend C multi-core and UB tiling for a measured operator workload.
---

# ascendc-tiling-design

Select the operator family and target architecture. The tiling reference includes multi-core/UB design and the selected elementwise and Welford routes.

Read and apply the skill workflow; do not invoke the skill itself as a Bash command. Commands inside a selected workflow can be run in their documented environment. Use the task's numerical and performance verdicts to validate any change.

## Contents

- [Ascend C 算子 Tiling 设计指南](#ascend-c-算子-tiling-设计指南)
- [EleWise - Tiling 详细计算](#elewise---tiling-详细计算)
- [Welford Online 算法（在线单遍）](#welford-online-算法在线单遍)

## Ascend C 算子 Tiling 设计指南

### 算子分类体系

| 类别 | 特征 | 典型算子 | 设计指南 |
|------|------|---------|---------|
| **Reduction 归约类** | 沿轴归约（含索引跟踪变体） | ReduceSum, Softmax, LayerNorm, ArgMax | ✅ [场景路由](../upstream.md#ascendc-tiling-design-reduction-patterns-md)（⚠️ 必须先读） / [算法](../upstream.md#ascendc-tiling-design-reduction-algorithms-md) |
| **Sort 排序类** | 排序、TopK、外部归并 | Sort, ArgSort, TopK | ✅ [场景路由](../upstream.md#ascendc-tiling-design-sort-patterns-md)（⚠️ 必须先读） / [多核两级归并方案](../upstream.md#ascendc-tiling-design-sort-alg-two-level-mrgsort-md) |
| **Elementwise 逐元素类** | 输入输出Shape相同，逐元素独立计算 | Sin, Cos, Abs, Exp | ✅ [场景路由](../upstream.md#ascendc-tiling-design-elewise-patterns-md)（⚠️ 必须先读） |
| Broadcast 广播类 | 输入Shape不同，需广播对齐 | Add, Mul, Sub | ✅ [场景路由](../upstream.md#ascendc-tiling-design-broadcast-patterns-md)（⚠️ 必须先读） |
| **Conversion 数据转换类** | 改变布局/形状，合并/拆分张量 | Transpose, Slice/StridedSlice, Pad, ChunkCat, Split | ✅ [场景路由](../upstream.md#ascendc-tiling-design-conversion-patterns-md)（⚠️ 必须先读；已展开 small-channel Transpose / Slice / Pad / ChunkCat，其余子场景待补） |
| Random 随机类 | 生成随机数，需种子管理 | RandomUniform, Dropout | 📋 规划中 |
| MatMul 矩阵乘类 | 矩阵乘法，高计算密度，用Cube单元 | MatMul, BatchMatMul, MatMulBias, GroupedMatmul | ✅ [场景路由](../upstream.md#ascendc-tiling-design-matmul-patterns-md)（⚠️ 必须先读平台路由） |
| Convolution 卷积类 | 空间卷积，滑动窗口计算 | Conv2D, DepthwiseConv | 📋 规划中 |
| FlashAttention 类 | 注意力计算 | FlashAttention, GQA, MHA, MLA（可叠加量化/稀疏 trait） | ✅ [场景路由](../upstream.md#ascendc-tiling-design-flashattention-patterns-md)（⚠️ 必须先读） |
| Scatter 累加散射类 | 按 index 散射写入/累加，updates 按行组织 | MaskedScatter, ScatterNdAdd, ScatterElementsV2, ScatterAdd, InplaceIndexAdd, scatter_reduce | ✅ [场景路由](../upstream.md#ascendc-tiling-design-scatter-patterns-md)（⚠️ 必须先读） |
| **Index 索引类**（A2/A3） | 以索引为核心访问数据，存在跨元素寻址 | Gather, GatherElements, IndexSelect, IndexAdd, RepeatInterleave, Take, Scatter, ScatterElements | ✅ [场景路由](../upstream.md#ascendc-tiling-design-index-patterns-md)（⚠️ 必须先读，按 P1~P7 分流到各 Pattern 文档） |
| NN 神经网络类(其它)| 神经网络专用，多种操作组合 | GroupNorm 等非 attention | 📋 规划中 |

> **Pad / 填充 / 镜像类算子：设计前必须先读 [conversion/pad.md](../upstream.md#ascendc-tiling-design-conversion-pad-md)**（该文的结构判据优先于 tile 参数）。

> **芯片代际说明**：`references/index/` 下的文档面向 **A2/A3 代际**（910B / 910C，`arch22`，`dav_c220` / `DAV_2201`，单核 UB 192KB）；
> 其中 `index/scatter.md` 为 A2/A3 的 UB 内标量散写方案，与面向 **A5 代际**（950，`arch35`，`DAV_3510`）的 [`references/scatter/patterns.md`](../upstream.md#ascendc-tiling-design-scatter-patterns-md) 分属两代，按目标芯片选用。
> 两代在 UB 容量与可用指令上存在实质差异（A2/A3 无硬件 Scatter 指令、只能走标量方案），Tiling 结论不可跨代照搬。

### 通用设计要素（所有类别必须）

以下设计要素是所有算子类别都必须考虑和完成的：

#### 1. 多核切分策略

**核心问题**：任务如何分配给多个 AI Core？

**设计要点**：
- 负载均衡：每个核处理的任务量尽量相等
- 数据局部性：相邻数据尽量分配给同一核
- 粒度适中：tile 不能太小（调度开销大），不能太大（并行度低）

**输出**：
- [ ] 总任务切分方式（按哪个维度切）
- [ ] 每个 AI Core 处理的任务量
- [ ] 使用的 AI Core 数量

#### 2. UB 切分策略

**核心问题**：单次能处理多少数据？

**设计要点**：
- UB 容量限制（DAV_2201: 192KB, DAV_3510: 248KB）
- 单次处理数据量
- 是否需要分 chunk 处理

**输出**：
- [ ] 单次处理的数据量
- [ ] 是否需要分 chunk
- [ ] chunk 大小计算公式

#### 3. Buffer 规划

**核心问题**：需要哪些 buffer？各多大？

**设计要点**：
- 输入 buffer（inQueue）
- 输出 buffer（outQueue）
- 中间计算 buffer（tmpBuf, workBuf 等）
- Double Buffer 优化
- **UB 复用与扩满**（必须遵守）：设计计算块时，尽可能实现 buffer 复用，减少临时 buffer 的申请；扩大 UB 使用量，实现尽可能用满所有可用 UB 空间。当多个计算步骤的 buffer 生命周期不重叠时，应复用同一 TBuf 而非申请新 buffer；在不超过 `GetCoreMemSize(UB)` 上限的前提下，增大 tile size 使 UB 利用率尽可能接近 100%

**输出**：
- [ ] Buffer 列表及用途
- [ ] 各 Buffer 大小计算公式
- [ ] Buffer 复用方案（哪些 buffer 可复用同一 TBuf）
- [ ] 总 UB 使用量及 UB 利用率（总分配 / `GetCoreMemSize(UB)`）

#### 4. 分支场景覆盖

**核心问题**：需要处理哪些不同场景？

**常见分支维度**：
- 数据类型：FP32 / FP16 / BF16 / INT8
- Shape 大小：大 shape / 小 shape
- 数据对齐：32字节对齐 / 非对齐
- 边界情况：最小值 / 最大值 / 特殊值

**输出**：
- [ ] 分支决策条件
- [ ] 各分支的处理策略
- [ ] 边界测试用例

## EleWise - Tiling 详细计算

> **适用场景**: 输入输出 Shape 完全相同，逐元素独立计算

---

### 一、多核切分（blockFormer / blockNum）

#### 核心思想

确保每个核处理的数据量 >= 最小阈值，并按 512 字节对齐。

#### Tiling 参数说明

| 参数 | 含义 | 计算公式 |
|------|------|---------|
| `dim0` | 元素总数量 | 所有维度相乘 |
| `coreNum` | 实际使用的核数 | `min(计算核数, 最大核数)` |
| `blockFormer` | 每个核的基础元素数（对齐到 512 元素） | `((dim0 / coreNum) + 511) / 512 * 512` |
| `blockNum` | 虚拟 block 数量 | `(dim0 + blockFormer - 1) / blockFormer` |

#### 计算步骤

```cpp
// Step 1: 计算核数（保证每个核至少处理 4KB 数据）
// 注意：MIN_TILING_BITS_SIZE_PER_CORE 和 minDtypeBits 单位都是 bits
coreNum = (dim0 * minDtypeBits + MIN_TILING_BITS_SIZE_PER_CORE - 1) / 
           MIN_TILING_BITS_SIZE_PER_CORE;
coreNum = min(coreNum, availableCoreNum);

// Step 2: 每个核的基础元素数，对齐到 512 元素
blockFormer = ((dim0 + coreNum - 1) / coreNum + ELEM_ALIGN_FACTOR - 1) / 
               ELEM_ALIGN_FACTOR * ELEM_ALIGN_FACTOR;

// Step 3: 计算虚拟 block 数量
blockNum = (dim0 + blockFormer - 1) / blockFormer;
```

#### 核间数据偏移

```cpp
template <typename DataType>
__aicore__ inline int64_t CalcBlockOffset() {
    return blockFormer * GetBlockIdx() * sizeof(DataType);
}
```

---

### 二、UB 切分（ubFormer / ubLoop / tail）

#### 核心思想

确保 UB 处理量是 256B 的整数倍（Vector 指令最优）。

#### Tiling 参数说明

| 参数 | 含义 | 计算公式 |
|------|------|---------|
| `ubFormer` | 每个 UB 块的基础大小（256B 对齐） | 对齐到 256B |
| `ubLoopOfFormerBlock` | 首块的 UB 循环次数 | `(blockFormer + ubFormer - 1) / ubFormer` |
| `ubTailOfFormerBlock` | 首块的尾部大小 | `blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer` |
| `ubLoopOfTailBlock` | 尾块的 UB 循环次数 | `(blockTail + ubFormer - 1) / ubFormer` |
| `ubTailOfTailBlock` | 尾块的尾部大小 | `blockTail - (ubLoopOfTailBlock - 1) * ubFormer` |

#### 计算步骤

```cpp
// Step 1: 计算 UB 能容纳的最大元素数
bufferDivisor = bufferNum * elemBytes;
maxElemNum = (ubSize - extraSize) * 8 / bufferDivisor;

// Step 2: 按 256B 对齐
alignFactor = REPEAT_BYTES * 8 / minDtypeBits;  // 如 FP32 = 64 元素
ubFormer = (maxElemNum / alignFactor) * alignFactor;

// Step 3: 计算循环次数
ubLoopOfFormerBlock = (blockFormer + ubFormer - 1) / ubFormer;
ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;
```

---

### 三、Kernel 执行模型

#### Process 循环结构

**核心逻辑**：区分首 block 和尾 block，因为它们的 UB 循环次数可能不同。

```cpp
// 1. 判断当前处理的是否是最后一个 block
bool isLastBlock = (blockIdx == blockNum - 1);

// 2. 获取当前 block 的循环次数和尾部大小
//    首 block 和尾 block 的循环次数/尾部大小可能不同
loopNum = isLastBlock ? ubLoopOfTailBlock : ubLoopOfFormerBlock;
tailNum = isLastBlock ? ubTailOfTailBlock : ubTailOfFormerBlock;

// 3. 主循环（处理完整的 UB 块）
for (uint64_t i = 0; i < loopNum - 1; i++) {
    ProcessTile(offset, ubFormer);
    offset += ubFormer;
}

// 4. 尾部处理（处理最后一个不完整的 UB 块）
ProcessTile(offset, tailNum);
```

**为什么要区分首/尾 block？**

- `blockFormer` 是按 512B 对齐的，可能略大于平均分配
- 最后一个 block 分配的原始数据量 `blockTail` 可能小于 `blockFormer`
- 导致 `ubLoopOfFormerBlock` ≠ `ubLoopOfTailBlock`，`ubTailOfFormerBlock` ≠ `ubTailOfTailBlock`

---

### 四、最小可执行 Tiling 模板

```cpp
struct TilingData {
    int64_t dim0;           // 元素总数量
    int32_t coreNum;        // 实际核数
    int64_t blockFormer;    // 每个核的基础数据量
    int64_t blockNum;       // block 数量
    int64_t ubFormer;       // UB 基础大小
    int64_t ubLoopOfFormerBlock;
    int64_t ubTailOfFormerBlock;
    int64_t ubLoopOfTailBlock;
    int64_t ubTailOfTailBlock;
};

TilingData ComputeTiling(int64_t dim0, int64_t elemBytes, int64_t ubSize, 
                         int64_t bufferNum, int64_t availableCoreNum) {
    TilingData tiling;
    
    // 常量
    constexpr int64_t MIN_TILING_BITS = 32768;       // 4KB，单位 bits
    constexpr int64_t ELEM_ALIGN_FACTOR = 512;       // 多核切分元素对齐因子
    constexpr int64_t ALIGN_256 = 256;               // UB 对齐字节数
    
    // 1. 多核切分（minDtypeBits = elemBytes * 8）
    tiling.coreNum = (dim0 * minDtypeBits + MIN_TILING_BITS - 1) / MIN_TILING_BITS;
    tiling.coreNum = std::min(tiling.coreNum, availableCoreNum);
    
    tiling.blockFormer = ((dim0 + tiling.coreNum - 1) / tiling.coreNum + ELEM_ALIGN_FACTOR - 1) / ELEM_ALIGN_FACTOR * ELEM_ALIGN_FACTOR;
    tiling.blockNum = (dim0 + tiling.blockFormer - 1) / tiling.blockFormer;
    
    // 2. UB 切分
    int64_t bufferDivisor = bufferNum * elemBytes;
    int64_t maxElemNum = (ubSize * 8) / bufferDivisor;
    int64_t alignFactor = ALIGN_256 * 8 / elemBytes;
    tiling.ubFormer = (maxElemNum / alignFactor) * alignFactor;
    
    // 3. 循环次数
    tiling.ubLoopOfFormerBlock = (tiling.blockFormer + tiling.ubFormer - 1) / tiling.ubFormer;
    tiling.ubTailOfFormerBlock = tiling.blockFormer - (tiling.ubLoopOfFormerBlock - 1) * tiling.ubFormer;
    
    int64_t blockTail = dim0 - (tiling.blockNum - 1) * tiling.blockFormer;
    tiling.ubLoopOfTailBlock = (blockTail + tiling.ubFormer - 1) / tiling.ubFormer;
    tiling.ubTailOfTailBlock = blockTail - (tiling.ubLoopOfTailBlock - 1) * tiling.ubFormer;
    
    tiling.dim0 = dim0;
    return tiling;
}
```

---

### 五、经验总结

| 经验 | 说明 | 代码模式 |
|------|------|----------|
| **最小粒度** | 每个核至少 4KB 数据，否则不值得开核 | `MIN_TILING_BITS = 32768` |
| **多核对齐** | 元素数对齐到 512 的倍数 | `blockFormer = (原始值 + 511) / 512 * 512` |
| **UB 对齐** | 按 256B 对齐，确保 Vector 指令效率 | `ubFormer = (原始值 / alignFactor) * alignFactor`，其中 alignFactor = 256 / elemBytes（如 FP32=4字节 则 alignFactor=64） |
| **跨核偏移** | 当前核 GM 偏移 = `blockFormer * blockIdx` | `CalcBlockOffset()` |

---

### 六、dtype 分支：FP16/BF16 升精度（Add/Sub）的 UB 预算

**何时启用**：patterns.md 的 Step 2 判定命中升精度分支时使用。

#### 1. Buffer 规划差异

升精度分支在原 dtype Queue 之外，额外引入 **K 份 `ubFormer * sizeof(float)` 的 FP32 中间 Buffer**。K 由 API 层的别名约束决定，tiling 阶段只需将其作为参数代入 UB 预算。

| 项 | 原 dtype 直算 | 升精度分支 |
|---|--------------|-----------|
| 输入/输出 Buffer dtype | 原 dtype（half/bf16） | 原 dtype（不变） |
| FP32 中间 Buffer 份数 | — | **K**（由 API 别名约束给出） |
| 单份 FP32 中间大小 | — | `ubFormer * sizeof(float)` |
| 总 UB 占用 | `bufferNum * ubFormer * elemBytes` | `bufferNum * ubFormer * elemBytes + K * ubFormer * sizeof(float)` |

#### 2. ubFormer 计算调整

升精度分支下 `bufferDivisor` 需同时包含半精度 Queue 和 FP32 中间 Buffer 两部分：

```cpp
// 原直算分支
bufferDivisor = bufferNum * elemBytes;

// 升精度分支：bufferNum 份半精度 + K 份 FP32
bufferDivisor = bufferNum * elemBytes + K * sizeof(float);

maxElemNum = (ubSize * 8) / bufferDivisor;
alignFactor = ALIGN_256 * 8 / elemBytes;       // 对齐仍按输入 dtype
ubFormer = (maxElemNum / alignFactor) * alignFactor;
```

K 的取值取决于 Compute 阶段选用 API 是否支持 dst/src 别名，该参数由 API 实现细节给出后回填。

> 升精度分支不新增 TilingData 字段，Kernel 侧按 dtype 模板参数静态选择分支即可。具体 Cast/Add/Sub 调用、RoundMode、别名写法等属 API 实现细节，tiling 阶段不涉及。

---

## Welford Online 算法（在线单遍）

**适用场景**: 分载模式下需要流式计算两个相关统计量（如 mean + variance），单遍扫描完成。

**优势**: 单遍扫描（vs TwoPass 两遍）、数值稳定性好、支持分组并行合并。

---

### 核心更新公式

```
初始: mean = 0, M2 = 0, count = 0

对每个新元素 x:
    count += 1
    delta1 = x - mean           ← 旧偏差
    mean = mean + delta1/count  ← 增量更新均值
    delta3 = x - mean           ← 新偏差
    M2 = M2 + delta1 * delta3   ← 增量更新方差

最终: var = M2 / (count - correction)
```

### 两组合并公式

当多个核或多个分组各自计算了局部 (mean, M2, count) 后，合并为全局结果：

```
合并 (mean_a, M2_a, count_a) 和 (mean_b, M2_b, count_b):

count_total = count_a + count_b
delta = mean_b - mean_a
mean_total = mean_a + delta * count_b / count_total
M2_total = M2_a + M2_b + delta² * count_a * count_b / count_total
```

### Group Welford（分组合并）

当分载的 chunk 数很大时，每 8 个 chunk 做一次中间合并，防止浮点误差累积。

### 向量化 Welford 更新（AscendC 实现示例）

```cpp
// 对 UB 中一个 chunk 的数据做 Welford 更新
void WelfordUpdate(LocalTensor<float>& x, int curLen,
                   LocalTensor<float>& mean, LocalTensor<float>& M2,
                   int& count) {
    for (int i = 0; i < curLen; i++) {
        count++;
        float scale = 1.0f / static_cast<float>(count);

        // delta1 = x - mean（向量化：对整个 A 维）
        Sub(delta1Buf, x[i * A_aligned], mean, A_aligned);

        // mean = mean + delta1 * scale
        Muls(tmpBuf, delta1Buf, scale, A_aligned);
        Add(mean, mean, tmpBuf, A_aligned);

        // delta3 = x - mean_new
        Sub(delta3Buf, x[i * A_aligned], mean, A_aligned);

        // M2 = M2 + delta1 * delta3
        Mul(tmpBuf, delta1Buf, delta3Buf, A_aligned);
        Add(M2, M2, tmpBuf, A_aligned);
    }
}
```

### 与 TwoPass 的选择

| 条件 | 推荐 |
|------|------|
| FullLoad + 两次顺序归约 | TwoPass（第一遍求 A，第二遍用 A 求 B，实现简单） |
| 分载 + 两次相关归约 | Welford（单遍流式，省一轮 IO） |
