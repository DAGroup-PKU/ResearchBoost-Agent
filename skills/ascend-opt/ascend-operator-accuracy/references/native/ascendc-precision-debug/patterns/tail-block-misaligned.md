# 尾块未处理 / 非对齐 shape

## 形态指纹

- 误差集中在输出尾部（bad_index_pattern = tail）。
- 对齐 shape（如 32B 整数倍）比对通过，非对齐 shape（如 fp32 的 1023 个元素）才错。
- first_bad_index 常出现在最后一个 tile/尾段起点附近。

## 根因机理

分别检查字节对齐与算法 tile 边界：DMA 是否满足 32B 对齐取决于元素数乘 dtype 字节数；算法尾块则取决于元素数是否为 tileElems 的整数倍。这两个条件不等价。常见错误：

- 尾块直接按整块长度搬运/计算，越界读到脏数据或写越界；
- 尾块被直接丢弃不处理；
- 多核切分时最后一个核的尾段长度算错。

## 验证方法

1. 固定 fp32 dtype：1024 个元素（4096B）通过，1023 个元素（4092B）失败 → 高度疑似。
2. dump 输出看首个错点位置是否 = floor(N/block)*block（尾段起点）。
3. 输入全 1、输出应为常数的算子：尾部值异常一目了然。

## 修复 pattern

```cpp
// tailGm points to the last tile; tailUb is a 32-byte-aligned UB allocation.
// Compute only tailLen valid elements and reserve aligned storage in tailUb.
const uint32_t tailLen = totalLen % tileElems;
if (tailLen > 0) {
    // UB -> GM: copyParams.blockLen uses bytes; tileElems uses elements.
    // No GM padding is written.
    const uint32_t tailBytes = tailLen * sizeof(T);
    AscendC::DataCopyExtParams copyParams{1, tailBytes, 0, 0, 0};
    AscendC::DataCopyPad(tailGm, tailUb, copyParams);
}
```

多核场景：逐核打印/检查每个核的 [start, len)，确认末核 len 未越界、无重叠。

## 反例（像但不是）

- 对齐 shape 也错 → 不是尾块，回 Step 4 重新匹配。
- 误差周期性出现在每块末尾（不只全局尾部）→ 更像块内处理或 broadcast 步长问题。
