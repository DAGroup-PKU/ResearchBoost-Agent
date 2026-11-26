---
name: torch-ascendc-op-extension
description: Connect an existing Ascend C direct-launch or ACLNN operator to PyTorch with an appropriate integration route.
---

# torch-ascendc-op-extension

Decide whether the source operator is a direct-launch kernel or an ACLNN registered operator. Follow the matching route; preserve schema, stream and output contracts.

Read and apply the skill workflow; do not invoke the skill itself as a Bash command. Commands inside a selected workflow can be run in their documented environment. Use the task's numerical and performance verdicts to validate any change.

## Contents

- [昇腾算子 PyTorch 接口（PTA）开发入口](#昇腾算子-pytorch-接口pta开发入口)
- [路线 A：Ascend C `<<<>>>` 直调工程的 TORCH_LIBRARY 对接](#路线-aascend-c--直调工程的-torch_library-对接)
- [路线 B：aclnn 注册算子的 PTA（PyTorch Adapter）接口开发](#路线-baclnn-注册算子的-ptapytorch-adapter接口开发)

## 昇腾算子 PyTorch 接口（PTA）开发入口

本 skill 只负责给**已能跑通的算子工程**加 PyTorch 接入层。算子工程按调用形态分两类，PTA 接口的做法完全不同且**不可混用**，因此先判定路线，再执行对应子流程。

### 共性前置条件

- 已有可构建/可运行的算子工程（编译产物或 `.run` 包已就绪）
- 环境已安装 torch、torch_npu；CANN 环境已 `source set_env.sh`（`ASCEND_HOME_PATH` 已设置）

### Step R：路线判定（必须先做，不许猜）

按证据判定，优先看工程里有什么文件、导出了什么符号：

| 证据（任一命中） | 路线 |
|---|---|
| kernel 源码里有 `<<<...>>>` 调用；有 `.asc` / `.cce` kernel 文件；CMake 直接编译 kernel 源码；已有 `op_extension/` 或 `TORCH_LIBRARY_FRAGMENT(npu, ...)` | **路线 A：kernel 直调** |
| 有 `op_host/` OpDef（`OP_ADD(...)`）+ `op_host/op_api/aclnn_<op>.h`；`build.sh` 产出 `custom_opp_<arch>.run`；`libcust_opapi.so` 导出 `aclnnXxxGetWorkspaceSize` / `aclnnXxx`；工程里没有 `<<<>>>` | **路线 B：aclnn 注册** |

取证命令（在算子工程根执行）：

```bash
grep -rn '<<<' --include='*.asc' --include='*.cpp' --include='*.cce' . | head   # 命中 → 路线 A
ls op_host/ op_host/op_api/ 2>/dev/null                                        # 有 OpDef + aclnn_*.h → 路线 B
nm -D $ASCEND_CUSTOM_OPP_PATH/../../lib64/libcust_opapi.so 2>/dev/null | grep aclnn | head
```

**判定规则**：

- 两类证据都不命中 → 算子工程还没建好，请先参考直调或注册算子工程的相关资料建立工程，再适配 PTA 层
- 两类证据都命中时，先根据任务要求和已有调用路径选择；若仍无法判断要接入哪一套，再询问用户。
- 用户只给了关键词没给工程：`TORCH_LIBRARY` / `torch.ops.npu` / `stream(true)` / `.so` 加载 → 倾向路线 A；`EXEC_NPU_CMD_V1` / `xops` / `NpuExtension` / `torchair converter` / `torch_npu.xxx` → 倾向路线 B；仍需向用户确认后再进入

### 路由表

| 路线 | 适用工程 | 子流程文件 |
|---|---|---|
| A：kernel 直调 | Ascend C `<<<>>>` 直调工程 | [直调子流程](#路线-aascend-c--直调工程的-torch_library-对接)（Step 0–6） |
| B：aclnn 注册 | aclnn 注册算子工程（`.run` + `libcust_opapi.so`） | [ACLNN 子流程](#路线-baclnn-注册算子的-ptapytorch-adapter接口开发)（Step 0–8） |

> 使用本技能随附的 [直调模板](templates/direct-invoke/) 或 [ACLNN 模板](templates/aclnn/)，也可以复用工程中符合接口契约的实现。

### 执行规则

1. 完成 Step R 判定并向用户说明依据，再按路由表进入对应的子流程
2. 严格按该文件的 Step 顺序执行，**只使用本路线子目录**下的资源：路线 A 用 [直调模板](templates/direct-invoke/)，路线 B 用 [ACLNN 模板](templates/aclnn/)，辅助资料见资源索引
3. **禁止跨路线取材**：把直调的 `stream(true)` / CMake 那套搬进 aclnn 路线会丢掉入 queue 语义，把 aclnn 的 `EXEC_NPU_CMD_V1` 搬进直调路线会因找不到 kernel 符号而链接失败
4. 排查问题或需要完整映射表时，按需读取资源索引中的辅助资料

### 两条路线红线对照

| | 路线 A：kernel 直调 | 路线 B：aclnn 注册 |
|---|---|---|
| 产物 | `libxxx_ops.so`，`torch.ops.load_library` 加载 | `xops` wheel，`import xops` 加载 |
| 构建 | CMake，编译 `.asc` kernel | `setup.py` + `NpuExtension`，**不编译 kernel** |
| 下发 | 函数调用 kernel + `stream(true)` 手动清 queue | `EXEC_NPU_CMD_V1` → `RunOpApiV2` 入 queue |
| 命名空间 | `torch.ops.npu.xxx` | `torch.ops.custom.xxx` / `torch_npu.xxx` |
| 图模式 | aclgraph（NPUGraph / `make_graphed_callables`），无 GE 入图 | torchair FX→GE converter → GE op_type |
| 共性 | 都必须注册 Meta backend（`torch.compile` / fx 追踪必需）；都不写 autograd 反向 | 同左 |

### 资源索引

| 资源 | 路径 | 说明 |
|---|---|---|
| 路线 A 子流程 | [直调子流程](#路线-aascend-c--直调工程的-torch_library-对接) | TORCH_LIBRARY 对接 Step 0–6 |
| 路线 B 子流程 | [ACLNN 子流程](#路线-baclnn-注册算子的-ptapytorch-adapter接口开发) | xops wheel + dlsym Step 0–8 |
| 路线 A 模板 | [templates/direct-invoke/](templates/direct-invoke/) | `ops_template.h`、`torch_template.cpp`、`register_template.cpp`、`CMakeLists_template.cmake` |
| 路线 B 模板 | [templates/aclnn/](templates/aclnn/) | `setup.py`、`package_init.py`、`converter_*.py`、`csrc/*`（`ops_common.{h,cpp}` 原样复制勿改） |
| 路线 A 参考 | [references/direct-invoke/anti_patterns.md](../upstream.md#torch-ascendc-op-extension-direct-invoke-anti-patterns-md)、[operation_checklist.md](../upstream.md#torch-ascendc-op-extension-direct-invoke-operation-checklist-md)、[troubleshooting.md](../upstream.md#torch-ascendc-op-extension-direct-invoke-troubleshooting-md) | stream 反模式、适配清单、踩坑 |
| 路线 B 参考 | [references/aclnn/schema_mapping.md](../upstream.md#torch-ascendc-op-extension-aclnn-schema-mapping-md)、[ops_common_internals.md](../upstream.md#torch-ascendc-op-extension-aclnn-ops-common-internals-md)、[troubleshooting.md](../upstream.md#torch-ascendc-op-extension-aclnn-troubleshooting-md) | schema/OpDef/aclnn 对齐、样板内部机制、踩坑 |

### 相邻 skill

- aclnn 注册算子工程从零搭建：参考注册算子工程相关资料
- 注册调用改直调：参考注册调用与直调接口迁移相关资料
- MC2/HCCL 通算融合算子的路线选择与精度判据：[ascendc-mc2-best-practice](../ascendc-mc2-best-practice/SKILL.md)

## 路线 A：Ascend C `<<<>>>` 直调工程的 TORCH_LIBRARY 对接

> 进入本路线的前提是已在 the skill entrypoint 完成路线判定：源工程是 `<<<>>>` 直调工程。若工程实际是 aclnn 注册算子（`op_host/` OpDef + `.run` 包 + `libcust_opapi.so`），退回 the skill entrypoint 改走 [aclnn-registry.md](#路线-baclnn-注册算子的-ptapytorch-adapter接口开发)。**两条路径不可混用。**

### 前置条件

- 已有可编译运行的 Ascend C `<<<>>>` 直调工程
- 环境已安装：torch、torch_npu
- CANN 环境已配置（`ASCEND_HOME_PATH` 已设置）

### 目标产出

无论源工程是什么目录结构，改造后都需要以下文件：

```
├── op_kernel/
│   ├── xxx_tiling.h         (Tiling 结构体 + 常量，kernel 和 host 共用)
│   └── xxx_kernel.asc       (纯 kernel 代码，引用 xxx_tiling.h)
├── op_host/
│   ├── xxx.asc              (host+main 可执行文件入口，引用 xxx_kernel.asc)
│   └── data_utils.h         (文件读写工具，位置以 include 路径可达为准)
├── op_extension/
│   ├── xxx_torch.cpp        (PyTorch 接入层：torch::Tensor 接口 + Tiling 计算 + kernel launch)
│   ├── register.cpp         (TORCH_LIBRARY 注册，含 Meta backend)
│   └── ops.h                (函数声明)
└── CMakeLists.txt           (双 target：可执行文件 + libxxx_ops.so)
```

**已存在的文件保留不动，缺失的按 Step 逐个补齐。**

### 架构原理

```
文件依赖:
  op_kernel/xxx_tiling.h              (单一数据源：Tiling 结构体 + 常量)
    ├── op_kernel/xxx_kernel.asc      (#include "xxx_tiling.h")
    │   └── op_host/xxx.asc           (#include "../op_kernel/xxx_kernel.asc"，间接引入)
    └── op_extension/xxx_torch.cpp (#include "../op_kernel/xxx_tiling.h")

Python 调用链:
  torch.ops.npu.xxx(x1, x2)
      │
      ▼ PyTorch dispatch（PrivateUse1 后端）
  ascend_kernel::xxx_torch()         ← op_extension/xxx_torch.cpp 实现
      │
      ├─ ComputeTiling()             ← 计算 Tiling 参数
      ├─ aclrtMemcpy(tiling→device)  ← Tiling 搬到 device
      └─ xxx_kernel(...)             ← 函数调用（等价于 <<<>>>）
            │
            ▼ NPU 执行
```

### Stream 同步模式

> **核心背景**：NPU 的 taskqueue 是设备端任务队列。"清 queue"指等待队列中已有任务完成后再执行当前任务；"入 queue"指将当前任务放入队列中按顺序执行。

本方案采用 CMake 多文件架构，`xxx_torch.cpp` 由 C++ 编译器编译（非 ASC），因此**不能使用 `<<<>>>` 语法**，只能将 kernel 作为普通 C 函数调用。在这种架构下，必须手动管理 stream 同步。

#### 推荐模式：`stream(true)` 清 queue

```cpp
auto aclStream = c10_npu::getCurrentNPUStream().stream(true);
xxx_kernel(blockNum, nullptr, aclStream, ...);
```

`stream(true)` 在返回 aclrtStream 前会清 queue（等待之前任务完成），确保不会乱序。输出分配用 `at::empty_like`（`zeros_like` 会触发 NPU 操作，可能乱序）。

#### 禁止模式

| 反模式 | 风险 | 修复 |
|--------|------|------|
| `stream(false)` + 直接调用 | **乱序**：不清 queue，kernel 先于之前操作执行 | 改用 `stream(true)` |
| lambda 内传 NPUStream + OpCommand | **死锁**：queue 等 lambda，lambda 等 queue 空 | lambda 内改用 `aclrtStream` |
| `zeros_like` 创建输出 | **乱序**：zeros_like 入 queue 但 kernel 不入 queue | 改用 `empty_like` |

#### 8 种启动方式速查

| # | 方式 | 判定 |
|---|------|------|
| 1 | 传 NPUStream 对象 + `<<<>>>` | **推荐**（单文件 .asc） |
| 2 | `stream(true)` + `<<<>>>` / 函数调用 | **正确**（**本方案唯一适用**） |
| 3 | `stream(false)` + OpCommand + `<<<>>>` | **推荐**（单文件 .asc） |
| 4 | `stream(false)` + 直接启动 | **反模式：乱序** |
| 5 | NPUStream + OpCommand | **反模式：死锁** |
| 6 | `stream(true)` + OpCommand | **正确**（单文件 .asc） |
| 7 | `stream(true)` + `zeros_like` | **反模式：乱序** |
| 8 | `stream()` 无参数 | **正确**（等价方式2） |

每种模式的完整代码、原理和死锁流程见 [references/direct-invoke/anti_patterns.md](../upstream.md#torch-ascendc-op-extension-direct-invoke-anti-patterns-md)。

### 改造步骤

#### Step 0: 拆分 kernel 与 host（仅单文件 .asc 工程需要）

**跳过条件**：kernel 和 host 已在不同文件中。

对单文件 `.asc` 工程（kernel + host + main 在一个文件里），执行以下操作：

1. **提取 `op_kernel/xxx_tiling.h`**：从原 `.asc` 中提取以下内容到新文件 `op_kernel/xxx_tiling.h`：
   - 命名常量（`constexpr`，如 `TILE_LENGTH`、`DOUBLE_BUFFER`）
   - Tiling 数据结构体（`struct XxxTilingData`）
   
   `xxx_tiling.h` 只含纯 C/C++ 语法，不含 `__aicore__`、`__gm__` 等 ASC 关键字。所有编译单元（kernel、host main、torch host）共用此文件。

2. **提取 `op_kernel/xxx_kernel.asc`**：从原 `.asc` 中剪切以下内容到新文件 `op_kernel/xxx_kernel.asc`：
   - `#include "kernel_operator.h"` 及 ASC 专用头文件
   - `#include "xxx_tiling.h"`（替代原有的内联常量和结构体）
   - Kernel 类（`class KernelXxx`）
   - 核函数入口（`extern "C" __global__ __vector__ void xxx_kernel(...)`）
   
   `xxx_kernel.asc` 只含 ASC 语法，不含任何 host 代码（main、文件读写、iostream 等）。

3. **改造原 `.asc`**（移至 `op_host/` 目录）：删除已剪切的 kernel 内容，改为：
   ```cpp
   #include "../op_kernel/xxx_kernel.asc"    // 引用拆出的纯 kernel 文件
   // ... 以下保留原有的 host 侧代码（main 等），不做任何修改
   ```

4. **确保 include 路径正确**：`op_host/xxx.asc` 依赖的辅助文件（如 `data_utils.h`）需在 CMakeLists.txt 的 `target_include_directories` 覆盖的路径内可被找到。

5. **清理旧文件**：删除已被替代的原始 `.asc` 文件（拆分后不再被编译引用），避免留下孤立文件混淆维护。

6. **验证**：编译可执行文件目标，确认拆分未破坏功能。

#### Step 1: 创建 `op_extension/ops.h`（函数声明）

**已存在则跳过。**

读取模板：[templates/direct-invoke/ops_template.h](templates/direct-invoke/ops_template.h)

在 `ascend_kernel` 命名空间中声明算子函数签名，供 `op_extension/register.cpp` 和 `op_extension/xxx_torch.cpp` 共用。

#### Step 2: 创建 `op_extension/xxx_torch.cpp`（PyTorch 接入层实现）

这是核心文件。读取源工程代码后，基于模板改造：[templates/direct-invoke/torch_template.cpp](templates/direct-invoke/torch_template.cpp)

模板使用 `#include "../op_kernel/xxx_tiling.h"` 引入 Tiling 结构体和常量（单一数据源）。`ComputeTiling` 函数是**示例占位**，应替换为源工程的实际计算逻辑。

**关键设计决策**：
- `#include "acl/acl.h"` 放在 torch 头文件**之前**——torch 的宏会干扰 ACL 头文件解析
- `#include "../op_kernel/xxx_tiling.h"` 引入 Tiling 结构体和常量
- 参数校验用 `x1.is_privateuseone()`（C++ API），不是 `is_npu()`
- 框架 stream：**必须用 `stream(true)`**（清 queue，防乱序），**禁止 `stream(false)`**。详见上方「Stream 同步模式」章节

**适配指引**：Tiling 计算、kernel 入口声明、kernel 调用方式等需从源码提取。详见 [references/direct-invoke/operation_checklist.md](../upstream.md#torch-ascendc-op-extension-direct-invoke-operation-checklist-md)。

#### Step 3: 创建 `op_extension/register.cpp`（算子注册）

**已存在则跳过。**

读取模板：[templates/direct-invoke/register_template.cpp](templates/direct-invoke/register_template.cpp)

必须包含三部分注册：

```cpp
// 1. 算子签名定义
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("xxx(Tensor x1, Tensor x2) -> Tensor");
}

// 2. NPU 后端实现绑定
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("xxx", TORCH_FN(ascend_kernel::xxx_torch));
}

// 3. Meta 后端注册（torch.compile / fx 必需）
at::Tensor xxx_meta(const at::Tensor& x1, const at::Tensor& x2) {
    return at::empty_like(x1);  // 简单算子：直接推导；多输出/shape变化需按实际逻辑
}
TORCH_LIBRARY_IMPL(npu, Meta, m) {
    m.impl("xxx", &xxx_meta);
}
```

**namespace 选择**：默认用 `npu`（`torch.ops.npu.xxx`）。如需避免与 torch_npu 内建算子冲突，可改用 `ascendc_ops`（`torch.ops.ascendc_ops.xxx`）。

#### Step 4: 更新 CMakeLists.txt

读取模板：[templates/direct-invoke/CMakeLists_template.cmake](templates/direct-invoke/CMakeLists_template.cmake)

双目标结构：保留原有可执行文件 Target，新增 `add_library(xxx_ops SHARED ...)` 编译 `.so`。`set(CMAKE_CXX_STANDARD 17)` 不可省略。`npu-arch` 按实际芯片修改：A2/A3=dav-2201, A5=dav-3510。

kernel 文件形态不同时的 CMakeLists 处理方式见 [references/direct-invoke/operation_checklist.md](../upstream.md#torch-ascendc-op-extension-direct-invoke-operation-checklist-md)。

**注意**：如果 CMakeLists.txt 中已有 PyTorch 相关的 Target，直接替换为新的 Target 2 配置，不要保留旧目标。

#### Step 5: 编译验证

```bash
cmake -S . -B build && cmake --build build -j4
```

编译成功后会产出：
- `build/<operator_name>_custom` — 原有可执行文件（直调验证）
- `build/libxxx_ops.so` — PyTorch 扩展库

#### Step 6: Python 测试

```python
import torch
import torch_npu

torch.ops.load_library("build/libxxx_ops.so")

# 根据算子签名调整输入数量、dtype 和验证逻辑
x1 = torch.randn(1024, 4096, dtype=torch.float32).npu()
x2 = torch.randn(1024, 4096, dtype=torch.float32).npu()

y = torch.ops.npu.xxx(x1, x2)    # 单输入算子改为 torch.ops.npu.xxx(x1)

assert y.is_npu                    # 注意：Python 中是 property 不是方法
assert torch.allclose(y.cpu(), x1.cpu() + x2.cpu())  # 验证逻辑按算子语义调整
```

多算子场景加载多个 `.so` 即可：

```python
torch.ops.load_library("/path/to/libadd1_ops.so")
torch.ops.load_library("/path/to/libmul1_ops.so")

y1 = torch.ops.npu.add1(x1, x2)
y2 = torch.ops.npu.mul1(x1, x2)
```

### 多算子扩展

一个 `.so` 可注册多个算子：

```
register.cpp:
  TORCH_LIBRARY_FRAGMENT(npu, m) {
      m.def("op_a(Tensor x1, Tensor x2) -> Tensor");
      m.def("op_b(Tensor x1, Tensor x2) -> Tensor");
  }
  TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
      m.impl("op_a", TORCH_FN(ascend_kernel::op_a));
      m.impl("op_b", TORCH_FN(ascend_kernel::op_b));
  }
  TORCH_LIBRARY_IMPL(npu, Meta, m) {
      m.impl("op_a", &op_a_meta);
      m.impl("op_b", &op_b_meta);
  }
```

### aclgraph 兼容

已注册 TORCH_LIBRARY 的自定义算子天然支持 aclgraph，无需额外改造。使用方式：

```python
# 方式1: NPUGraph
g = torch.npu.NPUGraph()
with torch.npu.graph(g):
    output = model(static_x, static_y)
g.replay()

# 方式2: make_graphed_callables
model = torch.npu.make_graphed_callables(model, (x, y))

# 方式3: npugraph_ex backend (需 torch_npu >= 7.3.0)
compiled = torch.compile(model, backend="npugraph_ex", fullgraph=True)
```

**前提**：算子已正确注册 Meta backend（见 Step 3），且 stream 同步使用 `stream(true)` 模式。

### 踩坑清单

遇到编译/链接/运行时问题时，读取 [references/direct-invoke/troubleshooting.md](../upstream.md#torch-ascendc-op-extension-direct-invoke-troubleshooting-md)。

### 参考资源

- aclnn 注册算子的 PTA 对接（另一条路线）：[aclnn-registry.md](#路线-baclnn-注册算子的-ptapytorch-adapter接口开发)
- 官方 aclgraph + TORCH_LIBRARY 示例：`torchair/examples/_kernel_extension_aclgraph/torch_library/`
- 官方 TORCH_LIBRARY 示例：`$ASC_DEVKIT_DIR/examples/01_simd_cpp_api/02_features/00_framework/00_pytorch/torch_library/`
- PyTorch 框架适配文档：`$ASC_DEVKIT_DIR/docs/zh/guide/programming_guide/advanced_programming/ai_framework_adaptation/pytorch_framework.md`
- PyTorch Custom Operators：https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html

## 路线 B：aclnn 注册算子的 PTA（PyTorch Adapter）接口开发

> 进入本路线的前提是已在 the skill entrypoint 完成路线判定：源工程是 aclnn 注册算子（无 `<<<>>>`，kernel 二进制在 `.run` 包里，PTA 层只 dlsym 符号、不编译 kernel）。若工程实际是 Ascend C `<<<>>>` 直调工程，退回 the skill entrypoint 改走 [direct-invoke.md](#路线-aascend-c--直调工程的-torch_library-对接)。**两条路径不可混用**：照搬直调那套会因为找不到 kernel 符号而链接/运行失败。

### 前置条件

- 已有**可构建的 aclnn 注册算子工程**：`op_host/`（OpDef + tiling + `op_api/aclnn_xxx.{h,cpp}`）+ `op_kernel/`，`bash build.sh` 能产出 `custom_opp_<arch>.run`
- `.run` 已安装，`libcust_opapi.so` 导出 `aclnnXxxGetWorkspaceSize` / `aclnnXxx` 两段式符号
- 环境已安装：torch、torch_npu、torchair（入图需要）
- CANN 环境已 `source set_env.sh`（`ASCEND_HOME_PATH`、`ASCEND_OPP_PATH` 已设置）

### 目标产出

```
torch_ops_extension/
├── setup.py                          # 构建 xops wheel（NpuExtension）
├── build_and_install.sh              # build bdist_wheel + pip3 install -I
└── xops/
    ├── __init__.py                   # import xops_lib；镜像 torch.ops.custom→torch_npu
    ├── converter/
    │   ├── __init__.py
    │   └── npu_x_custom_op.py        # torchair FX→GE converter（图模式）
    └── csrc/
        ├── ops_common.h              # 样板：EXEC_NPU_CMD_V1 + ConvertType + dlopen/dlsym（原样复制，勿改）
        ├── ops_common.cpp            # 样板：GetOpApiFuncAddrFromFeatureLib（原样复制，勿改）
        ├── ops_def_registration.cpp  # TORCH_LIBRARY(custom) schema 定义 + 空 pybind
        └── npu_x_custom_op.cpp       # NPU impl + Meta impl + TORCH_LIBRARY_IMPL
```

目录放在算子工程根下（与 `op_host/`、`op_kernel/`、`examples/` 平级）。**已存在的文件保留不动，缺失的按 Step 逐个补齐。**

### 架构与调用链

```
import xops ──▶ xops/__init__.py: from . import xops_lib   (加载 xops.xops_lib.*.so)
                    │  静态初始化执行 TORCH_LIBRARY 注册：
                    │   ① ops_def_registration.cpp: TORCH_LIBRARY(custom){ m.def("npu_x_custom_op(...)"); }
                    │   ② npu_x_custom_op.cpp:      TORCH_LIBRARY_IMPL(custom, PrivateUse1, ...)  绑 NPU
                    │                               TORCH_LIBRARY_IMPL(custom, Meta, ...)         绑 Meta
                    │   ③ __init__.py: 把 torch.ops.custom.* 镜像到 torch_npu 命名空间
                    ▼
eager:  torch.ops.custom.npu_x_custom_op(...)  ≡  torch_npu.npu_x_custom_op(...)
                    │  dispatch PrivateUse1
                    ▼  EXEC_NPU_CMD_V1(aclnnXCustomOp, ...)
                    │  dlsym aclnnXCustomOp{GetWorkspaceSize,} ← libcust_opapi.so
                    │  → ConvertType(torch Tensor→aclTensor*) → 查 workspace → 分配 → RunOpApiV2 下发
                    ▼  kernel 执行（二进制来自 .run 包）

图模式: torch.compile(backend=npu_backend) → FX 节点
                    │  @register_fx_node_ge_converter
                    ▼  torchair.ge.custom_op("XCustomOp", inputs, attrs, outputs) → GE → 同一 aclnn
```

### 命名红线

> **aclnn 导出符号与 GE op_type 都必须带自定义前缀**（本文统一用 `X`，如 `XAllGatherMatmul` / `aclnnXAllGatherMatmul`）。

`GetOpApiFuncAddr` 按裸符号名 dlsym，虽然搜索序里 `libopapi.so`（CANN 内置）排在自定义库之后，但 **GE 侧的 op_type 注册会被内置同名实现抢占**。症状极具迷惑性：调用成功、结果"正确"、kernel 却从未执行（新加的 printf 不打印、故意写坏 kernel 也不报错）。

三处名字必须严格对齐：

| 名字 | 出现位置 |
|------|---------|
| GE op_type `XCustomOp` | OpDef 的 `OP_ADD(XCustomOp)`、converter 的 `torchair.ge.custom_op("XCustomOp", ...)` |
| aclnn 符号 `aclnnXCustomOp` | `op_api/aclnn_xxx.h` 导出、impl cpp 的 `EXEC_NPU_CMD_V1(aclnnXCustomOp, ...)` |
| torch 算子名 `npu_x_custom_op` | schema `m.def`、`m.impl`、converter 的 `torch.ops.custom.npu_x_custom_op.default`、`__init__.py` 的 import |

### 改造步骤

#### Step 0: 采集算子契约

读取三份源文件，填出一张契约表（后续每个 Step 都从这张表取值）：

| 采集项 | 来源 |
|--------|------|
| aclnn 形参列表与顺序、可选参数、dtype | `op_host/op_api/aclnn_<op>.h` 的 `aclnnXxxGetWorkspaceSize` 声明 |
| GE op_type、Input/Output 名与顺序、Attr 名与类型/默认值 | `op_host/<op>_def.cpp` 的 OpDef |
| 输出 shape 是否可由框架推导 | `op_host/<op>_proto.cpp` 的 infershape |

**关键注意**：aclnn 形参名与 OpDef 的 Input/Attr 名常常不一致（如 aclnn `gatherIndex` ↔ OpDef `gather_index`；aclnn `output` ↔ OpDef `y`）。converter 里的 key 用 **OpDef 的名字**，`EXEC_NPU_CMD_V1` 的实参顺序用 **aclnn 的顺序**。两者的对齐规则见 [references/aclnn/schema_mapping.md](../upstream.md#torch-ascendc-op-extension-aclnn-schema-mapping-md)。

#### Step 1: 复制 `xops/csrc/ops_common.{h,cpp}`

从 [templates/aclnn/csrc/ops_common.h](templates/aclnn/csrc/ops_common.h) 和 [templates/aclnn/csrc/ops_common.cpp](templates/aclnn/csrc/ops_common.cpp) **原样复制，不要修改**。这是从 torch_npu 内部适配层摘出的通用样板，提供：

- `EXEC_NPU_CMD_V1(aclnn_api, ...)` 宏：dlsym 两段式符号 → `ConvertTypes` → 查 workspace → 分配 → `OpCommand::RunOpApiV2` 入 queue
- `GetOpApiFuncAddr` 五级搜索序（`ASCEND_CUSTOM_OPP_PATH` → `ASCEND_OPP_PATH/vendors` → feature libs → `libopapi.so` → 兜底）
- `ConvertType` 重载族（torch 类型 → acl 类型）

内部机制与可用重载清单见 [references/aclnn/ops_common_internals.md](../upstream.md#torch-ascendc-op-extension-aclnn-ops-common-internals-md)，**只在排查问题时读**。

#### Step 2: 创建 `xops/csrc/ops_def_registration.cpp`（schema 定义）

模板：[templates/aclnn/csrc/ops_def_registration.cpp](templates/aclnn/csrc/ops_def_registration.cpp)

```cpp
TORCH_LIBRARY(custom, m) {
    m.def("npu_x_custom_op(Tensor x1, Tensor x2, Tensor! out) -> Tensor");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}   // 空：不直接暴露 pybind 函数
```

schema 书写要点：

- 形参顺序与 aclnn 一一对应；输出也是入参（调用方预分配），用 `Tensor!` 标记可变，**漏了 `!` 会被 `torch.compile` 当纯函数消掉**
- 可选参数写 `Tensor? x`（C++ 侧 `const c10::optional<at::Tensor>&`），默认值只能给尾部参数
- 字符串属性写 `str s`（C++ 侧 `c10::string_view`），整型 `int` ↔ `int64_t`，浮点 `float` ↔ `double`，整型数组 `int[]` ↔ `at::IntArrayRef`
- 多输出返回 `-> (Tensor a, Tensor? b)`；单输出返回 `-> Tensor`
- **只用 `TORCH_LIBRARY`（不是 `_FRAGMENT`）注册一次 `custom` 命名空间**，全部算子的 `m.def` 集中在这一个文件里，避免重复注册命名空间

完整类型映射表（含 `Scalar` / `Tensor[]` / `SymInt[]` 等）见 [references/aclnn/schema_mapping.md](../upstream.md#torch-ascendc-op-extension-aclnn-schema-mapping-md) §2。

#### Step 3: 创建 `xops/csrc/npu_x_custom_op.cpp`（NPU + Meta 实现）

模板：[templates/aclnn/csrc/op_impl.cpp](templates/aclnn/csrc/op_impl.cpp)。一个算子一个文件，包含四段：NPU impl、Meta impl、`TORCH_LIBRARY_IMPL(custom, PrivateUse1)`、`TORCH_LIBRARY_IMPL(custom, Meta)`。

```cpp
at::Tensor npu_x_custom_op(const at::Tensor& x1, const at::Tensor& x2, at::Tensor& out)
{
    EXEC_NPU_CMD_V1(aclnnXCustomOp, x1, x2, out);
    return out;
}
```

**关键设计决策**：

- **实参顺序 = `aclnnXxxGetWorkspaceSize` 的形参顺序**，末尾的 `workspaceSize` / `executor` 由宏自动补齐，**不要手写**
- **不需要管 stream**：`EXEC_NPU_CMD_V1` 内部取 `getCurrentNPUStream().stream(false)` 并经 `RunOpApiV2` 入 queue，天然有序。直调路线的 `stream(true)` 清 queue 规则在这里**不适用**，照搬反而丢掉入 queue 语义
- **输出 in-place 预分配**：`Tensor! out` 由调用方分配，C++ 侧直接透传给 aclnn，不在 PTA 层 `at::empty`；Meta impl 只 `return out;` 透传，不做 shape 推导（推导在 host infershape）
- **Meta impl 必须注册**：`torch.compile` / fx 追踪必需，缺失会报 `Meta backend not registered`
- **`c10::string_view` 没有 `ConvertType` 重载**：字符串属性要先 `char *p = const_cast<char *>(sv.data());` 再传给宏
- **forward-only**：不写 autograd `Function`、不注册 backward。需要反向时另建一个独立算子并在 Python 侧组 `autograd.Function`

可选输入 / 多输出 / 输出改由 PTA 层分配的写法，以及一个九参数 MC2 算子的完整四方对照实例，见 [references/aclnn/schema_mapping.md](../upstream.md#torch-ascendc-op-extension-aclnn-schema-mapping-md) §3、§5。

#### Step 4: 创建 `xops/converter/npu_x_custom_op.py`（图模式）

模板：[templates/aclnn/converter_op.py](templates/aclnn/converter_op.py)。用 `@register_fx_node_ge_converter(torch.ops.custom.npu_x_custom_op.default)` 装饰一个与 schema 逐参对齐（末尾额外接 `*, meta_outputs=None`）的函数，函数体把参数拆成 `inputs` / `attrs` / `outputs` 三份交给 `torchair.ge.custom_op("XCustomOp", ...)`。

- **三份名字一律取自 OpDef，不是 aclnn 形参名**：OpDef 常用 snake_case（`gather_index`），aclnn 常用 camelCase（`gatherIndex`）；OpDef 中不存在的参数（仅 aclnn 需要的，如 `streamMode`）不要放进 `attrs`
- `outputs` 取 OpDef 的 Output 名与声明顺序；可选输入为 `None` 时**不要放进 `inputs`**，否则 GE 建图报缺输入
- attr 类型：`attr.Str` / `attr.Int` / `attr.Bool` / `attr.Float` / `attr.ListInt`
- **不需要入图可跳过 Step 4**，同时删掉 `converter/` 目录与 `__init__.py` 里的 converter import

> ⚠️ **不要照抄现成 converter**：MC2 通算融合类算子的 PTA 往往只验证了 eager、没验证入图，其 converter 里常见 attr key 直接沿用 aclnn 形参名、甚至混入 OpDef 中不存在的属性，照抄会在建图期报 attr/output 不存在。一律以自己的 OpDef 为准逐项核对。

#### Step 5: 创建 `xops/__init__.py` 与 `xops/converter/__init__.py`

模板：[templates/aclnn/package_init.py](templates/aclnn/package_init.py) → `xops/__init__.py`；[templates/aclnn/converter_init.py](templates/aclnn/converter_init.py) → `xops/converter/__init__.py`（内容可为空，但**文件不能省**：`setup.py` 用 `find_packages()`，缺它就找不到 `xops.converter` 包）。

`xops/__init__.py` 做三件事：`from . import xops_lib`（触发静态注册）→ `from .converter import npu_x_custom_op`（注册 converter）→ 遍历 `torch.ops.custom` 把非 `_` 开头的算子 `setattr` 到 `torch_npu`（于是 `torch_npu.npu_x_custom_op` 可用）。

**多算子时在 `from .converter import ...` 里逐个列出**，漏掉的算子 eager 可用但入图会报 "no converter registered"。

#### Step 6: 创建 `setup.py` 与 `build_and_install.sh`

`setup.py` 用模板 [templates/aclnn/setup.py](templates/aclnn/setup.py)：

- 用 `torch_npu.utils.cpp_extension.NpuExtension`，**不是**原生 `torch.utils.cpp_extension.CppExtension/load`
- `sources` 用 glob 收 `xops/csrc/*.cpp`——**新增算子只要把 cpp 丢进 csrc 目录即可，无需改 setup.py**
- 扩展模块名 `"xops.xops_lib"`，包名 `name="xops"`（两者的 `xops` 必须与 `__init__.py` 里 `from . import xops_lib` 一致）
- `extra_compile_args` 只加一条 ACL include 路径（`PYTORCH_NPU_INSTALL_PATH/include/third_party/acl/inc`）
- **不显式 link** `libascendcl` / `libaclnn` / `libopapi`（无 `libraries=` 参数）——全部运行时 dlopen

`build_and_install.sh` 就三条命令，直接写：

```bash
#!/bin/bash
rm -rf build                       # 不删会命中旧的编译缓存
python3 setup.py build bdist_wheel
pip3 install dist/*.whl -I         # -I 不可省，否则 pip 复用已装版本
```

#### Step 7: 构建与安装

```bash
# 0) CANN 环境（必须：setup.py 顶部 import torch_npu 会连带加载 libhccl.so）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 1) 先构建并安装算子（.run + libcust_opapi.so）
bash build.sh -n <算子工程目录名>          # 产出 build/custom_opp_<arch>.run
export ASCEND_CUSTOM_OPP_PATH=<vendors/custom_opp 绝对路径>

# 2) 构建 PTA wheel 并安装
cd torch_ops_extension
bash build_and_install.sh                  # python3 setup.py build bdist_wheel && pip3 install -I dist/*.whl

# 3) 冒烟验证：schema 注册成功
python3 -c "import xops, torch, torch_npu; print(torch.ops.custom.npu_x_custom_op._schemas)"
```

> 漏了 `source set_env.sh` 时的连锁失败：`ImportError: libhccl.so: cannot open shared object file` → `dist/` 不存在 → `pip3 install *.whl` 报找不到文件。看到后者先回头查前者。

#### Step 8: Python 调用验证

```python
import torch
import xops        # 触发注册；import xops 会自动 import torch_npu
import torch_npu

x1 = torch.randn(512, 12288, dtype=torch.float16).npu()
x2 = torch.randn(12288, 3904, dtype=torch.float16).npu()
out = torch.empty(512, 3904, dtype=torch.float16).npu()   # 输出须预分配

y = torch_npu.npu_x_custom_op(x1, x2, out)    # 等价于 torch.ops.custom.npu_x_custom_op(...)
```

**通信类算子的 `group`**：PTA 层不建 `ProcessGroup`/communicator，`group` 是 HCCL 通信域名字符串，直接透传给 aclnn。通信域由 torch_npu 分布式创建：

```python
pg = dist.new_group(backend="hccl", ranks=list(range(world_size)))
hccl_group = pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
```

图模式验证（可选）：

```python
import torchair
compiled = torch.compile(model, backend=torchair.get_npu_backend(), fullgraph=True)
```

### 多算子扩展

一个 `xops` wheel 可承载多个算子，每加一个算子做四处改动：

| 位置 | 改动 |
|------|------|
| `csrc/ops_def_registration.cpp` | 同一个 `TORCH_LIBRARY(custom, m)` 块里加一条 `m.def(...)` |
| `csrc/npu_<op>.cpp` | 新增文件（NPU impl + Meta impl + 两个 `TORCH_LIBRARY_IMPL`）；setup.py 的 glob 自动收录 |
| `converter/npu_<op>.py` | 新增 converter 文件 |
| `xops/__init__.py` | `from .converter import npu_op_a, npu_op_b` 里补上新算子 |

### 精度测试

不属于本 skill 职责。多卡通信算子的容差判据（先通后算 vs 先算后通、是否需逐 rank 校验）见 [ascendc-mc2-best-practice](../ascendc-mc2-best-practice/SKILL.md)，通用精度标准见 `opsthe task-provided numerical comparison contract`。

### 踩坑清单

遇到 import / 构建 / dlsym / 结果异常问题时，读取 [references/aclnn/troubleshooting.md](../upstream.md#torch-ascendc-op-extension-aclnn-troubleshooting-md)。

最高频的一条先记住：**改了 kernel 却跑出旧行为**（printf 不打印、把 kernel 故意写坏也不报错），两个成因是 op_type 未加前缀被内置实现抢占，以及 `libcust_opapi.so` 命中了嵌套目录里的旧副本；重装 PTA wheel 对第二种无效。展开见 troubleshooting 第 14、15 条。

### 参考资源

- aclnn 注册算子工程本身的搭建：参考注册算子工程相关资料
- `<<<>>>` 直调工程的 TORCH_LIBRARY 对接（另一条路线）：[direct-invoke.md](#路线-aascend-c--直调工程的-torch_library-对接)
