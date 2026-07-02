# Kernel Qgmm Mx
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_qgmm_mx.h)

## 功能说明
MX 量化 Grouped Matmul 的 Kernel 组件，基于 Tensor API 实现，仅支持 AIC 计算。
组件负责 group list 解析、group 间 `m/n/k` 更新、A/B/Scale/Bias/C 的 GM 偏移维护，以及 tail split 调度。

**实现框架参考**：[Kernel 公共框架](./kernel.md)

## 特殊约束

### 计算模式
仅支持 AIC，不支持 AIV 参与主计算流程。

### Block 依赖
通常与 `BlockMmad<GroupedMatmulWithScaleMx<...>, ...>` 搭配使用。

### 调度器限制
使用 `BlockSchedulerGmmSwatWithTailSplit` 负责 grouped matmul 的 tile 分发和 tail split。

### Scale 类型
ScaleA 和 ScaleB 固定按 `fp8_e8m0_t` 解释。

## 特殊类型别名

| 别名 | 含义 |
|------|------|
| `ProblemShape` | 整体问题规模 |
| `TupleShape` | 当前 group 的问题规模 |
| `BlockShape` | 单 tile 形状 |
| `SchedulerShape` | scheduler 使用的问题规模 |
| `BlockCoord` | tile 坐标 |

## 特殊数据结构

### GMMTiling
```cpp
struct GMMTiling {
    uint32_t groupNum;
    int64_t m;
    int64_t n;
    int64_t k;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t kAL1;
    uint32_t kBL1;
    uint32_t scaleKAL1;
    uint32_t scaleKBL1;
    uint8_t isBias;
    uint8_t dbL0C;
    int8_t groupType;
    uint8_t groupListType;
};
```

参数说明：

| 参数 | 说明 |
|------|------|
| `groupNum` | group 数量 |
| `m/n/k` | 初始问题规模 |
| `baseM/baseN/baseK` | 基础 tile 大小 |
| `kAL1/kBL1` | A/B 的 L1 K 轴切分 |
| `scaleKAL1/scaleKBL1` | ScaleA/ScaleB 的 L1 K 轴切分 tiling 字段；在 MX 量化中两者必须一致，并作为 BlockMmad 的共享 `scaleKL1` |
| `isBias` | 是否启用 bias |
| `dbL0C` | L0C 双缓冲模式；当前仅值 `2` 启用，其余值均视为禁用 |
| `groupType` | GMM tiling 兼容保留字段；当前 kernel 未读取该字段，split 方向由 `LayoutA` 对应的 `transA` 编译期路径决定（`!transA` 按 M，`transA` 按 K） |
| `groupListType` | offset、length 或 sparse |

`kAL1/kBL1/scaleKAL1/scaleKBL1` 需满足 BlockMmad 的 L1 参数约束：`kAL1` 与 `kBL1` 的较大值作为外层 K 窗口，较大值需为较小值的整数倍；tiling 需传入一致的 `scaleKAL1` 和 `scaleKBL1`，且不小于该外层窗口，并为该外层窗口的整数倍。Kernel 使用 `scaleKAL1` 作为共享 `scaleKL1`。

### Params
```cpp
struct Params {
    ProblemShape problemShape;
    BlockMmadParams mmadParams;
    BlockEpilogueParams epilogueParams;
    GM_ADDR groupListGmAddr;
    GMMTiling gmmParams;
};
```

## 特殊成员方法

### operator() 函数
```cpp
__aicore__ inline void operator()(const Params& params)
```

功能：
- 作为 kernel 入口，内部调用 `Run(params)`

### Init 函数
```cpp
__aicore__ inline void Init(const Params& params)
```

功能：
- 初始化 GM 基地址
- 读取 grouped matmul tiling 参数
- 构造首个 group 的 `problemShape_`

### Run 函数
```cpp
__aicore__ inline void Run(const Params& params)
```

功能：
- 遍历所有 group
- 逐 group 更新 `m/n/k`
- 逐 group 更新地址偏移
- 调用 scheduler 和 block 完成 tile 级计算

### SetMNK 函数
```cpp
__aicore__ inline void SetMNK(uint32_t groupIdx)
```

功能：
- 从 group list 中提取当前组切分值
- 当前 QGMM MX scalar 路径按 `LayoutA` 对应的 `transA` 编译期路径更新当前组的 `problemShape_`；`groupType` 在当前 kernel 中未参与该判断

### UpdateBaseOffsets 函数
```cpp
__aicore__ inline void UpdateBaseOffsets(uint32_t groupIdx)
```

功能：
- 根据上一组的 `m/n/k` 更新 A/B/ScaleA/ScaleB/Bias/C 偏移
- 兼容普通 group list 与 sparse group list

### ProcessSingleGroup 函数
```cpp
template <bool isLastGroupAndNeedSplit>
__aicore__ inline void ProcessSingleGroup(BlockScheduler& scheduler, uint32_t groupIdx)
```

功能：
- 构造当前 group 的 Tensor API GM Tensor
- 获取每个 tile 的坐标和形状
- 将 slice 后的 tensor 交给 `BlockMmad`

## 调用示例

### 组件组装
```cpp
using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
using DispatchPolicy = Blaze::Gemm::GroupedMatmulWithScaleMx<0>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;

using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
using QgmmKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

### 参数准备
```cpp
using Params = typename QgmmKernel::Params;

Params params = {
    {m, n, k, 0},
    {aGmAddr, bGmAddr, cGmAddr, biasGmAddr, scaleAGmAddr, scaleBGmAddr},
    {},
    groupListGmAddr,
    {groupNum, m, n, k, baseM, baseN, baseK, kAL1, kBL1, scaleKAL1, scaleKBL1,
     isBias, dbL0C, groupType, groupListType}
};
```

### Kernel 执行
```cpp
QgmmKernel kernel;
kernel(params);
```

## 调度流程
```text
读取首组 tiling
    -> 初始化 block scheduler
    -> 遍历 group
    -> 更新当前组的 m/n/k
    -> 更新 A/B/Scale/Bias/C 偏移
    -> scheduler 分发 tile
    -> block 执行单 tile 计算
    -> 末组按需执行 tail split
```

## 适用场景
- MX 量化 grouped matmul
- group 间 `m` 或 `k` 动态变化
- 需要 sparse group list / tail split 的 grouped matmul 场景
