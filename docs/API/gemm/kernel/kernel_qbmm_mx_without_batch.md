# Kernel Qbmm Mx Without Batch
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_qbmm_mx_without_batch.h)

## 功能说明
MX 量化单 Batch Matmul Kernel，仅支持 AIC 计算，支持 MxFP4/MxFP8 量化格式。该实现面向无 Batch 广播的标量路径，复用 `BlockMmadMX` 和 `BlockSchedulerQbmm`，裁剪多 Batch 地址偏移和广播循环。

**相关实现**：[Kernel Qbmm Mx](./kernel_qbmm_mx.md)

## 特殊约束

### 量化格式支持
支持以下量化数据类型：
- **MxFP4**：`fp4x2_e2m1_t`、`fp4x2_e1m2_t`
- **MxFP8**：`fp8_e5m2_t`、`fp8_e4m3fn_t`

### Scale 因子要求
必须提供两个 Scale 因子：
- `scaleAGmAddr`：A 矩阵的 per-token scale（`fp8_e8m0_t` 类型）
- `scaleBGmAddr`：B 矩阵的 per-group scale（`fp8_e8m0_t` 类型）

### Batch 限制
该类只处理单 Batch 输入，不包含 `batchA1`、`batchB1`、`batchC1` 等 Batch 广播参数。需要多 Batch 或广播场景时使用 `kernel_qbmm_mx.h` 中的 QBMM MX Kernel。

### 计算模式
仅支持 AIC 模式，不支持 AIV 计算（AIV 核直接返回）。

## 特殊数据结构

### Params
```
struct Params {
    ProblemShape problemShape;      // 问题 shape (m, n, k, batch)
    BlockMmadParams mmadParams;     // BlockMmad 参数（包含 GM 地址）
    L1Params l1Params;              // L1 参数（kL1, scaleKL1, l1BufNum）
    BlockSchedulerParams schParams; // scheduler 参数
    QBMMTiling qbmmParams;          // Without Batch QBMM 特有参数
};
```

### QBMMTiling
```
struct QBMMTiling {
    uint32_t baseM;   // L0 tile M 维度
    uint32_t baseN;   // L0 tile N 维度
    uint32_t baseK;   // L0 tile K 维度
    uint32_t isBias;  // 是否启用 bias
    uint32_t dbL0C;   // L0C 双缓冲标志
};
```

### BlockMmadParams
```
struct Params {
    GM_ADDR aGmAddr;      // A 矩阵 GM 地址
    GM_ADDR bGmAddr;      // B 矩阵 GM 地址
    GM_ADDR cGmAddr;      // C 矩阵 GM 地址
    GM_ADDR biasGmAddr;   // Bias GM 地址（可选）
    GM_ADDR scaleAGmAddr; // A 矩阵 Scale GM 地址
    GM_ADDR scaleBGmAddr; // B 矩阵 Scale GM 地址
};
```

## 特殊成员方法

### Run函数
```
__aicore__ inline void Run(const Params& params)
```
功能：执行单 Batch MX 量化矩阵乘。

执行流程：
1. AIV 核直接返回
2. 配置 Atomic Add（可选）
3. 创建 GM Tensor 和 BlockScheduler
4. 初始化 BlockMmadMX
5. 按 tile 循环调用 BlockMmadMX
6. 清理 Atomic Add（可选）

### SetL2Cache函数
```
template <typename TensorB, typename TensorC>
__aicore__ inline void SetL2Cache(
    const ProblemShape& problemShape,
    uint64_t baseM, uint64_t baseN,
    TensorB& gmB, TensorC& gmC)
```
功能：动态配置 L2 Cache。

说明：
- M tile 覆盖完整 M 维时，根据 B 的布局和对齐情况配置 B 的 L2 Cache
- 当前实现不配置 ScaleB 的 L2 Cache hint，ScaleB 使用 Tensor API 默认 Cache 策略
- Atomic Add 模式禁用 C 的 L2 Cache

## 调用示例

```
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<
    A_FULL_LOAD_MODE, false, Blaze::Gemm::KernelMmadWithScaleMxWithoutBatch>;

using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;

using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, void, BlockScheduler>;

using Params = typename QBMMKernel::Params;
Params params = {
    {m, n, k, 1},                         // problem shape
    {aGM, bGM, cGM, biasGM, scaleAGM, scaleBGM},
    {kL1, scaleKL1, l1BufNum},
    {baseM, baseN, mTailTile, nTailTile, mBaseTailSplitCnt, nBaseTailSplitCnt, mTailMain, nTailMain},
    {baseM, baseN, baseK, isBias, dbL0C}
};

QBMMKernel kernel;
kernel(params);
```

## 适用场景

- 单 Batch MX 量化推理
- 不需要 Batch 广播的标量路径
- 希望减少多 Batch 分支和地址偏移开销的 QBMM MX 场景
