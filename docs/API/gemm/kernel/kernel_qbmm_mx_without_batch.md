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

```cpp
struct Params {
    ProblemShape problemShape;      // 问题 shape (m, n, k, batch)
    BlockMmadParams mmadParams;     // BlockMmad 参数（包含 GM 地址）
    L1Params l1Params;              // L1 参数（kL1, scaleKL1, l1BufNum）
    BlockSchedulerParams schParams; // scheduler 参数
    QBMMTiling qbmmParams;          // Without Batch QBMM 特有参数
};
```

### QBMMTiling

```cpp
struct QBMMTiling {
    uint32_t baseM;   // L0 tile M 维度
    uint32_t baseN;   // L0 tile N 维度
    uint32_t baseK;   // L0 tile K 维度
    uint32_t isBias;  // 是否启用 bias
    uint32_t dbL0C;   // L0C 双缓冲标志
    uint32_t bMustHitL2 = 1U; // B 是否必须保留在 L2 Cache
};
```

`bMustHitL2` 为 1 时，B 矩阵的 `L2CacheHint` 设置为 `NORMAL`；为 0 时，Kernel 根据当前 tile 动态设置为 `NORMAL` 或 `DISABLE`。仅当当前 M tile 覆盖完整 M，且 B 已转置或当前 N tile 按 128 Bytes 对齐时，设置为 `DISABLE`。

### BlockMmadParams

```cpp
struct Params {
    GM_ADDR aGmAddr;      // A 矩阵 GM 地址
    GM_ADDR bGmAddr;      // B 矩阵 GM 地址
    GM_ADDR cGmAddr;      // C 矩阵 GM 地址
    GM_ADDR biasGmAddr;   // Bias GM 地址（可选）
    GM_ADDR scaleAGmAddr; // A 矩阵 Scale GM 地址
    GM_ADDR scaleBGmAddr; // B 矩阵 Scale GM 地址
};
```

## 公共接口

### operator() 函数

```cpp
__aicore__ inline void operator()(const Params& params)
```

该方法是 Kernel 的公开调用入口，内部调用私有的 `Run(params)`。外部代码应通过 `kernel(params)` 执行计算。

## 内部实现

### Run 函数

```cpp
__aicore__ inline void Run(const Params& params)
```

`Run` 是私有方法，用于执行单 Batch MX 量化矩阵乘，仅供 `operator()` 内部调用。

执行流程：

1. AIV 核直接返回
2. 配置 Atomic Add，并在该模式下禁用 C 的 L2 Cache（可选）
3. 绑定 GM 地址并创建 BlockScheduler
4. 初始化 BlockMmadMX
5. 调用 `Process` 创建 GM Tensor，并按 tile 通过 `ProcessOneBlock` 调用 BlockMmadMX
6. 清理 Atomic Add（可选）

### SetBL2Cache 函数

```cpp
template <typename TensorB>
__aicore__ inline void SetBL2Cache(
    const ProblemShape& problemShape,
    uint64_t currentBasicBlockM,
    uint64_t currentBasicBlockN,
    uint32_t bMustHitL2,
    TensorB& gmB)
```

`SetBL2Cache` 是私有方法，用于动态配置 B 矩阵的 L2 Cache hint。

说明：

- `bMustHitL2` 非 0 时，B 的 L2 Cache hint 保持为 `NORMAL`
- `bMustHitL2` 为 0、M tile 覆盖完整 M 维，且 B 已转置或当前 N tile 满足相应量化格式的对齐要求时，将 B 的 L2 Cache hint 设置为 `DISABLE`
- 当前实现不配置 ScaleB 的 L2 Cache hint，ScaleB 使用 Tensor API 默认 Cache 策略

## 调用示例

```cpp
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
    {baseM, baseN, baseK, isBias, dbL0C, bMustHitL2}
};

QBMMKernel kernel;
kernel(params);
```

## 适用场景

- 单 Batch MX 量化推理
- 不需要 Batch 广播的标量路径
- 希望减少多 Batch 分支和地址偏移开销的 QBMM MX 场景
