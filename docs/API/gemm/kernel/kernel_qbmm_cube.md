# Kernel Qbmm Cube
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_qbmm_cube.h)

## 功能说明
Fixpipe 量化 Batch Matmul Kernel，仅支持 AIC 计算。该 Kernel 组合 `BlockMmadA8W8FixpipeQuant` 与 `BlockSchedulerQuantBatchMatmulV3`，完成量化 A/B 矩阵乘、Batch 广播、Bias 处理、scale 反量化与 GM 输出。`BlockMmadA8W8FixpipeQuant` 沿用 A8W8 路径命名，实际输入类型由 `AType/BType` 决定。

**继承自**：[Kernel Matmul 基础框架](./kernel.md)

## 特殊约束

### 量化格式支持
支持 Tensor API Cube Mmad/Fixpipe 路径可处理的量化输入类型，典型包括：
- `int8_t`：A8W8 输入，L0C 通常累加为 `int32_t`
- `hifloat8_t`：HiFloat8 输入，L0C 通常累加为 `float`
- `fp8_e4m3fn_t`、`fp8_e5m2_t`：FP8 输入，支持 E4M3FN/E5M2 同型或混合组合，L0C 通常累加为 `float`
- C 输出由 `CType` 指定，支持 `half`、`bfloat16_t`、`float` 或 `int32_t`，最终以 Tensor API Fixpipe 静态检查为准

### Scale 因子要求
Scale 模式由 `x1QuantMode` 和 `x2QuantMode` 决定，取值来自 `QuantMode`：

| 枚举值 | 取值 | 说明 |
|--------|------|------|
| `DEFAULT` | `0x0` | 不启用对应 scale |
| `PERTENSOR_MODE` | `0x1` | per-tensor scalar scale |
| `PERCHANNEL_MODE` | `0x2` | per-channel scale |
| `PERTOKEN_MODE` | `0x4` | per-token scale |
| `MX_PERGROUP_MODE` | `0x8` | MX per-group scale |
| `PERBLOCK_MODE` | `0x10` | per-block scale |
| `PERGROUP_MODE` | `0x20` | per-group scale |

`x1QuantMode` 描述 A 矩阵的 scale 模式，`x2QuantMode` 描述 B 矩阵的 scale 模式。当前 Kernel 支持以下组合：

| A 矩阵模式 (`x1QuantMode`) | B 矩阵模式 (`x2QuantMode`) | 适用场景与 Scale 处理 |
|----------------------------|----------------------------|-----------------------|
| `DEFAULT` | `DEFAULT` | A、B 均不提供反量化 scale。用于 `CType = int32_t` 的场景，Block 将 L0C 累加结果直接写入 GM。 |
| `DEFAULT` | `PERCHANNEL_MODE` | A 不提供 scale，B 的每个 N 通道分别使用一个 scale。`scaleBGmAddr` 指向 B 的 per-channel scale 数组；Kernel 根据当前 block 负责的 N 范围截取对应 scale Tensor 并传给 Block，Block 将 scale 搬入 L1，Fixpipe 搬出结果时按通道完成反量化。 |
| `DEFAULT` | `PERTENSOR_MODE` | A 不提供 scale，整个 B 矩阵共用一个 scalar scale。Kernel 从 `scaleBGmAddr` 读取该值，将其转换并封装为 Fixpipe 使用的 `uint64_t scaleScalar_`，再传给 Block。支持的 scale 存储类型为 `uint64_t/int64_t`、`bfloat16_t` 或 `uint32_t`。Block 在 Fixpipe 搬出结果时使用该 scalar scale 完成反量化。 |
| `PERTENSOR_MODE` | `PERTENSOR_MODE` | A、B 各自提供一个 per-tensor scalar scale，二者的存储类型均为 `float`。Kernel 分别从 `scaleAGmAddr` 和 `scaleBGmAddr` 读取 scale，计算两者乘积，将乘积转换并封装为 Fixpipe 使用的 `uint64_t scaleScalar_`，再传给 Block。Block 在 Fixpipe 搬出结果时使用该乘积 scale 完成反量化。 |

其他组合当前模板未提供对应 scale 数据流。

当输出 `CType = int32_t` 时，无论传入 Block 的是 scalar scale 还是 per-channel scale，Block 都不会使用 scale，也不会执行 Fixpipe 反量化，而是将 L0C 中的 `int32_t` 累加结果直接写入 GM。

### 计算模式
仅支持 AIC 模式，不支持 AIV 计算（AIV 核在 `Init` 中直接返回）。

### BlockMmad 限制
仅支持调度策略为 `MatmulWithScaleFixpipeQuant` 的 `BlockMmad`，即 `BlockMmad::DispatchPolicy::ScheduleType` 必须为 `KernelMmadWithScaleFixpipeQuant`。

### BlockScheduler 限制
使用 `BlockSchedulerQuantBatchMatmulV3` 调度器，支持 block 切分、尾块切分和多 Batch 维度遍历。

### Batch 维度限制
支持 4 维 Batch（`batchA1/A2/A3/A4`、`batchB1/B2/B3/B4`、`batchC1/C2/C3/C4`）。多 Batch 路径按 C 的 Batch 空间遍历，并根据 A/B 到 C 的广播倍数计算 GM 地址偏移。

### Atomic Add 模式
可选 Atomic Add 模式（`IS_ATOMIC_ADD = true`）。Kernel 在计算开始时配置 atomic add，结束时恢复 atomic none。

## 模板参数

### 模板定义
```
template <
    class ProblemShape,      // 问题形状类型
    class BlockMmad,         // Fixpipe Quant BlockMmad
    class BlockEpilogue,     // 后处理组件（通常为空）
    class BlockScheduler>    // BlockSchedulerQuantBatchMatmulV3 调度器
```

### 模板参数说明
| 参数 | 说明 |
|------|------|
| ProblemShape | 问题形状类型，包含 m、n、k、b |
| BlockMmad | Fixpipe Quant BlockMmad，基于 `MatmulWithScaleFixpipeQuant` |
| BlockEpilogue | 后处理组件，当前路径通常使用 `BlockEpilogueEmpty` |
| BlockScheduler | QBMM V3 调度器 |

## 特殊类型别名

| 类型 | 说明 |
|------|------|
| BlockMmadParams | BlockMmad 参数类型，包含 GM 地址 |
| AType/BType/CType/BiasType | 继承自 BlockMmad 的数据类型 |
| X2ScaleType | Kernel 内部使用的 scalar scale 类型，固定为 `uint64_t` |
| ScaleGmType | BlockMmad 中 `BTypeTuple` 的第 1 个类型 |
| WEIGHT_NZ | B 矩阵是否为 NZ 格式（继承自 BlockMmad） |
| TRANS_A | A 矩阵是否转置（继承自 BlockMmad） |
| TRANS_B | B 矩阵是否转置（继承自 BlockMmad） |
| IS_ATOMIC_ADD | 是否启用 Atomic Add 模式（继承自 BlockMmad::DispatchPolicy） |
| C0_SIZE | A/B C0 对齐大小 |
| MakeLayoutA/MakeLayoutB/MakeLayoutC | A/B/C GM Tensor Layout 构建器 |

## 特殊数据结构

### Params
```
struct Params {
    ProblemShape problemShape;      // 问题 shape (m, n, k, batch)
    BlockMmadParams mmadParams;     // BlockMmad 参数（包含 GM 地址）
    BlockSchedulerParams schParams; // scheduler 参数
    QBMMTiling qbmmParams;          // QBMM Cube 参数
};
```

### QBMMTiling
```
struct QBMMTiling {
    uint32_t batchA1, batchA2, batchA3, batchA4; // A 矩阵 Batch 维度
    uint32_t batchB1, batchB2, batchB3, batchB4; // B 矩阵 Batch 维度
    uint32_t batchC1, batchC2, batchC3, batchC4; // C 矩阵 Batch 维度
    uint32_t biasThreeDim;                       // Bias 是否按 Batch 展开
    uint32_t x1QuantMode;                        // x1 scale 量化模式
    uint32_t x2QuantMode;                        // x2 scale 量化模式
    uint32_t kAL1;                               // A 的 L1 K 轴切分
    uint32_t kBL1;                               // B 的 L1 K 轴切分
    uint32_t nBufferNum;                         // A/B L1 缓冲数量
    uint32_t baseM, baseN, baseK;                // L0 tile 形状
    uint32_t isBias;                             // 是否启用 bias
    uint32_t dbL0C;                              // L0C 双缓冲标志
    uint32_t bMustHitL2 = 1U;                    // B 是否必须保留在 L2 Cache
};
```

`bMustHitL2` 为 1 时，B 矩阵的 `L2CacheHint` 设置为 `NORMAL`；为 0 时，Kernel 根据当前 tile 动态设置为 `NORMAL` 或 `DISABLE`。仅当当前 M tile 覆盖完整 M，且 B 已转置或当前 N tile 按 128 Bytes 对齐时，设置为 `DISABLE`。

### BlockMmadParams
```
struct Params {
    GM_ADDR aGmAddr;      // A 矩阵 GM 地址
    GM_ADDR bGmAddr;      // B 矩阵 GM 地址
    GM_ADDR cGmAddr;      // C 矩阵 GM 地址
    GM_ADDR biasGmAddr;   // Bias GM 地址（可选）
    GM_ADDR scaleAGmAddr; // x1 scale GM 地址
    GM_ADDR scaleBGmAddr; // x2 scale GM 地址
};
```

## 执行流程

Kernel 对外通过 `operator()` 执行完整 QBMM Cube 计算，内部流程概括如下：
1. 解析 `Params`，设置 A/B/C/Bias/Scale 的 GM 地址和 scale 模式。
2. 按 `baseM/baseN/baseK/kAL1/kBL1/nBufferNum/x2QuantMode/isBias/dbL0C` 初始化 BlockMmad。
3. 由 `BlockSchedulerQuantBatchMatmulV3` 负责 block 坐标、尾块和 Batch 遍历。
4. 单 Batch 场景直接遍历 block；多 Batch 场景按 C 的 4 维 Batch 空间遍历，并根据广播关系更新 A/B/C 地址偏移。
5. per-channel scale 场景向 BlockMmad 传入当前 N 分片 scale Tensor；per-tensor scale 场景传入折算后的 scalar scale。
6. Atomic Add 模式下，Kernel 在计算前后分别配置和恢复 atomic 状态。

## 调用示例

### 组件组装
```
// 以下以 int8_t A/B 为例，可按 Tensor API 支持组合替换为 hifloat8_t 或 fp8 类型。
using AType = int8_t;
using BType = int8_t;
using CType = bfloat16_t;
using BiasType = int32_t;
using X2ScaleType = uint64_t;

using LayoutA = AscendC::Te::NDExtLayoutPtn;
using LayoutB = AscendC::Te::NDExtLayoutPtn;
using LayoutC = AscendC::Te::NDExtLayoutPtn;
using LayoutBias = AscendC::Te::NDExtLayoutPtn;
using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

using BTypeTuple = AscendC::Std::tuple<BType, X2ScaleType>;
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<0, false>;

using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BTypeTuple, LayoutB, CType, LayoutC, BiasType, LayoutBias>;

using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
    ProblemShape, 0, LayoutA, LayoutB, AType>;

using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;

using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

### 参数准备
```
using Params = typename QBMMKernel::Params;
Params params;

params.problemShape = {m, n, k, batch};
params.mmadParams.aGmAddr = aGM;
params.mmadParams.bGmAddr = bGM;
params.mmadParams.cGmAddr = cGM;
params.mmadParams.biasGmAddr = biasGM;
params.mmadParams.scaleAGmAddr = pertokenScaleGM;
params.mmadParams.scaleBGmAddr = scaleGM;

params.schParams = {
    baseM, baseN,
    mTailTile, nTailTile,
    mBaseTailSplitCnt, nBaseTailSplitCnt,
    mTailMain, nTailMain
};

params.qbmmParams = {
    batchA1, batchA2, batchA3, batchA4,
    batchB1, batchB2, batchB3, batchB4,
    batchC1, batchC2, batchC3, batchC4,
    biasThreeDim,
    x1QuantMode, x2QuantMode,
    kAL1, kBL1, nBufferNum,
    baseM, baseN, baseK,
    isBias, dbL0C, bMustHitL2
};
```

### Kernel 执行
```
QBMMKernel kernel;
kernel(params);  // 或 kernel.Run(params)
```

## 数据流

### 单 Batch 流程
```
Params
    ↓
解析 GM 地址和 scale
    ↓
BlockScheduler 获取 block
    ↓
Slice A/B/C/Bias/Scale
    ↓
BlockMmadA8W8FixpipeQuant
    ↓
GM 输出
```

### 多 Batch 流程
```
4 维 Batch C 空间遍历
    ↓
按广播倍数计算 A/B Batch 偏移
    ↓
更新 A/B/C GM 地址偏移
    ↓
进入单 Batch block 处理流程
```

### Scale 处理流程

Scale 的处理分为 Kernel 准备和 Block 搬出两个阶段。Kernel 根据 `x1QuantMode/x2QuantMode` 确定 scale 的来源和形态，并将 scalar scale 或当前 N 分片的 per-channel scale Tensor 传给 Block。Mmad 计算始终先在 L0C 中完成累加；仅当输出类型不是 `int32_t` 时，Block 才在结果搬出阶段使用 scale 进行 Fixpipe 反量化。

- `CType = int32_t`：Mmad 完成后，Block 将 L0C 中的 `int32_t` 累加结果直接写入 GM。该路径不使用传入 Block 的 scale，不执行 Fixpipe 反量化，也不把累加结果转换为浮点类型。
- `CType = half/bfloat16_t/float`，且 B 使用 per-channel scale：Kernel 根据当前 block 的 N 方向起始位置和长度，从 `scaleBGmAddr` 中取得对应通道的 scale Tensor。Block 将该 Tensor 从 GM 搬入 L1，Fixpipe 搬出 L0C 结果时对每个 N 通道应用各自的 scale，并转换为 `CType` 后写入 GM。
- `CType = half/bfloat16_t/float`，且仅 B 使用 per-tensor scale：Kernel 从 `scaleBGmAddr` 读取整个 B 矩阵共用的 scalar scale，将其转换并封装为 `uint64_t scaleScalar_` 后传给 Block。Fixpipe 搬出 L0C 结果时对所有元素使用同一个 scale，并转换为 `CType` 后写入 GM。
- `CType = half/bfloat16_t/float`，且 A、B 均使用 per-tensor scale：Kernel 从 `scaleAGmAddr` 和 `scaleBGmAddr` 分别读取 A、B 的 `float` scalar scale，先计算二者的乘积，再将乘积转换并封装为 `uint64_t scaleScalar_` 后传给 Block。Fixpipe 使用该乘积 scale 完成反量化和输出类型转换，然后将结果写入 GM。

## 性能优化建议

### Block 与 L1 配置
- `baseM/baseN/baseK` 应与 BlockMmad 的 L0/L1 切分匹配。
- `kAL1/kBL1` 可根据 A/B 复用关系配置；详见 [Block Mmad A8W8 Fixpipe Quant](../block/block_mmad_a8w8_fixpipe_quant.md)。
- `nBufferNum = 4` 可提高搬运流水并行度，`nBufferNum = 2` 支持 A/B 不同 K-L1 窗口。

### Batch 广播
- 多 Batch 场景下，A/B/C 的 4 维 Batch 参数需满足广播关系。
- `biasThreeDim = 1` 时，Bias 随 C Batch 偏移；否则所有 Batch 复用同一 Bias。

### 尾块切分
- 尾轮由 scheduler 更新尾块切分，用于提升尾块场景下的核利用率。
- `mTailTile/nTailTile` 应与 scheduler 参数保持一致。
