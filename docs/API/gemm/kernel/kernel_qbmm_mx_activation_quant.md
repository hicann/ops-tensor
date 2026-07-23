# Kernel Qbmm Mx Activation Quant
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_qbmm_mx_activation_quant.h)

## 功能说明
基于 MX 量化 Batch Matmul 改造的 **CV 融合 Kernel**，**AIC（cube）+ AIV（vector）双核协同**：AIC 执行 MxFP8 量化矩阵乘，通过 DualDst fixpipe 将 L0C 结果搬到 UB；AIV 执行 Gelu 激活 + 动态 MX 量化后处理，输出 MxFP8。相比 [kernel_qbmm_mx](./kernel_qbmm_mx.md)（仅 AIC、直写 GM），本 Kernel 将矩阵乘与激活量化融合到同一 Kernel 调用中，通过 cube 流水掩盖 vector 流水实现性能优化。

**继承自**：[Kernel Matmul 基础框架](./kernel.md)

## 特殊约束

### 量化格式支持
支持以下量化数据类型：
- **MxFP8**：`fp8_e5m2_t`、`fp8_e4m3fn_t`（8-bit 浮点）

### Scale 因子要求
必须提供两个 Scale 因子：
- `scaleAGmAddr`：A 矩阵的 缩放因子（`fp8_e8m0_t` 类型）
- `scaleBGmAddr`：B 矩阵的 缩放因子（`fp8_e8m0_t` 类型）

### 计算模式
AIC + AIV 双核：
- `ASCEND_IS_AIC`：执行 cube 量化矩阵乘，L0C→UB（DualDst fixpipe），`NotifyVector()` 通知 AIV
- `ASCEND_IS_AIV`：`WaitForCube()` 等待 AIC 结果，执行 Gelu 激活 + 动态 MX 量化，写最终结果到 GM

### BlockScheduler 限制
仅支持 `BlockSchedulerQbmm` 调度器，支持多 Batch 维度切分与尾块切分（mTailTile / nTailTile）。

### L2 Cache 动态配置
根据 tile 形状动态启用/禁用 L2 Cache：
- 大 tile（`curBaseM >= problemShape.m`）：禁用 L2 Cache
- 小 tile：启用 L2 Cache

### Batch 维度限制
支持 4 维 Batch（batchA1/A2/A3/A4、batchB1/B2/B3/B4、batchC1/C2/C3/C4），需满足广播规则。 x1Scale/x2Scale 的 Batch 维度须分别与 A 矩阵/B 矩阵一致。

### Atomic Add 模式
可选 Atomic Add 模式（`IS_ATOMIC_ADD = true`），用于多核并行累加场景。

### UB 对齐要求
- AIC 写入 UB 的结果需 32 元素对齐（`baseN` 按 32 对齐），满足 AIV 每 32 个元素做一次 MX 量化
- UB Tensor 使用 `NDExtLayoutPtn`，行数为 `(baseM + 1) & ~1`（M 向 2 对齐）

### DualDst 模式要求
`BlockMmad` 的 DispatchPolicy 必须设置 `IsDualDst_ = true`：
```
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<FullLoadMode, false, ScheduleType, true>;
```
此时 BlockMmad 在 L0C→UB 时使用 `CustomCopyL0C2UBTrait`（`DUAL_DST_SPLIT_M`），将结果写入 UB 地址 0 供 AIV 读取。

## 特殊成员方法

### 构造函数
```
__aicore__ inline GemmUniversal()
```
功能：构造 GemmUniversal 对象。

### 析构函数
```
__aicore__ inline ~GemmUniversal()
```
功能：析构 GemmUniversal 对象。

### 特殊模板参数
```
template <
    class ProblemShape,      // 问题形状类型 (m, n, k, batch)
    class BlockMmad,         // BlockMmadMX（需 IsDualDst_=true），ScheduleType 必须为 KernelMmadWithScaleMx
    class BlockEpilogue,     // BlockEpilogueGeluQuant（AIV Gelu 激活 + 动态 MX 量化）
    class BlockScheduler>    // BlockSchedulerQbmm 调度器
```

### 模板参数说明
| 参数 | 说明 |
|------|------|
| ProblemShape | 问题形状类型，包含 m、n、k、b（batch） |
| BlockMmad | BlockMmadMX 组件，基于 `MatmulWithScaleMx<..., true>` 调度策略（IsDualDst_=true） |
| BlockEpilogue | `BlockEpilogueGeluQuant`，AIV 侧 Gelu 激活 + 动态 MX 量化后处理 |
| BlockScheduler | BlockSchedulerQbmm 调度器 |

## 特殊类型别名

| 类型 | 说明 |
|------|------|
| WEIGHT_NZ | B 矩阵是否为 NZ 格式（由 LayoutB 判断） |
| TRANS_A | A 矩阵是否转置（由 LayoutA 判断） |
| TRANS_B | B 矩阵是否转置（由 LayoutB 判断） |
| IS_ATOMIC_ADD | 是否启用 Atomic Add 模式（继承自 BlockMmad::DispatchPolicy） |
| C0_SIZE | C0 对齐大小（FP4: 64，FP8: 32） |
| SCALE_C0 | Scale C0 对齐大小（固定为 2，定义于 `common_utils.h`） |
| MakeLayoutScaleA | ScaleA Layout 构建器（根据 TRANS_A 选择 ScaleADN/ScaleAND） |
| MakeLayoutScaleB | ScaleB Layout 构建器（根据 TRANS_B 选择 ScaleBDN/ScaleBND） |

## 特殊数据结构

### Params
```
struct Params {
    ProblemShape problemShape;       // 问题 shape (m, n, k, batch)
    BlockMmadParams mmadParams;      // mmad 参数（包含 GM 地址 + Scale 地址）
    BlockEpilogueParams epilogueParams; // epilogue 参数（Gelu + MX Quant）
    L1Params l1Params;               // L1 参数（kL1, scaleKL1, l1BufNum）
    BlockSchedulerParams schParams;  // scheduler 参数（含 mTailTile / nTailTile）
    QBMMTiling qbmmParams;           // QBMM 特有参数
};
```

### QBMMTiling
```
struct QBMMTiling {
    uint32_t batchA1, batchA2, batchA3, batchA4;  // A 矩阵/ScaleA 的 Batch 维度
    uint32_t batchB1, batchB2, batchB3, batchB4;  // B 矩阵/ScaleB 的 Batch 维度
    uint32_t batchC1, batchC2, batchC3, batchC4;  // C 矩阵 Batch 维度
    uint32_t biasThreeDim;                        // Bias 是否为 3 维
    uint32_t baseM, baseN, baseK;                 // L0 tile 形状
    uint32_t isBias;                              // 是否启用 bias
    uint32_t dbL0C;                               // L0C 双缓冲标志
};
```

### BlockMmadParams（MX 特有）
```
struct Params {
    GM_ADDR aGmAddr;             // A 矩阵 GM 地址
    GM_ADDR bGmAddr;             // B 矩阵 GM 地址
    GM_ADDR cGmAddr;             // C 矩阵 GM 地址（此处为 UB 地址）
    GM_ADDR biasGmAddr;          // Bias GM 地址（可选）
    GM_ADDR scaleAGmAddr;        // A 矩阵 Scale GM 地址
    GM_ADDR scaleBGmAddr;        // B 矩阵 Scale GM 地址
};
```

## 特殊成员方法

### Init函数
```
__aicore__ inline void Init(const Params& params)
```
功能：初始化 Kernel，提取问题规模、GM 地址、Batch 参数。
执行流程：
1. 设置 Bias 标志：根据 `qbmmParams.isBias` 判断
2. 设置 BiasThreeDim 标志：根据 `qbmmParams.biasThreeDim` 判断
3. 调用 `ResetGmAddr` 设置 GM 地址

### Run函数
```
__aicore__ inline void Run(const Params& params)
```
功能：执行 CV 融合量化 Batch Matmul Kernel 计算。
执行流程：
1. Atomic Add 配置：如果 `IS_ATOMIC_ADD = true`，调用 `SetAtomicAdd<float>`
2. 调用 `Init(params)` 设置参数
3. 创建 BlockScheduler 实例
4. 初始化 BlockMmadMX 组件（`baseM/baseN/baseK` + `dbL0C`）
5. 初始化 BlockEpilogueGeluQuant 并调用 `UpdateNextProblem`
6. 判断 Batch 数量：
   - 单 Batch（`b == 1`）：调用 `ProcessSingleBatch`，结束后调用 `End()`
   - 多 Batch：调用 `ProcessWithBatch`，结束后调用 `End()`
7. 清理 Atomic Add：如果启用，调用 `SetAtomicNone`

### ProcessSingleBatch函数
```
__aicore__ inline void ProcessSingleBatch(
    const Params& params, BlockScheduler& bs,
    uint64_t restBatch, bool isTailRound)
```
功能：处理单个 Batch 的矩阵乘 + 激活量化计算。
执行流程：
1. 构建 Layout：A、B、ScaleA、ScaleB、Bias、C
2. 创建 GM Tensor 与 UB Tensor（地址 0，`NDExtLayoutPtn`，行数 `(baseM+1)&~1`）
3. 动态配置 L2 Cache
4. Tile 循环处理：
   - 获取 tile 坐标 (mPos, nPos) 与形状 (baseM, baseN)
   - Slice GM Tensor 到当前 tile
   - **AIC**：必要时 `WaitForVector()`；调用 BlockMmadMX 执行量化矩阵乘（L0C→UB DualDst）；`NotifyVector()`
   - **AIV**：`WaitForCube()`；调用 `BlockEpilogueGeluQuant`（Gelu 激活 + 动态 MX 量化，写回 GM）；`NotifyCube()`

### ProcessWithBatch函数
```
__aicore__ inline void ProcessWithBatch(const Params& params, BlockScheduler& bs)
```
功能：处理多 Batch 的矩阵乘计算。
执行流程：
1. 计算 Batch 广播倍数：multiA1C1、multiB1C1 等
2. 4 维 Batch 循环（batchC1/C2/C3/C4）：
   - 更新 Batch 偏移：`batchCOffset_`、`batchAOffset_`、`batchBOffset_`，x1Scale/x2Scale 分别复用 `batchAOffset_`/`batchBOffset_`
   - 调用 `AddBatchOffset` 更新 GM 地址与 epilogue 输出偏移
   - 调用 `ProcessSingleBatch` 处理当前 Batch

### AddBatchOffset函数
```
__aicore__ inline void AddBatchOffset(
    const Params& params, uint64_t aBatchElementStride, uint64_t bBatchElementStride,
    uint64_t cBatchStride, uint64_t scaleABatchStride, uint64_t scaleBBatchStride,
    uint64_t biasBatchStride)
```
功能：更新 Batch 偏移后的 GM 地址。
执行流程：
1. 调用 `ResetGmAddr` 重置到基址
2. 按偏移量更新 A/B/C/Bias/ScaleA/ScaleB 的 GM 地址（FP4 时 A/B 右移 1 位）
3. 调用 `epilogueOp_.UpdateGlobalAddr` 更新 epilogue 输出偏移（`batchCOffset_ * m * n`、`batchCOffset_ * m * scaleN`）

### SetL2Cache函数
```
template <typename TensorB, typename TensorC>
__aicore__ inline void SetL2Cache(
    const ProblemShape& problemShape, uint64_t baseM, uint64_t baseN,
    TensorB& gmB, TensorC& gmC)
```
功能：动态配置 L2 Cache。
说明：
- 同 Batch 且 M tile 覆盖完整 M 维时，根据 B 的布局和对齐情况配置 B 的 L2 Cache
- 当前实现不再配置 ScaleB 的 L2 Cache hint，ScaleB 使用 Tensor API 默认 Cache 策略
- Atomic Add 模式：禁用 C 的 L2 Cache

### AIC<->AIV 同步
| 方法 | 作用 | flag |
|------|------|------|
| NotifyVector | AIC 通知 AIV（PIPE_FIX） | `AIC_SYNC_AIV_FLAG=4` 与 `+FLAG_ID_MAX(16)` |
| WaitForVector | AIC 等待 AIV（PIPE_FIX） | `AIV_SYNC_AIC_FLAG=6` 与 `+FLAG_ID_MAX(16)` |
| NotifyCube | AIV 通知 AIC（PIPE_V） | `AIV_SYNC_AIC_FLAG=6` |
| WaitForCube | AIV 等待 AIC（PIPE_V） | `AIC_SYNC_AIV_FLAG=4` |

### End函数
```
__aicore__ inline void End()
```
功能：AIC 侧在所有 tile 处理完成后，若已与 AIV 建立同步通信（`isVecSetSyncCom_`），等待 AIV 完成最后一个 tile 的后处理。

## 调用示例

### 组件组装
```
// 定义量化数据类型
using AType = fp8_e4m3fn_t;       // 或 fp8_e5m2_t
using BType = fp8_e4m3fn_t;       // 或 fp8_e4m3fn_t
using CType = float;
using BiasType = float;
using OutType = fp8_e8m0_t;     // epilogue 输出类型

// 定义 Layout
using LayoutA = AscendC::Te::NDExtLayoutPtn;
using LayoutB = AscendC::Te::NZLayoutPtn;
using LayoutC = AscendC::Te::NDExtLayoutPtn;
using LayoutBias = AscendC::Te::NDExtLayoutPtn;

// 定义调度策略（IsDualDst_ = true）
using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<
    A_FULL_LOAD_MODE, false, Blaze::Gemm::KernelMmadWithScaleMx, true>;

// 定义 BlockMmadMX（DualDst 模式）
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;

// 定义 BlockEpilogueGeluQuant
using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueGeluQuant<OutType, CType>;

// 定义 BlockScheduler
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQbmm<ProblemShape>;

// 定义 Kernel
using QBMMKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

### 参数准备
```
using Params = typename QBMMKernel::Params;
Params params;
params.problemShape = {m, n, k, batch};
params.mmadParams = {aGM, bGM, cGM, biasGM, scaleAGM, scaleBGM};
params.epilogueParams = {yGM, yScaleGM, baseM, baseN, geluAlg, quantAlg, fp4RoundMode};
params.l1Params = {kL1, scaleKL1, l1BufNum};
params.schParams = {baseM, baseN, baseK, mTailTile, nTailTile, ...};
params.qbmmParams = {batchA1, batchA2, ..., batchC4, biasThreeDim, baseM, baseN, baseK, isBias, dbL0C};
```

### Kernel 执行
```
QBMMKernel qbmm;
qbmm(params);  // 或 qbmm.Run(params);
```

## 数据流

### 存储层次
```
AIC:  GM(A/B/ScaleA/ScaleB/Bias) → L1(量化数据+Scale) → L0A/L0B → L0C(float) --DualDst fixpipe--> UB
                                                                                              │ (CrossCore Notify)
AIV:  UB(float) --Gelu激活--> bf16 --动态MX量化--> GM(y: MxFP8, yScale: fp8_e8m0)
```

### Batch 处理流程
```
Batch 循环（4 维）
    ↓
更新 Batch 偏移（batchA/B/COffset）
    ↓
重置 GM 地址 + epilogue 输出偏移（AddBatchOffset）
    ↓
处理单个 Batch（ProcessSingleBatch）
    ↓
Tile 循环 → AIC: BlockMmadMX(L0C→UB) + AIV: Gelu+MXQuant(UB→GM)
```

### CV 融合计算流程
```
AIC: 加载量化数据(A/B) + Scale(ScaleA/ScaleB) → Mmad 计算 → L0C(float) → DualDst fixpipe → UB
                                                                                  ↓ CrossCore Notify
AIV: UB(float) → Gelu 激活 → bf16 → 动态 MX 量化 → GM(MxFP8 + fp8_e8m0 scale)
```

## 性能优化建议

### L1 缓冲配置
- `l1BufNum`：建议 2、3 或 4，平衡 L1 容量和流水线并行度
- `kL1`：建议对齐到 `MXFP_DIVISOR_SIZE`（64）

### Scale KL1 配置
- `scaleKL1`：建议为 `kL1` 的整数倍
- Scale 数据复用：Scale 常驻 L1，减少搬运开销

### 全载模式选择
- **非全载模式**：每次迭代重新加载 A/B 块
- **A 全载模式**：A 矩阵常驻 L1，适用于大 K、小 M 场景

### L0C->UB DualDst
- DualDst fixpipe 将 L0C 结果直接搬到 UB，避免 GM 中转，降低延迟
- cube 流水掩盖 vector 流水，融合后整体性能接近纯 cube 计算

### Batch 维度设计
- Batch 维度需满足广播规则：`batchA = batchC × multiA`、`batchB = batchC × multiB`
- 4 维 Batch 灵活支持多种广播场景

### L2 Cache 配置
- 大 tile 场景自动禁用 L2 Cache，避免缓存污染
- Atomic Add 模式自动禁用 C 的 L2 Cache

### UB 对齐
- L0C -> UB Dualdst 拷贝时, M 必须为偶数(向 2 对齐)
- `baseN` 需 32 元素对齐，满足 AIV 每 32 个元素做一次 MX 量化
- 尾块不足 32 元素时由 epilogue 自动Padding为 0

## 适用场景

- **量化推理融合**：MxFP8 量化矩阵乘 + Gelu 激活 + 动态 MX 量化，单 Kernel 完成
- **Batch Matmul**：多 Batch 维度支持
- **Scale 因子处理**：per-token 和 per-group scale
- **广播 Batch**：A/B/C 不同 Batch 维度的广播计算
