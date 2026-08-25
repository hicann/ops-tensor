# Block Epilogue Flat Quant
> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_flat_quant.h)

## 功能说明

FlatQuant AIV 侧 MX FP4 量化后处理组件。从 UB 读取 AIC 通过 Fixpipe 写入的 bf16 中间结果，执行 MX FP4 量化（per-group eMax → E8M0 scale → FP4 量化），输出量化后的 FP4 packed 数据（int8）和 scale（E8M0）到 GM。

仅在 3510 架构（`__NPU_ARCH__ == 3510`）下编译。

**计算位置**：AIV 核

## 量化流程

```
bf16 输入（UB，由 AIC Fixpipe 写入）
    ↓
1. 清零 scale tensor
2. 非 16 对齐时清除脏数据（GatherMask）
3. 尾块保存/清零/恢复（GROUP_SIZE=32 对齐）
    ↓
4. ComputeMxQuant:
   a. ExpMax：每 32 元素组提取 bf16 指数部分，ReduceMax 得到 eMax
   b. Scale：根据 eMax 计算 E8M0 scale 和 dequant scale
   c. Quant：bf16 × dequant scale → Cast FP4（fp4x2_e2m1_t）
    ↓
5. ComputeTransLayout：scale 布局转换（行优先→block 对齐）
    ↓
6. CopyOutputFromUbToGm：量化结果（int8）写回 GM
7. CopyScaleFromUbToGm：scale（E8M0）写回 GM
```

## 模板参数

```cpp
template <
    typename DataTypeIn_,
    typename DataTypeOut_,
    typename DataTypeScale_,
    typename FusionOp_ = Fusion::DefaultFusion<DataTypeOut_, DataTypeIn_>>
class BlockEpilogueFlatQuant;
```

| 参数 | 说明 |
|------|------|
| `DataTypeIn_` | 输入数据类型，当前为 `bfloat16_t` |
| `DataTypeOut_` | 量化输出数据类型，当前为 `int8_t`（FP4 packed） |
| `DataTypeScale_` | Scale 数据类型，当前为 `uint8_t`（E8M0） |
| `FusionOp_` | 融合操作类型，默认为 `Fusion::DefaultFusion` |

## 类型别名

| 类型 | 说明 |
|------|------|
| DataTypeIn | 输入数据类型（继承自模板参数） |
| DataTypeOut | 量化输出数据类型（继承自模板参数） |
| DataTypeScale | Scale 数据类型（继承自模板参数） |
| FusionOp | 融合操作类型（继承自模板参数） |
| BlockShape | `Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>` |
| ProblemShape | `Shape<int64_t, int64_t, int64_t, int64_t>` (m, n, k, batch) |

## Params 参数结构

### 结构定义
```cpp
struct Params {
    GM_ADDR outGmAddr{nullptr};       // 量化输出 GM 地址（int8/FP4 packed）
    GM_ADDR scaleGmAddr{nullptr};    // Scale 输出 GM 地址（E8M0）
    ProblemShape problemShape{};     // 问题规模 (m, n, k, batch)
    float dstTypeMax{0.0f};          // 量化目标最大值
    float invDstTypeMax{0.0f};       // 量化目标最大值的倒数
};
```

### 参数详解

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| outGmAddr | GM_ADDR | 量化输出 GM 地址 | outGM |
| scaleGmAddr | GM_ADDR | Scale 输出 GM 地址 | scaleGM |
| problemShape | ProblemShape | 问题规模 (m, n, k, batch) | {128, 128, 64, 1} |
| dstTypeMax | float | 量化目标最大值，决定 scale 计算路径 | 6.0f |
| invDstTypeMax | float | 量化目标最大值的倒数 | 1.0f / 6.0f |

### dstTypeMax 路径选择

| dstTypeMax | Scale 计算路径 | 说明 |
|-----------|---------------|------|
| 0.0f | `ScaleVf` | 默认 FP4 E2M1 路径，标准 eMax→scale |
| 6.0f | `ScaleVfDynamic` | 动态 FP4 路径，addValueBit = `ADD_VALUE_FOR_BF16_MAN1` (0x003f) |
| 7.0f | `ScaleVfDynamic` | 动态 FP4 路径，addValueBit = `ADD_VALUE_FOR_BF16_MAN2` (0x001f) |
| 其他 | `ScaleVfcuBLAS` | cuBLAS 兼容路径，使用 invDstTypeMax |

### eMax 计算路径选择

| dstTypeMax | eMax 计算路径 | 说明 |
|-----------|-------------|------|
| 6.0f ~ 12.0f | `ExpMaxVfcuBLAS` | cuBLAS 兼容，取绝对值后提取指数 |
| 其他 | `ExpMaxVf` | 标准路径，直接提取 bf16 指数部分 |

## 常量定义

### Constant 命名空间

| 常量 | 值 | 说明 |
|------|------|------|
| GATHER_PATTERN | 7 | GatherMask 模式（清除脏数据） |
| CEIL_SIZE | 16 | 对齐粒度 |
| GROUP_SIZE | 32 | MX 量化分组大小（每 32 元素一个 scale） |
| VEC_N_LEN | 64 | N 轴向量长度 |
| MN_SIZE | 64 * 1024 | xTensor UB 空间大小（元素数） |
| OUT_SIZE | 32 * 1024 | yTensor UB 空间大小（元素数） |
| EMAX_SIZE | 2 * 1024 | eMaxTensor UB 空间大小（元素数） |
| MAX_EXP_FOR_BF16 | 0x7f80 | BF16 指数掩码 |
| BF16_EXP_BIAS | 0x7f00 | BF16 指数偏置 |
| SHR_NUM_FOR_BF16 | 7 | BF16 指数右移位数 |
| FP4_E2M1_MAX_EXP | 0x0100 | FP4 E2M1 最大指数 |
| NAN_CUSTOMIZATION | 0x7f81 | 自定义 NaN 值 |
| SPECIAL_EXP_THRESHOLD | 0x0040 | 特殊指数阈值 |
| BLOCK_SCALE | 2 | Scale block 大小 |
| SCALE_STORE_STRIDE | 32 | Scale 存储步长 |

## UB 空间布局

使用单个 `TBuf<VECCALC>` 分配全部 UB 空间，内部按偏移划分：

```
UB 空间布局：
┌──────────────────────────────────────────────────────────────┐
│ xTensor_ (bfloat16_t)     │ 偏移 0, 大小 MN_SIZE (64K 元素)   │
│ yTensor_ (int8_t)         │ 偏移 MN_SIZE, 大小 OUT_SIZE (32K) │
│ eMaxTensor_ (uint16_t)    │ 偏移 MN_SIZE+OUT_SIZE, EMAX_SIZE  │
│ deQuantScaleTensor_       │ 偏移 ...+EMAX_SIZE, EMAX_SIZE     │
│ scaleTensor_ (int8_t)     │ 偏移 ...+EMAX_SIZE, EMAX_SIZE     │
│ scaleBlockTensor_ (int8_t)│ 偏移 ...+EMAX_SIZE, EMAX_SIZE     │
└──────────────────────────────────────────────────────────────┘
```

## 公共成员方法（Public API）

### 构造函数
```cpp
__aicore__ inline BlockEpilogueFlatQuant()
```
功能：构造 BlockEpilogueFlatQuant 对象。

### Init函数
```cpp
__aicore__ inline void Init(Params const& params)
```
功能：初始化 Epilogue 组件。

执行流程：
1. 设置 GM GlobalTensor（cGlobal_, scaleGlobal_）
2. 解析 problemShape：m, n, k（batch）
3. 设置 dstTypeMax_ / invDstTypeMax_ / addValueBit_
4. 计算对齐参数：mCeil, nCeil, alignM_
5. 初始化 UB Buffer 和 Tensor
6. 获取 V_MTE3 / MTE3_V 事件 ID

### operator函数
```cpp
__aicore__ inline void operator()(uint64_t startBatchIdx, uint64_t iterBatch)
```
功能：执行 MX FP4 量化后处理。

参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| startBatchIdx | uint64_t | Batch 起始索引 |
| iterBatch | uint64_t | 迭代数量 |

执行流程：
```
for iter in [0, iterBatch):
    Quant(startBatchIdx + iter, iter)
```

## 私有方法

### Quant
```cpp
__aicore__ inline void Quant(uint64_t batchIdx, uint64_t iterIdx)
```
功能：对单个 batch 执行完整 MX 量化流程。

执行流程：
1. 计算输出偏移：yOffset = batchIdx * m * n，scaleOffset = batchIdx * CeilDiv(mn, MXFP_DIVISOR) * 2
2. ClearScaleTensor：清零 scale 和 scaleBlock tensor
3. 非 16 对齐时 ClearDirtyData：GatherMask 清除脏数据
4. 尾块处理：GROUP_SIZE(32) 不整除时保存/清零/恢复尾部数据
5. ComputeMxQuant：eMax → Scale → Quant
6. 尾块恢复
7. ComputeTransLayout：scale 布局转换
8. V_MTE3 同步后 CopyOutputFromUbToGm / CopyScaleFromUbToGm

### ComputeMxQuant
```cpp
__aicore__ inline void ComputeMxQuant(
    LocalTensor<bfloat16_t>& xTensor, LocalTensor<int8_t>& yTensor,
    LocalTensor<uint16_t>& eMaxTensor, LocalTensor<int8_t>& scaleTensor,
    LocalTensor<uint16_t>& deQuantScaleTensor, uint32_t totalDataInUB, uint64_t inputOffset)
```
功能：执行 MX 量化核心计算（eMax → Scale → Quant）。

### ComputeTransLayout
```cpp
__aicore__ inline void ComputeTransLayout(
    LocalTensor<int8_t>& scaleTensor, LocalTensor<int8_t>& scaleBlockTensor,
    uint16_t m, uint16_t n)
```
功能：Scale 布局转换，从行优先转换为 block 对齐（32B）格式。

## SIMD 向量函数

以下函数使用 `__simd_vf__` 内联向量指令实现：

| 函数 | 说明 |
|------|------|
| `ExpMaxVf` | 标准 eMax 计算：提取 bf16 指数 → ReduceMax |
| `ExpMaxVfcuBLAS` | cuBLAS 兼容 eMax：取绝对值 → 提取指数 → ReduceMax |
| `ScaleVf` | 默认 FP4 E2M1 scale 计算 |
| `ScaleVfDynamic` | 动态 FP4 scale 计算（dstTypeMax=6/7） |
| `ScaleVfcuBLAS` | cuBLAS 兼容 scale 计算（使用 invDstTypeMax） |
| `QuantVf` | 量化：bf16 × dequant scale → Cast FP4（fp4x2_e2m1_t） |
| `TransLayoutVf` | Scale 布局转换（逐行搬运到 32B 对齐） |
| `SaveTailVf` | 保存尾部数据（GROUP_SIZE 不整除时） |
| `ClearTailVf` | 清零尾部数据 |
| `RestoreTailVf` | 恢复尾部数据 |

## 数据流

### 量化数据流
```
bf16 输入 (UB, xTensor_)
    ↓ ExpMaxVf / ExpMaxVfcuBLAS
eMax (uint16_t, eMaxTensor_)    —— 每 32 元素组 1 个 eMax
    ↓ ScaleVf / ScaleVfDynamic / ScaleVfcuBLAS
scale (E8M0, scaleTensor_)      —— 量化 scale
dequant scale (uint16_t, deQuantScaleTensor_)  —— 反量化 scale
    ↓ QuantVf
FP4 packed (int8_t, yTensor_)   —— 量化结果
    ↓ TransLayoutVf
scale block (int8_t, scaleBlockTensor_)  —— 布局转换后的 scale
    ↓ CopyOutputFromUbToGm / CopyScaleFromUbToGm
GM (量化输出 + scale)
```

### 事件同步
| 事件 | 用途 |
|------|------|
| V_MTE3 (eventIdVToMte3_) | Vector 计算完成 → MTE3 搬运 |
| MTE3_V (eventIdMte3ToV_) | MTE3 搬运完成 → Vector 继续 |

## 调用示例

### 组件组装
```cpp
using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueFlatQuant<
    bfloat16_t, int8_t, uint8_t>;
```

### 参数准备
```cpp
BlockEpilogue::Params params = {
    .outGmAddr = outGM,
    .scaleGmAddr = scaleGM,
    .problemShape = {m, n, k, batch},
    .dstTypeMax = 6.0f,
    .invDstTypeMax = 1.0f / 6.0f
};
```

### 组件初始化与执行
```cpp
BlockEpilogue epilogue;
epilogue.Init(params);

// 由 Kernel 层调用，startBatchIdx 和 iterBatch 来自 Scheduler
epilogue(batchOffset, iterBatch);
```

## 约束

- 仅在 3510 架构下编译（`__NPU_ARCH__ == 3510`）
- 输入数据必须为 `bfloat16_t`，由 AIC 通过 Fixpipe L0C→UB 写入
- 输出为 FP4 packed（`fp4x2_e2m1_t`，存储为 `int8_t`）
- Scale 为 E8M0 格式（`uint8_t`）
- MX 分组大小固定为 32（`GROUP_SIZE`）
- N 轴非 16 对齐时自动执行脏数据清除
- 尾部数据非 GROUP_SIZE 整除时自动保存/恢复

## 性能优化建议

### dstTypeMax 选择
- `dstTypeMax = 0.0f`：标准 FP4 E2M1 路径，适用通用场景
- `dstTypeMax = 6.0f / 7.0f`：动态 FP4 路径，支持更灵活的量化范围
- cuBLAS 兼容路径：使用 `invDstTypeMax` 参数，与 cuBLAS 量化行为对齐

### UB 空间利用
- 全部 UB 空间通过单个 TBuf 分配，内部按偏移划分
- xTensor 占用最大空间（64K 元素），支持较大的 M×N tile
- scale 相关 tensor 复用 EMAX_SIZE 空间

### 适用场景
- FlatQuant Kernel 的 AIV 侧 MX FP4 量化后处理
- 需要将 bf16 矩阵乘结果在线量化为 FP4 的场景
- Attention 场景中的动态量化需求
