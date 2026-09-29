# Block Epilogue SwiGLU MX Quant

> [代码位置](../../../../include/blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h)

## 功能说明

`BlockEpilogueSwigluMxQuant` 在 AIV 上读取 AIC 通过 DualDst Fixpipe 写入 UB 的
gate/linear 两组 FP32 结果。输入语义为 `C=[gate | linear]`，组件计算
`SiLU(gate) * linear`，随后执行动态 MXFP8 量化并输出 `y` 与 `yScale`。

ND 和 WeightNZ 路径都将两个等宽半区连续写入 UB。

## 模板参数

```cpp
template <
    typename DataTypeOut_,
    typename DataTypeIn_ = float,
    typename DataTypeScale_ = fp8_e8m0_t>
class BlockEpilogueSwigluMxQuant;
```

| 参数 | 说明 |
|---|---|
| `DataTypeOut_` | 输出类型，支持 `fp8_e4m3fn_t` 或 `fp8_e5m2_t` |
| `DataTypeIn_` | AIC 中间结果类型，当前为 `float` |
| `DataTypeScale_` | 输出 scale 类型，当前为 `fp8_e8m0_t` |

## 数据结构

### Params

```cpp
struct Params {
    GM_ADDR yGmAddr{nullptr};
    GM_ADDR yScaleGmAddr{nullptr};
    uint32_t baseM{0};
    uint32_t baseN{0};
    uint8_t scaleAlg{0};
};
```

| 字段 | 说明 |
|---|---|
| `yGmAddr` | MXFP8 输出地址 |
| `yScaleGmAddr` | E8M0 scale 输出地址 |
| `baseM` / `baseN` | 保留给现有调用方的 tile 参数，当前 epilogue 不读取 |
| `scaleAlg` | 0 表示 OCP scale，1 表示 cuBLAS/BLAS scale |

调用示例：

```cpp
typename Epilogue::Params params{};
params.yGmAddr = y;
params.yScaleGmAddr = yScale;
params.baseM = baseM;
params.baseN = outputBaseN;
params.scaleAlg = scaleAlg;
```

### OutputOffsets

```cpp
struct OutputOffsets {
    int64_t yOffset{0};
    int64_t yScaleOffset{0};
};
```

偏移均以对应输出元素为单位。

## 输入 UB 布局

每行拼接后的布局为：

```text
[gate: H][linear: H]
```

`H` 是当前输出 tile 的列数，调用方保证按 32 列对齐。WeightNZ 的 gate 和 linear
分别从各自完整的分形边界搬运，无额外前缀。

若原始矩阵乘宽度为 `N`，则 `H=N/2`。最终输出为
`y=[..., M, H]`、`yScale=[..., M, ceil(H/64), 2]`；最后一维中的两个值分别对应
同一 64 元素存储组内的两个 32 元素量化组。

## 公共接口

### GetL0c2UbTensor

```cpp
__aicore__ inline auto GetL0c2UbTensor(
    int64_t rows,
    int64_t cols,
    L0c2UbTensorType tensorType)
```

兼容已有调用方的单半区 UB Tensor 构造接口。`tensorType` 可取 `SWISH_INPUT` 或
`GATE_INPUT`，分别使用两个固定输入缓冲区。这两个枚举名沿用历史命名：按本文
`C=[gate | linear]` 的语义，`SWISH_INPUT` 对应送入 SiLU 的 gate 半区，`GATE_INPUT`
对应作为乘数的 linear 半区。当前 QBMM SwiGLU 融合 Kernel 使用下面的
`GetConcatL0c2UbTensor` 连续布局，不调用该兼容接口；新接入者不应混用两种布局。

### GetConcatL0c2UbTensor

```cpp
__aicore__ inline auto GetConcatL0c2UbTensor(int64_t rows, int64_t cols)
```

根据当前输出 tile 的逻辑列数构造 AIC 写入、AIV 读取的 ND_EXT UB Tensor。
`rows` 会按 Split-M 要求向 2 对齐。

### Init

```cpp
__aicore__ inline void Init(const Params& params)
```

初始化输出类型的最大指数与倒数范围，并规划内部 UB 区域。

### UpdateNextProblem

```cpp
__aicore__ inline void UpdateNextProblem(const ProblemShape& problemShape)
```

更新逻辑输出宽度，并计算公开 `yScale[..., ceil(N/64), 2]` 布局所需的行步长。

### UpdateGlobalAddr

```cpp
__aicore__ inline void UpdateGlobalAddr(const OutputOffsets& baseOffsets)
```

按 Batch 或当前问题的基准偏移更新 `y`、`yScale` GM 地址。

### operator()

```cpp
__aicore__ inline void operator()(
    const BlockShape& blockShape,
    const OutputOffsets& outputOffsets)
```

依次完成 SwiGLU、BF16 中间转换、MX scale 计算、MXFP8 量化以及 `y/yScale`
写回。AIC 调用该接口时直接返回。

## Scale 算法选择

| `scaleAlg` | 算法 | 行为与适用场景 |
|---|---|---|
| 0 | OCP | 按组内最大指数直接生成2的整数次幂scale，不会根据目标FP8最大有限值额外上调scale。归一化结果位于FP8表示边界外时，输出由底层FP8类型转换语义决定。 |
| 1 | cuBLAS/BLAS | 按目标FP8最大有限值计算scale，并将E8M0指数向上取整，可降低边界值量化溢出的风险。输入可能触及FP8表示边界且要求有限输出时建议使用。 |

两种算法都以32个元素为一组。SwiGLU结果先以RNE舍入为BF16，再计算scale和量化结果；
这一量化阶段与独立的DynamicMxQuant、SwigluMxQuant采用相同算法语义。

## 约束

- 输入 gate/linear 的逻辑列数必须相同。
- qbmmswigluquant 的原始矩阵乘 `N` 必须为正数且是 64 的倍数，epilogue 的逻辑 `N` 为原始 `N/2`。
- 单 tile 的逻辑元素数不能超过 `64 * 256`。
- 仅支持 MXFP8 输出；产品侧 `scaleAlg` 取值为 0 或 1。

## 相关 Kernel

- [多 Batch QBMM MX Activation Quant](../../gemm/kernel/kernel_qbmm_mx_activation_quant.md)
- [QBMM MX Activation Quant（统一 Kernel，单 Batch 复用）](../../gemm/kernel/kernel_qbmm_mx_activation_quant.md)
