# Gelu（Tile 级 GELU 激活）
> [代码位置](../../../../../include/blaze/epilogue/tile/arch35/gelu.h)

## 功能说明

Tile 级 GELU 激活组件，提供两种算法，供多个 Block Epilogue 复用：

- **GeluTanh**：tanh 近似，`0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`，
  等价 sigmoid 形式 `x / (1 + exp(-√(8/π)·(x + 0.044715·x³)))`，纯寄存器指令实现
- **GeluErf**：erf 精确形式，`0.5·x·(1 + erf(x/√2))`，高层 `AscendC::Erf`
  （分段多项式近似）+ 寄存器组装

采用两层设计：
- **公共接口（`__aicore__`）**：接收 MakeTensor 构造的 UB 张量，内部经
  `.Data().Get()` 提取 `__ubuf__` 指针后委托 Vf；入口含 static_assert
  类型/排布门禁与 `if ASCEND_IS_AIC` 早退
- **私有实现（`__simd_vf__` / `__simd_callee__`）**：Reg API 寄存器级计算。
  `GeluTanhVf`（fp32 输入直接加载）与 `GeluTanhCastInVf`（16F 输入 unpack+加宽）
  共享 `GeluTanhCoreCallee` 数学尾段（类内定义，复用 shift_w4_to_w8.h 的
  callee 组合模式）

## 特殊约束

### 架构
仅 dav-3510（`__NPU_ARCH__ == 3510`），经
[compute.h](../compute.md) 派发引入，Block 层禁止直接 include 本头文件。

### 数据类型
`DataTypeIn` / `DataTypeOut` 各自支持 `float`、`bfloat16_t`、`half`，
**任意 3×3 组合**。类型转换策略：

| 输入 | 计算 | 输出 | 缩窄策略 |
|------|------|------|---------|
| float | fp32 | float | 直接存储，无缩窄 |
| float | fp32 | bfloat16_t | `CT_32F_TO_16F`（NO_SAT，bf16 与 fp32 指数域相同不会溢出） |
| float | fp32 | half | `CT_32F_TO_16F_SAT`（SAT，超过 65504 钳位而非 inf） |
| bfloat16_t / half | 先 `CT_16F_TO_32F` 加宽到 fp32 | 同上 | 同上 |

缩窄舍入模式均为 `CAST_RINT`（就近舍入），由按输出类型选择的
`CT_32F_TO_OUT` 编译期决定。

### 张量契约（static_assert 门禁）
- **元素类型**：src 元素须等于 `DataTypeIn`，dst 元素须等于 `DataTypeOut`
  （GeluErf 的三块 temp 须为 `float`），防止张量传错被 reinterpret_cast 静默吞掉
- **内存位置**：所有张量必须为 UB
- **排布**：所有张量必须为 `NDExtLayoutPtn`（仅支持 ND）
- **行距**：src/dst/temp 的 rowPitch 均须为 `Gemm::Align32(n)` 元素
  （Vf 内按 `mIdx * nAligned` 计算行偏移）

### temp buffer 生命周期
GeluErf 的 `erfTensor` / `fp32Tensor` / `geluFp32Tensor` 由调用方（Block 层）
分配和管理，Tile 只做临时写入。

## 特殊类型

### Gelu
```cpp
template <typename DataTypeOut_, typename DataTypeIn_>
class Gelu {
public:
    template <typename SrcTensor, typename DstTensor>
    __aicore__ inline void GeluTanh(const SrcTensor& srcTensor, const DstTensor& dstTensor,
                                    uint16_t mSize, uint16_t nSize);

    template <typename SrcTensor, typename DstTensor, typename ErfTensor,
              typename Fp32Tensor, typename GeluFp32Tensor>
    __aicore__ inline void GeluErf(const SrcTensor& srcTensor, const DstTensor& dstTensor,
                                   const ErfTensor& erfTensor, const Fp32Tensor& fp32Tensor,
                                   const GeluFp32Tensor& geluFp32Tensor, uint16_t mSize, uint16_t nSize);
};
```

### Gelu::GeluTanh
| 参数 | 类型 | 说明 |
|------|------|------|
| srcTensor | SrcTensor | 输入 UB Tensor，[mSize, nSize]，rowPitch = Align32(n) |
| dstTensor | DstTensor | 输出 UB Tensor，与 src 同形同行距 |
| mSize | uint16_t | 行数 |
| nSize | uint16_t | 每行有效元素数 |

执行流程：fp32 输入直接 `DataCopy` 加载；16F 输入 `DIST_UNPACK_B16` 加载 +
`CT_16F_TO_32F` 加宽 → `GeluTanhCoreCallee`（Mul→Mul→Axpy→Muls→Exp→Adds→Div
七指令数学链 → 按输出类型缩窄 → store）。

### Gelu::GeluErf
| 参数 | 类型 | 说明 |
|------|------|------|
| srcTensor | SrcTensor | 输入 UB Tensor，[mSize, nSize] |
| dstTensor | DstTensor | 输出 UB Tensor，与 src 同形同行距 |
| erfTensor | ErfTensor | temp：erf 结果缓冲（float，每行复用） |
| fp32Tensor | Fp32Tensor | temp：16F 输入的 fp32 加宽缓冲（float 输入时仍需传入） |
| geluFp32Tensor | GeluFp32Tensor | temp：x/√2 缓冲（float） |
| mSize / nSize | uint16_t | 形状 |

执行流程（逐行）：`Muls(x, 1/√2)` → `AscendC::Erf`（分段多项式近似）→
`GeluErfVf` 组装 `(1+erf)·(0.5·x)` → 缩窄 store。

## 使用示例

```cpp
#include "blaze/epilogue/tile/compute.h"

// Block 层持有 UB 字节偏移（geluResUbOffset_ / erfTmpUbOffset_ ...）
const uint32_t nAligned = Gemm::Align32(static_cast<uint32_t>(nSize));
auto layout = Gemm::MakeNDExtLayout(static_cast<int64_t>(mSize),
                                    static_cast<int64_t>(nSize),
                                    static_cast<int64_t>(nAligned));
auto srcTensor = AscendC::Te::MakeTensor(
    AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(0), layout);
auto dstTensor = AscendC::Te::MakeTensor(
    AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, bfloat16_t>(geluResUbOffset_), layout);

Blaze::Epilogue::Block::Gelu<bfloat16_t, float> gelu;
gelu.GeluTanh(srcTensor, dstTensor, mSize, nSize);   // tanh 近似
// 或 erf 精确形式（额外传入三块 fp32 temp）：
// gelu.GeluErf(srcTensor, dstTensor, erfTensor, fp32Tensor, geluFp32Tensor, mSize, nSize);
```

## 数据流

```
UB（src, rowPitch=Align32(n)）
    ↓ [16F] unpack + CT_16F_TO_32F 加宽        ↓ [fp32] 直接加载
    └──────────────┬───────────────────────────┘
                   ↓
        GeluTanhCoreCallee / 逐行 Erf 组装（fp32 寄存器计算）
                   ↓
        CT_32F_TO_OUT 缩窄（half→SAT / bf16→NO_SAT / float→直存）
                   ↓
UB（dst, rowPitch=Align32(n)）
```

## 与 Block 层的关系

`Gelu` 是首个从 Block 内联实现抽取的 Epilogue Tile，当前被两个 Block 复用：
- [BlockEpilogueGeluMxQuant](../../block/block_epilogue_gelu_mx_quant.md)：
  `GeluErf` / `GeluTanh` 按 `geluAlg` 运行时选择
- [BlockEpilogueGeluTanhMxQuant](../../block/block_epilogue_gelu_tanh_mx_quant.md)：
  固定 `GeluTanh`，输出 bfloat16 激活结果供 MX 量化

Block 层负责 UB 偏移管理、AIC/AIV 跨核同步与 slot 双缓冲，Tile 层只做纯计算。
