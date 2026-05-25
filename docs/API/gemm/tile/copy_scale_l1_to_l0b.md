# Copy Scale L1 to L0B
> [代码位置](../../../../include/blaze/gemm/tile/copy_scale_l1_to_l0b.h)

## 功能说明
MX 量化 ScaleB 因子拷贝 Tile，用于将 ScaleB 从 L1 拷贝到 L0B。支持 fp8_e8m0_t 数据类型，用于 MX 量化矩阵乘。

## 特殊约束

### 数据类型支持
仅支持 fp8_e8m0_t 数据类型：
- **源类型**：`__cbuf__ fp8_e8m0_t`（L1 缓冲）
- **目标类型**：`__cb__ fp8_e8m0_t`（L0B 缓冲）

### 布局要求
ScaleB L1 布局：
- **Shape**：`(n1, k/64, n0, 2)`
- **布局**：NN → NN
- **Stride**：`(2, k/64*n0*2), (1, n0*2)`

ScaleB L0B 布局：
- **Shape**：`(2, k/64), (n0, n1)`
- **Stride**：自动计算

### 坐标参数
- **nStartPosition**：`ceilDiv(coord.n, 16)`
- **kStartPosition**：`ceilDiv(coord.k, 2)`
- **nStep**：`n1`（N 轴大分形个数）
- **kStep**：`k/64`

## 特殊类型

### CopyL12L0MxScaleB3510
```
struct CopyL12L0MxScaleB3510 {
    template <typename Tp, const Tp& traits, typename T, typename U, class Coord>
    __aicore__ inline static void Copy(const T& dst, const U& src, const Coord& coord);
};
```

功能：ScaleB L1→L0B 拷贝，用于 MX 量化。

### CopyTraits 特化
```
template <>
struct AscendC::Te::CopyTraits<Blaze::Gemm::Tile::CopyL12L0MxScaleB3510>
    : public CopyTraits<
          Blaze::Gemm::Tile::CopyL12L0MxScaleB3510, CopyL12L0ATraitDefault, Blaze::Gemm::Tile::CopyL12L0MxScaleB3510,
          CopyL12L0ATraitDefault> {};
```

功能：CopyTraits 针对 CopyL12L0MxScaleB3510 的特化。

## 特殊成员方法

### Copy
```
template <typename Tp, const Tp& traits, typename T, typename U, class Coord>
__aicore__ inline static void Copy(const T& dst, const U& src, const Coord& coord)
```
功能：拷贝 ScaleB 从 L1 到 L0B。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| dst | T | 目标 Tensor（L0B） |
| src | U | 源 Tensor（L1） |
| coord | Coord | 坐标 `(k, n)` |

执行流程：
1. 计算起始位置：`nStartPosition`, `kStartPosition`
2. 计算步长：`nStep`, `kStep`
3. 计算 stride：`srcStride`, `dstStride`
4. 执行拷贝：`asc_copy_l12l0b_mx(mxDstAddr, src.Data().Get(), ...)`

## 使用示例

### 在 BlockMmadMX 中使用
```
using ScaleType = fp8_e8m0_t;

auto tensorScaleBL1 = AscendC::Te::MakeTensor(...);  // L1 ScaleB Tensor
auto tensorScaleBL0 = AscendC::Te::MakeTensor(...);  // L0B ScaleB Tensor
auto coord = AscendC::Te::MakeCoord(kOffset, nOffset);

auto copyL12L0ScaleB = AscendC::Te::MakeCopy(Blaze::Gemm::Tile::CopyL12L0MxScaleB3510{});
AscendC::Te::Copy(copyL12L0ScaleB, tensorScaleBL0, tensorScaleBL1, coord);
```

## 数据流

```
L1（ScaleB，fp8_e8m0_t）
    ↓
CopyL12L0MxScaleB3510
    ↓
L0B（ScaleB，fp8_e8m0_t）
    ↓
Mmad（MX 模式，ScaleB 用于反量化）
```

说明：ScaleB 用于 B 矩阵的反量化，在 Mmad 计算时自动应用。