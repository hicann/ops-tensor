# Copy Scale L1 to L0A
> [代码位置](../../../../include/blaze/gemm/tile/copy_scale_l1_to_l0a.h)

## 功能说明
MX 量化 ScaleA 因子拷贝 Tile，用于将 ScaleA 从 L1 拷贝到 L0A。支持 fp8_e8m0_t 数据类型，用于 MX 量化矩阵乘。

## 特殊约束

### 数据类型支持
仅支持 fp8_e8m0_t 数据类型：
- **源类型**：`__cbuf__ fp8_e8m0_t`（L1 缓冲）
- **目标类型**：`__ca__ fp8_e8m0_t`（L0A 缓冲）

### 布局要求
ScaleA L1 布局：
- **Shape**：`(m1, k/64, m0, 2)`
- **布局**：ZZ → ZZ
- **Stride**：`(2, k/64*m0*2), (1, m0*2)`

ScaleA L0A 布局：
- **Shape**：`(m0, m1), (2, k/64)`
- **Stride**：自动计算

### 坐标参数
- **mStartPosition**：`ceilDiv(coord.m, 16)`
- **kStartPosition**：`ceilDiv(coord.k, 2)`
- **mStep**：`m1`（M 轴大分形个数）
- **kStep**：`k/64`

## 特殊类型

### CopyL12L0MxScaleA3510
```
struct CopyL12L0MxScaleA3510 {
    template <typename Tp, const Tp& traits, typename T, typename U, class Coord>
    __aicore__ inline static void Copy(const T& dst, const U& src, const Coord& coord);
};
```

功能：ScaleA L1→L0A 拷贝，用于 MX 量化。

### CopyTraits 特化
```
template <>
struct AscendC::Te::CopyTraits<Blaze::Gemm::Tile::CopyL12L0MxScaleA3510>
    : public CopyTraits<
          Blaze::Gemm::Tile::CopyL12L0MxScaleA3510, CopyL12L0ATraitDefault, Blaze::Gemm::Tile::CopyL12L0MxScaleA3510,
          CopyL12L0ATraitDefault> {};
```

功能：CopyTraits 针对 CopyL12L0MxScaleA3510 的特化。

## 特殊成员方法

### Copy
```
template <typename Tp, const Tp& traits, typename T, typename U, class Coord>
__aicore__ inline static void Copy(const T& dst, const U& src, const Coord& coord)
```
功能：拷贝 ScaleA 从 L1 到 L0A。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| dst | T | 目标 Tensor（L0A） |
| src | U | 源 Tensor（L1） |
| coord | Coord | 坐标 `(m, k)` |

执行流程：
1. 计算起始位置：`mStartPosition`, `kStartPosition`
2. 计算步长：`mStep`, `kStep`
3. 计算 stride：`srcStride`, `dstStride`
4. 执行拷贝：`asc_copy_l12l0a_mx(mxDstAddr, src.Data().Get(), ...)`

## 使用示例

### 在 BlockMmadMX 中使用
```
using ScaleType = fp8_e8m0_t;

auto tensorScaleAL1 = AscendC::Te::MakeTensor(...);  // L1 ScaleA Tensor
auto tensorScaleAL0 = AscendC::Te::MakeTensor(...);  // L0A ScaleA Tensor
auto coord = AscendC::Te::MakeCoord(mOffset, kOffset);

auto copyL12L0ScaleA = AscendC::Te::MakeCopy(Blaze::Gemm::Tile::CopyL12L0MxScaleA3510{});
AscendC::Te::Copy(copyL12L0ScaleA, tensorScaleAL0, tensorScaleAL1, coord);
```

## 数据流

```
L1（ScaleA，fp8_e8m0_t）
    ↓
CopyL12L0MxScaleA3510
    ↓
L0A（ScaleA，fp8_e8m0_t）
    ↓
Mmad（MX 模式，ScaleA 用于反量化）
```

说明：ScaleA 用于 A 矩阵的反量化，在 Mmad 计算时自动应用。