# Copy UB To L1

> [代码位置](../../../../include/blaze/gemm/tile/arch35/copy_ub_to_l1.h)

## 功能说明

该组件将带分组填充的 FP16/BF16 权重从 UB 搬运到 L1，供 WQMM 权重反量化使用。
调用方通过 `asc::te::make_copy(Blaze::Gemm::Tile::CopyPaddedUBToL1{})` 创建 Copy atom，
再使用 `asc::te::copy(copyUB2L1, dstTensor, srcTensor)` 执行搬运。
`CopyPaddedUBToL1` 根据源布局 pattern 选择对应的搬运分支。
该实现直接从源、目标的层次化布局提取搬运参数，调用 `asc_copy_ub2l1` 完成数据搬运。

## 使用限制

- 仅在 `__NPU_ARCH__ == 3510` 时由聚合头 `blaze/gemm/tile/datamove.h` 引入。
- 源 Tensor 位于 UB，目标 Tensor 位于 L1；两端元素类型相同，支持 FP16 或 BF16。
- 源布局 pattern 为 `Blaze::Gemm::zn_row_padding_layout_ptn` 或
  `Blaze::Gemm::nz_col_padding_layout_ptn`。
- 源 pattern 决定搬运分支；目标布局提供对应分组轴的实际 L1 行距。两端完整 shape 可以不同，
  但每个分组内的数据排列必须兼容。
- 调用方保证范围非空、地址及连续段长度和行距按 32 字节对齐、段间空隙非负，
  搬运参数可由指令字段表示，且缓冲区容量覆盖包含填充的物理范围。

## 布局构造接口

源布局由 [layout_struct.h](../../../../include/blaze/gemm/utils/layout_struct.h) 中的两个辅助类构造：

```cpp
template <typename T>
struct ZnRowPaddingUBLayout {
    __aicore__ inline auto operator()(int64_t kSize, int64_t nSize, int64_t groupPitch) const;
};

template <typename T>
struct NzColPaddingUBLayout {
    __aicore__ inline auto operator()(int64_t kSize, int64_t nSize, int64_t groupPitch) const;
};
```

两个类均位于 `Blaze::Gemm` 命名空间，`T` 为 FP16 或 BF16。
`kSize`、`nSize` 是权重逻辑坐标 `(K,N)` 下的有效长度；`groupPitch` 为包含填充的实际 UB 分组行距，
三者均以元素计。令 `C0=32/sizeof(T)=16`，构造结果如下：

| 构造类 | 源 pattern | shape | stride，单位为元素 |
| --- | --- | --- | --- |
| `ZnRowPaddingUBLayout<T>` | `zn_row_padding_layout_ptn` | `((16,ceil(kSize/16)),(1,nSize))` | `((1,groupPitch),(16,16))` |
| `NzColPaddingUBLayout<T>` | `nz_col_padding_layout_ptn` | `((1,kSize),(16,ceil(nSize/16)))` | `((16,16),(1,groupPitch))` |

ZN 保留有效 N 长度并对齐 K，NZ 保留有效 K 长度并对齐 N。
WQMM VF 的 UB 分组行距为 `65*16` 个元素，尾块也使用该行距，不能根据有效 shape 缩小间距。

## 搬运参数

对于源 shape `((k0,k1),(n0,n1))`，按源 pattern 提取以下参数。
行距分别从源、目标 layout 的对应 stride 中读取，单位为元素。

| 源 pattern | 连续段数量 | 每段元素数 | 分组行距 |
| --- | --- | --- | --- |
| `zn_row_padding_layout_ptn` | `k1` | `k0*n0*n1` | `stride[0][1]` |
| `nz_col_padding_layout_ptn` | `n1` | `k0*k1*n0` | `stride[1][1]` |

设每段元素数为 `segmentElements`，两端行距为 `srcPitch`、`dstPitch`。
指令的 `block_len`、`src_gap` 和 `dst_gap` 以 32 字节为单位：

```text
block_len = segmentElements * sizeof(T) / 32
src_gap   = (srcPitch - segmentElements) * sizeof(T) / 32
dst_gap   = (dstPitch - segmentElements) * sizeof(T) / 32
```

## 使用方式

调用方包含 `blaze/gemm/tile/datamove.h`，使用上述辅助类构造源布局，并为目标提供兼容的 L1 布局。
以下展示 ZN 分组的调用片段，`ubWeight` 和 `l1Weight` 为已分配的类型化指针，
`l1Layout` 描述目标 L1 的实际分组排列和行距。

```cpp
auto ubLayout = Blaze::Gemm::ZnRowPaddingUBLayout<T>{}(kSize, nSize, groupPitch);
auto srcTensor = asc::te::make_tensor(
    asc::te::make_mem_ptr<asc::te::location::ub>(ubWeight), ubLayout);
auto dstTensor = asc::te::make_tensor(
    asc::te::make_mem_ptr<asc::te::location::l1>(l1Weight), l1Layout);
auto copyUB2L1 = asc::te::make_copy(Blaze::Gemm::Tile::CopyPaddedUBToL1{});
asc::te::copy(copyUB2L1, dstTensor, srcTensor);
```

NZ 分组使用 `NzColPaddingUBLayout<T>` 构造源布局，并提供对应的 L1 NZ 布局。
`CopyPaddedUBToL1` 通过 `copy_traits` 注册，默认使用 Tensor API 的 `ub_to_l1_trait_default`。
