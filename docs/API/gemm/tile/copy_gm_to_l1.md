# Copy GM To L1
> [代码位置](../../../../include/blaze/gemm/tile/copy_gm_to_l1.h)

## 功能说明
A 矩阵 ND slice 非连续场景的 GM->L1 搬运 Tile，配合 `MatmulMultiBlockBasic` 的 `NON_CONTIGUOUS_TYPE_SLICE` 路径使用。

该组件将三维 GM layout `[ndNum, [sliceM, curK]]` 按 ND2NZ 方式搬运到 L1 NZ layout，GM stride 为 `[srcNdStride, [k, 1]]`。

## 特殊约束

### 调用场景
仅用于 A 矩阵 ND slice 非连续输入。普通连续输入继续使用默认 `CopyGM2L1`。

### 架构支持
当前实现位于 `tile/arch35/copy_gm_to_l1.h`，仅在 `__NPU_ARCH__ == 3510` 时引入。

### 数据类型
支持 `half` 和 `float` 数据类型。

## 特殊类型

### CopySliceGM2L1
```cpp
struct CopySliceGM2L1 {
    template <typename Tp, const Tp& traits, typename T, typename U>
    __aicore__ inline static void Copy(const T& dst, const U& src);
};
```

功能：将 A 矩阵 slice 后的三维 GM Tensor 搬运到 L1 Tensor。

## 使用方式

```cpp
auto copyGM2L1Slice = AscendC::Te::MakeCopy(Blaze::Gemm::Tile::CopySliceGM2L1{});
AscendC::Te::Copy(copyGM2L1Slice, tensorAL1, gmTileASlice);
```

说明：
- `gmTileASlice` 的 shape 为 `[curM / sliceM, [sliceM, curK]]`
- `tensorAL1` 的 layout 为 A 矩阵 L1 NZ layout
- 该路径由 `BlockMmadMatmulBasic` 在 `NON_CONTIGUOUS_TYPE_SLICE` 场景下自动选择
