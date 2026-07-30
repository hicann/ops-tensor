# Tile 层组件概述

## API 清单

| 组件名 | 描述 |
| :----------------------------------------------------------- | :------: |
| [tile_mmad_mx](./tile_mmad_mx.md) | MX Mmad Trait 定义，用于量化矩阵乘计算 |
| [pad_mx_k_l1](./pad_mx_kl1.md) | MX K 轴 Padding，用于 L1 数据对齐补零 |
| [copy_gm_to_l1](./copy_gm_to_l1.md) | A 矩阵 ND slice 非连续场景的 GM->L1 搬运 |
| [tile_weight_quant_mx_preprocess](./tile_weight_quant_mx_preprocess.md) | packed FP4 ND/NZ 转换、bias 预缩放和 UB/L1 布局契约 |

## 核心组件关系

```
BlockMmadMX（量化矩阵乘）
    ├── TileMmadMX（MX Mmad Trait）
    │       ├── MmadTraitMX trait 定义
    │       └── 支持 FP4/FP8 量化计算
    └── PadMxKAL1 / PadMxKBL1（K 轴 Padding）
            ├── NZ/ZN 布局补零
            └── 对齐到 C0_SIZE

GemmUniversal（KernelMixWithWeightPrologue）
    └── Weight Quant MX Preprocess
            ├── CopyGM2UBWeight
            ├── ShiftW4ToW8
            ├── CopyUB2L1Weight8Bit
            └── ScaleMxBias
```

## 使用流程

1. **PadMxKL1**：在 GM→L1 搬运后，对 K 轴尾部进行补零对齐
2. **TileMmadMX**：执行 MX Mmad 计算（使用 MmadTraitMX trait）
3. **CopyGM2UBWeight**：将 packed FP4 从 GM 搬入 UB
4. **ShiftW4ToW8 / ScaleMxBias**：转换 FP4 权重，并对可选 bias 预缩放
5. **CopyUB2L1Weight8Bit**：将转换后的 FP8 权重从 UB 写入共享 L1

## 与 Block 层的关系

Tile 层是 Block 层的底层辅助组件，提供：
- **数据对齐**：PadMxKL1 确保 K 轴对齐到 C0_SIZE
- **Mmad Trait**：TileMmadMX 定义 MX 量化计算 trait
- **Weight 前处理**：Weight Quant MX Preprocess 为 `GemmUniversal` Weight Prologue 路径提供 AIV 搬运和转换

详见：[Block Mmad MX](../block/block_mmad_qbmm_mx.md)、
[Block Mmad Weight Prologue MX](../block/block_mmad_weight_prologue_mx.md) 和
[Kernel Matmul Mix Weight Prologue](../kernel/kernel_matmul_mix_weight_prologue.md)。
