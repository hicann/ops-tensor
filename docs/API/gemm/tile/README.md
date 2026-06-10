# Tile 层组件概述

## API 清单

| 组件名 | 描述 |
| :----------------------------------------------------------- | :------: |
| [tile_mmad_mx](./tile_mmad_mx.md) | MX Mmad Trait 定义，用于量化矩阵乘计算 |
| [pad_mx_k_l1](./pad_mx_kl1.md) | MX K 轴 Padding，用于 L1 数据对齐补零 |

## 核心组件关系

```
BlockMmadMX（量化矩阵乘）
    ├── TileMmadMX（MX Mmad Trait）
    │       ├── MmadTraitMX trait 定义
    │       └── 支持 FP4/FP8 量化计算
    └── PadMxKAL1 / PadMxKBL1（K 轴 Padding）
            ├── NZ/ZN 布局补零
            └── 对齐到 C0_SIZE
```

## 使用流程

1. **PadMxKL1**：在 GM→L1 搬运后，对 K 轴尾部进行补零对齐
2. **TileMmadMX**：执行 MX Mmad 计算（使用 MmadTraitMX trait）

## 与 Block 层的关系

Tile 层是 Block 层的底层辅助组件，提供：
- **数据对齐**：PadMxKL1 确保 K 轴对齐到 C0_SIZE
- **Mmad Trait**：TileMmadMX 定义 MX 量化计算 trait

详见：[Block Mmad MX](../block/block_mmad_qbmm_mx.md)