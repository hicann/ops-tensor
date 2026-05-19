# Tile Mmad MX
> [代码位置](../../../include/blaze/tile/tile_mmad_mx.h)

## 功能说明
MX 量化矩阵乘 Tile，定义 MmadTraitMX trait，用于支持 MxFP4/MxFP8 量化矩阵乘计算。

## 特殊约束

### 数据类型支持
仅支持 MxFP4 和 MxFP8 量化数据类型：
- **MxFP4**：`fp4x2_e2m1_t`、`fp4x2_e1m2_t`
- **MxFP8**：`fp8_e5m2_t`、`fp8_e4m3fn_t`

### Trait 定义
```
constexpr MmadTrait MX_MMAD_TRAIT = MmadTrait{0, false, false, true, MmadType::MX};
```

说明：
- `MmadType::MX`：标识 MX 量化计算模式
- 自动应用于 Mmad 计算

## 特殊类型

### MmadTraitMX
```
struct MmadTraitMX {
    using TraitType = MmadTrait;
    static constexpr const TraitType value = MX_MMAD_TRAIT;
};
```

功能：MX Mmad Trait 定义，用于量化矩阵乘。

### MmadTraits 特化
```
template <>
struct MmadTraits<MmadOperation, MmadTraitMX>
    : public MmadTraits<MmadOperation, MmadTraitDefault, MmadOpWith, MmadTraitMX> {};
```

功能：MmadTraits 针对 MmadTraitMX 的特化，继承默认 trait。

## 使用示例

### 在 BlockMmadMX 中使用
```
// Mmad 计算（自动使用 MmadTraitMX）
AscendC::Te::Mmad(
    AscendC::Te::MmadAtom<
        AscendC::Te::MmadTraits<
            AscendC::Te::MmadOperation,
            AscendC::Te::MmadTraitMX>>{},
    tensorL0C, tensorAL0, tensorBL0);
```

说明：
- `MmadTraitMX` 自动应用于 MX 量化计算
- 无需手动指定 trait 参数

## 数据流

```
L0A（量化数据） + L0B（量化数据）
    ↓
Mmad（MmadTraitMX）
    ↓
L0C（float 结果）
```

说明：量化数据在 Mmad 计算时自动反量化为 float。