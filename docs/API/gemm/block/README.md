# Gemm/Block 类模板概述

## API 清单

### BlockMmad（矩阵乘计算）
| 组件名 | 描述 |
| :----------------------------------------------------------- | :------: |
| [block_mmad_matmul_basic](./block_mmad_matmul_basic.md) | 基础矩阵乘 Block，基于 Tensor API，支持 L1/L0 双缓冲 |
| [block_mmad_qbmm_mx](./block_mmad_qbmm_mx.md) | MX 量化矩阵乘 Block，支持 Scale 因子、MxFP4/MxFP8 量化 |
| [block_mmad_a8w8_mix](./block_mmad_a8w8_mix.md) | MIX 模板 A8W8 量化矩阵乘 Block，int32 累加 + L0C→UB（fixpipe NoQuant），不做 scale/bias |
| [block_mmad_qgmm_mx](./block_mmad_qgmm_mx.md) | MX 量化 Grouped Matmul Block，支持 group list、ScaleA/ScaleB |
| [block_mmad_qbmm_mx_l0c_pingpong](./block_mmad_qbmm_mx_l0c_pingpong.md) | MX 量化矩阵乘 L0C PingPong Block，支持 N 方向拆分、Scale 复用和 SplitK 写回控制 |
| [block_mmad_matmul_streamk](./block_mmad_matmul_streamk.md) | StreamK 矩阵乘 Block，支持 workspace 输出、K 轴切分 |

### BlockScheduler（任务调度）
| 组件名 | 描述 |
| :----------------------------------------------------------- | :------: |
| [block_scheduler](./block_scheduler.md) | 公共框架：tile 切分、Z 型扫描、尾块处理 |
| [block_scheduler_matmul_basic](./block_scheduler_matmul_basic.md) | Basic 调度器：尾块切分、SplitK 切分、L2Cache 配置 |
| [block_scheduler_matmul_streamk](./block_scheduler_matmul_streamk.md) | StreamK 调度器：DP+SK 混合策略、K 轴切分 |
| [block_scheduler_qbmm_mx](./block_scheduler_qbmm_mx.md) | QBMM 调度器：Batch 维度切分、量化对齐 |
| [block_scheduler_gmm_aswt_with_tail_split](./block_scheduler_gmm_aswt_with_tail_split.md) | QGMM 调度器：group 间连续分核、SWAT 扫描、末组 tail split |

## 公共框架

### BlockMmad 公共框架
所有 BlockMmad 组件基于 [block_mmad.md](./block_mmad.md) 公共框架实现，包含统一的：
- 模板参数
- 数据结构（Params）
- 核心方法（Init、operator）

详见：[block_mmad.md](./block_mmad.md)

### BlockScheduler 公共框架
所有 BlockScheduler 组件基于 [block_scheduler.md](./block_scheduler.md) 公共框架实现，包含统一的：
- 模板参数（ProblemShape）
- 类型别名（BlockShape、BlockCoord）
- Z 型扫描策略
- 尾块处理

详见：[block_scheduler.md](./block_scheduler.md)

## 核心组件关系

```
BlockMmad
    ├── DispatchPolicy (调度策略)
    │       ├── MatmulMultiBlockBasic (Basic)
    │       ├── MatmulMultiBlockWithStreamK (StreamK)
    │       ├── MatmulWithScaleMx (QBMM MX 量化)
    │       ├── MatmulWithScaleMix (QBMM MIX A8W8 量化)
    │       ├── MatmulWithScaleMxL0CPingpong (QBMM MX L0C PingPong)
    │       └── GroupedMatmulWithScaleMx (QGMM MX 量化)
    ├── 数据类型 (AType, BType, CType, BiasType)
    ├── 布局类型 (LayoutA, LayoutB, LayoutC, LayoutBias)
    └── 计算流程
            ├── GM → L1 → L0 数据搬运
            ├── Mmad 计算
            └── L0C → GM/workspace 结果搬出
```

## 实现差异对比

| Block 类型 | 调度策略 | 输出目标 | 量化支持 | Scale 支持 | L1 双缓冲 | L0C 双缓冲 | Bias 支持 | AIC-AIV 同步 | 适用场景 |
|-----------|---------|---------|---------|-----------|---------|-----------|---------|-------------|---------|
| BlockMmadBasic | MatmulMultiBlockBasic | GM | 不支持 | 不支持 | 可配置 (1 或 2) | 可配置 | 支持 | 无 | Basic Kernel |
| BlockMmadStreamK | MatmulMultiBlockWithStreamK | GM 或 workspace | 不支持 | 不支持 | 固定双缓冲 | 固定单缓冲 | 支持 | 无（Kernel 层处理） | StreamK Kernel |
| BlockMmadMx | MatmulWithScaleMx | GM | MxFP4/MxFP8 | ScaleA + ScaleB | 可配置 (2 或 4) | 可配置 | 支持 | 无 | QBMM MX Kernel |
| BlockMmadA8W8Mix | MatmulWithScaleMix | UB (L0C→UB) | int8 (A8W8) | 不在本层（由 epilogue 处理） | 可配置 (2 或 4) | 可配置 | 不在本层 | 无（Kernel 层处理） | QBMM MIX Kernel |
| BlockMmadQGmmMx | GroupedMatmulWithScaleMx | GM | MxFP4/MxFP8 | ScaleA + ScaleB | 固定双缓冲 | 可配置 | 支持 | 无 | QGMM MX Kernel |
| BlockMmadMxL0CPingpong | MatmulWithScaleMxL0CPingpong | GM | MxFP4/MxFP8 | ScaleA + ScaleB | 可配置 (2 或 4) | 固定双缓冲 | 支持 | 无 | QBMM MX L0C PingPong Kernel |

## 使用流程

1. **查看公共框架**：了解模板参数和核心接口 → [block_mmad.md](./block_mmad.md)
2. **选择具体实现**：根据 Kernel 类型选择 Basic、StreamK、MX 或 L0C PingPong MX
3. **定义调度策略**：选择 DispatchPolicy（TensorApi、StreamK、ScaleMx 或 ScaleMxL0CPingpong）
4. **组装组件**：定义数据类型、布局类型
5. **初始化**：调用 Init 设置 tile 形状、缓冲策略
6. **执行计算**：调用 operator 执行矩阵乘
