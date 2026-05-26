# Blaze GEMM Library 架构概述

## 简介

Blaze (Basic Linear Algebra optimized Engine) 是一个高性能矩阵乘库，专为华为 Ascend NPU 平台优化。采用分层架构设计，提供灵活的组件组装能力，支持多种矩阵乘场景（Basic、StreamK、MX 量化）。

## 整体架构

```
+-----------------------------------------------------------------------+
|                          Application Layer                             |
|                     (User Code / Solution / Tiling)                    |
+-----------------------------------------------------------------------+
|  +--------------------------------------------------------------+      |
|  |                         Kernel Layer                         |      |
|  |  +-------------+  +-------------+  +-------------+           |      |
|  |  |KernelMatmul |  |KernelStreamK|  |KernelQbmmMx |           |      |
|  |  |   Basic     |  |             |  |             |           |      |
|  |  |+------------+  +-------------+  +-------------+           |      |
|  +--------------------------------------------------------------+      |
|
|        |                 |                 |                           |
|        V                 V                 V                           |
|  +--------------------------------------------------------------+      |
|  |                       Block Layer                             |     |
|  |  +-------------+  +-------------+  +-------------+            |     |
|  |  | Scheduler   |  |    Mmad     |  |  Epilogue   |            |     |
|  |  +-------------+  +-------------+  +-------------+            |     |
|  +--------------------------------------------------------------+      |
|                              |                                         |
|                              V                                         |
|  +--------------------------------------------------------------+      |
|  |                        Tile Layer                             |     |
|  |  +-------------+  +-------------+  +-------------+            |     |
|  |  | PadMxKL1    |  | CopyScale   |  | TileMmadMX  |            |     |
|  |  +-------------+  +-------------+  +-------------+            |     |
|  +--------------------------------------------------------------+      |
|                              |                                         |
|                              V                                         |
|  +--------------------------------------------------------------+      |
|  |                       Utils Layer                             |     |
|  |  +-------------+  +-------------+  +-------------+            |     |
|  |  | CommonUtils |  | LayoutUtils |  |  Constant   |            |     |
|  |  +-------------+  +-------------+  +-------------+            |     |
|  +--------------------------------------------------------------+      |
+-----------------------------------------------------------------------+
                                |
                                V
+-----------------------------------------------------------------------+
|                         Hardware Layer                                 |
|  +-----------+  +-----------+  +-----------+  +-----------+            |
|  |    GM     |  |    L1     |  |  L0A/L0B  |  |    L0C    |            |
|  +-----------+  +-----------+  +-----------+  +-----------+            |
|                              |                                         |
|                              V                                         |
|  +--------------------------------------------------------+            |
|  |                     Compute Units                      |            |
|  |  +----------+  +----------+  +----------+              |            |
|  |  |   AIC    |  |   AIV    |  |   Mmad   |              |            |
|  |  +----------+  +----------+  +----------+              |            |
|  +--------------------------------------------------------+            |
+-----------------------------------------------------------------------+
```

## 层级详解

### Kernel Layer（Kernel 层）
职责：整体计算流程编排，GM Tensor 创建，多 Block 并行调度。

详细文档：[Kernel 层 API](kernel/README.md)

### Block Layer（Block 层）
职责：单个 Block 的矩阵乘计算、任务调度、后处理。

详细文档：[Block 层 API](block/README.md) | [Epilogue 层 API](epilogue/README.md)

### Tile Layer（Tile 层）
职责：底层辅助组件，数据对齐、Scale 搬运、Trait 定义。

详细文档：[Tile 层 API](tile/README.md)

### Policy Layer（策略层）
职责：调度策略、配置模式定义。

详细文档：[Policy 层](../include/blaze/policy/dispatch_policy.h)

### Utils Layer（工具层）
职责：公共工具函数、布局判断、常量定义。


详细文档：[Utils 层](../include/blaze/utils/)

## 数据流示意图

### Basic Kernel 数据流
```
┌───────────────────────────────────────────────────────────────┐
│                      GM (A, B, Bias)                          │
└────────────────────────────┬──────────────────────────────────┘
                             │ GM→L1 (BlockScheduler 调度)
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                       L1 (双缓冲)                              │
│               A0|A1  B0|B1  Bias0|Bias1                       │
└────────────────────────────┬──────────────────────────────────┘
                             │ L1→L0 (BlockMmad 迭代)
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                   L0A/L0B (双缓冲)                             │
│                     A_L0  B_L0                                │
└────────────────────────────┬──────────────────────────────────┘
                             │ Mmad 计算
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                          L0C                                  │
│                      C 结果 (float)                           │
└────────────────────────────┬──────────────────────────────────┘
                             │ Fixpipe (L0C→GM)
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                       GM (C 输出)                             │
└───────────────────────────────────────────────────────────────┘
```


## 组件组装示意图

```
┌───────────────────────────────────────────────────────────────┐
│                    Kernel 组装示例                             │
├───────────────────────────────────────────────────────────────┤
│ // 定义数据类型和布局                                           │
│ using AType = half;                                            │
│ using BType = half;                                            │
│ using LayoutA = AscendC::Te::NZLayoutPtn;                      │
│ using LayoutB = AscendC::Te::NZLayoutPtn;                      │
│                                                                │
│ // 定义 ProblemShape                                           │
│ using ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;│
│                                                                │
│ // 组装 BlockScheduler                                         │
│ using BlockScheduler = BlockSchedulerMatmulBasic<...>;         │
│                                                                │
│ // 组装 BlockMmad                                              │
│ using BlockMmad = BlockMmad<                                   │
│     DispatchPolicy, AType, LayoutA,                            │
│     BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;     │
│                                                                │
│ // 组装 BlockEpilogue                                          │
│ using BlockEpilogue = BlockEpilogueEmpty;                      │
│                                                                │
│ // 组装 Kernel                                                 │
│ using Kernel = KernelMatmulBasic<                              │
│     ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;   │
└───────────────────────────────────────────────────────────────┘
```

## 目录结构

```
blaze/
├── include/blaze/
│   ├── epilogue/            # Epilogue 层组件
│   │   ├── block_epilogue_empty.h
│   │   └── block_epilogue_matmul_streamk.h
|   └── gemm/
│       ├── kernel/              # Kernel 层组件
│       │   ├── kernel_matmul_basic.h
│       │   ├── kernel_matmul_streamk.h
│       │   └── kernel_qbmm_mx.h
│       │
│       ├── block/               # Block 层组件
│       │   ├── block_mmad.h          # BlockMmad 基类
│       │   ├── block_mmad_matmul_basic.h    # Matmul Basic BlockMmad
│       │   ├── block_mmad_matmul_streamk.h  # Matmul StreamK BlockMmad
│       │   ├── block_mmad_mx.h       # MX BlockMmad
│       │   ├── block_scheduler_matmul_basic.h
│       │   ├── block_scheduler_matmul_streamk.h
│       │   └── block_scheduler_qbmm.h
│       │
│       ├── tile/                # Tile 层组件
│       │   ├── tile_mmad_mx.h
│       │   ├── pad_mx_kl1.h
│       │   ├── copy_scale_l1_to_l0a.h
│       │   └── copy_scale_l1_to_l0b.h
│       │
│       ├── policy/              # Policy 层
│       │   └── dispatch_policy.h
│       │
│       └── utils/               # Utils 层
│       ├── common_utils.h
│       ├── layout_utils.h
│       └── quant_batch_matmul_constant.h
│
└── docs/API/                # API 文档
    ├── README.md            # 本文档
    ├── epilogue/            # Epilogue 层文档
    |   └── block/                # Block 层文档
    └── gemm/                # Gemm 层文档
        ├── kernel/              # Kernel 层文档
        ├── block/               # Block 层文档
        └── tile/                # Tile 层文档
```

## 快速开始

### 1. 查看架构图
了解 Blaze 分层设计和数据流 → 本文档

### 2. 选择 Kernel 类型
根据场景选择 Basic、StreamK 或 MX → [Kernel 层 API](kernel/README.md)

### 3. 组装 Block 组件
选择 BlockScheduler、BlockMmad、BlockEpilogue → [Block 层 API](block/README.md)

### 4. 了解底层组件
Tile 层辅助组件 → [Tile 层 API](tile/README.md)

### 5. 配置策略
调度策略和配置 → [Policy 层](../../include/blaze/gemm/policy/dispatch_policy.h)

## 版本信息

- **平台**：华为 Ascend NPU（3510 架构）
- **编译器**：ASC_DEVKIT_MAJOR >= 9
- **量化格式**：MxFP4（fp4x2_e2m1_t, fp4x2_e1m2_t）、MxFP8（fp8_e5m2_t, fp8_e4m3fn_t）
- **Scale 格式**：fp8_e8m0_t（E8M0 浮点）