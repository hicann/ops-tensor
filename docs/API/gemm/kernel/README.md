# Gemm/Kernel 类模板概述

## API 清单

| 组件名 | 描述 |
| :----------------------------------------------------------- | :------: |
| [kernel_matmul_basic](./kernel_matmul_basic.md) | 基础矩阵乘 Kernel，仅 AIC 计算，无 workspace |
| [kernel_qbmm_mx_basic](./kernel_qbmm_mx_basic.md) | MX 量化 Batch Matmul，支持 MxFP4/MxFP8 量化 |
| [kernel_matmul_streamk](./kernel_matmul_streamk.md) | StreamK 矩阵乘 Kernel，AIC+AIV 双核计算，支持 workspace |

## 公共框架

所有 Kernel 组件基于 [kernel.md](./kernel.md) 公共框架实现，包含统一的：
- 模板参数
- 数据结构（Params、Arguments）
- 核心方法（Init、operator）

详见：[kernel.md](./kernel.md)

## 核心组件关系

```
KernelMatmul
    ├── BlockScheduler (任务调度)
    │       ├── Tile 切分策略
    │       ├── Block 分配
    │       └── HF32/L2Cache 配置
    ├── BlockMmad (矩阵乘计算)
    │       ├── GM → L1 → L0 数据搬运
    │       ├── Mmad 计算
    │       └── L0C → GM 结果搬出
    └── BlockEpilogue (后处理)
            └── Empty 或 StreamK 实现
```

## 实现差异对比

| Kernel 类型 | 计算模式 | 量化支持 | Scale 支持 | BlockEpilogue | Workspace | Batch 支持 | BlockScheduler | AIC-AIV 同步 | 适用场景 |
|------------|---------|---------|-----------|---------------|-----------|-----------|---------------|-------------|---------|
| KernelMatmulBasic | 仅 AIC | 不支持 | 不支持 | BlockEpilogueEmpty | 不需要 | 单 batch | MatmulBasic | 无 | 通用 Matmul |
| KernelQbmmMx | 仅 AIC | MX FP4/MX FP8 | ScaleA + ScaleB | 无 | 不需要 | 多 batch | BlockSchedulerQbmm | 无 | 量化 Batch Matmul |
| KernelMatmulStreamK | AIC + AIV 双核 | 不支持 | 不支持 | BlockEpilogueStreamK | 需要 | 单 batch | StreamK Scheduler | 有 | 切 K 场景 Matmul |

## 使用流程

1. **查看公共框架**：了解模板参数和核心接口 → [kernel.md](./kernel.md)
2. **选择具体实现**：根据场景选择 Basic、QBMM MX 或 StreamK
3. **查看特殊约束**：了解各实现的特有约束和方法
4. **组装组件**：定义 ProblemShape、BlockMmad、BlockEpilogue、BlockScheduler
5. **准备参数**：构造 Params 结构体
6. **执行 Kernel**：实例化并调用 operator()