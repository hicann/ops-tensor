# Gemm/Kernel 模板概览

## API 清单

| 组件名 | 说明 |
| :--- | :---: |
| [kernel_matmul_basic](./kernel_matmul_basic.md) | 基础矩阵乘 Kernel，仅 AIC 计算，无 workspace |
| [kernel_qbmm_mx](./kernel_qbmm_mx.md) | MX 量化 Batch Matmul，支持 MxFP4/MxFP8 |
| [kernel_qbmm_mx_without_batch](./kernel_qbmm_mx_without_batch.md) | MX 量化单 Batch Matmul，裁剪 Batch 广播路径 |
| [kernel_qgmm_mx_basic](./kernel_qgmm_mx_basic.md) | MX 量化 Grouped Matmul，支持 group list 与 tail split |
| [kernel_matmul_streamk](./kernel_matmul_streamk.md) | StreamK 矩阵乘 Kernel，AIC+AIV 双核计算，支持 workspace |

## 公共框架

所有 Kernel 组件均基于 [kernel.md](./kernel.md) 公共框架实现，统一包含：
- 模板参数
- 数据结构，如 `Params`、`Arguments`
- 核心方法，如 `Init`、`operator()`

详见：[kernel.md](./kernel.md)

## 核心组件关系

```text
KernelMatmul
    -> BlockScheduler
    -> BlockMmad
    -> BlockEpilogue
```

## 实现差异

| Kernel 类型 | 计算模式 | 量化支持 | Workspace | 典型场景 |
| :--- | :---: | :---: | :---: | :---: |
| KernelMatmulBasic | AIC | 否 | 否 | 通用 Matmul |
| KernelQbmmMx | AIC | 是 | 否 | 量化 Batch Matmul |
| GemmUniversal | AIC | 是 | 否 | 量化 Grouped Matmul |
| KernelMatmulStreamK | AIC + AIV | 否 | 是 | 大 K 或 StreamK 场景 |

## 使用流程

1. 查看公共框架：[kernel.md](./kernel.md)
2. 选择具体实现：Basic、QBMM MX、QGMM MX 或 StreamK
3. 组装 `ProblemShape`、`BlockMmad`、`BlockEpilogue`、`BlockScheduler`
4. 构造 `Params`
5. 实例化 Kernel 并调用 `operator()`
