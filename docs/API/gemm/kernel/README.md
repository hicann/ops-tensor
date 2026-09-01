# Gemm/Kernel 模板概览

## API 清单

| 组件名 | 说明 |
| :--- | :---: |
| [kernel_matmul_basic](./kernel_matmul_basic.md) | 基础矩阵乘 Kernel，仅 AIC 计算，无 workspace |
| [kernel_matmul_bl1_full_load](./kernel_matmul_bl1_full_load.md) | B 矩阵 L1 全载 Kernel，支持 ON_THE_FLY（AIC 直销）和 Fixpipe（AIC+AIV 双核）两种输出 |
| [kernel_matmul_fixpipe_opti](./kernel_matmul_fixpipe_opti.md) | Fixpipe 非全载 Kernel，AIC+AIV 双核，根据 FULL_LOAD_MODE 自动选择非全载/B全载 calc BlockMmad |
| [kernel_qbmm_cube](./kernel_qbmm_cube.md) | Fixpipe 量化 Batch Matmul，支持 int8/HiFloat8/FP8 输入与 per-tensor/per-channel scale |
| [kernel_qbmm_mx](./kernel_qbmm_mx.md) | MX 量化 Batch Matmul，支持 MxFP4/MxFP8 |
| [kernel_qbmm_mx_without_batch](./kernel_qbmm_mx_without_batch.md) | MX 量化单 Batch Matmul，裁剪 Batch 广播路径 |
| [kernel_qbmm_mx_activation_quant](./kernel_qbmm_mx_activation_quant.md) | MX 量化 Batch Matmul + Gelu 激活 + 动态 MX 量化融合 Kernel，AIC+AIV 双核 |
| [kernel_qbmm_mix](./kernel_qbmm_mix.md) | MIX 模板 A8W8 量化 Batch Matmul，AIC cube + AIV dequant 后处理 |
| [kernel_qbmm_mix_without_batch](./kernel_qbmm_mix_without_batch.md) | MIX 模板 A8W8 量化单 Batch Matmul，裁剪 Batch 广播路径 |
| [kernel_qgmm_mx_basic](./kernel_qgmm_mx_basic.md) | MX 量化 Grouped Matmul，支持 group list 与 tail split |
| [kernel_qgmm_mix_fixpipe_quant](./kernel_qgmm_mix_fixpipe_quant.md) | Fixpipe 量化 Grouped Matmul，支持 per-channel/per-group 和可选 offset 后处理 |
| [kernel_matmul_streamk](./kernel_matmul_streamk.md) | StreamK 矩阵乘 Kernel，AIC+AIV 双核计算，支持 workspace |
| [kernel_qbmm_streamk](./kernel_qbmm_streamk.md) | MX 量化 StreamK Kernel，支持单 Batch MxFP4/MxFP8 workspace 归约 |
| [kernel_qbmm_pertensor_streamk](./kernel_qbmm_pertensor_streamk.md) | QBMM per-tensor StreamK Kernel，AIC raw partial + AIV 统一反量化 |
| [kernel_matmul_mix_weight_prologue](./kernel_matmul_mix_weight_prologue.md) | AIV 权重前处理 + AIC MX MMAD 的 Mix Kernel |
| [kernel_wqgmm_mix_weight_prologue](./kernel_wqgmm_mix_weight_prologue.md) | Grouped MX A8W4 Mix Kernel，支持 E2M1/E1M2、FP16/BF16 输出、可选 Bias 和单/多 Weight |
| [kernel_matmul_with_scale_add](./kernel_matmul_with_scale_add.md) | FusedMatMul scale_add Kernel，AIC 矩阵乘 + AIV 缩放相加后处理 |

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

| Kernel 类型 | 计算模式 | 量化支持 | Scale 支持 | BlockEpilogue | Workspace | Batch 支持 | BlockScheduler | AIC-AIV 同步 | 适用场景 |
|------------|---------|---------|-----------|---------------|-----------|-----------|---------------|-------------|---------|
| KernelMatmulBasic | 仅 AIC | 不支持 | 不支持 | BlockEpilogueEmpty | 不需要 | 单 batch | MatmulBasic | 无 | 通用 Matmul |
| KernelMatmulBL1FullLoad | 仅 AIC / AIC+AIV | 不支持 | 不支持 | BlockEpilogueEmpty / BlockEpilogueFixpipe | 不需要 | 单 batch | MatmulBasic | 有（Fixpipe） | B 全载 Matmul，大 K/N 场景 |
| KernelMatmulFixpipeOpti | AIC+AIV 双核 | 不支持 | 不支持 | BlockEpilogueFixpipe | 不需要 | 单 batch | MatmulBasic | 有 | 非全载 Fixpipe，小 K 场景 |
| KernelQbmmCube | 仅 AIC | int8/HiFloat8/FP8 | X2 scale + Fixpipe | 无 | 不需要 | 多 batch | BlockSchedulerQuantBatchMatmulV3 | 无 | Fixpipe 量化 Batch Matmul |
| KernelQbmmMx | 仅 AIC | MX FP4/MX FP8 | ScaleA + ScaleB | 无 | 不需要 | 多 batch | BlockSchedulerQbmm | 无 | 量化 Batch Matmul |
| KernelQbmmMxActivationQuant | AIC + AIV 双核 | MX FP4/MX FP8 | ScaleA + ScaleB | BlockEpilogueGeluQuant | 不需要 | 多 batch | BlockSchedulerQbmm | 有 | 量化 Matmul + Gelu 激活 + 动态 MX 量化融合 |
| KernelQbmmMxWithoutBatch | 仅 AIC | MX FP4/MX FP8 | ScaleA + ScaleB | 无 | 不需要 | 单 batch | BlockSchedulerQbmm | 无 | 量化单 Batch Matmul |
| KernelQbmmMix | AIC + AIV 双核 | int8 (A8W8) | x2Scale + x1Scale(可选) | BlockEpilogueDequant | 不需要 | 多 batch | BlockSchedulerQbmm | 有 | int8 量化 Batch Matmul（ND/WeightNz） |
| KernelQbmmMixWithoutBatch | AIC + AIV 双核 | int8 (A8W8) | x2Scale + x1Scale(可选) | BlockEpilogueDequant | 不需要 | 单 batch | BlockSchedulerQbmm | 有 | int8 量化单 Batch Matmul（ND/WeightNz） |
| KernelQgmmMx | 仅 AIC | MX FP4/MX FP8 | ScaleA + ScaleB | 无 | 不需要 | group list | BlockSchedulerGmmSwatWithTailSplit | 无 | 量化 Grouped Matmul |
| KernelMatmulStreamK | AIC + AIV 双核 | 不支持 | 不支持 | BlockEpilogueStreamK | 需要 | 单 batch | StreamK Scheduler | 有 | 切 K 场景 Matmul |
| KernelQbmmStreamK | AIC + AIV 双核 | MX FP4/MX FP8 | ScaleA + ScaleB | BlockEpilogueStreamK（复用） | 需要 | 单 batch | StreamK Scheduler（复用） | 有 | 量化切 K 场景 Matmul |
| KernelQbmmPertensorStreamK | AIC + AIV 双核 | int8/FP8/HiFloat8 | X2 per-tensor + 可选 X1 scale | BlockEpilogueQbmmPertensorStreamK | 需要 | 单 batch | StreamK Scheduler | 有 | QBMM per-tensor StreamK |
| GemmUniversal (Weight Prologue) | AIC + AIV 双核 | FP8 激活 + packed FP4 权重 | ScaleA + ScaleB | `void` | 不需要 | 单 batch | Matmul SWAT | 有（ready/free 标志） | MXA8W4 Weight ND/NZ |
| GmmWeightQuantMxKernel | AIC + AIV 双核 | FP8 激活 + packed FP4 E2M1/E1M2 权重 | ScaleA + ScaleB | `void` | 不需要 | group list | BlockSchedulerWqgmmNResplit | 有（ready/free 标志） | MX A8W4 Grouped Matmul |
| KernelMatmulWithScaleAdd | AIC + AIV 双核 | 不支持 | alpha + beta | BlockEpilogueFmmWithScaleAdd | 不需要 | 多 batch | MatmulBasic | 有 | FusedMatMul scale_add |

## 使用流程

1. **查看公共框架**：了解模板参数和核心接口 → [kernel.md](./kernel.md)
2. **选择具体实现**：根据场景选择 Basic、TBMM、QBMM Cube、QBMM MX、QBMM MIX、Weight Prologue、QGMM MX 或 StreamK
3. **查看特殊约束**：了解各实现的特有约束和方法
4. **组装组件**：定义 ProblemShape、BlockMmad、BlockEpilogue、BlockScheduler
5. **准备参数**：构造 Params 结构体
6. **执行 Kernel**：实例化并调用 operator()
