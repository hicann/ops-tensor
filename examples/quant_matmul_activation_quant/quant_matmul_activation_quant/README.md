# quant_matmul_activation_quant Example

## 概述

本示例演示基于 Blaze 框架的 MX 量化矩阵乘 + Gelu 激活 + 动态 MX 量化融合算子在昇腾 NPU 上的实现。AIC 执行 MxFP8 量化矩阵乘（DualDst fixpipe 将 L0C 结果搬到 UB），AIV 执行 Gelu 激活和动态 MX 量化，输出量化结果（Y dtype 跟随 A）和 MX Scale。通过 AIC+AIV 双核协同，cube 流水掩盖 vector 流水，提升整体性能。

- **算子**: quant_matmul_activation_quant
- **场景**: AIC MX GEMM + AIV Gelu 激活 + 动态 MX 量化融合
- **算法特点**: A/B 双侧 MX FP8 量化输入，DualDst L0C→UB，OCP 动态 MX 量化输出
- **参考实现**: 基于 Blaze 框架 `blaze/gemm/kernel/kernel_qbmm_mx_activation_quant.h`

## 支持架构

| 架构 | SoC | 支持状态 |
|------|-----|----------|
| dav-3510 | Ascend950 | ✅ |

## 使用约束

- 输入 A shape: `[M, K]`（transA=false）或 `[K, M]`（transA=true）
- 输入 B shape: `[K, N]`（transB=false）或 `[N, K]`（transB=true），权重固定 NZ 布局
- 输出 Y shape: `[M, N]`，输出 Y_scale shape: `[M, ceil(N/64)*2]`
- A/B dtype: `fp8_e4m3`、`fp8_e5m2`（A 和 B 可为不同 FP8 子类型）
- C dtype（epilogue 输出结果）= A dtype
- 输出 Y dtype 跟随 A dtype
- ScaleA/ScaleB/Y_scale dtype: `fp8_e8m0`
- Bias dtype: `float32`

## CSV 驱动测试

### 执行方式

通过统一入口驱动，自动完成编译、数据生成、kernel 执行和精度验证：

```bash

bash examples/common/run.sh --ops=quant_matmul_activation_quant --target=quant_matmul_activation_quant
```

### 测试用例定义

测试用例定义在 `quant_matmul_activation_quant.csv` 中，格式如下：

```csv
nz_basic_fp8e4m3,64,128,128,0,fp8_e4m3,fp8_e4m3,false,false,64,128,64,64,64,2,1,false
nz_bias_fp8e4m3,64,128,128,128,fp8_e4m3,fp8_e4m3,false,false,64,128,64,64,64,2,1,false
nz_fp8e5m2_fp8_e4m3,64,128,128,0,fp8_e5m2,fp8_e4m3,false,false,64,128,64,64,64,2,1,false
nz_fp8e4m3_fp8_e4m3,64,128,128,0,fp8_e4m3,fp8_e4m3,false,false,64,128,64,64,64,2,1,false
nz_fp8e5m2_fp8e4m3_bias,64,128,128,128,fp8_e5m2,fp8_e4m3,false,false,64,128,64,64,64,2,1,false
nz_fp8e4m3_afl,64,128,128,0,fp8_e4m3,fp8_e4m3,false,false,64,128,64,64,64,2,2,true
nz_fp8e5m2_afl_bias,64,128,128,128,fp8_e5m2,fp8_e4m3,false,false,64,128,64,64,64,2,2,true
nz_fp8e4m3_dbl0c2,128,256,256,0,fp8_e4m3,fp8_e4m3,false,false,128,128,128,128,128,2,2,false
nz_fp8e4m3_transB,64,128,128,0,fp8_e4m3,fp8_e4m3,false,true,64,128,64,64,64,2,1,false
nz_fp8e4m3_transA,64,128,128,0,fp8_e4m3,fp8_e4m3,true,false,64,128,64,64,64,2,1,false
nz_fp8e4m3_transA_transB_bias,64,128,128,128,fp8_e4m3,fp8_e4m3,true,true,64,128,64,64,64,2,1,false
nz_fp8e5m2_dbl0c2_afl,128,256,256,256,fp8_e5m2,fp8_e4m3,false,false,128,128,128,128,128,2,2,true
nz_large_fp8e4m3,256,512,256,0,fp8_e4m3,fp8_e4m3,false,false,128,128,128,256,256,2,1,false
nz_fp8e4m3_fp8_e4m3_transB_bias,128,256,256,256,fp8_e4m3,fp8_e4m3,false,true,128,128,128,128,128,2,1,false
```

**列说明**：

| 列 | 说明 |
|----|------|
| casename | 用例名称 |
| m, k, n | 矩阵维度 |
| bias | bias 元素数量，必须为 n 或 0 |
| a_dtype, b_dtype | A/B 量化 dtype: fp8_e4m3, fp8_e5m2 |
| transA, transB | A/B 矩阵是否转置 |
| base_m, base_n, base_k | Cube 基础分块大小 |
| tile_k_l1 | L1 中 K 方向的 Tile 大小 |
| scale_k_l1 | L1 中 Scale K 方向的 Tile 大小 |
| l1_buffers | L1 Buffer 数量 |
| db_l0c | L0C double buffer 开关（1=关，2=开） |
| a_full_load | A 全载到 L1（true/false） |

### 结果输出

执行完成后结果写入 `quant_matmul_activation_quant_result.csv`。

## 数据与校验

### 输入数据

由 `examples/quant_matmul_activation_quant/scripts/gen_data.py` 在 `scripts/input/` 下生成：

- `input_a.bin`: A 矩阵（FP8）
- `input_b.bin`: B 矩阵（FP8，NZ 格式）
- `scale_a.bin`: ScaleA（E8M0，布局随 transA 变化）
- `scale_b.bin`: ScaleB（E8M0，布局随 transB 变化）
- `bias.bin`: bias 向量（FP32，bias>0 时生成）
- `golden_y.bin`: NumPy 计算得到的 Golden 输出（dtype 跟随 A）
- `golden_y_scale.bin`: NumPy 计算得到的 E8M0 Golden Scale

### 输出数据

- `scripts/output/npu_y.bin`: NPU 计算得到的量化输出
- `scripts/output/npu_y_scale.bin`: NPU 计算得到的 E8M0 输出 Scale

### 验证标准

由 `examples/quant_matmul_activation_quant/scripts/verify_result.py` 执行校验。Y 和 Y_scale 均转换为 float32 后逐点比较，超过 `atol` 时记为误差点：

| dtype | rtol | atol |
|-------|------|------|
| fp8_e4m3 | 1e-3 | 1.0 |
| fp8_e5m2 | 1e-3 | 1.0 |
| fp8_e8m0 (Y_scale) | 1e-3 | 1.0 |

## 代码结构

```text
quant_matmul_activation_quant/
├── quant_matmul_activation_quant.cpp           # kernel 实现
├── quant_matmul_activation_quant.conf          # 参数路由配置
├── quant_matmul_activation_quant.csv           # CSV 测试用例
└── README.md                                   # 本文档
```

构建配置在 op 层 `examples/quant_matmul_activation_quant/CMakeLists.txt` 中统一管理；运行通过 `examples/common/run.sh` 统一调度，数据生成和精度校验由 `examples/quant_matmul_activation_quant/scripts/` 下的 `gen_data.py` 和 `verify_result.py` 执行。

## Blaze 组件

| 组件 | 头文件 | 职责 |
|------|--------|------|
| Kernel | `blaze/gemm/kernel/kernel_qbmm_mx_activation_quant.h` | AIC+AIV 融合 kernel 入口 |
| Block MMAD | `blaze/gemm/block/block_mmad_qbmm_mx.h` | Block 级 MX 量化矩阵乘 |
| Block Scheduler | `blaze/gemm/block/block_scheduler_qbmm.h` | QBMM 调度器 |
| Epilogue | `blaze/epilogue/block/block_epilogue_gelu_mx_quant.h` | Gelu 激活 + 动态 MX 量化 |
| Dispatch Policy | `blaze/gemm/policy/dispatch_policy.h` | MatmulWithScaleMx (DualDst) |
