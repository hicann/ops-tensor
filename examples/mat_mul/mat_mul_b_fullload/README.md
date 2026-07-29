# mat_mul_b_fullload Example

## 概述

本示例演示基于 Blaze 框架的 MatMul 矩阵乘法算子在昇腾 NPU 上的 BFullLoad 实现。B 矩阵完整加载到 L1 缓存中，A 矩阵按 K 维度分块流式加载。适用于 N 维度较小、B 矩阵可以完整放入 L1 缓存的场景。

- **算子**: mat_mul
- **场景**: mat_mul_b_fullload
- **算法特点**: B 矩阵全量加载到 L1，A 矩阵流式加载，支持 FP16/BF16/FP32/HF32 及 Weight NZ 格式
- **参考实现**: 基于 Blaze 框架 `blaze/gemm/kernel/kernel_matmul_b_full_load.h`

## 支持架构

| 架构 | SoC | 支持状态 |
|------|-----|----------|
| dav-3510 | Ascend950 | ✅ |

## 使用约束

- 输入 A shape: `[M, K]`（transA=false）或 `[K, M]`（transA=true）
- 输入 B shape: `[K, N]`（transB=false）或 `[N, K]`（transB=true）
- 输出 C shape: `[M, N]`
- 数据类型: float16, bfloat16, float32
- B 矩阵需要足够小以完整放入 L1 缓存

## CSV 驱动测试

### 执行方式

```bash
bash run.sh --case=mat_mul_b_fullload.csv
```

### 测试用例定义

```csv
casename,m,k,n,bias,dtype,transA,transB,hf32,format
mat_mul_b_fullload_fp16,256,512,32,32,float16,false,false,false,"(ND,ND)"
mat_mul_b_fullload_bf16,256,512,32,32,bfloat16,false,false,false,"(ND,ND)"
mat_mul_b_fullload_fp32,256,512,32,32,float32,false,false,false,"(ND,ND)"
mat_mul_b_fullload_hf32,256,512,32,32,float32,false,false,true,"(ND,ND)"
mat_mul_b_fullload_weightNz,256,512,32,0,float16,false,false,false,"(ND,NZ)"
```

## 代码结构

```
mat_mul_b_fullload/
├── CMakeLists.txt                  # 构建配置
├── mat_mul_b_fullload.cpp          # 统一 kernel 实现
├── mat_mul_b_fullload.csv          # CSV 测试用例
├── parse_csv.py                    # CSV 解析与批量执行
├── run.sh                          # 运行脚本
└── README.md                       # 本文档
```

## Blaze 组件

| 组件 | 头文件 | 职责 |
|------|--------|------|
| Kernel | `blaze/gemm/kernel/kernel_matmul_b_full_load.h` | BFullLoad kernel 入口 |
| Block MMAD | `blaze/gemm/block/block_mmad_matmul_b_fullLoad_fixpipe_opti.h` | B 全量加载的 Block 级矩阵乘 |
| Block Scheduler | `blaze/gemm/block/block_scheduler_matmul_basic.h` | Basic 调度器 (B_FULL_LOAD_MODE) |
| Epilogue | `blaze/epilogue/block/block_epilogue_empty.h` | 空后处理 |
| Dispatch Policy | `blaze/gemm/policy/dispatch_policy.h` | MatmulMultiBlockFullLoadOrFixpipe |
