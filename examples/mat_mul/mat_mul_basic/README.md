# mat_mul_basic Example

## 概述

本示例演示基于 Blaze 框架的 MatMul 矩阵乘法算子在昇腾 NPU 上的 Basic 多块实现。Basic 算法采用标准的多块分块策略，将矩阵乘计算任务均匀分配到多个 NPU 核心，适用于 M*N 足够大以充分利用 NPU 核心的场景。

- **算子**: mat_mul
- **场景**: mat_mul_basic
- **算法特点**: 标准多块分块，支持 FP16/BF16/FP32/HF32 及 Weight NZ 格式
- **参考实现**: 基于 Blaze 框架 `blaze/gemm/kernel/kernel_matmul_basic.h`

## 支持架构

| 架构 | SoC | 支持状态 |
|------|-----|----------|
| dav-3510 | Ascend950 | ✅ |

## 使用约束

- 输入 A shape: `[M, K]`（transA=false）或 `[K, M]`（transA=true）
- 输入 B shape: `[K, N]`（transB=false）或 `[N, K]`（transB=true）
- 输出 C shape: `[M, N]`
- 数据类型: float16, bfloat16, float32
- Weight NZ 场景: B 矩阵需使用 NZ 格式存储，仅支持 float16/bfloat16

## CSV 驱动测试

### 执行方式

```bash
bash run.sh --case=mat_mul_basic.csv
```

### 测试用例定义

```csv
casename,m,k,n,bias,dtype,transA,transB,hf32,format
mat_mul_basic_fp16,128,512,128,128,float16,false,false,false,"(ND,ND)"
mat_mul_basic_bf16,128,512,128,128,bfloat16,false,false,false,"(ND,ND)"
mat_mul_basic_fp32,128,512,128,128,float32,false,false,false,"(ND,ND)"
mat_mul_basic_hf32,128,512,128,128,float32,false,false,true,"(ND,ND)"
mat_mul_basic_weightNz,128,512,128,0,float16,false,false,false,"(ND,NZ)"
```

## 代码结构

```
mat_mul_basic/
├── CMakeLists.txt                  # 构建配置
├── mat_mul_basic.cpp               # 统一 kernel 实现
├── mat_mul_basic.csv               # CSV 测试用例
├── parse_csv.py                    # CSV 解析与批量执行
├── run.sh                          # 运行脚本
└── README.md                       # 本文档
```

## Blaze 组件

| 组件 | 头文件 | 职责 |
|------|--------|------|
| Kernel | `blaze/gemm/kernel/kernel_matmul_basic.h` | 完整 kernel 入口 |
| Block MMAD | `blaze/gemm/block/block_mmad_matmul_basic.h` | Block 级矩阵乘 |
| Block Scheduler | `blaze/gemm/block/block_scheduler_matmul_basic.h` | Basic 调度器 |
| Epilogue | `blaze/epilogue/block/block_epilogue_empty.h` | 空后处理 |
| Dispatch Policy | `blaze/gemm/policy/dispatch_policy.h` | MatmulMultiBlockBasic |
