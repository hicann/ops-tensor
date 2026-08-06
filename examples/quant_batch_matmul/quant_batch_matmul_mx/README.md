# quant_batch_matmul_mx Example

## 概述

本示例演示基于 Blaze 框架的 MX 量化 Batch MatMul 算子在昇腾 NPU 上的实现。本样例对 A、B 矩阵进行 MX 量化（FP8/FP4 激活 × FP8/FP4 权重），两侧各携带独立的 E8M0 MX Scale。

- **算子**: quant_batch_matmul
- **场景**: quant_batch_matmul_mx
- **算法特点**: A/B 双侧 MX 量化，支持 FP8×FP8、FP4×FP4，多种 C dtype
- **参考实现**: 基于 Blaze 框架 `blaze/gemm/kernel/kernel_qbmm_mx.h` + `blaze/gemm/block/block_mmad_qbmm_mx.h`

## 支持架构

| 架构     | SoC       | 支持状态 |
| -------- | --------- | -------- |
| dav-3510 | Ascend950 | ✅       |

## 使用约束

- 输入 A shape: `[M, K]`（transA=false）或 `[K, M]`（transA=true）
- 输入 B shape: `[K, N]`（transB=false）或 `[N, K]`（transB=true）
- 输出 C shape: `[M, N]`
- A/B dtype: fp8_e4m3, fp8_e5m2, fp4_e2m1（A 和 B 可为不同 FP8 子类型，但 FP4 与 FP8 不可混用）
- C dtype: float16, bfloat16, float32
- Bias dtype: float32
- ScaleA shape: `[M, scaleK]`，ScaleB shape: `[scaleK, N]`，`scaleK = ceil(K/64) * 2`

## CSV 驱动测试

### 执行方式

通过 `run.sh --case=<csv>` 驱动，自动完成编译、数据生成、kernel 执行和精度验证：

```bash
bash run.sh --case=quant_batch_matmul_mx.csv
```

### 测试用例定义

测试用例定义在 `quant_batch_matmul_mx.csv` 中，格式如下：

```csv
casename,m,k,n,bias,a_dtype,b_dtype,c_dtype,transA,transB,format,base_m,base_n,base_k,tile_k_l1,scale_k_l1,l1_buffers,db_l0c,a_full_load
qbmm_mx_5344_1260_1976_FT_fp4e2m1_fp4e2m1_bfloat16_bias,5344,1260,1976,1976,fp4_e2m1,fp4_e2m1,bfloat16,false,true,"(ND,ND)",256,256,256,512,1536,3,1,false
qbmm_mx_10240_1024_2624_FF_fp8e4m3_fp8e4m3_bfloat16_NZ,10240,1024,2624,0,fp8_e4m3,fp8_e4m3,bfloat16,false,false,"(ND,NZ)",256,256,128,256,1024,3,1,false
```

**列说明**：

| 列                     | 说明                                      |
| ---------------------- | ----------------------------------------- |
| casename               | 用例名称                                  |
| m, k, n                | 矩阵维度                                  |
| bias                   | bias 向量大小，必须等于 n 或 0（无 bias） |
| a_dtype, b_dtype       | A/B 量化 dtype                            |
| c_dtype                | 输出 dtype                                |
| transA                 | A 矩阵是否转置                            |
| transB                 | B 矩阵是否转置                            |
| format                 | 输入格式: (ND,ND) 或 (ND,NZ)             |
| base_m, base_n, base_k | L0 tile shape                             |
| tile_k_l1              | L1 K 方向 tile 大小                       |
| scale_k_l1             | L1 scaleK 方向 tile 大小                  |
| l1_buffers             | L1 buffer 数（2/3/4）                     |
| db_l0c                 | L0C double buffer 开关（1=关，2=开）      |
| a_full_load            | A 全载到 L1（true/false）                 |

### 结果输出

执行完成后结果写入 `quant_batch_matmul_mx_result.csv`。

## 数据与校验

### 输入数据

由 `../scripts/gen_data.py` 生成：

- `input/input_a.bin`: A 矩阵（FP8 或打包 FP4）
- `input/input_b.bin`: B 矩阵（FP8 或打包 FP4，ND 或 NZ 格式）
- `input/scale_a.bin`: ScaleA（E8M0，布局随 transA 变化）
- `input/scale_b.bin`: ScaleB（E8M0，布局随 transB 变化）
- `input/bias.bin`: bias 向量（FP32，bias>0 时生成）
- `input/initial_c.bin`: 初始化 C（全零）

### 输出数据

- `output/npu_out.bin`: NPU 计算结果

### 验证标准

由 `../scripts/verify_result.py` 执行，支持 float16/bfloat16/float32 三种输出 dtype 的精度比对。

## 覆盖的场景

| 场景维度             | 用例                                           |
| -------------------- | ---------------------------------------------- |
| A/B dtype            | FP8(e4m3,e5m2)×FP8(e4m3,e5m2) , FP4×FP4 |
| C dtype              | float16/float32/bfloat16                       |
| transA/transB        | FF/FT/TF/TT 四种组合                           |
| format               | (ND,ND) / (ND,NZ)                             |
| L1 Buffer 数         | 2 (Double), 3 (Triple), 4 (Quad)               |
| A Full Load          | 开/关                                          |
| dbL0C (L0C PingPong) | 1/2                                            |
| K 维 L1 多次循环     | kL1 < K                                        |
| ScaleK 多次循环      | scaleKL1 > kL1                                 |
| 尾块处理             | M/N/K 尾块非对齐                               |
| 多核                 | 大 shape 利用全部 AIC core                     |
| Bias                 | 有/无                                          |

## 代码结构

```
quant_batch_matmul/
├── CMakeLists.txt                  # 构建配置
├── scripts/
│   ├── gen_data.py                 # 数据生成（full_mx 路径）
│   └── verify_result.py            # 精度校验（float16/bfloat16/float32）
└── quant_batch_matmul_mx/
    ├── CMakeLists.txt              # 构建配置
    ├── quant_batch_matmul_mx.cpp   # 统一 kernel 实现
    ├── quant_batch_matmul_mx.csv   # CSV 测试用例
    ├── parse_csv.py                # CSV 解析与批量执行
    ├── run.sh                      # 运行脚本
    └── README.md                   # 本文档
```

## Blaze 组件

本场景使用以下 Blaze 组件:

| 组件            | 头文件                                          | 职责                         |
| --------------- | ----------------------------------------------- | ---------------------------- |
| Kernel          | `blaze/gemm/kernel/kernel_qbmm_mx.h`          | MX quant batch matmul kernel |
| Block MMAD      | `blaze/gemm/block/block_mmad_qbmm_mx.h`       | Block 级 MX 矩阵乘           |
| Block Scheduler | `blaze/gemm/block/block_scheduler_qbmm.h`     | QBMM V3 调度器               |
| Epilogue        | `blaze/epilogue/block/block_epilogue_empty.h` | 空 epilogue                  |
| Dispatch Policy | `blaze/gemm/policy/dispatch_policy.h`         | MatmulWithScaleMx 派发策略   |
