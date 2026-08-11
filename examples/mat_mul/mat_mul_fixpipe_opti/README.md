# mat_mul_fixpipe_opti Example

## 概述

本示例演示基于 Blaze 框架的 MatMul Fixpipe 优化（非全载）在昇腾 NPU 上的实现。Fixpipe 模式将 L0C 输出通过 fixpipe 搬运到 UB，再经 AIV 搬出到 GM，支持 AIC/AIV 跨核同步。

- **算子**: mat_mul
- **场景**: mat_mul_fixpipe_opti
- **算法特点**: A/B 均流水线加载，L0C→UB→GM 的 fixpipe 输出路径，AIC/AIV 跨核同步
- **参考实现**: 基于 Blaze 框架 `blaze/gemm/block/block_mmad_matmul_fixpipe_opti.h`

## 支持架构

| 架构 | SoC | 支持状态 |
|------|-----|----------|
| dav-3510 | Ascend950 | ✅ |

## 使用约束

- 输入 A shape: `[M, K]`
- 输入 B shape: `[K, N]`
- 输出 C shape: `[M, N]`
- 数据类型: float16, float32
- A 和 B 均通过 l1Stages 流水线缓冲加载

## CSV 驱动测试

### 执行方式

通过 `run.sh --case=<csv>` 驱动，自动完成编译、数据生成、kernel 执行和精度验证：

```bash
bash run.sh --case=mat_mul_fixpipe_opti.csv
```

### 测试用例定义

测试用例定义在 `mat_mul_fixpipe_opti.csv` 中，格式如下：

```csv
casename,m,k,n,dtype
mat_mul_fixpipe_opti_fp16,512,512,512,float16
mat_mul_fixpipe_opti_fp32,256,256,256,float32
```

**列说明**：

| 列 | 说明 |
|----|------|
| casename | 用例名称 |
| m, k, n | 矩阵维度 |
| dtype | 数据类型：float16 / float32 |

### 结果输出

执行完成后结果写入 `mat_mul_fixpipe_opti_result.csv`。

## 数据与校验

### 输入数据

由 `../scripts/gen_data.py` 生成:

- `input/input_a.bin`: A 矩阵
- `input/input_b.bin`: B 矩阵
- `output/cpu_output.bin`: CPU 参考结果

### 输出数据

- `output/npu_out.bin`: NPU 计算结果

### 验证标准

由 `../scripts/verify_result.py` 执行:

| dtype | ratio_tol | error_ratio_tol |
|-------|-----------|-----------------|
| float16 | 5e-3 | 5e-3 |
| float32 | 1e-4 | 1e-4 |

- 超差比例 < error_ratio_tol

## 代码结构

```
mat_mul_fixpipe_opti/
├── CMakeLists.txt                  # 构建配置
├── mat_mul_fixpipe_opti.cpp        # 统一 kernel 实现
├── mat_mul_fixpipe_opti.csv        # CSV 测试用例
├── parse_csv.py                    # CSV 解析与批量执行
├── run.sh                          # 运行脚本
└── README.md                       # 本文档
```

## Blaze 组件

本场景使用以下 Blaze 组件:

| 组件 | 头文件 | 职责 |
|------|--------|------|
| Kernel | `blaze/gemm/kernel/kernel_matmul_fixpipe_opti.h` | 完整 kernel 入口 |
| Block MMAD | `blaze/gemm/block/block_mmad_matmul_fixpipe_opti.h` | Block 级矩阵乘（fixpipe） |
| Block Scheduler | `blaze/gemm/block/block_scheduler_matmul_basic.h` | 基础调度器 |
| Epilogue | `blaze/epilogue/block/block_epilogue_fixpipe.h` | Fixpipe 后处理 |
| Dispatch Policy | `blaze/gemm/policy/dispatch_policy.h` | 派发策略 |

## Fixpipe 算法说明

Fixpipe 优化模式将 L0C 计算结果通过 fixpipe 路径搬出：
- A 和 B 均通过 l1Stages 流水线从 GM 加载到 L1
- L0C 结果通过 `CopyL0C2UB` 搬移到 UB
- UB 数据由 AIV 核通过 fixpipe 搬出到 GM
- AIC/AIV 通过 CrossCore 同步标志协调搬出时序
