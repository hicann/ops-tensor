# QGMM MX 样例

## 概述

本样例演示 QGMM MX 分组矩阵乘算子在昇腾 NPU 上的运行方式，并对算子输出进行数值正确性校验。

## 支持架构

| 架构 | SoC | 支持状态 |
|------|-----|----------|
| dav-3510 | Ascend950 | ✅ |

## 功能说明

本样例支持以下功能：

- MXFP4 E2M1、MXFP4 E1M2、MXFP8 E4M3 和 MXFP8 E5M2 数据类型；
- ND 和 NZ 数据格式；
- M 轴分组，以及合法组合下的 B 矩阵转置；
- 不同的 E、M、N 和 K 组合，其中 E 表示分组数量；
- 连续存储的单张量权重，以及 NZ 场景下的多张量权重和缩放因子；
- MXFP4 E2M1 场景下的可选偏置输入；
- Length、Offset 和 Sparse 三种 Group List 类型；
- L1 双缓冲和三缓冲。

样例使用固定随机种子生成可复现的非零 MX 数据，并采用以下流程验证数值正确性：

1. `../scripts/gen_data.py` 根据 CSV 参数生成输入数据和 NumPy `golden_c.bin`；
2. `qgmm_mx.cpp` 读取输入数据，在 NPU 上执行内核并写出 `npu_out.bin`；
3. `../scripts/verify_result.py` 使用相对误差和绝对误差比较 NPU 输出与 CPU golden。

M 轴分组场景分别计算各组矩阵乘并按 M 轴拼接结果。

## 使用约束

- E、M、N 和 K 必须为正整数；
- M 轴分组时 A 矩阵不转置；
- B 矩阵支持 ND 和 NZ 数据格式，其中 NZ 表示 FRACTAL_NZ 格式；
- B 矩阵是否转置通过 `transB` 配置；
- Weight ND 场景使用单tensor权重，Weight NZ 场景支持单tensor和多tensor权重；
- 多tensor权重的数量与分组数量一致，各权重的 K 轴和 N 轴分别相同；
- MXFP4 E1M2 用例仅覆盖 Weight NZ 场景；
- MXFP4 E2M1 场景要求 K 为偶数且不等于 2，B 矩阵不转置时要求 N 为偶数；
- L1 buffer仅支持 2 或 3。

## CSV 驱动测试

测试用例通过 `qgmm_mx.csv` 配置，各字段含义如下：

| 字段 | 说明 |
|------|------|
| `casename` | 测试用例名称 |
| `e` | 分组数量 |
| `m` | 每个分组的 M 维度 |
| `k` | K 维度 |
| `n` | N 维度 |
| `dtype` | MX 数据类型 |
| `transA` | A 矩阵是否转置，当前样例配置为 `false` |
| `transB` | B 矩阵是否转置 |
| `format` | A、B 矩阵的数据格式及对应布局 |
| `weight_mode` | 权重存储模式，支持 `single` 和 `multi` |
| `bias` | 是否启用偏置 |
| `group_list_type` | Group List 类型 |
| `l1_buffer_stage` | L1 buffer数 |

当前样例覆盖 MX 量化的 M 轴分组。GroupList 中的数值表示 M 轴上各组的大小或累计偏移。

## 编译和运行

在代码仓根目录执行：

```bash
bash build.sh --examples --ops=gmm --target=qgmm_mx
```

构建脚本会读取 `qgmm_mx.csv` 并批量执行测试用例，结果保存在 `qgmm_mx_result.csv`。每个用例的输入、CPU golden 和 NPU 输出保存在 `output/<casename>/` 目录中；结果文件中的 `stage` 字段用于区分参数校验、数据生成、内核运行和精度校验阶段。

也可以进入样例目录直接运行：

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
cd examples/gmm/qgmm_mx
bash run.sh
```

运行指定的 CSV 文件：

```bash
bash run.sh --case=qgmm_mx.csv
```

仅执行编译：

```bash
bash run.sh --build-only
```

跳过编译并执行指定的 CSV 文件：

```bash
bash run.sh --skip-build --case=qgmm_mx.csv
```

## 内核单元测试

在代码仓根目录执行：

```bash
bash build.sh --opkernel -u --ops=qgmm
```

内核单元测试覆盖数据类型、数据格式、矩阵形状、M 轴分组、B 矩阵转置、Group List、偏置、L0C 双缓冲、L1 三缓冲以及 NZ 多张量权重和缩放因子等场景。单元测试用于检查各内核分支能否正常执行，数值正确性由 NPU 样例验证。

## 目录结构

```text
gmm/
├── scripts/
│   ├── gen_data.py      # 输入数据和 CPU golden 生成脚本
│   └── verify_result.py # NPU 输出精度校验脚本
└── qgmm_mx/
    ├── CMakeLists.txt   # 构建配置
    ├── parse_csv.py     # CSV 解析及生成、运行、校验流程调度脚本
    ├── qgmm_mx.cpp      # NPU 内核执行和输出落盘
    ├── qgmm_mx.csv      # 测试用例配置
    ├── README.md        # 使用说明
    └── run.sh           # 编译和运行脚本
```
