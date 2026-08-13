# Quant Grouped MatMul MX 样例

## 概述

本样例演示 Quant Grouped MatMul MX 算子在昇腾 NPU 上的运行方式，并对算子输出进行数值正确性校验。

## 支持架构

| 架构 | SoC | 支持状态 |
|------|-----|----------|
| dav-3510 | Ascend950 | ✅ |

## 功能说明

本样例支持以下功能：

- MXFP4 E2M1、MXFP4 E1M2、MXFP8 E4M3 和 MXFP8 E5M2 数据类型；
- ND 和 NZ 数据格式；
- M 轴分组，以及合法组合下的 B 矩阵转置；
- 不同的 `groupNum`、M、N 和 K 组合；
- 连续存储的单张量权重，以及 NZ 场景下的多张量权重和缩放因子；
- MXFP4 E2M1 场景下的可选偏置输入；
- Length、Offset 和 Sparse 三种 Group List 类型；
- L1 双缓冲和三缓冲。

样例使用固定随机种子生成可复现的非零 MX 数据，并采用以下流程验证数值正确性：

1. `../scripts/gen_data.py` 根据 CSV 参数生成输入数据和 NumPy `golden_c.bin`；
2. `quant_grouped_mat_mul_mx.cpp` 读取输入数据，在 NPU 上执行内核并写出 `npu_out.bin`；
3. `../scripts/verify_result.py` 使用相对误差和绝对误差比较 NPU 输出与 CPU golden。

M 轴分组场景分别计算各组矩阵乘并按 M 轴拼接结果。

## 使用约束

- `groupNum`、M、N 和 K 必须为正整数；
- M 轴分组时 A 矩阵不转置；
- B 矩阵支持 ND 和 NZ 数据格式，其中 NZ 表示 FRACTAL_NZ 格式；
- A、B 的数据排布分别通过 `layoutA` 和 `layoutB` 配置；
- Weight ND 场景使用单tensor权重，Weight NZ 场景支持单tensor和多tensor权重；
- 多tensor权重的数量与分组数量一致，各权重的 K 轴和 N 轴分别相同；
- MXFP4 E1M2 用例仅覆盖 Weight NZ 场景；
- MXFP4 E2M1 场景要求 K 为偶数且不等于 2，B 矩阵不转置时要求 N 为偶数；
- L1 buffer仅支持 2 或 3。

## CSV 驱动测试

测试用例通过 `quant_grouped_mat_mul_mx.csv` 配置。`groupNum` 至 `singleW` 与 `QgmmTilingData`
中的字段同名，并严格按照结构体的声明顺序排列：

| 字段 | 说明 |
|------|------|
| `caseName` | 测试用例名称 |
| `groupNum` | 分组数量 |
| `m` | 每个分组的 M 维度 |
| `n` | N 维度 |
| `k` | K 维度 |
| `baseM` | L0 计算块的 M 大小 |
| `baseN` | L0 计算块的 N 大小 |
| `baseK` | L0 计算块的 K 大小 |
| `kAL1` | A 在 L1 中的 K 切分大小 |
| `kBL1` | B 在 L1 中的 K 切分大小 |
| `scaleKAL1` | ScaleA 在 L1 中的 K 切分大小 |
| `scaleKBL1` | ScaleB 在 L1 中的 K 切分大小 |
| `isBias` | 是否启用偏置，0 表示关闭，1 表示启用 |
| `dbL0C` | L0C buffer 数量 |
| `l1BufferStage` | L1 buffer 数量，支持 2 或 3 |
| `groupType` | 分组轴类型，0 表示 M 轴分组，2 表示 K 轴分组 |
| `groupListType` | Group List 类型，0 表示 Offset，1 表示 Length，2 表示 Sparse |
| `singleW` | 权重存储模式，1 表示单张量，0 表示多张量 |
| `dtype` | MX 数据类型 |
| `layoutA` | A 的数据排布，支持 ND 和 DN |
| `layoutB` | B 的数据排布，支持 ND、DN、NZ 和 ZN |
| `groupList` | Group List 原始值，以分号分隔；Length 填各组长度，Offset 填累计偏移，Sparse 填索引/长度对 |
| `aFullLoad` | A 是否全载入，0 表示关闭，1 表示启用 |

当前样例覆盖 MX 量化的 M 轴和 K 轴分组。Tiling 参数由每条 CSV case 独立配置，不在 kernel 中按分组类型固定。
`groupList` 中的分组长度必须为正，并完整覆盖分组轴；M 轴分组的总长度为 `groupNum*m`，K 轴分组的总长度为 `k`。

## 编译和运行

在代码仓根目录执行：

```bash
bash build.sh --examples --ops=grouped_mat_mul --target=quant_grouped_mat_mul_mx
```

构建脚本会读取 `quant_grouped_mat_mul_mx.csv` 并批量执行测试用例，结果保存在
`quant_grouped_mat_mul_mx_result.csv`。每个用例的输入、CPU golden 和 NPU 输出保存在
`output/<caseName>/` 目录中；结果文件中的 `stage` 字段用于区分参数校验、数据生成、内核运行和精度校验阶段。

也可以进入样例目录直接运行：

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
cd examples/grouped_mat_mul/quant_grouped_mat_mul_mx
bash run.sh
```

运行指定的 CSV 文件：

```bash
bash run.sh --case=quant_grouped_mat_mul_mx.csv
```

仅执行编译：

```bash
bash run.sh --build-only
```

跳过编译并执行指定的 CSV 文件：

```bash
bash run.sh --skip-build --case=quant_grouped_mat_mul_mx.csv
```

## 内核单元测试

在代码仓根目录执行：

```bash
bash build.sh --opkernel -u --ops=quant_grouped_mat_mul
```

内核单元测试覆盖数据类型、数据格式、矩阵形状、M 轴分组、B 矩阵转置、Group List、偏置、L0C 双缓冲、L1 三缓冲以及 NZ 多张量权重和缩放因子等场景。单元测试用于检查各内核分支能否正常执行，数值正确性由 NPU 样例验证。

## 目录结构

```text
grouped_mat_mul/
├── scripts/
│   ├── gen_data.py      # 输入数据和 CPU golden 生成脚本
│   └── verify_result.py # NPU 输出精度校验脚本
└── quant_grouped_mat_mul_mx/
    ├── CMakeLists.txt   # 构建配置
    ├── parse_csv.py     # CSV 解析及生成、运行、校验流程调度脚本
    ├── quant_grouped_mat_mul_mx.cpp # NPU 内核执行和输出落盘
    ├── quant_grouped_mat_mul_mx.csv # 测试用例配置
    ├── README.md        # 使用说明
    └── run.sh           # 编译和运行脚本
```
