# ops-tensor Examples

## 1. 功能说明

examples 目录提供基于 Blaze header-only GEMM 框架的算子样例程序，用于：

- 验证算子在 NPU 上的编译、执行和精度正确性
- 演示如何使用 Blaze 框架编写 StreamK、WeightNz 等高级 GEMM kernel
- 提供端到端的测试数据生成、执行、精度验证工作流
- 支持多种数据类型（float16/bfloat16/float32）的测试和验证

每个样例包含完整的 host 侧代码（数据读取、ACL 内存管理、kernel 调度）和 device 侧 kernel（Blaze 模板实例化），通过 CSV 测试用例表驱动执行。

## 2. 环境依赖

examples 的完整编译与运行涉及三类依赖：**系统工具链**、**CANN / 昇腾环境**、**Python 依赖**。`run.sh` 的 `preflight()` 函数会在执行前自动检查关键依赖（`ASCEND_HOME_PATH`、`bisheng`、`g++`、`python3`、`cmake`），缺失时报错退出。

### 2.1 系统工具链

| 依赖 | 最低版本 | 用途 | 检查方式 |
|------|---------|------|---------|
| cmake | 3.16 | 构建系统（`examples/CMakeLists.txt` 中 `cmake_minimum_required(VERSION 3.16)`） | `cmake --version` |
| g++ | 默认标准 ≥ C++14 | Host 侧 C++ 编译器（`project(... LANGUAGES CXX)`）| `g++ --version` |
| bisheng | — | ASC 语言编译器，编译 device 侧 kernel（`find_package(ASC REQUIRED)`） | `bisheng --version` |
| make | — | 由 cmake 调用的底层构建工具 | `make --version` |
| python3 | 3.7+ | 驱动 CSV 解析、数据生成、精度验证脚本 | `python3 --version` |

> `bisheng` 编译器随 CANN Toolkit 安装，`source set_env.sh` 后自动加入 PATH。

### 2.2 CANN / 昇腾环境

| 依赖 | 要求 | 说明 |
|------|------|------|
| CANN Toolkit | ≥ 9.1.0 | 提供 ACL 运行时、bisheng 编译器、头文件 |
| NPU 硬件 | Ascend950 | 当前 examples 仅支持 Ascend950芯片 |

### 2.3 Python 依赖

数据生成（`gen_data.py`）和精度验证（`verify_result.py`）脚本依赖以下 Python 包：

| 包 | 用途 | 安装命令 |
|----|------|---------|
| numpy | 二进制数据读写、数组操作 | `pip install numpy` |
| torch | CPU golden 参考计算（`torch.matmul`/`torch.addmm`）、dtype 映射 | `pip install torch` |

> **注意**：仓库根目录的 `requirements.txt` 列出了主项目依赖（pyyaml、sympy 等），但 examples 的 Python 脚本额外需要 `numpy` 和 `torch`，需单独安装。

### 2.4 依赖检查清单

执行前可运行以下命令快速验证环境：

```bash
# 1. CANN 环境
echo $ASCEND_HOME_PATH && [ -d "$ASCEND_HOME_PATH" ] && echo "OK" || echo "MISSING"

# 2. 编译器
cmake --version | head -1
g++ --version | head -1
bisheng --version 2>&1 | head -1

# 3. Python 依赖
python3 -c "import numpy; print('numpy', numpy.__version__)"
python3 -c "import torch; print('torch', torch.__version__)"

# 4. NPU 设备
npu-smi info | head -20
```

> `run.sh` 执行时会自动运行 preflight 检查，缺失关键依赖会报错退出并给出修复提示。

## 3. 目录结构与层级说明

```
examples/
├── CMakeLists.txt              # L1: 全局 CMake 配置（编译器、链接库、宏定义）
├── README.md                   # 本文件
├── common/                     # 公共头文件
│   └── data_utils.h            #   ACL_CHECK 宏、文件读写工具
│
└── {op}/                       # 算子级目录（L2），如 mat_mul
    ├── CMakeLists.txt          #   L2: 加载子场景目录
    ├── scripts/                #   算子级脚本（数据生成 + 精度验证）
    │   ├── gen_data.py         #     输入数据生成 + CPU golden 计算
    │   └── verify_result.py    #     NPU 输出 vs CPU golden 精度比对
    │
    └── {example}/             # 场景级目录（L3），如 mat_mul_streamk
        ├── CMakeLists.txt      #   L3: 注册可执行样例
        ├── run.sh              #   执行脚本（编译 + 运行 + 验证 + 清理）
        ├── {example}.cpp      #   样例源码，如 mat_mul_streamk.cpp
        ├── {example}.csv      #   CSV 测试用例表
        ├── parse_csv.py        #   CSV 解析与批量执行
        ├── README.md           #   场景说明文档
        └── build/              #   编译产物（自动生成）
```

**层级关系**：

| 层级 | 目录 | CMakeLists.txt 职责 |
|------|------|---------------------|
| L1 | `examples/` | 编译器发现、链接库配置、`ops_example_add_executable` 宏定义 |
| L2 | `examples/{op}/` | `add_subdirectory()` 加载子场景 |
| L3 | `examples/{op}/{example}/` | `ops_example_add_executable()` 注册可执行样例 |

## 4. 执行方法

### 4.1 通过 build.sh 执行（推荐）

从仓库根目录执行，`build.sh --examples` 会自动发现样例目录下的同名 CSV 文件（`{example}.csv`），以 CSV 批量模式驱动 `run.sh` 完成编译、数据生成、kernel 执行和精度验证：

```bash
# 运行指定算子下的所有场景（自动读取各场景的 {example}.csv）
bash build.sh --examples --ops=mat_mul

# 运行指定场景
bash build.sh --examples --ops=mat_mul --target=mat_mul_streamk

# 运行所有算子的所有场景
bash build.sh --examples
```

`build.sh` 对每个场景的调用流程：

```
build.sh --examples --ops=mat_mul --target=mat_mul_streamk
    │
    └─→ 自动发现 mat_mul_streamk.csv
    │
    └─→ bash run.sh --case=mat_mul_streamk.csv
            │
            ├─→ cmake + make                                    # 编译
            ├─→ parse_csv.py 读取 CSV，逐条执行:
            │       ├─ gen_data.py m k n transA transB dtype bias format  # 数据生成
            │       ├─ ./mat_mul_streamk m k n ...               # kernel 执行
            │       └─ verify_result.py m n dtype [--hf32]       # 精度验证
            └─→ 结果写入 {example}_result.csv
```

### 4.2 通过 run.sh 执行

进入场景目录，通过 `--case` 参数指定 CSV 测试用例表：

```bash
cd examples/mat_mul/mat_mul_streamk
bash run.sh --case=mat_mul_streamk.csv
```

`run.sh` 会自动完成编译、CSV 解析、逐条执行（数据生成 → kernel → 验证）和结果汇总。

### 4.3 CSV 测试用例格式

每个场景目录下有一个与场景同名的 CSV 文件（如 `mat_mul_streamk.csv`），定义所有测试用例：

```csv
casename,m,k,n,bias,dtype,transA,transB,hf32,format
mat_mul_streamk_fp16,100,8192,100,100,float16,false,false,false,"(ND,ND)"
mat_mul_streamk_bf16,100,8192,100,100,bfloat16,false,false,false,"(ND,ND)"
mat_mul_streamk_fp32,100,8192,100,100,float32,false,false,false,"(ND,ND)"
mat_mul_streamk_hf32,100,8192,100,100,float32,false,false,true,"(ND,ND)"
mat_mul_streamk_weightNz,100,8192,100,0,float16,false,false,false,"(ND,NZ)"
```

| 列 | 说明 |
|----|------|
| casename | 用例名称 |
| m, k, n | 矩阵维度 |
| bias | bias 向量大小，必须等于 n 或 0（无 bias） |
| dtype | 数据类型：float16 / bfloat16 / float32 |
| transA | A 矩阵是否转置 |
| transB | B 矩阵是否转置 |
| hf32 | 是否启用 HF32 模式（仅 float32 有效） |
| format | 输入格式：(ND,ND) 或 (ND,NZ) |

### 4.4 两种方式的关联

- `build.sh --examples` 是顶层入口，自动发现场景目录下的 `{example}.csv`，以 `--case` 参数委托给 `run.sh`
- `run.sh --case=<csv>` 是场景级执行器，负责完整的编译 → CSV 解析 → 逐条执行 → 结果汇总流程
- 两种方式均通过 CSV 驱动，不涉及手动传参

## 5. 新增 Examples

### 5.1 新增算子级（如 `new_op`）

```bash
# 1. 创建算子目录
mkdir -p examples/new_op/scripts

# 2. 创建算子级 CMakeLists.txt
cat > examples/new_op/CMakeLists.txt << 'EOF'
add_subdirectory(new_op_basic)
EOF

# 3. 创建数据生成和验证脚本
# examples/new_op/scripts/gen_data.py
# examples/new_op/scripts/verify_result.py

# 4. 在 examples/CMakeLists.txt 中注册算子
# 末尾追加: add_subdirectory(new_op)
```

### 5.2 新增场景级（如 `new_op/new_op_basic`）

```bash
# 1. 创建场景目录
mkdir -p examples/new_op/new_op_basic

# 2. 创建场景级 CMakeLists.txt，注册样例
cat > examples/new_op/new_op_basic/CMakeLists.txt << 'EOF'
ops_example_add_executable(new_op_basic new_op_basic.cpp)
EOF

# 3. 创建样例源码 new_op_basic.cpp
# 4. 创建 CSV 测试用例表 new_op_basic.csv
# 5. 创建 run.sh（可参考 mat_mul_streamk/run.sh）
# 6. 创建 parse_csv.py（可参考 mat_mul_streamk/parse_csv.py）
# 7. 在 examples/new_op/CMakeLists.txt 中添加 add_subdirectory(new_op_basic)
```

注册后即可通过以下方式运行：

```bash
bash build.sh --examples --ops=new_op --target=new_op_basic
```
