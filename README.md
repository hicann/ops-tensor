# ops-tensor

## 🔥 最新动态
- [2026/03] 完成项目基础架构搭建，支持 Add 算子实现和测试，支持一键编译、测试和打包。
- [2026/03] 建立完整的测试框架，支持单元测试、超时控制和自动化测试统计。
- [2026/03] 实现标准化的打包流程，生成 .run 安装包，支持 install/uninstall/upgrade 完整生命周期管理。

## 🚀 概述

ops-tensor 是 [CANN](https://hiascend.com/software/cann) （Compute Architecture for Neural Networks）算子库中提供张量类计算的基础算子库，采用模块化设计，支持灵活的算子开发和管理。

### 主要特性

- ✅ **模块化设计** - 支持动态添加算子模块，每个算子独立开发、编译和测试
- ✅ **标准 CMake 构建** - 跨平台编译支持，统一的构建流程
- ✅ **完整测试体系** - 基于自定义测试框架，支持自动化测试和超时控制
- ✅ **便捷打包** - 一键生成 .run 安装包，支持 install/uninstall/upgrade
- ✅ **版本管理** - 安装信息记录和版本追踪，支持升级管理
- ✅ **轻量高效** - 简洁的架构设计，避免过度工程化

## 📝 版本配套

当前仓库已验证通过的 CANN Toolkit 如下：

| CANN 版本 | 发布时间 | 分支 |
| --- | --- | --- |
| [CANN 9.0.0-beta.2](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0-beta.2) | `2026/03/30` | master |

请根据实际 CPU 架构，从上述链接目录中自行选择对应的 `.run` 安装包。

toolkit 安装包文件名格式如下：

- `Ascend-cann-toolkit_${cann_version}_linux-aarch64.run`
- `Ascend-cann-toolkit_${cann_version}_linux-x86_64.run`

1. **安装社区版 CANN Toolkit**

    ```bash
    # 确保安装包具有可执行权限
    chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
    # 安装命令
    ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --force --install-path=${install_path}
    ```
    - `${cann_version}`：表示 toolkit 安装包版本号，需满足上文的最低版本要求。
    - `${arch}`：表示 CPU 架构，如 `aarch64`、`x86_64`。
    - `${install_path}`：表示指定安装路径，默认安装在 `/usr/local/Ascend` 目录。

2. **配置环境变量**

   安装完成后，请先执行：

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

   请将 `${install_path}` 替换为 toolkit 的实际安装目录，例如 `/usr/local/Ascend` 或 `${HOME}/Ascend`。

## ⚡️ 快速入门

### 编译与测试

详细的 build.sh 参数说明请参考 [build 参数说明](docs/zh/context/build.md)。

```bash
# 编译所有算子（默认 8 线程）
./build.sh

# 编译指定算子
./build.sh --ops=add

# 编译并运行测试
./build.sh --run

# 编译并打包成 .run 文件
./build.sh --pkg

# 查看完整帮助信息
./build.sh --help
```

### 安装

```bash
# 标准安装（需要 root 权限）
sudo ./cann-950-ops-tensor_9.0.0_linux-*.run

# 查看安装包信息
./cann-950-ops-tensor_9.0.0_linux-*.run --help

# 安装到自定义路径
sudo ./cann-950-ops-tensor_9.0.0_linux-*.run --install-path=/opt/ascend

# 卸载
sudo ./cann-950-ops-tensor_9.0.0_linux-*.run --uninstall

# 升级
sudo ./cann-950-ops-tensor_9.0.0_linux-*.run --upgrade
```

## 📖 项目说明

### 支持的算子

当前支持的算子列表：

| 算子名称 | 描述 | 状态 |
|---------|------|------|
| [Add](src/add/) | 张量加法运算 | ✅ 已实现 |

更多算子正在持续开发中...

### SoC 支持矩阵

| SoC 型号 | SOC_VERSION | 支持状态 |
|---------|-------------|---------|
| Ascend950 | ascend950dt_9595 | ✅ 默认支持 |
| Ascend910B | ascend910b3 | ❌ 暂不支持 |
| Ascend910_93 | ascend910_93 | ❌ 暂不支持 |
| Ascend910 | ascend910 | ❌ 暂不支持 |
| Ascend310P | ascend310p | ❌ 暂不支持 |
| Ascend310B | ascend310b | ❌ 暂不支持 |

## 🔍 目录结构

```
ops-tensor/
├── cmake/                      # CMake 配置文件
│   ├── func.cmake             # 公共函数（算子注册等）
│   ├── init_env.cmake         # 环境初始化
│   ├── variables.cmake        # 变量定义
│   ├── package.cmake          # 打包配置
│   ├── makeself_built_in.cmake # .run 包生成脚本
│   └── third_party/           # 第三方依赖
├── include/                    # 公共头文件
│   ├── cann_ops_tensor.h      # API 头文件
│   └── cann_ops_tensor_types.h # 类型定义头文件
├── scripts/                    # 脚本目录
│   ├── check_build_dependencies.py  # 依赖检查脚本
│   ├── generate_version_info.py    # 版本信息生成
│   └── package/               # 打包相关脚本
│       ├── common/           # 通用打包工具
│       └── cfg/              # 打包配置
├── lib/                        # 基础设施库（算子开发依赖）
│   ├── core/                  # 核心模块
│   │   ├── handle.cpp/hpp     # 句柄管理
│   │   ├── operation_descriptor.cpp/hpp  # 算子描述符
│   │   ├── plan.cpp/hpp       # 执行计划
│   │   ├── plan_preference.cpp/hpp # 计划偏好
│   │   └── tensor_descriptor.cpp/hpp   # 张量描述符
│   ├── elementwise/           # 元素算子基础实现
│   │   ├── elementwise.cpp/hpp
│   │   └── elementwise_binary.cpp/hpp
│   ├── utils/                 # 工具函数
│   │   ├── type_utils.hpp     # 类型工具
│   │   ├── utils.cpp          # 通用工具
│   │   └── validation.cpp/hpp # 验证工具
│   └── CMakeLists.txt
├── src/                        # 源代码目录
│   ├── add/                   # Add 算子实现
│   │   ├── add_solution.cpp   # 解决方案实现（Tiling 计算、解决方案注册）
│   │   ├── add_kernel.cpp     # Kernel 端实现
│   │   ├── arch35/            # 架构特定代码（可选）
│   │   │   └── add_struct.h   # 数据结构定义（也可定义在 .cpp 中）
│   │   ├── tests/             # 算子测试
│   │   │   ├── add_test.h
│   │   │   └── add_test.cpp
│   │   └── CMakeLists.txt
│   ├── ...                    # 其他算子
│   └── CMakeLists.txt
├── tests/                      # 测试框架
│   ├── test_common.h          # 测试框架头文件
│   ├── test_common.cpp        # 测试框架实现
│   ├── test_elementwise.cpp   # 元素算子测试辅助
│   ├── all_tests.cpp.in       # 测试入口模板
│   └── CMakeLists.txt
├── build.sh                    # 编译脚本
├── install_deps.sh             # 依赖安装脚本
├── CMakeLists.txt              # 主 CMake 配置
├── version.cmake               # CMake 版本配置
├── version.info                # 版本信息
└── README.md                   # 本文件
```

**说明**：
- **lib/** - 基础设施库，提供算子开发所需的核心功能：
  - `core/`: 算子描述、执行计划、张量抽象等基础设施
  - `elementwise/`: 元素算子通用实现
  - `utils/`: 工具函数和验证逻辑
  - **所有算子实现都依赖 lib 模块提供的基础设施**
- `<op>_solution.cpp` - 解决方案实现（Tiling 计算、内存管理、解决方案注册）
- `<op>_kernel.cpp` - Kernel 核函数实现
- `arch35/<op>_struct.h` - Tiling 数据结构（可选，也可定义在 solution.cpp 中）
- `arch35/` 目录是可选的，仅在需要区分不同 SOC 架构时使用
- 测试文件强烈推荐，但不是必需的

**说明**：
- `<op>_solution.cpp` - 解决方案实现（Tiling 计算、内存管理、解决方案注册）
- `<op>_kernel.cpp` - Kernel 核函数实现
- `arch35/<op>_struct.h` - Tiling 数据结构（可选，也可定义在 solution.cpp 中）
- `arch35/` 目录是可选的，仅在需要区分不同 SOC 架构时使用
- 测试文件强烈推荐，但不是必需的

## 🛠️ 开发指南

### 添加新算子

详细的算子开发指南请参考 [算子开发指南](docs/zh/develop/operator_development_guide.md)，包括：

- 完整的目录结构说明
- 解决方案实现模板
- Tiling 数据结构定义
- 解决方案注册机制
- 完整开发流程

**快速开始**：

1. **创建目录**
```bash
mkdir -p src/my_op/arch35 (可选)
mkdir -p src/my_op/tests
```

2. **编写算子实现**
创建 `src/my_op/my_op_solution.cpp` 和 `src/my_op/my_op_kernel.cpp`，包含：
- 解决方案部分：Tiling 计算、内存管理、解决方案注册
- Kernel 部分：核函数实现

3. **创建 CMakeLists.txt**
```cmake
register_operator(NAME my_op ARCH_DIR arch35)
```

4. **编写测试**（推荐）
参考 [测试编写指南](docs/zh/develop/test_writing_guide.md)

5. **编译验证**
```bash
./build.sh --ops=my_op --run
```

完整示例参考 `src/add/` 目录。

### 编写测试

ops-tensor 提供了轻量级、自动化的测试框架。详细的测试编写指南请参考 [测试编写指南](docs/zh/develop/test_writing_guide.md)，包括：

- 测试框架特性
- 测试文件结构
- 核心宏和函数说明
- 完整编写步骤和示例
- 最佳实践和常见问题



## 💬 相关信息

- **许可证**: [CANN Open Software License Agreement Version 2.0](LICENSE)
- **安全声明**: [SECURITY.md](SECURITY.md)
- **贡献指南**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **所属 SIG**: [CANN Community](https://gitcode.com/cann/community)

## 🤝 联系我们

本项目功能和文档正在持续更新和完善中，欢迎您关注最新版本。

- **问题反馈**: 通过 [Issues](https://gitcode.com/cann/ops-tensor/issues) 提交问题
- **社区互动**: 通过 [Discussions](https://gitcode.com/cann/ops-tensor/discussions) 参与交流
- **技术专栏**: 通过 [Wiki](https://gitcode.com/cann/ops-tensor/wiki) 获取技术文章
