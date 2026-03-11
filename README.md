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

本项目源码会跟随 CANN 软件版本发布，关于 CANN 软件版本与本项目标签的对应关系请参阅 [release 仓库](https://gitcode.com/cann/release-management) 中的相应版本说明。

**当前版本：** v1.0.0 (2025-03-02)

为确保您的源码定制开发顺利进行，请选择配套的 CANN 版本，使用 master 分支可能存在版本不匹配的风险。

## ⚡️ 快速入门

### 环境要求

- CANN 8.0 及以上版本
- 支持 Ascend950、Ascend910B、Ascend910_93、Ascend910、Ascend310P、Ascend310B 等 SoC
- Linux x86_64/AArch64 平台

### 环境配置

```bash
# 设置 CANN 环境变量
source /usr/local/Ascend/cann/set_env.sh

# 验证环境
echo $ASCEND_HOME_PATH
```

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
sudo ./cann-ops-tensor-1.0.0-linux-*.run

# 查看安装包信息
./cann-ops-tensor-1.0.0-linux-*.run --help

# 安装到自定义路径
sudo ./cann-ops-tensor-1.0.0-linux-*.run --install-path=/opt/ascend

# 卸载
sudo ./cann-ops-tensor-1.0.0-linux-*.run --uninstall

# 升级
sudo ./cann-ops-tensor-1.0.0-linux-*.run --upgrade
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
| Ascend910B | ascend910b3 | ✅ 支持 |
| Ascend910_93 | ascend910_93 | ✅ 支持 |
| Ascend910 | ascend910 | ✅ 支持 |
| Ascend310P | ascend310p | ✅ 支持 |
| Ascend310B | ascend310b | ✅ 支持 |

## 🔍 目录结构

```
ops-tensor/
├── cmake/                      # CMake 配置文件
│   ├── func.cmake             # 公共函数（算子注册等）
│   ├── init_env.cmake         # 环境初始化
│   ├── package.cmake          # 打包配置
│   └── run_package.cmake      # 打包脚本
├── include/                    # 公共头文件
│   └── cann_ops_tensor.h      # API 头文件
├── scripts/                    # 脚本目录
│   └── package/               # 打包相关脚本
├── src/                        # 源代码目录
│   ├── add/                   # Add 算子实现
│   │   ├── add_host.cpp       # Host 端实现
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
│   ├── all_tests.cpp.in       # 测试入口模板
│   └── CMakeLists.txt
├── build.sh                    # 编译脚本
├── CMakeLists.txt              # 主 CMake 配置
├── version.info                # 版本信息
└── README.md                   # 本文件
```

**说明**：
- Host 和 Kernel 可以合并为一个 `.cpp` 文件
- TilingData 可以定义在 `.cpp` 文件中，也可以独立为 `_struct.h`
- `arch35/` 目录是可选的，仅在需要区分不同 SOC 架构时使用
- 测试文件强烈推荐，但不是必需的

## 🛠️ 开发指南

### 添加新算子

详细的算子开发指南请参考 [算子开发指南](docs/zh/develop/operator_development_guide.md)，包括：

- 完整的目录结构说明
- Host + Kernel 实现模板
- Tiling 数据结构定义
- 完整开发流程

**快速开始**：

1. **创建目录**
```bash
mkdir -p src/my_op/arch35 (可选)
mkdir -p src/my_op/tests
```

2. **编写算子实现**
创建 `src/my_op/my_op.cpp`，包含：
- Host 部分：对外接口、Tiling 计算、内存管理
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

## 📦 打包说明

### Run 包结构

```
cann-ops-tensor-1.0.0-linux-Ascend950-x86_64.run
    ↓ 解压

/tmp/extract/
├── lib/
│   └── libops_tensor.so           # 核心动态库
├── include/
│   └── cann_ops_tensor.h          # API 头文件
└── share/
    └── info/
        └── ops_tensor/
            ├── version.info       # 版本信息
            ├── scripts/           # 管理脚本
            │   ├── install.sh
            │   ├── uninstall.sh
            │   └── help.info
            └── ascend_install.info # 安装信息（安装后生成）
```

### 安装后的路径

```
/usr/local/Ascend/cann/
├── lib/
│   └── libops_tensor.so
├── include/
│   └── cann_ops_tensor.h
└── share/
    └── info/
        └── ops_tensor/
            ├── version.info
            ├── scripts/
            └── ascend_install.info
```

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
