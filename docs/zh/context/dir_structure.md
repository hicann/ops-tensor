# 项目目录

> 本章罗列的部分目录是可选的，请以实际交付件为准。尤其**单算子目录**，不同场景下交付件有差异。

项目全量目录层级介绍如下：

```
├── cmake                                               # 项目工程编译目录
│   ├── func.cmake                                      # 公共函数
│   ├── init_env.cmake                                  # 环境初始化
│   ├── makeself_built_in.cmake                         # makeself内置配置
│   ├── package.cmake                                   # 打包配置
│   ├── variables.cmake                                 # 变量定义
│   └── third_party                                     # 第三方依赖配置
│       └── makeself-fetch.cmake                        # makeself获取脚本
├── docs                                                # 项目相关文档目录
│   ├── README.md                                       # 文档目录索引
│   ├── implementation.md                               # 实现说明文档
│   └── zh                                              # 中文文档目录
│       ├── op_list.md                                  # 算子列表
│       ├── context                                     # 公共文档目录
│       ├── invocation                                  # 算子调用文档目录
│       ├── develop                                     # 算子开发文档目录
│       └── debug                                       # 调试调优文档目录
├── include                                             # 头文件目录
│   ├── cann_ops_tensor.h                               # API头文件
│   └── cann_ops_tensor_types.h                         # 类型定义头文件
├── lib                                                 # 框架代码目录
│   ├── CMakeLists.txt                                  # 库编译配置
│   ├── core                                            # 核心功能模块
│   │   ├── handle.cpp/hpp                              # 句柄管理
│   │   ├── operation_descriptor.cpp/hpp                # 操作描述符
│   │   ├── plan.cpp/hpp                                # 计划管理
│   │   ├── plan_preference.cpp/hpp                     # 计划偏好设置
│   │   └── tensor_descriptor.cpp/hpp                   # 张量描述符
│   ├── elementwise                                     # 逐元素运算模块
│   │   ├── elementwise.cpp/hpp                         # 逐元素运算基础
│   │   └── elementwise_binary.cpp                      # 二元逐元素运算
│   └── utils                                           # 工具模块
│       ├── type_utils.hpp                              # 类型工具
│       ├── utils.cpp                                   # 通用工具函数
│       ├── validation.cpp/hpp                          # 参数验证
├── scripts                                             # 脚本目录，包含自定义算子、Kernel构建相关配置文件
│   ├── check_build_dependencies.py                     # 构建依赖检查脚本
│   ├── generate_version_info.py                        # 版本信息生成脚本
│   └── package                                         # 打包相关脚本
│       ├── common                                      # 公共打包脚本
│       ├── latest_manager                              # 版本管理
│       ├── module                                      # 模块化打包
│       ├── package.py                                  # 打包Python脚本
│       └── ops_tensor/                                 # ops_tensor打包配置
│           ├── ops_tensor.xml                          # 打包配置文件
│           └── scripts/                                # 安装/卸载脚本
├── src                                                 # 源码目录
│   ├── CMakeLists.txt                                  # 算子编译入口
│   ├── add                                             # add算子目录
│   │   ├── CMakeLists.txt                              # 算子编译配置文件
│   │   ├── add_kernel.cpp                              # Kernel实现文件
│   │   ├── add_solution.cpp                            # Solution实现文件
│   │   ├── arch35                                      # Ascend950特有算子代码
│   │   └── tests                                       # 算子测试用例目录
│   └── [其他算子目录...]                               # 其他算子遵循相同结构
├── tests                                               # 项目级测试目录
│   ├── CMakeLists.txt                                  # 测试编译配置
│   ├── all_tests.cpp.in                                # 测试入口模板
│   ├── test_common.cpp/h                               # 测试公共代码
│   └── test_elementwise.cpp                            # 逐元素运算测试
├── CMakeLists.txt                                      # 项目工程cmakelist入口
├── CHANGELOG.md                                        # 变更日志
├── CONTRIBUTING.md                                     # 项目贡献指南文件
├── LICENSE                                             # 许可证文件
├── OAT.xml                                             # OAT测试配置
├── QUICKSTART.md                                       # 快速入门指南
├── README.md                                           # 项目工程总介绍文档
├── SECURITY.md                                         # 安全声明文件
├── build.sh                                            # 项目工程编译脚本
├── install_deps.sh                                     # 项目依赖安装脚本
├── requirements.txt                                    # Python依赖列表
├── Third_Party_Open_Source_Software_List.yaml          # 第三方开源软件列表
├── Third_Party_Open_Source_Software_Notice             # 第三方开源软件声明
├── version.cmake                                       # 版本信息(CMake格式)
├── version.info                                        # 版本信息文件
├── .clang-format                                       # 代码风格配置
├── .gitignore                                          # Git忽略规则
└── classify_rule.yaml                                  # 分类规则配置
```

## 目录说明

### 核心目录

| 目录/文件 | 说明 |
| :--- | :--- |
| `src/` | 算子源码目录，包含所有算子的实现代码 |
| `src/add/` | add算子目录，实现张量加法运算 |
| `lib/` | 框架代码目录，提供核心基础功能 |
| `lib/core/` | 核心功能模块，包含句柄、描述符、计划等 |
| `lib/elementwise/` | 逐元素运算模块，提供基础元素级操作 |
| `lib/utils/` | 工具模块，包含类型工具、验证等 |
| `include/` | API头文件目录 |
| `cmake/` | CMake编译配置文件 |

### 文档目录

| 目录/文件 | 说明 |
| :--- | :--- |
| `docs/` | 项目文档目录 |
| `docs/zh/` | 中文文档目录 |
| `docs/zh/context/` | 公共文档，如环境部署、目录介绍等 |
| `docs/zh/invocation/` | 算子调用相关文档 |
| `docs/zh/develop/` | 算子开发相关文档 |
| `docs/zh/debug/` | 调试调优相关文档 |
| `docs/implementation.md` | 实现说明文档 |

### 测试目录

| 目录/文件 | 说明 |
| :--- | :--- |
| `tests/` | 项目级测试目录 |
| `tests/test_common.cpp/h` | 测试公共代码 |
| `src/*/tests/` | 各算子测试用例目录 |

### 构建相关

| 目录/文件 | 说明 |
| :--- | :--- |
| `build.sh` | 项目编译脚本，支持多种编译选项 |
| `install_deps.sh` | 项目依赖安装脚本 |
| `CMakeLists.txt` | CMake配置文件 |
| `version.info` | 版本信息文件 |
| `version.cmake` | 版本信息(CMake格式) |

### 脚本目录

| 目录/文件 | 说明 |
| :--- | :--- |
| `scripts/` | 脚本目录 |
| `scripts/check_build_dependencies.py` | 构建依赖检查脚本 |
| `scripts/generate_version_info.py` | 版本信息生成脚本 |
| `scripts/package/` | 打包相关脚本 |
| `scripts/package/package.py` | 打包Python脚本 |

### 配置文件

| 文件 | 说明 |
| :--- | :--- |
| `.clang-format` | 代码风格配置 |
| `.gitignore` | Git忽略规则 |
| `classify_rule.yaml` | 分类规则配置 |
| `requirements.txt` | Python依赖列表 |
| `OAT.xml` | OAT测试配置 |

## 算子目录结构

每个算子目录（如`src/add/`）的典型结构如下：

```
${op_name}/                              # 算子名的小写下划线形式
├── CMakeLists.txt                       # 算子编译配置文件
├── ${op_name}_kernel.cpp                # Kernel实现文件
├── ${op_name}_solution.cpp              # Solution实现文件
├── arch35/                              # Ascend950特有实现
│   └── ${op_name}_struct.h              # 算子结构定义
└── tests/                               # 测试用例目录
    ├── ${op_name}_test.cpp              # 算子测试用例
    └── ${op_name}_test.h                # 测试头文件
```

> **说明**：不同算子的交付件可能有差异，请以实际目录为准。

## 框架目录结构

`lib/` 目录包含框架层的基础功能模块，为所有算子提供统一的抽象接口和基础设施：

```
lib/
├── CMakeLists.txt                       # 库编译配置
├── core/                                # 核心功能模块
│   ├── handle.cpp/hpp                   # 句柄管理
│   ├── operation_descriptor.cpp/hpp     # 操作描述符
│   ├── plan.cpp/hpp                     # 计划管理
│   ├── plan_preference.cpp/hpp          # 计划偏好设置
│   └── tensor_descriptor.cpp/hpp        # 张量描述符
├── elementwise/                         # 逐元素运算模块
│   ├── elementwise.cpp/hpp              # 逐元素运算基础
│   └── elementwise_binary.cpp           # 二元逐元素运算
└── utils/                               # 工具模块
    ├── type_utils.hpp                   # 类型工具
    ├── utils.cpp                        # 通用工具函数
    └── validation.cpp/hpp               # 参数验证
```

### 核心功能模块说明

| 模块 | 文件 | 说明 |
| :--- | :--- | :--- |
| **句柄管理** | `handle.cpp/hpp` | 管理库的上下文句柄，维护全局状态 |
| **操作描述符** | `operation_descriptor.cpp/hpp` | 描述算子操作的参数和属性 |
| **计划管理** | `plan.cpp/hpp` | 管理算子执行计划，支持计划缓存 |
| **计划偏好** | `plan_preference.cpp/hpp` | 设置计划生成的偏好选项 |
| **张量描述符** | `tensor_descriptor.cpp/hpp` | 描述张量的形状、数据类型等属性 |
| **逐元素运算** | `elementwise/` | 提供通用的逐元素运算实现 |
| **工具函数** | `utils/` | 提供类型转换、参数验证等工具函数 |
