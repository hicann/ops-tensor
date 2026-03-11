# 项目目录

> 本章罗列的部分目录是可选的，请以实际交付件为准。尤其**单算子目录**，不同场景下交付件有差异。

项目全量目录层级介绍如下：

```
├── cmake                                               # 项目工程编译目录
│   ├── func.cmake                                      # 公共函数
│   ├── init_env.cmake                                  # 环境初始化
│   ├── package.cmake                                   # 打包配置
│   ├── run_package.cmake                               # 打包脚本
│   └── third_party                                     # 第三方依赖配置
│       └── opbase.cmake                                # opbase依赖配置
├── docs                                                # 项目相关文档目录
│   ├── README.md                                       # 文档目录索引
│   └── zh                                              # 中文文档目录
│       ├── op_list.md                                  # 算子列表
│       ├── context                                     # 公共文档目录
│       ├── invocation                                  # 算子调用文档目录
│       ├── develop                                     # 算子开发文档目录
│       └── debug                                       # 调试调优文档目录
├── include                                             # 头文件目录
│   └── cann_ops_tensor.h                               # API头文件
├── scripts                                             # 脚本目录，包含自定义算子、Kernel构建相关配置文件
│   └── package                                         # 打包相关脚本
│       └── ops_tensor/
│           ├── ops_tensor.xml                          # 打包配置文件
│           └── scripts/                                # 安装/卸载脚本
├── src                                                 # 源码目录
│   ├── CMakeLists.txt                                  # 算子编译入口
│   ├── add                                             # add算子目录
│   │   ├── CMakeLists.txt                              # 算子编译配置文件
│   │   ├── add_kernel.cpp                              # Kernel实现文件
│   │   ├── add_host.cpp                                # Host侧代码
│   │   ├── arch35                                      # Ascend950特有算子代码
│   │   │   └── add_struct.h                            # 算子结构定义
│   │   └── tests                                       # 算子测试用例目录
│   │       ├── add_test.cpp
│   │       └── add_test.h
│   └── common                                          # 算子公共代码
│       └── tests
├── tests                                               # 项目级测试目录
├── third_party                                         # 第三方依赖目录
│   ├── makeself                                        # makeself打包工具
│   └── pkg                                             # 预打包的依赖文件
├── CMakeLists.txt                                      # 项目工程cmakelist入口
├── CONTRIBUTING.md                                     # 项目贡献指南文件
├── README.md                                           # 项目工程总介绍文档
├── QUICKSTART.md                                       # 快速入门指南
├── SECURITY.md                                         # 项目安全声明文件
├── build.sh                                            # 项目工程编译脚本
└── version.info                                        # 项目版本信息
```

## 目录说明

### 核心目录

| 目录/文件 | 说明 |
| :--- | :--- |
| `src/` | 算子源码目录，包含所有算子的实现代码 |
| `src/add/` | add算子目录，实现张量加法运算 |
| `src/common/` | 算子公共代码，包含通用工具函数和数据结构 |
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

### 构建相关

| 文件 | 说明 |
| :--- | :--- |
| `build.sh` | 项目编译脚本，支持多种编译选项 |
| `CMakeLists.txt` | CMake配置文件 |
| `version.info` | 版本信息文件 |

## 算子目录结构

每个算子目录（如`src/add/`）的典型结构如下：

```
${op_name}/                              # 算子名的小写下划线形式
├── CMakeLists.txt                       # 算子编译配置文件
├── ${op_name}_kernel.cpp                # Kernel实现文件
├── ${op_name}_host.cpp                  # Host侧代码
├── arch35/                              # Ascend950特有实现
│   └── ${op_name}_struct.h              # 算子结构定义
└── tests/                               # 测试用例目录
    ├── ${op_name}_test.cpp              # 算子测试用例
    └── ${op_name}_test.h                # 测试头文件
```

> **说明**：不同算子的交付件可能有差异，请以实际目录为准。
