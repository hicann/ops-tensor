# ops-tensor

## 🔥Latest News

- [2026/06] 支持 `GroupedMatmul` 算子。
- [2026/05] 支持 `MatMul`、`BatchMatMul` 算子。
- [2026/04] `Blaze` 为算子开发提供基础线性代数加速能力，支持 `QuantBatchMatMul`。

## 🚀概述

ops-tensor 内置了 **Blaze**（Basic Linear Algebra optimiZed Engine）基础线性代数加速引擎，为融合算子开发提供高性能 MM 公共能力，构建 NPU 通用高性能 CUBE 底座。采用模块化设计，支持灵活的算子开发、测试和部署。

## 🔍目录结构

ops-tensor 代码目录结构如下：

```
ops-tensor/
├── cmake/                               # 项目工程编译目录
├── CMakeLists.txt                        # 编译配置文件
├── docs/                                 # 项目文档介绍
│   └ API/                                # API文档
├── include/                              # 项目公共头文件
│   ├── blaze/                            # Blaze 高性能线性代数引擎
│   └ tensor_api/                         # Tensor 抽象（Layout/Shape/Coord）
├── tests/                                # UT/ST 测试工程目录
│   └ ut/op_kernel/                       # Kernel UT 测试
├── build.sh                              # 编译脚本
├── README.md                              # 项目说明
├── QUICKSTART.md                          # 快速入门
├── QUICK_OP_INVOCATION.md                 # 算子开发指南
```

## 🔧支持的算子

| 算子名称 | 描述 | 状态 |
|---------|------|------|
| MatMul | 矩阵乘运算 | ✅ 已实现 |
| BatchMatMul | 批量矩阵乘运算 | ✅ 已实现 |
| QuantBatchMatMul | 量化批量矩阵乘运算 | ✅ 已实现 |
| GroupedMatmul | 分组矩阵乘运算 | ✅ 已实现 |

## 💻SoC 支持

Ascend 950PR / Ascend 950DT

## ⚡️快速入门

若您希望快速体验项目，请访问[快速入门](./QUICKSTART.md)获取简易教程，包括环境搭建、编译执行、本地验证等操作。

- [环境准备](./QUICKSTART.md#环境准备)：安装软件包之前，需要完成搭建基础环境，包括第三方依赖等；基础环境搭建后需要完成社区版CANN软件包安装、环境变量配置等。
- [源码下载](./QUICKSTART.md#源码下载)：本项目源码下载。
- [编译执行](./QUICKSTART.md#编译执行)：环境准备好后，可对源码修改编译生成可执行的文件。
- [Kernel UT 测试](./QUICKSTART.md#Kernel-UT测试)：基于项目根目录的 build.sh 脚本，可执行 Kernel UT 用例，快速验证功能。
- [打包安装](./QUICKSTART.md#打包安装)：编译并打包生成 .run 安装包。

## 📖文档介绍

| 文档 | 说明 |
|------|------|
|[快速入门](./QUICKSTART.md)|快速体验项目的简易教程。|
|[算子开发指南](./QUICK_OP_INVOCATION.md)|使用 ops-tensor 实现算子开发的教程。|
|[Blaze 模块](./include/blaze/README.md)|Blaze 高性能线性代数引擎介绍。|
|[API 文档](./docs/API/README.md)|API 详细说明。|

## 📝相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)