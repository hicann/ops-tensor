# ops-tensor

## 项目简介

ops-tensor 是一个轻量级的算子库项目，提供基本的张量操作算子实现。本项目采用模块化设计，支持灵活地添加和管理算子。

## 主要特性

- ✅ **模块化设计** - 支持动态添加算子模块
- ✅ **标准 CMake 构建** - 跨平台编译支持
- ✅ **完整测试体系** - 基于 CTest 的单元测试
- ✅ **便捷打包** - 一键生成 .run 安装包
- ✅ **企业级安装** - 支持 install/uninstall/upgrade
- ✅ **版本管理** - 安装信息记录和版本追踪

---

## 与 ops-math 的对比

| 功能 | ops-math | ops-tensor | 说明 |
|------|----------|------------|------|
| **基本功能** | | | |
| 安装/卸载/升级 | ✅ | ✅ | 均支持完整的安装管理 |
| 版本管理 | ✅ | ✅ | ascend_install.info 记录安装信息 |
| 日志系统 | ✅ | ✅ | 结构化日志到文件 |
| 错误码 | ✅ | ✅ | 标准化错误码定义 |
| 权限处理 | ✅ | ✅ | root/用户自动适配 |
| 静默模式 | ✅ | ✅ | --quiet 参数 |
| **高级功能** | | | |
| --full 分类安装 | ✅ | ❌ | ops-tensor 结构简单，不需要 |
| Docker 支持 | ✅ | ❌ | 无 Docker 特殊文件 |
| 异构安装 | ✅ | ❌ | 无跨架构需求 |
| 预检查脚本 | ✅ | ❌ | 暂不需要 |
| XML 配置 | 复杂（多文件） | 简单（单文件） | ops-tensor 更简洁 |
| package.py | ✅ | ❌ | ops-tensor 使用 CMake 直接生成 |
| **复杂度** | 企业级 | 简洁高效 | 各取所需 |

**设计理念：**
- **ops-math** - 面向企业级复杂场景，支持多种部署模式和特殊需求
- **ops-tensor** - 面向简单场景，够用即可，避免过度设计

---

## 快速开始

### 编译

```bash
# 编译所有算子
./build.sh

# 编译指定算子
./build.sh --ops=add

# 编译并运行测试
./build.sh --run
```

### 打包

```bash
# 编译并打包成 .run 文件
./build.sh --pkg

# 输出文件：build/cann-ops-tensor-1.0.0-linux-Ascend910B-x86_64.run
```

---

## 安装与使用

### 安装

```bash
# 标准安装（root）
sudo ./cann-ops-tensor-1.0.0-linux-*.run

# 用户安装
./cann-ops-tensor-1.0.0-linux-*.run

# 安装到自定义路径
sudo ./cann-ops-tensor-1.0.0-linux-*.run --install-path=/opt/ascend
```

### 卸载

```bash
# 方式1：使用 .run 包
sudo ./cann-ops-tensor-*.run --uninstall

# 方式2：直接运行卸载脚本
sudo /usr/local/Ascend/cann/share/info/ops_tensor/scripts/uninstall.sh
```

### 升级

```bash
sudo ./cann-ops-tensor-*.run --upgrade
```

---

## Run 包结构

### 内部结构（解压后）

```
cann-ops-tensor-1.0.0-linux-Ascend910B-x86_64.run
    ↓ 解压（--noexec --target /tmp/extract）

/tmp/extract/
│
├── lib/                                    # 库文件目录
│   └── libops_tensor.so                    # ops-tensor 核心动态库
│                                            # 功能：提供算子实现，供用户程序链接
│
├── include/                                # 头文件目录
│   └── ops_tensor.h                        # 公共 API 头文件
│                                            # 功能：声明接口，用户代码 include 使用
│
└── share/
    └── info/
        └── ops_tensor/                      # ops-tensor 信息目录
            │
            ├── version.info                # 版本信息文件
            │   # Version=1.0.0
            │   # ReleaseDate=2025-03-02
            │   # 功能：记录包版本和发布信息
            │
            ├── scripts/                    # 脚本目录
            │   │
            │   ├── install.sh              # 安装脚本
            │   │   # 功能：执行文件拷贝、权限设置、环境配置
            │   │   # 参数：--install, --uninstall, --upgrade, --quiet, --install-path
            │   │
            │   ├── uninstall.sh            # 卸载脚本
            │   │   # 功能：删除文件、清理目录、删除安装信息
            │   │   # 参数：--install-path, --quiet
            │   │
            │   └── help.info               # 帮助信息
            │       # 功能：显示使用方法和命令说明
            │
            └── ascend_install.info         # 安装信息文件（安装后生成）
                # USERNAME=root
                # OPS_TENSOR_INSTALL_PATH=/usr/local/Ascend/cann
                # OPS_TENSOR_VERSION=1.0.0
                # 功能：记录安装历史，供卸载/升级使用
```

---

## 安装后的路径结构

### CANN 环境中的路径

```
/usr/local/Ascend/cann/                     # CANN 安装根目录（版本目录）
│
├── lib/                                    # 库文件目录
│   └── libops_tensor.so                    # ✅ ops-tensor 动态库
│   # 权限：550 (r-xr-x---)
│   # 用途：用户程序链接时使用 -lops_tensor -L/usr/local/Ascend/cann/lib
│
├── include/                                # 头文件目录
│   └── ops_tensor.h                        # ✅ 公共 API 头文件
│   # 权限：440 (r--r-----)
│   # 用途：用户代码中使用 #include "ops_tensor.h"
│
└── share/
    └── info/
        └── ops_tensor/                      # ops-tensor 信息目录
            │
            ├── version.info                # ✅ 版本信息
            │
            ├── scripts/                    # ✅ 脚本目录
            │   ├── install.sh              # 权限：555 (r-xr-xr-x)
            │   ├── uninstall.sh            # 权限：555 (r-xr-xr-x)
            │   └── help.info               # 权限：444 (r--r--r--)
            │
            └── ascend_install.info         # ✅ 安装信息文件
                # 权限：644 (rw-r--r--)
                # 功能：记录安装元数据
```

### 与其他 CANN 组件共存

```
/usr/local/Ascend/cann/
│
├── lib/
│   ├── libops_tensor.so                    # ✅ ops-tensor 库
│   ├── libopapi_math.so                   # ops-math 库（如果安装）
│   ├── libge_ir.so                        # 图执行器库
│   └── ...                                 # 其他 CANN 库
│
├── include/
│   ├── ops_tensor.h                        # ✅ ops-tensor 头文件
│   ├── aclnnop/                            # ops-math 头文件
│   ├── graph/                              # 图相关头文件
│   └── ...                                 # 其他头文件
│
└── share/info/
    ├── ops_tensor/                         # ✅ ops-tensor 信息
    ├── ops_math/                           # ops-math 信息
    └── ...                                 # 其他组件信息
```

---

## 文件权限说明

| 文件类型 | 权限 | 属主 | 说明 |
|---------|------|------|------|
| 动态库 (.so) | 550 | root:root | 用户可读可执行 |
| 头文件 (.h) | 440 | root:root | 用户可读 |
| 脚本 (.sh) | 555 | root:root | 所有人可读可执行 |
| 版本信息 | 440 | root:root | 用户可读 |
| 安装信息 | 644 | root:root | 可读写 |

---

## 使用示例

### 用户代码

```cpp
#include "ops_tensor.h"
#include <iostream>

int main() {
    // 创建张量
    Tensor* tensor = ops_tensor_create(/*...*/);

    // 使用张量操作
    // ...

    // 销毁张量
    ops_tensor_destroy(tensor);
    return 0;
}
```

### 编译链接

```bash
g++ -o my_app main.cpp \
    -I/usr/local/Ascend/cann/include \
    -L/usr/local/Ascend/cann/lib \
    -lops_tensor
```

### 运行

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/cann/lib:$LD_LIBRARY_PATH
./my_app
```

---

## 项目结构

```
ops-tensor/
├── include/                 # 公共头文件
├── src/                     # 源代码
│   ├── add/                # Add 算子
│   └── ...
├── tests/                   # 测试代码
├── cmake/                   # CMake 配置
│   ├── func.cmake          # 公共函数
│   ├── package.cmake       # 打包配置
│   └── run_package.cmake   # 打包脚本
├── scripts/
│   └── package/            # 打包相关
│       └── ops_tensor/
│           ├── ops_tensor.xml
│           └── scripts/
├── build.sh                 # 构建脚本
├── version.info             # 版本信息
└── README.md                # 本文件
```

---

## 许可证

[待添加]

---

## 联系方式

[待添加]
