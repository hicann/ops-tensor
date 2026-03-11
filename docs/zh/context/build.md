# build.sh 参数说明

## 简介

build.sh 是 ops-tensor 项目的构建脚本，位于项目根目录下。该脚本通过配置不同参数实现多种功能，包括编译算子、运行测试、生成安装包等。

## 使用方法

### 1. 配置环境变量

在使用 build.sh 之前，需要先配置 CANN 环境变量：

```bash
# 默认路径安装
source /usr/local/Ascend/cann/set_env.sh

# 验证环境变量
echo $ASCEND_HOME_PATH
```

### 2. 构建命令格式

```bash
./build.sh [OPTIONS]
```

## 参数说明

build.sh 支持多种功能，可通过 `--help` 参数查看所有选项：

```bash
./build.sh --help
```

| 参数 | 必选/可选 | 说明 |
|------|----------|------|
| `--ops=OP_LIST` | 可选 | 指定要编译的算子列表，多个算子用逗号分隔（如：`--ops=add,sub`）。不指定时编译所有算子。 |
| `--run` | 可选 | 编译后执行测试。需要配合 `BUILD_TESTING=ON` 使用。 |
| `--pkg` | 可选 | 编译并打包成 .run 安装包。 |
| `--soc=SOC` | 可选 | 指定目标 SoC 型号，支持大小写不敏感输入（如：`--soc=ascend950` 或 `--soc=Ascend950`）。默认为 `Ascend950`。 |
| `-j[N]` | 可选 | 指定编译线程数，默认为 8（如：`-j16`）。若线程数超过 CPU 核心数，会自动调整为 CPU 核心数。 |
| `--test-timeout=N` | 可选 | 指定测试超时时间（单位：秒），默认为 300。仅在 `--run` 模式下有效。 |
| `-h, --help` | 可选 | 显示帮助信息。 |

## 支持的 SoC 型号

| SoC 型号 | SOC_VERSION（CANN 编译器） | 说明 |
|---------|---------------------------|------|
| Ascend950 | ascend950dt_9595 | 默认支持（dav-3510） |
| Ascend910B | ascend910b3 | dav-2201 |
| Ascend910_93 | ascend910_93 | dav-2201 |
| Ascend910 | ascend910 | dav-2101 |
| Ascend310P | ascend310p | dav-2101 |
| Ascend310B | ascend310b | dav-2101 |

## 使用示例

### 基本编译

```bash
# 编译所有算子（默认 8 线程）
./build.sh

# 编译指定算子
./build.sh --ops=add

# 编译多个算子
./build.sh --ops=add,sub

# 使用 16 线程编译
./build.sh -j16
```

### 编译并测试

```bash
# 编译所有算子并运行测试
./build.sh --run

# 编译指定算子并运行测试
./build.sh --ops=add --run

# 指定测试超时时间（600 秒）
./build.sh --run --test-timeout=600
```

### 打包

```bash
# 编译所有算子并打包（默认 SoC: Ascend950）
./build.sh --pkg

# 编译指定算子并打包
./build.sh --ops=add --pkg

# 为指定 SoC 打包
./build.sh --soc=Ascend910B --pkg

# 大小写不敏感
./build.sh --soc=ascend910b --pkg
```

### 组合使用

```bash
# 编译 add 算子、运行测试、使用 16 线程
./build.sh --ops=add --run -j16

# 编译所有算子、打包、指定 SoC
./build.sh --pkg --soc=Ascend910B -j16
```

## 行为说明

| 命令 | 行为 |
|------|------|
| 无参数 | 编译所有算子，不执行测试 |
| `--ops=add` | 只编译 add 算子，不执行测试 |
| `--ops=add,sub` | 编译 add 和 sub 算子，不执行测试 |
| `--run` | 编译所有算子，并执行所有算子的测试 |
| `--ops=add --run` | 编译 add 算子，并执行 add 算子的测试 |
| `--ops=add,sub --run` | 编译 add、sub 算子，并执行这些算子的测试 |
| `--pkg` | 编译所有算子并打包成 .run 文件（默认 SoC: Ascend950） |
| `--ops=add --pkg` | 编译 add 算子并打包成 .run 文件 |
| `--soc=ascend950 --pkg` | 为 Ascend950 芯片打包（支持小写） |

## 输出说明

### 编译输出

编译成功后，生成的文件位于 `build/` 目录：

```
build/
├── libops_tensor.so         # 动态库
├── tests/
│   └── all_ops_test         # 测试可执行文件
└── ...
```

### 打包输出

打包成功后，会在 `build/` 目录生成 .run 文件：

```
build_out/cann-{soc}-ops-tensor_{version}_linux-{arch}.run
```

例如：
- `cann-950-ops-tensor_9.0.0_linux-x86_64.run`
- `cann-910b-ops-tensor_9.0.0_linux-aarch64.run`

## 注意事项

1. **环境变量要求**：必须设置 `ASCEND_HOME_PATH` 环境变量，否则脚本会报错退出。

2. **线程数限制**：如果指定的线程数超过 CPU 核心数，脚本会自动调整为 CPU 核心数。

3. **测试超时**：默认测试超时时间为 300 秒，可根据实际情况调整。

4. **SoC 大小写**：`--soc` 参数支持大小写不敏感输入，脚本会自动标准化为首字母大写、其余小写的格式。

5. **算子验证**：如果使用 `--ops` 指定了不存在的算子，脚本会列出所有可用的算子并报错退出。

## 相关文档

- [环境部署](quick_install.md)
- [算子调用](../invocation/quick_op_invocation.md)
- [算子开发](../develop/aicore_develop_guide.md)
