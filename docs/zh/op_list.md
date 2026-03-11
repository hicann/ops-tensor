# 算子列表

> 说明：
> - **算子目录**：目录名为算子名小写下划线形式，每个目录承载该算子所有交付件，包括代码实现、文档等，目录介绍参见[项目目录](./context/dir_structure.md)。
> - **算子执行硬件单元**：项目中提到的算子指AI Core算子。

项目提供的所有算子分类和算子列表如下：

| 算子分类 | 算子目录 | op_kernel | op_host | 算子执行硬件单元 | 说明 |
| :--- | :--- | :---: | :---: | :--- | :--- |
| tensor | [add](#add) | √ | √ | AI Core | 张量加法运算，支持FP32、FP16数据类型。 |

## 算子详细说明

### add

对两个张量执行逐元素加法运算。

**目录位置**：`src/add/`

**支持的芯片**：Ascend 950

**支持的数据类型**：FP32、FP16

**输入**：
- x1：第一个输入张量
- x2：第二个输入张量

**输出**：
- y：输出张量，y = x1 + x2

## 算子分类说明

### tensor 类算子

ops-tensor是[CANN](https://hiascend.com/software/cann)（Compute Architecture for Neural Networks）算子库中提供的**张量操作库**，主要提供以下功能：

- **add**：张量加法运算，对两个张量执行逐元素加法。

## 支持的芯片版本

| 芯片类型 | 支持状态 |
| :--- | :---: |
| Ascend 950PR | √ |
| Ascend 950DT | √ |

> **说明**：本项目当前主要支持 Ascend 950 系列芯片，其他芯片版本的支持情况请关注后续更新。

## 快速开始

如需快速体验算子调用，请参考：
- [环境部署](./context/quick_install.md)：搭建基础环境
- [算子调用](./invocation/quick_op_invocation.md)：编译部署并调用算子
