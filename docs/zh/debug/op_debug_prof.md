# 算子调试调优

本文档介绍ops-tensor项目中常见的算子调试和调优方法。

## 概述

在算子开发过程中，可能会遇到以下问题：

- 算子功能异常，输出结果不正确
- 算子性能不达标，需要优化
- 算子运行时报错

针对这些问题，本文档提供以下调试和调优方法：

- [日志调试](#日志调试)：通过日志定位问题
- [性能调优](#性能调优)：优化算子性能

## 日志调试

### 编译时日志

使用`-v`参数查看详细编译输出：

```bash
bash build.sh -v --run
```

### 运行时日志

CANN提供了多种日志级别，可通过环境变量配置：

```bash
# 设置日志级别
export ASCEND_GLOBAL_LOG_LEVEL=3  # 0-debug, 1-info, 2-warning, 3-error

# 设置日志路径
export ASCEND_GLOBAL_EVENT_ENABLE=0
```

### 常见错误排查

| 错误信息 | 可能原因 | 解决方案 |
| :--- | :--- | :--- |
| `Out of memory` | 内存不足 | 减小tile size或增加workspace |
| `Invalid tiling parameters` | Tiling参数错误 | 检查TilingData结构体定义 |
| `Kernel launch failed` | Kernel启动失败 | 检查核函数定义和参数 |

## 性能调优

### 1. Tiling优化

合理设置Tiling参数可以提高算子性能：

- **Block切分**：根据核数均匀分配数据
- **UB切分**：充分利用Unified Buffer空间
- **对齐要求**：确保数据地址和大小满足硬件对齐要求

### 2. 内存优化

- 减少内存拷贝次数
- 使用双缓冲（Double Buffer）技术
- 合理规划workspace大小

### 3. 计算优化

- 向量化计算：使用Ascend C向量指令
- 流水并行：合理使用多队列
- 指令融合：减少中间结果存储

## 调试工具

### msProf性能分析

使用msProf工具进行性能分析：

```bash
msprof --output=./prof_out ./your_program
```

### DumpTensor数据导出

导出算子中间结果进行调试：

```bash
export ASCEND_WORK_PATH=./dump_out
export ASCEND_GLOBAL_LOG_LEVEL=0
```

## 更多帮助

- [CANN 开发文档](https://www.hiascend.com/document)
- [Ascend C 性能优化指南](https://hiascend.com/document/redirect/CannCommunityAscendCPerf)
