# Gemm/Epilogue 类模板概述

## API 清单

| 组件名 | 描述 |
| :----------------------------------------------------------- | :------: |
| [block_epilogue_empty](./block/block_epilogue_empty.md) | 空后处理组件，用于不支持后处理的 Kernel |
| [block_epilogue_streamk](./block/block_epilogue_matmul_streamk.md) | StreamK 后处理组件，支持 workspace 汇聚、类型转换、ReLU |
| [block_epilogue_dequant](./block/block_epilogue_dequant.md) | MIX 模板 dequant 向量后处理，AIV 侧 int32→fp32 × scale + bias → bf16/fp16/fp32 |
| [block_epilogue_qbmm_pertensor_streamk](./block/block_epilogue_qbmm_pertensor_streamk.md) | QBMM per-tensor StreamK 专用后处理，归约 raw partial 后应用 scale/bias |
| [block_epilogue_per_token_scale](./block/block_epilogue_per_token_scale.md) | 应用 per-token scale 和可选 row-sum offset 修正 |
| [block_epilogue_fmm_with_scale_add](./block/block_epilogue_fmm_with_scale_add.md) | FusedMatMul scale_add 向量后处理，执行 alpha × (x1@x2) + beta × x3 |

## 公共框架

所有 BlockEpilogue 组件基于 [block_epilogue.md](./block/block_epilogue.md) 公共框架实现，包含统一的：
- 类型别名
- 数据结构（Params）
- 核心方法（Init、Run、operator）

详见：[block_epilogue.md](./block/block_epilogue.md)

## 核心组件关系

```
BlockEpilogue
    ├── BlockShape (Block 形状)
    ├── BlockCoord (Block 坐标)
    ├── Params (参数结构)
    └── 核心方法
            ├── Init (初始化)
            ├── Run (执行后处理)
            └── operator() (调用接口)
```

## 实现差异对比

| Epilogue 类型 | 计算内容 | 计算位置 | Workspace | 类型转换 | ReLU | 适用场景 |
|--------------|---------|---------|-----------|---------|-----|---------|
| BlockEpilogueEmpty | 无（空实现） | 无 | 不支持 | 无 | 不支持 | Basic Kernel |
| BlockEpilogueStreamK | workspace 汇聚、Add、Cast、ReLU | AIV 核 | 支持 | 支持 float → half/bf16 | 可选支持 | StreamK Kernel |
| BlockEpilogueDequant | dequant：int32→fp32 × x2Scale [× x1Scale] + bias | AIV 核 | 不支持 | 支持 fp32 → bf16/fp16/fp32 | 不支持 | QBMM MIX Kernel |
| BlockEpilogueQbmmPertensorStreamK | workspace 归约 + dequant + bias | AIV 核 | 支持 | 支持 int32/fp32 → bf16/fp16/fp32 | 不支持 | QBMM per-tensor StreamK |
| BlockEpilogueFmmWithScaleAdd | alpha × (x1@x2) + beta × x3 | AIV 核 | 不支持 | 支持 fp32 → bf16/fp16 | 不支持 | FusedMatMul scale_add Kernel |

## 使用流程

1. **查看公共框架**：了解类型别名和核心接口 → [block_epilogue.md](./block/block_epilogue.md)
2. **选择具体实现**：根据 Kernel 类型选择 Empty 或 StreamK
3. **组装组件**：在 Kernel 模板参数中定义 BlockEpilogue 类型
4. **初始化**：调用 Init 设置参数（StreamK 需要额外参数）
5. **执行后处理**：调用 Run 或 operator 执行（Empty 无实际效果）

## 设计说明

### 为什么需要 BlockEpilogueEmpty
1. **模板参数要求**：Kernel 模板需要 BlockEpilogue 参数
2. **接口一致性**：保持与其他 Epilogue 组件相同的接口
3. **扩展性**：未来可替换为实际的后处理组件
4. **零开销**：空实现不会引入额外计算开销

### BlockEpilogue 在 Kernel 中的作用
- **Basic Kernel**：使用 BlockEpilogueEmpty，无后处理
- **StreamK Kernel**：使用 BlockEpilogueStreamK，在 AIV 核执行后处理
  - 从 workspace 读取 AIC 计算的中间结果
  - 执行 Add 汇聚（K 轴切分）
  - 执行类型转换（float → half/bf16）
  - 执行可选的 ReLU 激活
  - 输出最终结果到 GM
