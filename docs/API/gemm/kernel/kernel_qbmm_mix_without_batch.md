# Kernel Qbmm Mix Without Batch
> [代码位置](../../../../include/blaze/gemm/kernel/kernel_qbmm_mix_without_batch.h)

## 功能说明
MIX 模板量化 Matmul Kernel（无 Batch 变体），与 [kernel_qbmm_mix](./kernel_qbmm_mix.md) 对称，裁剪掉 4 维 Batch 广播路径，提供轻量化的单 Batch 调度。**AIC（cube）+ AIV（vector）双核协同**：AIC 做 int32 矩阵乘并 fixpipe（NoQuant）搬 L0C→UB，AIV 在向量上做 dequant + x2Scale [* x1Scale] + bias，输出 bf16/fp16/fp32。支持 int8（per-token / per-channel / per-tensor）与 WeightNz（FRACTAL_NZ）。

**继承自**：[Kernel Matmul 基础框架](./kernel.md)
**配套组件**：[block_mmad_a8w8_mix](../block/block_mmad_a8w8_mix.md) + [block_epilogue_dequant](../../epilogue/block/block_epilogue_dequant.md)

## 与带 Batch 版本的差异
| 维度 | kernel_qbmm_mix | kernel_qbmm_mix_without_batch |
|------|-----------------|-------------------------------|
| 类名 | `GemmUniversal<...>`（SFINAE 特化） | `QbmmMixWithoutBatch` |
| Batch | 4 维 Batch 广播 + 尾块 latch | 仅单 Batch，无 Batch 偏移逻辑 |
| 尾块切分 | 跨 Batch latch（needUpdateTail_ + restBatch） | 单轮判断即可 |
| QBMMTiling | batchA/B/C 等 12 个字段 | groupSizeM/N/K（无 batch 字段） |
| 偏移计算 | 含 `batchCOffset_` | `mPos * n + nPos` |

## 特殊约束
- AIC + AIV 双核（同带 Batch 版本）。
- 仅支持 `BlockSchedulerQbmm`，单 Batch；支持尾块切分（mTailTile / nTailTile）。
- 量化与权重格式约束同 [kernel_qbmm_mix](./kernel_qbmm_mix.md)。

## 特殊数据结构

### Params
```
struct Params {
    ProblemShape problemShape;       // 问题 shape (m, n, k)
    BlockMmadParams mmParams;        // mmad 参数（A/B/C GM 地址）
    BlockSchedulerParams schParams;  // scheduler 参数（含 mTailTile / nTailTile）
    QBMMTiling qbmmParams;           // QBMM tiling
    EpilogueParams epilogueParams;   // dequant epilogue 参数
};
```

### QBMMTiling
```
struct QBMMTiling {
    uint32_t groupSizeM, groupSizeN, groupSizeK;  // group 切分
    uint32_t kAL1, kBL1;       // A/B 的 L1 K 轴切分
    uint32_t nBufferNum;       // L1 缓冲数量
    uint32_t baseM, baseN, baseK;  // L0 tile 形状
    uint32_t isBias;           // 是否启用 bias
    uint32_t dbL0C;            // L0C 双缓冲标志
};
```

## 特殊成员方法

### operator函数
```
__aicore__ inline void operator()(const Params& params)
```
执行流程：
1. 构造 BlockScheduler。
2. AIC：用 `{baseM, baseN, baseK}` 与 `kAL1/kBL1/nBufferNum/dbL0C` 初始化 BlockMmad。
3. AIV：用 `epilogueParams` 初始化 BlockEpilogueDequant。
4. 调用 `Run(params, bs)`。

### Run函数
```
__aicore__ inline void Run(const Params& params, BlockScheduler& bs)
```
执行流程：
1. 构建 A/B 的 GM Tensor。
2. 尾块判断：`(GetEndBlockIdx()+1) * mTailTile * nTailTile <= GetBlockNum()` 时 `UpdateTailTile`。
3. Tile 循环：
   - **AIC**：必要时 `WaitForVector()`；Slice A/B；按 `CeilAlign(curN, L0C_ALIGN)` 对齐 UB 行距；BlockMmad 写 L0C→UB；`NotifyVector()`。
   - **AIV**：`WaitForCube()`；调用 epilogue（偏移 `scale=nPos, ptScale=mPos, bias=nPos, C=mPos*n+nPos`）；`NotifyCube()`。
4. 收尾：AIC 若有 tile，`WaitForVector()`。

### AIC<->AIV 同步
同 [kernel_qbmm_mix](./kernel_qbmm_mix.md)：NotifyVector / WaitForVector（PIPE_FIX，flag 0 与 0+16），NotifyCube / WaitForCube（PIPE_V，flag 0）。

## 调用示例
```
using QBMMKernel = Blaze::Gemm::Kernel::QbmmMixWithoutBatch<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
QBMMKernel qbmm;
qbmm(params);
```

## 适用场景
- arch35（Ascend910D）单 Batch int8 量化 Matmul。
- 无需 Batch 广播、追求更轻调度路径的量化推理与 WeightNz 场景。
