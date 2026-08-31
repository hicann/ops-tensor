# BlockAttnResPrepare Kernel

> [代码位置](../../../../include/blaze/attention/kernel/kernel_block_attn_res_prepare.h)

`KernelBlockAttnResPrepare` 是以下组件的公开组合别名：

```cpp
AttentionUniversal<ProblemShape, BlockMmadTuple, BlockEpilogue, BlockScheduler>
```

`ProblemShape` 按 `[S,N,D,T]` 解释，`BlockMmadTuple` 的元素 0/1 分别为 MM1 和 MM2。编译期通过
`KernelBlockAttnResPrepareSchedule` 选择对应的 `AttentionUniversal` 特化。

## 参数组织

```cpp
struct Params {
    ProblemShape problemShape;
    Mm1Params mm1Params;
    Mm2Params mm2Params;
    BlockEpilogueParams epilogueParams;
    BlockSchedulerParams schedulerParams;
};
```

这种组织与 `GemmUniversal` 一致。ops-transformer 显式组装每个组件的参数；Tensor Kernel 不再接收专用
扁平 Tiling 结构，也不在 Kernel 内重新计算 MM1/MM2 的 L1/L0 参数。

## 职责

- 把组件参数中的 GM 地址绑定为强类型 Tensor；
- 读取并把 `validBlocks` 裁剪到 `[0,N]`；
- 把物理 AIC/AIV index 映射为共享 workspace 的逻辑核 index；
- 构造 Scheduler，并消费其 `BlockInfo` 和 `AivRowRange`；
- 构造 residual、dot、E、max、sum、output 等语义化 Tensor；
- 编排 MM1、AIV Epilogue、MM2 和 mode-4 跨核同步。

Host Tiling 负责形状、容量和切分合法性，Kernel 不重复执行 `HasSafeStorageContract` 一类运行时校验。

## 流程

```text
AIC: MM1(Q * V^T) -> dot workspace
                  |
                  v
AIV: reduce(V^2) -> RMS normalize(dot) -> softmax -> E workspace
                  |
                  v
AIC: MM2(E * V) -> numerator GM
```

同一逻辑组的一个 AIC 和两个 AIV 共享一段 per-core workspace。Kernel 使用三组握手：dot ready、E ready、
E buffer free。MM2 通过 Fixpipe 直接写最终 GM。

`validBlocks <= 0` 时跳过 Cube，仅由 AIV0 把 `numerator`、`logitMax`、`expSum` 写 0。
