# Kernel Matmul StreamK
> [代码位置](../../../include/blaze/kernel/kernel_matmul_streamk.h)

## 功能说明
StreamK 矩阵乘 Kernel，支持 AIC + AIV 双核协同计算。支持 workspace（用于 K 轴切分中间结果）、DP+SK 混合策略、AIC-AIV 跨核同步，适用于大矩阵、高并行度场景。

**继承自**：[Kernel Matmul 基础框架](./kernel.md)

## 特殊约束

### BlockEpilogue 限制
仅支持 `BlockEpilogueStreamK` 系列组件：
```
BlockEpilogueStreamK<float, float, MatmulMultiBlockWithStreamK<ON_THE_FLY>>
BlockEpilogueStreamK<float, float, MatmulMultiBlockWithStreamK<ND_FIXPIPE_1_2>>
BlockEpilogueStreamK<float, half, MatmulMultiBlockWithStreamK<ON_THE_FLY>>
BlockEpilogueStreamK<float, half, MatmulMultiBlockWithStreamK<ND_FIXPIPE_1_2>>
BlockEpilogueStreamK<float, bfloat16_t, ...>
```

不支持 `BlockEpilogueEmpty`。

### 计算模式
支持 AIC + AIV 双核协同：
- **AIC 核**：执行矩阵乘计算（BlockMmadStreamK）
- **AIV 核**：执行后处理（BlockEpilogueStreamK）

### Workspace 必需
必须提供 workspace 用于存储 K 轴切分的中间结果：
- AIC 计算结果输出到 workspace
- AIV 从 workspace 读取并汇聚（Add）
- 最终结果输出到 GM

### BlockScheduler 限制
仅支持 `BlockSchedulerStreamK` 调度器，支持 DP+SK 混合策略。

### HF32 模式
可选 HF32 计算模式，通过 scheduler 参数控制。

### AIC-AIV 同步
使用 `CrossCoreSetFlag` 和 `CrossCoreWaitFlag` 进行跨核同步：
- AIC 完成后设置同步标志
- AIV 等待同步标志后执行后处理

### L0C2Out 模式
支持两种 Fixpipe 输出模式：
- **ON_THE_FLY**：实时输出模式
- **ND_FIXPIPE_1_2**：ND 1v2 优化模式（stride 对齐到 32B）

## 特殊静态常量

| 常量 | 说明 |
|------|------|
| AIC_SYNC_AIV_MODE_4 | 同步模式（MODE_4） |
| AIV_SYNC_AIC_FLAG | AIV 同步 AIC 标志 ID |
| AIC_SYNC_AIV_FLAG | AIC 同步 AIV 标志 ID |
| FLAG_ID_MAX | 标志 ID 最大值（16） |
| BLOCK_BASE_M | Block 基础 M 维度（256） |
| BLOCK_BASE_N | Block 基础 N 维度（256） |

## 特殊类型别名

| 类型 | 说明 |
|------|------|
| BlockMmadOp | BlockMmadStreamK 组件 |
| BlockEpilogueParams | BlockEpilogueStreamK 参数 |
| WorkspaceType | Workspace 数据类型（float） |

## 特殊数据结构

### Params
```
struct Params {
    ProblemShape problemShape;          // 问题 shape (m, n, k, batch)
    BlockMmadParams mmadParams;         // mmad 参数（包含 workspace 地址）
    BlockEpilogueParams epilogueParams; // epilogue 参数
    BlockSchedulerParams schParams;     // scheduler 参数
};
```

### BlockMmadParams（StreamK 特有）
```
struct GmParams {
    GM_ADDR aGmAddr;         // A 矩阵 GM 地址
    GM_ADDR bGmAddr;         // B 矩阵 GM 地址
    GM_ADDR cGmAddr;         // C 矩阵 GM 地址（可选，DP 模式）
    GM_ADDR biasGmAddr;      // Bias GM 地址（可选）
    GM_ADDR workspaceGmAddr; // Workspace GM 地址（必需）
};
```

说明：workspace 用于存储 SK 模式下的中间计算结果。

## 特殊成员方法

### 构造函数
```
__aicore__ inline KernelMatmulStreamK()
```
功能：构造 KernelMatmulStreamK 对象。

### 析构函数
```
__aicore__ inline ~KernelMatmulStreamK()
```
功能：析构 KernelMatmulStreamK 对象。

### Init函数
```
__aicore__ inline void Init(Params const& params)
```
功能：初始化 Kernel，提取问题规模、GM 地址、workspace 地址。
执行流程：
1. 设置问题规模 `problemShape_`
2. 提取 BlockMmad 参数（包含 workspace 地址）
3. 设置 A、B、C、workspace 的 GM 地址
4. 判断 bias 地址是否为 nullptr

### operator函数
```
__aicore__ inline void operator()(Params const& params)
```
功能：执行 StreamK 矩阵乘 Kernel 计算。
执行流程：
**公共部分**：
1. 调用 `Init(params)` 设置参数
2. 创建 BlockScheduler 实例
3. 获取 L1/L0 tile 形状、tile 数量

**AIC 核执行流程**：
1. Block 索引检查：超出实际数量则设置同步标志并返回
2. HF32 模式设置
3. BlockMmadStreamK 初始化
4. Layout 构建：A、B、C、Bias
5. GM Tensor 创建
6. Tile 循环处理：
   - **DP 模式**：结果输出到 GM
   - **SK 模式**：结果输出到 workspace
   - **Preload**：SK 模式下预加载下一轮 tile
7. AIC-AIV 同步：设置 `AIC_SYNC_AIV_FLAG`
8. 清理：关闭 HF32/MM Layout Transform

**AIV 核执行流程**：
1. Block 索引检查：超出处理范围则等待同步并返回
2. 等待 AIC 同步标志 `AIC_SYNC_AIV_FLAG`
3. 调用 `SyncAll` 全核同步
4. BlockEpilogueStreamK 初始化
5. 执行后处理：
   - 从 workspace 读取中间结果
   - 执行 Add 汇聚（K 轴切分累加）
   - 执行类型转换（float → half/bf16）
   - 执行可选 ReLU
   - 输出到 GM

### DP+SK 混合策略
```
// DP 模式：前 (mTileNum * nTileNum - tailMNTileNum) 个 tile
// SK 模式：后 tailMNTileNum * skKTileNum 个 tile
```

说明：
- **DP（Data Parallel）模式**：每个 tile 完整计算，结果直接输出到 GM
- **SK（StreamK）模式**：K 轴切分，结果输出到 workspace，AIV 汇聚

### Preload 优化
```
if (!bs.CheckIsSkScene(0)) { // SK Preload in DP+SK
    if (tileIdx % usedCoreNum_ < tailSKTotalTileNum &&
        (CeilDiv(tileIdx + 1, usedCoreNum_) == (CeilDiv(tileNum, usedCoreNum_) - 1))) {
        tmpTileIdx = tileIdx + usedCoreNum_;  // Preload 下一轮 SK tile
    }
}
```

说明：在最后一轮 DP tile 时，预加载 SK tile 数据到 L1，减少 SK 阶段加载延迟。

## 调用示例

### 组件组装
```
// 定义数据类型
using AType = half;
using BType = half;
using CType = float;
using BiasType = float;

// 定义 Layout
using LayoutA = AscendC::Te::NDExtLayoutPtn;
using LayoutB = AscendC::Te::NZLayoutPtn;
using LayoutC = AscendC::Te::NDExtLayoutPtn;

// 定义调度策略（ON_THE_FLY 或 ND_FIXPIPE_1_2）
using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockWithStreamK<Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY>;

// 定义 BlockMmadStreamK
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;

// 定义 BlockEpilogueStreamK
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueStreamK<float, half, DispatchPolicy>;

// 定义 BlockScheduler
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerStreamK<ProblemShape>;

// 定义 Kernel
using StreamKKernel = Blaze::Gemm::Kernel::KernelMatmulStreamK<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

### 参数准备
```
using Params = typename StreamKKernel::Params;
Params params = {
    {m, n, k, batch},               // problem shape
    {aGM, bGM, cGM, biasGM, workspaceGM}, // mmad params（包含 workspace）
    {cGM, workspaceGM},              // epilogue params
    {usedCoreNum, baseM, baseN, baseK, singleCoreK, kL1, isHf32} // scheduler params
};
```

### Kernel 执行
```
StreamKKernel streamk;
streamk(params);
```

## 数据流

### 存储层次
```
GM (A/B/Bias) → BlockScheduler (DP+SK 混合调度) → L1 → L0 → L0C
                                                         ↓
                                         DP: → GM (C)
                                         SK: → Workspace → AIV → GM (C)
```

### DP 模式流程
```
AIC: GM → L1 → L0 → Mmad → L0C → GM (C)
```

### SK 模式流程
```
AIC: GM → L1 → L0 → Mmad → L0C → Workspace (K 轴切分)
AIV: Workspace → Add 汇聚 → Cast → ReLU → GM (C)
```

### Kernel 执行流程
```
BlockScheduler 初始化（DP+SK 策略）
    ↓
DP Tile 循环（完整 tile）
    ↓
AIC 计算 → 输出到 GM
    ↓
SK Tile 循环（K 轴切分）
    ↓
AIC 计算 → 输出到 Workspace
    ↓
AIC-AIV 同步（CrossCoreSetFlag）
    ↓
AIV 后处理 → 输出到 GM
```

### Workspace 布局
```
workspace 布局：
offsetWorkspace = ((tileIdx % usedCoreNum) / skKTileNum) * skKTileNum + kCntIndex) * BLOCK_BASE_M * BLOCK_BASE_N
```

说明：每个 (m, n) tile 的 K 轴切分结果按顺序存储在 workspace 中。

## 性能优化建议

### DP+SK 混合策略配置
- `tailMNTileNum`：最后一批 tile 数量（进入 SK 模式）
- `skSingleCoreK`：SK 模式下每个核处理的 K 维大小
- 建议：`skSingleCoreK` 约为 `k_ / 4`，平衡 DP 和 SK 负载

### Preload 优化
- 启用 Preload：在最后一轮 DP tile 时预加载 SK tile
- 减少 SK 阶段数据加载延迟

### L0C2Out 模式选择
- **ON_THE_FLY**：实时输出，适用于小矩阵
- **ND_FIXPIPE_1_2**：ND 1v2 优化，适用于大矩阵（stride 对齐）

### AIC-AIV 同步
- 使用 `CrossCoreSetFlag`（AIC 设置）和 `CrossCoreWaitFlag`（AIV 等待）
- 使用 `SyncAll` 全核同步

### Workspace 配置
- workspace 大小：`tailMNTileNum × skKTileNum × BLOCK_BASE_M × BLOCK_BASE_N × sizeof(float)`
- 建议：workspace 大小约为 `m × n × (k / skKTileNum) × sizeof(float)` 的尾块部分

### 适用场景
- **大矩阵场景**：(m × n × k) 较大，需要多核并行
- **K 轴切分场景**：K 维度远大于 M/N
- **高并行度场景**：需要充分利用 AIC 和 AIV 双核