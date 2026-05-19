# Kernel Matmul Basic
> [代码位置](../../../include/blaze/kernel/kernel_matmul_basic.h)

## 功能说明
基础矩阵乘 Kernel，仅支持 AIC 计算，无 AIV 参与，不支持 workspace。适用于小矩阵、简单计算场景，集成 BlockScheduler 调度、BlockMmad 计算和 BlockEpilogueEmpty 后处理组件。

**继承自**：[Kernel Matmul 基础框架](./kernel.md)

## 特殊约束

### BlockEpilogue 限制
仅支持 `Block::BlockEpilogueEmpty`，不支持任何后处理操作。
```
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
```

### 计算模式
仅在 AIC 核函数中执行，不支持 AIV 计算（AIV 核直接返回）。
```
if ASCEND_IS_AIV {
    return;  // AIV 核直接返回，不执行任何计算
}
```

### Workspace 不支持
不支持 workspace，无法存储中间结果，适用于完整 tile 计算场景。

### 非连续输入
不支持带 stride 的非连续输入（仅支持连续 ND/NZ layout）。

### FP32 大 K
不支持 FP32 大 K 场景（K 轴切分受硬件限制），K 值过大时需使用 StreamK Kernel。

## 特殊成员方法

### 构造函数
```
__aicore__ inline KernelMatmulBasic()
```
功能：构造 KernelMatmulBasic 对象。

### 析构函数
```
__aicore__ inline ~KernelMatmulBasic()
```
功能：析构 KernelMatmulBasic 对象。

### UnsetHf32函数
```
__aicore__ inline void UnsetHf32(bool isHf32)
```
功能：关闭 HF32 模式。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| isHf32 | bool | 是否启用 HF32 模式 |

说明：
- 当 `isHf32 = true` 时，调用 `AscendC::SetHF32Mode(0)` 关闭 HF32 模式
- Basic Kernel 在计算完成后自动调用此函数清理 HF32 状态

### HF32 模式设置流程
```
// Kernel 开始时
if (isHf32) {
    AscendC::SetHF32Mode(1);
    AscendC::SetHF32TransMode(1);
}

// Kernel 结束时
UnsetHf32(isHf32);  // 关闭 HF32 模式
```

### MM Layout Transform
```
SetMMLayoutTransform(true);   // 调用 BlockMmad 前设置为列主序（适配 Fixpipe）
// ... BlockMmad 计算 ...
SetMMLayoutTransform(false);  // 计算后关闭
```
说明：MM Layout Transform 用于适配 Fixpipe 输出格式，确保 L0C 数据正确搬出到 GM。

### L2 Cache 配置
```
// 根据 scheduler 参数禁用 A/B 的 L2 Cache
if (bs.GetBL2CacheDisable()) {
    gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
}
if (bs.GetAL2CacheDisable()) {
    gmA.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
}
```
说明：可选禁用 A/B 矩阵的 L2 Cache，避免大矩阵场景下的缓存污染。

## Tile 循环策略

### Basic 特有的循环策略
```
for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
    // 每个 block 处理 tileIdx = curBlockIdx, curBlockIdx + blockNum, curBlockIdx + 2*blockNum, ...
    // 多 block 并行处理不同 tile
    // 每个 tile 独立调用 BlockMmad 执行矩阵乘
}
```

说明：
- **简单 stride 策略**：block 按固定 stride（blockNum）遍历 tile
- **无 workspace**：每个 tile 完整计算，不依赖中间结果
- **无 AIC-AIV 同步**：单核计算，无需跨核同步

### BlockScheduler 功能（Basic 特有）
| 功能 | 说明 |
|------|------|
| GetTileNum | 获取总 tile 数量 |
| GetTileL1Shape | 获取 L1 tile 形状 |
| GetTileL0Shape | 获取 L0 tile 形状 |
| GetBlockNum | 计算实际需要的 block 数量 |
| GetBlockCoord | 获取 tile 的坐标 |
| GetBlockShape | 获取 tile 的形状 |
| Gethf32Flag | 获取 HF32 模式标志 |
| GetL1BuferNum | 获取 L1 缓冲数量 |
| GetL0cDB | 获取 L0C 双缓冲标志 |
| GetBL2CacheDisable | 获取 B 矩阵 L2 Cache 禁用标志 |
| GetAL2CacheDisable | 获取 A 矩阵 L2 Cache 禁用标志 |

## 调用示例

### Kernel 组装与调用

```cpp
// ============== 1. 类型定义 ==============
using AType = half;                              // A 矩阵数据类型
using BType = half;                              // B 矩阵数据类型
using CType = half;                              // C 矩阵输出类型
using BiasType = half;                           // Bias 数据类型（可选）
using LayoutA = AscendC::Te::NZLayoutPtn;        // A 矩阵布局（NZ/ND）
using LayoutB = AscendC::Te::NZLayoutPtn;        // B 矩阵布局（NZ/ND）
using LayoutC = AscendC::Te::NDLayoutPtn;        // C 矩阵布局（ND）
using LayoutBias = LayoutC;                      // Bias 布局

// ============== 2. ProblemShape 定义 ==============
// 形状：(m, n, k, batch)，batch=0 或 1 表示单 batch
using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

// ============== 3. BlockScheduler 组装 ==============
// FullLoadMode: 0=非全载（默认）, 1=A全载, 2=B全载
constexpr int64_t FULL_LOAD_MODE = 0;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<
    ProblemShape, FULL_LOAD_MODE>;

// ============== 4. BlockMmad 组装 ==============
// DispatchPolicy: 调度策略，FusedOpType: 融合操作类型
constexpr uint64_t FUSED_OP_TYPE = 0;
using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<FULL_LOAD_MODE, FUSED_OP_TYPE>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA,
    BType, LayoutB, CType, LayoutC,
    BiasType, LayoutBias>;

// ============== 5. BlockEpilogue 组装 ==============
// Basic Kernel 仅支持 Empty，不支持后处理
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;

// ============== 6. Kernel 组装 ==============
using MatmulKernel = Blaze::Gemm::Kernel::KernelMatmulBasic<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

// ============== 7. Params 构造 ==============
using Params = typename MatmulKernel::Params;
Params params;

// --- ProblemShape 参数 ---
params.problemShape = {m, n, k, batch};          // (M, N, K, Batch)

// --- BlockMmad 参数 ---
params.mmadParams.aGmAddr = aGM;                 // A 矩阵 GM 地址
params.mmadParams.bGmAddr = bGM;                 // B 矩阵 GM 地址
params.mmadParams.cGmAddr = cGM;                 // C 矩阵 GM 地址
params.mmadParams.biasGmAddr = biasGM;           // Bias GM 地址（可选，nullptr 表示无 bias）

// --- BlockScheduler 参数 ---
// L1 tile 形状：决定每个 tile 的 M/N/K 轴尺寸
params.schedulerParams.mL1 = 256;                // M 轴 L1 tile 尺寸
params.schedulerParams.nL1 = 256;                // N 轴 L1 tile 尺寸
params.schedulerParams.kL1 = 128;                // K 轴 L1 tile 尺寸

// L0 base 形状：决定每次 Mmad 计算的 M/N/K 轴尺寸
params.schedulerParams.baseM = 128;             // M 轴 L0 base 尺寸
params.schedulerParams.baseN = 128;             // N 轴 L0 base 尺寸
params.schedulerParams.baseK = 64;              // K 轴 L0 base 尺寸

// 尾块切分（Batch=1 场景，提升尾块并行度）
params.schedulerParams.mTailCnt = 2;            // M 轴尾块切分数量（建议 1~4）
params.schedulerParams.nTailCnt = 2;            // N 轴尾块切分数量（建议 1~4）

// L1 尾块切分（矩阵不能被 tile 整除时使用）
params.schedulerParams.mBaseTailSplitCnt = 1;   // M 轴 L1 尾块切分数量（建议 1）
params.schedulerParams.nBaseTailSplitCnt = 1;   // N 轴 L1 尾块切分数量（建议 1）
params.schedulerParams.mTailMain = 1;           // M 轴 L1 尾块主尺寸
params.schedulerParams.nTailMain = 1;           // N 轴 L1 尾块主尺寸

// 双缓冲配置（提升数据搬运与计算并行度）
params.schedulerParams.l1BufferNum = 2;         // L1 缓冲数量（1=单缓冲, 2=双缓冲）
params.schedulerParams.l0cDB = 2;               // L0C 双缓冲（1=单缓冲, 2=双缓冲）
params.schedulerParams.ubDB = 2;                // UB 双缓冲（1=单缓冲, 2=双缓冲）

// HF32 模式（可选，用于特定精度场景）
params.schedulerParams.isHf32 = 0;              // HF32 模式标志（0=关闭）

// L2Cache 配置（可选，控制 A/B 矩阵 L2Cache 行为）
params.schedulerParams.l2CacheDisable = 
    Blaze::Gemm::L2CacheMode::L2_CACHE_DEFAULT; // L2Cache 使能（默认）
// 其他选项：
//   A_L2_CACHE_DISABLE     禁用 A 矩阵 L2Cache
//   B_L2_CACHE_DISABLE     禁用 B 矩阵 L2Cache
//   ALL_L2_CACHE_DISABLE   禁用所有 L2Cache

// 非连续场景参数（连续 ND 格式不需要设置）
params.schedulerParams.sliceM = 0;              // M 轴 slice 尺寸（非连续场景）
params.schedulerParams.srcNdStride = 0;         // stride（非连续场景）
params.schedulerParams.innerBatch = 1;          // transpose 内轴 batch

// --- BlockEpilogue 参数 ---
// Empty Epilogue 无需设置参数
params.epilogueParams = {};

// ============== 8. Kernel 调用 ==============
MatmulKernel mm;
mm(params);                                      // 执行矩阵乘计算
```

### 参数详解

#### ProblemShape 参数
| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| m | int64_t | M 轴尺寸 | 1024 |
| n | int64_t | N 轴尺寸 | 1024 |
| k | int64_t | K 轴尺寸 | 512 |
| batch | int64_t | Batch 数量（0 或 1 为单 batch） | 1 |

#### BlockMmad 参数
| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| aGmAddr | GM_ADDR | A 矩阵 GM 地址 | aGM |
| bGmAddr | GM_ADDR | B 矩阵 GM 地址 | bGM |
| cGmAddr | GM_ADDR | C 矩阵 GM 地址 | cGM |
| biasGmAddr | GM_ADDR | Bias GM 地址（nullptr 表示无 bias） | biasGM 或 nullptr |

#### BlockScheduler 参数
详见 [BlockSchedulerMatmulBasic 参数详解](../block/block_scheduler_matmul_basic.md#params-参数结构)

#### 常用配置示例

**小矩阵场景**：
```cpp
params.schedulerParams.mL1 = 128;
params.schedulerParams.nL1 = 128;
params.schedulerParams.kL1 = 64;
params.schedulerParams.baseM = 64;
params.schedulerParams.baseN = 64;
params.schedulerParams.baseK = 32;
params.schedulerParams.l1BufferNum = 1;  // 单缓冲
params.schedulerParams.l0cDB = 1;
params.schedulerParams.ubDB = 1;
```

**大矩阵场景**：
```cpp
params.schedulerParams.mL1 = 256;
params.schedulerParams.nL1 = 256;
params.schedulerParams.kL1 = 128;
params.schedulerParams.baseM = 128;
params.schedulerParams.baseN = 128;
params.schedulerParams.baseK = 64;
params.schedulerParams.l1BufferNum = 2;  // 双缓冲
params.schedulerParams.l0cDB = 2;
params.schedulerParams.ubDB = 2;
params.schedulerParams.l2CacheDisable = Blaze::Gemm::L2CacheMode::ALL_L2_CACHE_DISABLE;
```

**尾块优化场景（Batch=1）**：
```cpp
params.schedulerParams.mTailCnt = 4;  // 尾块切为 4×4 = 16 份
params.schedulerParams.nTailCnt = 4;  // 16 个 Block 并行处理尾块
```

## 数据流

### 存储层次
```
GM (A/B/Bias) → BlockScheduler (tile 切分) → L1 (双缓冲) → L0A/L0B (双缓冲) → L0C (双缓冲) → GM (C)
```

### Kernel 执行流程
```
BlockScheduler 初始化
    ↓
获取 tile 数量和形状 (L1/L0 tile)
    ↓
HF32 模式设置（可选）
    ↓
BlockMmad 初始化 (设置缓冲策略)
    ↓
创建 GM Tensor (ND/NZ layout)
    ↓
配置 L2 Cache (可选禁用)
    ↓
遍历 tile → BlockMmad 执行 (每个 tile 独立计算)
    ↓
清理 (关闭 HF32/MM Layout Transform)
```

## 性能优化建议（Basic 特有）

### L2 Cache 配置
- **大矩阵场景**：建议禁用 L2 Cache 避免缓存污染
- **小矩阵场景**：可保留 L2 Cache 提升数据复用

### Bias 预处理
- **常量 bias**：可预先处理减少运行时开销
- **动态 bias**：保持实时加载

### NZ 格式优化
- **权重矩阵（B）**：优先使用 NZ 格式，提升 L1/L0 搬运效率
- **激活矩阵（A）**：使用 ND 格式即可

### K 轴切分
- `kL1` 和 `baseK` 应根据数据局部性和复用率优化
- 避免 K 轴切分过于细碎导致搬运开销增加

### 适用场景
- **小矩阵**：(m × n × k) 较小时，Basic Kernel 更高效
- **简单计算**：无复杂后处理需求时，Basic Kernel 足够
- **单核场景**：仅需 AIC 计算时，Basic Kernel 更简洁