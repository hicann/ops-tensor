# Block Scheduler Matmul Basic
> [代码位置](../../../../include/blaze/gemm/block/block_scheduler_matmul_basic.h)

## 功能说明
MatmulBasic 内置调度器，支持 tile 切分、block 分配、Z 型扫描、尾块切分、单核 SplitK 切分等。适用于 Basic Kernel 和通用矩阵乘场景。

**继承自**：[Block Scheduler 公共框架](./block_scheduler.md)

## 模板参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| ProblemShape_ | Shape<int64_t, int64_t, int64_t, int64_t> | - | 问题规模 `(m, n, k, batch)` |
| FullLoadMode_ | int64_t | 0 | 全载模式：0=非全载, 1=A全载, 2=B全载 |

## 全载模式

| 值 | 常量 | 说明 | 适用场景 |
|----|------|------|----------|
| 0 | - | 非全载模式（默认） | 通用场景，支持 SplitK |
| 1 | A_FULL_LOAD_MODE | A 矩阵全载 | A 矩阵较小，可完全载入 L1 |
| 2 | B_FULL_LOAD_MODE | B 矩阵全载 | B 矩阵较小，可完全载入 L1 |

## Params 参数结构

### 结构定义
```cpp
struct Params {
    // L1 tile 形状（必填）
    uint32_t mL1 = 0;        // M 轴 L1 tile 尺寸
    uint32_t nL1 = 0;        // N 轴 L1 tile 尺寸
    uint32_t kL1 = 0;        // K 轴 L1 tile 尺寸

    // L0 base 形状（必填）
    uint32_t baseM = 0;      // M 轴 L0 base 尺寸
    uint32_t baseN = 0;      // N 轴 L0 base 尺寸
    uint32_t baseK = 0;      // K 轴 L0 base 尺寸

    // 尾块切分（Batch=1 场景，可选）
    uint32_t mTailCnt = 0;   // M 轴尾块切分数量
    uint32_t nTailCnt = 0;   // N 轴尾块切分数量

    // L1 尾块切分（可选）
    uint32_t mBaseTailSplitCnt = 1;  // M 轴 L1 尾块切分数量
    uint32_t nBaseTailSplitCnt = 1;  // N 轴 L1 尾块切分数量
    uint32_t mTailMain = 1;          // M 轴 L1 尾块主尺寸
    uint32_t nTailMain = 1;          // N 轴 L1 尾块主尺寸

    // 其他配置（可选）
    uint8_t isHf32 = 0;              // HF32 模式标志
    uint8_t l1BufferNum = 0;         // L1 缓冲数量（双缓冲=2）
    uint8_t l0cDB = 1;               // L0C 双缓冲（1=单缓冲, 2=双缓冲）
    uint8_t ubDB = 1;                // UB 双缓冲（1=单缓冲, 2=双缓冲）
    L2CacheMode l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT;  // L2Cache 配置

    // 非连续场景（可选）
    uint32_t sliceM = 0;             // 非连续场景 M 轴 slice 尺寸
    uint32_t srcNdStride = 0;        // 非连续场景 stride
    uint32_t innerBatch = 1;         // 非连续 transpose 场景内轴 batch
};
```

### 参数详解

#### 1. L1 Tile 形状 (mL1, nL1, kL1)

**作用**：将大矩阵切分为多个 L1 tile，每个 Block 处理一个 tile。

**传值建议**：
| 参数 | 建议值 | 说明 |
|------|--------|------|
| mL1 | 128~256 | M 轴 tile 尺寸，建议与 baseM 成倍数关系 |
| nL1 | 128~256 | N 轴 tile 尺寸，建议与 baseN 成倍数关系 |
| kL1 | 64~128 | K 轴 tile 尺寸，建议与 baseK 成倍数关系 |

**常用配置**：
```
// 小矩阵场景
mL1=128, nL1=128, kL1=64

// 大矩阵场景
mL1=256, nL1=256, kL1=128
```

#### 2. L0 Base 形状 (baseM, baseN, baseK)

**作用**：L1 tile 进一步切分为 L0 block，每次 Mmad 计算一个 L0 block。

**传值建议**：
| 参数 | 建议值 | 说明 |
|------|--------|------|
| baseM | 64~128 | M 轴 L0 尺寸，建议 mL1/baseM 为整数 |
| baseN | 64~128 | N 轴 L0 尺寸，建议 nL1/baseN 为整数 |
| baseK | 32~64 | K 轴 L0 尺寸，建议 kL1/baseK 为整数 |

**常用配置**：
```
// 配合小 L1 tile
baseM=64, baseN=64, baseK=32

// 配合大 L1 tile
baseM=128, baseN=128, baseK=64
```

#### 3. 尾块切分 (mTailCnt, nTailCnt)

**作用**：Batch=1 场景下，最后一个 tile 的尾块进一步切分给多个 Block 处理，提高并行度。

**触发条件**：
- `batch_ == 1`
- `tileIdx / blockNum_ == perCoreBlockNum_ - 1`（最后一个 tile）

**传值建议**：
| 参数 | 建议值 | 说明 |
|------|--------|------|
| mTailCnt | 1~4 | M 轴尾块切分数量，建议不超过 4 |
| nTailCnt | 1~4 | N 轴尾块切分数量，建议不超过 4 |

**示例**：
```
// 不切分（默认）
mTailCnt=0, nTailCnt=0  // 实际会被设为 1

// 2x2 切分
mTailCnt=2, nTailCnt=2  // 尾块切为 4 份，4 个 Block 并行处理
```

**示意图**：
```
尾块切分示意（mTailCnt=2, nTailCnt=2）

┌───────────────────────────────────┐
│           尾块 (mL1TailLast)      │
│  ┌──────────┐  ┌──────────┐       │
│  │ Block 0  │  │ Block 1  │       │  ← M 轴切分
│  │ (0,0)    │  │ (0,1)    │       │
│  └──────────┘  └──────────┘       │
│  ┌──────────┐  ┌──────────┐       │
│  │ Block 2  │  │ Block 3  │       │  ← M 轴切分
│  │ (1,0)    │  │ (1,1)    │       │
│  └──────────┘  └──────────┘       │
│        ↑           ↑              │
│     N轴切分     N轴切分            │
└───────────────────────────────────┘

切分计算：
  splitBlkM = CeilDiv(mL1TailLast, mTailCnt)
  splitBlkN = CeilDiv(nL1TailLast, nTailCnt)
  tailCnt = mTailCnt × nTailCnt
```

#### 4. L1 尾块切分 (mBaseTailSplitCnt, nBaseTailSplitCnt, mTailMain, nTailMain)

**作用**：当矩阵 M/N 轴不能被 mL1/nL1 整除时，尾块区域进一步切分。

**传值建议**：
| 参数 | 建议值 | 说明 |
|------|--------|------|
| mBaseTailSplitCnt | 1 | M 轴 L1 尾块切分数量，建议为 1（不切分） |
| nBaseTailSplitCnt | 1 | N 轴 L1 尾块切分数量，建议为 1（不切分） |
| mTailMain | 1 | M 轴尾块主尺寸（当切分数量>1 时使用） |
| nTailMain | 1 | N 轴尾块主尺寸（当切分数量>1 时使用） |

**示意图**：
```
L1 尾块切分示意（mBaseTailSplitCnt=2, nBaseTailSplitCnt=1）

┌────────────────────────────────────────────────────────────┐
│                    矩阵 M×N                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Normal   │ │ Normal   │ │ Normal   │ │ Normal   │       │ ← mL1NormCnt 个正常 tile
│  │ mL1×nL1  │ │ mL1×nL1  │ │ mL1×nL1  │ │ mL1×nL1  │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Normal   │ │ Normal   │ │ Normal   │ │ Normal   │       │
│  │ mL1×nL1  │ │ mL1×nL1  │ │ mL1×nL1  │ │ mL1×nL1  │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ TailMain │ │ TailMain │ │ TailMain │ │ TailMain │       │ ← mBaseTailSplitCnt-1 个主尾块
│  │mTailMain │ │mTailMain │ │mTailMain │ │mTailMain │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ TailLast │ │ TailLast │ │ TailLast │ │ TailLast │       │ ← 最后一个尾块
│  │mL1TailLast││mL1TailLast││mL1TailLast││mL1TailLast│      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└────────────────────────────────────────────────────────────┘

计算公式：
  mL1NormCnt = mTileNum_ - mBaseTailSplitCnt
  mL1TailMain = (mBaseTailSplitCnt > 1) ? mTailMain : tailL1M
  mL1TailLast = tailL1M - (mBaseTailSplitCnt - 1) × mL1TailMain
```

#### 5. 双缓冲配置 (l1BufferNum, l0cDB, ubDB)

**作用**：启用双缓冲可以提高数据搬运和计算的并行度。

**传值建议**：
| 参数 | 建议值 | 说明 |
|------|--------|------|
| l1BufferNum | 2 | L1 双缓冲，建议大 tile 场景启用 |
| l0cDB | 2 | L0C 双缓冲，建议大 tile 场景启用 |
| ubDB | 2 | UB 双缓冲，建议大 tile 场景启用 |

**示例**：
```
// 单缓冲（小矩阵场景）
l1BufferNum=1, l0cDB=1, ubDB=1

// 双缓冲（大矩阵场景）
l1BufferNum=2, l0cDB=2, ubDB=2
```

#### 6. L2Cache 配置 (l2CacheDisable)

**作用**：控制 A/B 矩阵的 L2Cache 行为，某些场景禁用 L2Cache 可提高性能。

**可选值**：
| 常量 | 说明 | 适用场景 |
|------|------|----------|
| L2_CACHE_DEFAULT | L2Cache 使能（默认） | 通用场景 |
| A_L2_CACHE_DISABLE | 禁用 A 矩阵 L2Cache | A 矩阵复用少 |
| B_L2_CACHE_DISABLE | 禁用 B 矩阵 L2Cache | B 矩阵复用少 |
| ALL_L2_CACHE_DISABLE | 禁用所有 L2Cache | 小矩阵场景 |

**示例**：
```
// 默认配置
l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT

// 禁用 A 矩阵 L2Cache
l2CacheDisable = L2CacheMode::A_L2_CACHE_DISABLE

// 禁用所有 L2Cache
l2CacheDisable = L2CacheMode::ALL_L2_CACHE_DISABLE
```

#### 7. 非连续场景参数 (sliceM, srcNdStride, innerBatch)

**作用**：处理非连续 ND 格式的矩阵数据。

**传值建议**：
| 参数 | 说明 | 使用场景 |
|------|------|----------|
| sliceM | M 轴 slice 尺寸 | 非 ND 连续格式 |
| srcNdStride | M 轴 stride | 非 ND 连续格式 |
| innerBatch | transpose 内轴 batch | transpose 场景 |

**判断逻辑**：
```
isSlice_ = (srcNdStride != 1 && sliceM != 0)
```

**示例**：
```
// 连续 ND 格式（默认）
sliceM=0, srcNdStride=0  // isSlice_ = false

// 非 ND 连续格式
sliceM=64, srcNdStride=128  // isSlice_ = true
```

## SplitK 切分

### 触发条件
```
isFp32_ && !isHf32_ && isNdFormat_ && k_ > fp32SplitKThreshold && FullLoadMode_ == 0
```

### 阈值配置
| 常量 | 值 | 说明 |
|------|-----|------|
| FP32_K_SWITCH_THRESHOLD | 268435456 | 大 K 阈值切换点 |
| FP32_SPLIT_K_THRESHOLD1 | 1024 | 小 K 场景切分阈值 |
| FP32_SPLIT_K_THRESHOLD2 | 8192 | 大 K 场景切分阈值 |

### 切分逻辑
```
if (k_ > FP32_K_SWITCH_THRESHOLD) {
    splitSingleK_ = FP32_SPLIT_K_THRESHOLD2;  // 8192
} else {
    splitSingleK_ = FP32_SPLIT_K_THRESHOLD1;  // 1024
}

splitSingleKRound_ = CeilDiv(k_, splitSingleK_);
splitSingleKTail_ = k_ % splitSingleK_ + splitSingleK_;
```

### 示意图
```
SplitK 切分示意（k=20480, splitSingleK_=8192）

┌─────────────────────────────────────────────────────────────┐
│                      K 轴 (k=20480)                          │
│  ┌─────────────────┐                                         │
│  │  Round 0        │  kOffset=0                              │
│  │  0~8191         │  blkK_=8192                             │
│  └─────────────────┘                                         │
│  ┌─────────────────┐                                         │
│  │  Round 1        │  kOffset=8192                           │
│  │  8192~16383     │  blkK_=8192                             │
│  └─────────────────┘                                         │
│  ┌─────────────────┐                                         │
│  │  Round 2 (Tail) │  kOffset=16384                          │
│  │  16384~20479    │  blkK_=4096                             │
│  └─────────────────┘                                         │
└─────────────────────────────────────────────────────────────┘

splitSingleKRound_ = 3
splitSingleKTail_ = 4096
```

## Z 型扫描

### 扫描逻辑
```
// 奇数行反向扫描
if (rowIdx % 2 != 0) {
    nTileIdx_ = nTileNum_ - 1 - nTileIdx_;
}
```

### 示意图
```
Z 型扫描示意（mTileNum_=4, nTileNum_=4）

     N轴 →
   ┌──┬──┬──┬──┐
   │0 │1 │2 │3 │  ← Row 0（正向）
M  ├──┼──┼──┼──┤
轴 │7 │6 │5 │4 │  ← Row 1（反向）
↓  ├──┼──┼──┼──┤
   │8 │9 │10│11│  ← Row 2（正向）
   ├──┼──┼──┼──┤
   │15│14│13│12│  ← Row 3（反向）
   └──┴──┴──┴──┘

扫描顺序：0→1→2→3→7→6→5→4→8→9→10→11→15→14→13→12
```

### 窗口扫描
```
mainWindow_ = 4  （窗口长度）
mainRow_ = mTileNum_ / mainWindow_ - 1
tailWindow_ = mTileNum_ - mainRow_ * mainWindow_

示例（mTileNum_=10）：
  mainWindow_ = 4
  mainRow_ = 2
  tailWindow_ = 2

扫描区域：
  Row 0-1：mainWindow_=4 正常扫描
  Row 2：tailWindow_=2 扫描
```

## 构造函数

```cpp
__aicore__ inline BlockSchedulerMatmulBasic(
    const ProblemShape& shape,  // 问题规模 (m, n, k, batch)
    int64_t blockIdx,           // 当前 Block 索引
    int64_t blockNum,           // 总 Block 数量
    const Params& params,       // 参数
    bool isFp32 = false,        // 是否为 FP32
    bool isNdFormat = true)     // 是否为 ND 格式
```

### 参数说明
| 参数 | 类型 | 说明 |
|------|------|------|
| shape | ProblemShape | 问题规模 `(m, n, k, batch)` |
| blockIdx | int64_t | 当前 Block 索引（`GetBlockIdx()`） |
| blockNum | int64_t | 总 Block 数量 |
| params | Params | 调度参数 |
| isFp32 | bool | 是否为 FP32（影响 SplitK 切分） |
| isNdFormat | bool | 是否为 ND 格式（影响尾块切分） |

### 执行流程
```
1. 设置问题规模：k_, batch_, innerBatch_
2. 设置 L1/L0 形状：mL1_, nL1_, kL1_, baseM_, baseN_, baseK_
3. 计算 tile 数量：mTileNum_, nTileNum_, kTileNum_, tileNum_
4. 计算 L1 尾块参数：mL1NormCnt_, mL1TailMain_, mL1TailLast_, nL1NormCnt_, nL1TailMain_, nL1TailLast_
5. 判断 SplitK 切分：isSplitSingleK_, splitSingleK_, splitSingleKRound_, splitSingleKTail_
6. 判断非连续场景：isSlice_
7. 计算尾块切分：mTailCnt_, nTailCnt_, tailCnt_（batch=1 场景）
8. 计算扫描窗口：mainWindow_, mainRow_, tailWindow_
```

## 成员方法

### DisableSplitSingleK
```cpp
__aicore__ inline void DisableSplitSingleK()
```
功能：禁用 SplitK 切分。

### GetTileNum
```cpp
__aicore__ inline int64_t GetTileNum()
```
功能：返回总 tile 数量（`tileNum_ * batch_`）。

### Gethf32Flag
```cpp
__aicore__ inline bool Gethf32Flag()
```
功能：返回 HF32 模式标志（`isHf32_ > 0`）。

### GetL1BuferNum_
```cpp
__aicore__ inline uint64_t GetL1BuferNum_()
```
功能：返回 L1 缓冲数量。

### GetTileL1Shape
```cpp
__aicore__ inline Shape<int64_t, int64_t, int64_t, int64_t> GetTileL1Shape()
```
功能：返回 L1 tile 形状 `{mL1_, nL1_, kL1_, 1}`。

### GetTileL0Shape
```cpp
__aicore__ inline Shape<int64_t, int64_t, int64_t, int64_t> GetTileL0Shape()
```
功能：返回 L0 tile 形状 `{baseM_, baseN_, baseK_, 1}`。

### GetBlockNum
```cpp
__aicore__ inline int64_t GetBlockNum(ProblemShape shape, int64_t blockNum)
```
功能：返回实际使用的 Block 数量（不超过 tile 总数）。
返回值：`min(tileNum_ * batch_, blockNum)`

### GetBlockShape
```cpp
template <bool TransB_ = false, class B_T>
__aicore__ inline BlockL1L0Shape GetBlockShape(
    int64_t tileIdx, int64_t mOffset = 0, int64_t nOffset = 0, int64_t kOffset = 0)
```
功能：返回当前 tile 的 Block 形状。
返回值：`BlockL1L0Shape {mL1, nL1, k, batch, mL0, nL0}`

### GetBlockCoord
```cpp
__aicore__ inline BlockCoord GetBlockCoord(int tileIdx)
```
功能：返回当前 tile 的 Block 坐标。
返回值：`BlockCoord {mOffset, nOffset, mOffsetNonContiguous, batchIdx}`

### GetSplitKBlockCoord
```cpp
__aicore__ inline BlockCoord GetSplitKBlockCoord(int tileIdx)
```
功能：返回 SplitK 场景的 Block 坐标。
返回值：`BlockCoord {mOffset, nOffset, kOffset, batchIdx}`

### GetSplitOffset
```cpp
__aicore__ inline Shape<int64_t, int64_t> GetSplitOffset()
```
功能：返回尾块切分偏移 `{mSplitOffset_, nSplitOffset_}`。

### GetNonContinuousParams
```cpp
__aicore__ inline Shape<int64_t, int64_t, int64_t> GetNonContinuousParams()
```
功能：返回非连续场景参数 `{sliceM_, srcNdStride_, innerBatch_}`。

### GetTailParams
```cpp
__aicore__ inline Shape<int64_t, int64_t, int64_t, int64_t> GetTailParams()
```
功能：返回尾块参数 `{mL1NormCnt_, mL1TailMain_, nL1NormCnt_, nL1TailMain_}`。

### GetL0cDB / GetUbDB
```cpp
__aicore__ inline bool GetL0cDB()  // 返回 L0C 双缓冲标志（l0cDB_ > 1）
__aicore__ inline bool GetUbDB()   // 返回 UB 双缓冲标志（ubDB_ > 1）
```

### GetAL2CacheDisable / GetBL2CacheDisable
```cpp
__aicore__ inline bool GetAL2CacheDisable()  // 返回 A 矩阵 L2Cache 禁用标志
__aicore__ inline bool GetBL2CacheDisable()  // 返回 B 矩阵 L2Cache 禁用标志
```

## 调用示例

### 组件组装
```cpp
using ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;
constexpr int64_t FULL_LOAD_MODE = 0;  // 非全载模式
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, FULL_LOAD_MODE>;
```

### 参数准备
```cpp
BlockScheduler::Params params = {
    // L1 tile 形状
    .mL1 = 256,
    .nL1 = 256,
    .kL1 = 128,

    // L0 base 形状
    .baseM = 128,
    .baseN = 128,
    .baseK = 64,

    // 尾块切分（Batch=1 场景）
    .mTailCnt = 2,
    .nTailCnt = 2,

    // L1 尾块切分
    .mBaseTailSplitCnt = 1,
    .nBaseTailSplitCnt = 1,
    .mTailMain = 1,
    .nTailMain = 1,

    // 双缓冲
    .l1BufferNum = 2,
    .l0cDB = 2,
    .ubDB = 2,

    // L2Cache
    .l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT,

    // 非连续场景（连续 ND 格式不需要设置）
    .sliceM = 0,
    .srcNdStride = 0,
    .innerBatch = 1
};
```

### 组件初始化
```cpp
ProblemShape shape{m, n, k, batch};
int64_t blockIdx = GetBlockIdx();
int64_t blockNum = GetBlockNum();
bool isFp32 = false;      // 非 FP32 场景
bool isNdFormat = true;   // ND 格式

BlockScheduler scheduler(shape, blockIdx, blockNum, params, isFp32, isNdFormat);
```

### 获取 tile 数量
```cpp
int64_t tileNum = scheduler.GetTileNum();
for (int64_t tileIdx = blockIdx; tileIdx < tileNum; tileIdx += blockNum) {
    // 处理 tile
}
```

### 获取 Block 形状
```cpp
using B_T = half;
bool TransB = false;
auto blockShape = scheduler.GetBlockShape<TransB, B_T>(tileIdx, mOffset, nOffset, kOffset);
int64_t mL1 = Get<0>(blockShape);
int64_t nL1 = Get<1>(blockShape);
int64_t kL1 = Get<2>(blockShape);
int64_t mL0 = Get<4>(blockShape);
int64_t nL0 = Get<5>(blockShape);
```

### 获取 Block 坐标
```cpp
// 正常场景
auto blockCoord = scheduler.GetBlockCoord(tileIdx);
int64_t mOffset = Get<0>(blockCoord);
int64_t nOffset = Get<1>(blockCoord);
int64_t batchIdx = Get<3>(blockCoord);

// SplitK 场景
auto splitKCoord = scheduler.GetSplitKBlockCoord(tileIdx);
int64_t kOffset = Get<2>(splitKCoord);
```

## 适用场景

| 场景 | 配置建议 |
|------|----------|
| Basic Kernel | FullLoadMode=0，默认配置 |
| FP32 大 K | isFp32=true，启用 SplitK |
| 尾块优化 | batch=1，设置 mTailCnt/nTailCnt |
| 小矩阵 | 禁用 L2Cache，减小 tile 尺寸 |
| 大矩阵 | 启用双缓冲，增大 tile 尺寸 |