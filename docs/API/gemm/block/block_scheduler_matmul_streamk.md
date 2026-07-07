# Block Scheduler StreamK
> [代码位置](../../../../include/blaze/gemm/block/block_scheduler_matmul_streamk.h)

## 功能说明
StreamK 调度器，支持 DP+SK 混合策略。将问题规模切分为 DP（Data Parallel）模式和 SK（StreamK）模式的 tile，适用于 StreamK Kernel 的 AIC+AIV 双核协同计算场景。

**继承自**：[Block Scheduler 公共框架](./block_scheduler.md)

## 模板参数

### 模板参数列表
```cpp
template <typename ProblemShape_>
class BlockSchedulerMatmulStreamK;
```

### 参数详解

| 参数 | 类型约束 | 默认值 | 说明 |
|------|----------|--------|------|
| ProblemShape_ | `Shape<int64_t, int64_t, int64_t, int64_t>` | - | 问题规模 `(m, n, k, batch)` |

### ProblemShape 参数详解

**类型定义**：
```cpp
using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
```

**参数组成**：
| 索引 | 参数 | 类型 | 说明 | 示例值 |
|------|------|------|------|--------|
| 0 | m | int64_t | M 轴尺寸（矩阵 A 的行数） | 1024 |
| 1 | n | int64_t | N 轴尺寸（矩阵 B 的列数） | 1024 |
| 2 | k | int64_t | K 轴尺寸（矩阵 A 的列数 = 矩阵 B 的行数） | 512 |
| 3 | batch | int64_t | Batch 数量（0 或 1 表示单 batch） | 1 |

**使用示例**：
```cpp
// 定义 ProblemShape 类型
using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

// 创建 ProblemShape 实例
ProblemShape problemShape{1024, 1024, 512, 1};  // (M, N, K, Batch)

// 获取各维度值
int64_t m = Get<0>(problemShape);
int64_t n = Get<1>(problemShape);
int64_t k = Get<2>(problemShape);
int64_t batch = Get<3>(problemShape);
```

**参数约束**：
- `m > 0`：M 轴尺寸必须大于 0
- `n > 0`：N 轴尺寸必须大于 0
- `k > 0`：K 轴尺寸必须大于 0
- `batch >= 0`：Batch 数量，0 或 1 表示单 batch
- `m`, `n`, `k` 建议与 `baseM`, `baseN`, `baseK` 成倍数关系，减少尾块开销

**特殊说明**：
- **单 Batch 场景**：`batch = 0` 或 `batch = 1`，不进行 Batch 循环
- **多 Batch 场景**：`batch > 1`，Kernel 层会进行 Batch 循环处理
- StreamK 调度器会根据 ProblemShape 计算 tile 切分和 DP+SK 混合策略

## 特殊类型别名

| 类型 | 说明 |
|------|------|
| BlockShape | Block 形状：`Shape<int64_t, int64_t, int64_t, int64_t>` |
| BlockCoord | Block 坐标：`Coord<int64_t, int64_t, int64_t, int64_t>` (mTileIdx, nTileIdx, kTileIdx, 0) |
| ProblemShape | 问题规模类型（模板参数） |

## Params 参数结构

### 结构定义
```cpp
struct Params {
    int64_t usedCoreNum{0};                                   // 使用的核数
    int64_t baseM{0};                                         // L0 M 维度 base 大小
    int64_t baseN{0};                                         // L0 N 维度 base 大小
    int64_t baseK{0};                                         // L0 K 维度 base 大小
    int64_t singleCoreK{0};                                   // SK 模式下单核处理的 K 大小
    int64_t kL1{0};                                           // L1 K 维度大小
    uint8_t isHf32{0};                                        // HF32模式标志 (0=关闭, 1=开启)
    uint32_t l2CacheMode = L2_CACHE_DEFAULT; // L2 Cache 配置
};
```

说明：
- `usedCoreNum`：参与计算的 AIC 核数量
- `baseM` / `baseN`：L0 M/N 维度 base 形状，同时作为 L1 M/N 维度
- `baseK`：L0 K 维度 base 形状
- `singleCoreK`：SK 模式下单核处理的 K 维大小（用于 K 轴切分）
- `kL1`：L1 K 维度大小
- `isHf32`：HF32 模式标志（uint8_t 类型，0=关闭，1=开启）
- `l2CacheMode`：L2 Cache 配置模式（uint32_t 类型）

## 特殊成员变量

| 变量 | 说明 |
|------|------|
| usedCoreNum_ | 使用的核数 |
| mTileNum_ | M 轴 tile 数量 |
| nTileNum_ | N 轴 tile 数量 |
| skKTileNum_ | SK 模式 K 轴 tile 数量 |
| tileNum_ | 总 tile 数量（DP tile + SK tile） |
| totalMNTileNumInDP_ | DP 模式 tile 数量 |
| batch_ | Batch 数量 |
| m_ | M 维度大小 |
| n_ | N 维度大小 |
| k_ | K 维度大小 |
| mTileIdx_ | 当前 M 轴 tile 索引 |
| nTileIdx_ | 当前 N 轴 tile 索引 |
| kTileIdx_ | 当前 K 轴切分索引（SK 模式） |
| curKTileNum_ | 当前 K 轴 tile 数量（DP=1, SK=skKTileNum_） |
| mL1_ | L1 M 维度大小（等于 baseM） |
| nL1_ | L1 N 维度大小（等于 baseN） |
| skSingleCoreK_ | SK 模式单核 K 大小 |

## 特殊成员方法

### 构造函数
```
__aicore__ inline BlockSchedulerMatmulStreamK(const ProblemShape& shape, const Params& params)
```
功能：初始化 BlockSchedulerMatmulStreamK，计算 DP+SK 混合 tile 切分。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| shape | ProblemShape | 问题规模 `(m, n, k, batch)` |
| params | Params | 调度参数（usedCoreNum, baseM, baseN, baseK, singleCoreK, kL1, isHf32, l2CacheMode） |

执行流程：
1. 设置问题规模：`m_`, `n_`, `k_`, `batch_`
2. 设置 L1/L0 形状：`mL1_ = baseM`, `nL1_ = baseN`, `skSingleCoreK_ = singleCoreK`
3. 计算 tile 数量：`mTileNum_ = CeilDiv(m_, mL1_)`, `nTileNum_ = CeilDiv(n_, nL1_)`, `skKTileNum_ = CeilDiv(k_, skSingleCoreK_)`
4. 计算 DP+SK tile：
   - `tailMNTileNum = (mTileNum_ × nTileNum_) % usedCoreNum_`（SK 模式 MN tile 数量）
   - `totalMNTileNumInDP_ = mTileNum_ × nTileNum_ - tailMNTileNum`（DP 模式 tile 数量）
   - `tileNum_ = totalMNTileNumInDP_ + tailMNTileNum × skKTileNum_`（总 tile 数量）

### GetTileNum
```cpp
__aicore__ inline int64_t GetTileNum()
```
功能：返回总 tile 数量（`tileNum_ × batch_`）。

### GetMNKTileNum
```cpp
__aicore__ inline Shape<int64_t, int64_t, int64_t, int64_t> GetMNKTileNum()
```
功能：返回 M/N/K tile 数量 `{mTileNum_, nTileNum_, skKTileNum_, 1}`。

### GetBlockNum
```cpp
__aicore__ inline int64_t GetBlockNum(ProblemShape shape)
```
功能：返回实际使用的 Block 数量（不超过 tile 总数）。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| shape | ProblemShape | 问题规模 |

返回值：`min(tileNum_ × batch_, AscendC::GetBlockNum())`

### GetBlockShape
```cpp
__aicore__ inline BlockShape GetBlockShape(int64_t tileIdx)
```
功能：返回当前 tile 的 Block 形状。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| tileIdx | int64_t | tile 索引 |

返回值：`BlockShape {blkM, blkN, blkK, 0}`
特殊逻辑：
- **尾块判断**：`mTileIdx_ == (mTileNum_ - 1)` 或 `nTileIdx_ == (nTileNum_ - 1)`
- **K 切分尾块**：`kTileIdx_ == (curKTileNum_ - 1)`
- **DP 模式**：`blkK = k_`（完整 K）
- **SK 模式**：`blkK = skSingleCoreK_` 或 `tailSingleCoreK`

### GetBlockCoord
```cpp
__aicore__ inline BlockCoord GetBlockCoord(int64_t tileIdx)
```
功能：返回当前 tile 的 Block 坐标。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| tileIdx | int64_t | tile 索引 |

返回值：`BlockCoord {mTileIdx_, nTileIdx_, kTileIdx_, 0}`
说明：K 轴索引 `kTileIdx_` 仅在 SK 模式有效（DP 模式为 0）。

### GetCurKSingleCore
```cpp
__aicore__ inline int64_t GetCurKSingleCore(int64_t tileIdx)
```
功能：返回当前 tile 的单核 K 大小。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| tileIdx | int64_t | tile 索引 |

返回值：
- **DP 模式**：`k_`（完整 K）
- **SK 模式**：`skSingleCoreK_`（切分 K）

### CheckIsSkScene
```cpp
__aicore__ inline bool CheckIsSkScene(int64_t tileIdx)
```
功能：判断当前 tile 是否为 SK 模式。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| tileIdx | int64_t | tile 索引 |

返回值：
- **true**：SK 模式（K 轴切分）
- **false**：DP 模式（完整 tile）

判断逻辑：
```
CeilDiv((tileIdx + 1), usedCoreNum_) == CeilDiv(tileNum_, usedCoreNum_)
```

### UpdateMNTileIdx
```
__aicore__ inline void UpdateMNTileIdx(int64_t tileIdx)
```
功能：更新当前 tile 的 M/N/K tile 索引。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| tileIdx | int64_t | tile 索引 |

执行流程：
1. **判断 DP/SK 模式**：`CheckIsSkScene(tileIdx)`
2. **设置 K 轴 tile 数量**：`curKTileNum_ = (SK ? skKTileNum_ : 1)`
3. **计算 mnIdxInCurLoop**：
   - **SK 模式**：`kTileIdx_ = (tileIdx % usedCoreNum_) % curKTileNum_`, `mnIdxInCurLoop = (tileIdx % usedCoreNum_) / curKTileNum_ + totalMNTileNumInDP_`
   - **DP 模式**：`kTileIdx_ = 0`, `mnIdxInCurLoop = tileIdx / curKTileNum_`
4. **Z 型扫描**：计算 `mTileIdx_`, `nTileIdx_`
5. **反向扫描**：奇数行反向（`nTileIdx_ = nTileNum_ - 1 - nTileIdx_`）

## 调用示例

### 组件组装
```
using ProblemShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
```

### 参数准备
```
BlockScheduler::Params params = {
    usedCoreNum,     // 使用的核数（如 8）
    baseM,           // L0 M 维度 base（如 256）
    baseN,           // L0 N 维度 base（如 256）
    baseK,           // L0 K 维度 base（如 32）
    singleCoreK,     // SK 模式单核 K 大小（如 k_ / 4）
    kL1,             // L1 K 维度（如 baseK）
    isHf32,          // HF32 模式（uint8_t，0 或 1）
    l2CacheMode      // L2 Cache 配置
};
```

### 组件初始化
```
ProblemShape shape{m, n, k, batch};
BlockScheduler scheduler(shape, params);
```

### 获取 tile 数量
```cpp
int64_t tileNum = scheduler.GetTileNum();
int64_t blockNum = scheduler.GetBlockNum(shape);
for (int64_t tileIdx = GetBlockIdx(); tileIdx < tileNum; tileIdx += blockNum) {
    // 处理 tile
}
```

### 判断 DP/SK 模式
```
bool isSkScene = scheduler.CheckIsSkScene(tileIdx);
if (isSkScene) {
    // SK 模式：输出到 workspace
} else {
    // DP 模式：输出到 GM
}
```

### 获取单核形状
```
auto singleCoreShape = scheduler.GetBlockShape(tileIdx);
int64_t blkM = Get<0>(singleCoreShape);
int64_t blkN = Get<1>(singleCoreShape);
int64_t blkK = Get<2>(singleCoreShape);
```

### 获取单核坐标
```
auto singleCoreCoord = scheduler.GetBlockCoord(tileIdx);
int64_t mTileIdx = Get<0>(singleCoreCoord);
int64_t nTileIdx = Get<1>(singleCoreCoord);
int64_t kTileIdx = Get<2>(singleCoreCoord);  // SK 模式有效
```

### 获取当前 K 大小
```cpp
int64_t curK = scheduler.GetCurKSingleCore(tileIdx);
// DP 模式：curK = k_
// SK 模式：curK = skSingleCoreK_
```

### 获取 MNK tile 数量
```cpp
auto mnkTileNum = scheduler.GetMNKTileNum();
int64_t mTileNum = Get<0>(mnkTileNum);
int64_t nTileNum = Get<1>(mnkTileNum);
int64_t skKTileNum = Get<2>(mnkTileNum);
```

## 数据流

### DP+SK 混合策略流程
```
问题规模 (m, n, k, batch)
    ↓
tile 切分 (mTileNum, nTileNum, skKTileNum)
    ↓
DP tile 数量 = mTileNum × nTileNum - tailMNTileNum
    ↓
SK tile 数量 = tailMNTileNum × skKTileNum
    ↓
总 tile 数量 = DP tile + SK tile
    ↓
tile 索引判断 (CheckIsSkScene)
    ↓
DP 模式：完整 tile，输出到 GM
SK 模式：K 轴切分，输出到 workspace
```

### DP 模式流程
```
tileIdx 判断：CeilDiv((tileIdx + 1), usedCoreNum) < CeilDiv(tileNum, usedCoreNum)
    ↓
curKTileNum = 1（不切分 K）
    ↓
kTileIdx = 0
    ↓
GetBlockShape：blkK = k_（完整 K）
    ↓
BlockMmad：输出到 GM
```

### SK 模式流程
```
tileIdx 判断：CeilDiv((tileIdx + 1), usedCoreNum) == CeilDiv(tileNum, usedCoreNum)
    ↓
curKTileNum = skKTileNum（K 轴切分）
    ↓
kTileIdx = (tileIdx % usedCoreNum) % curKTileNum
    ↓
GetBlockShape：blkK = skSingleCoreK_ 或 tailSingleCoreK
    ↓
BlockMmad：输出到 workspace
    ↓
BlockEpilogue（AIV）：workspace 汇聚 → GM
```

### Z 型扫描流程
```
mnIdxInCurLoop（DP/SK 模式的 MN 索引）
    ↓
rowIdx = mnIdxInCurLoop / nTileNum / mainWindow
    ↓
rowIdx < mainRow：mTileIdx = rowIdx × mainWindow + mnIdxInCurLoop % mainWindow
    ↓
rowIdx == mainRow：尾窗口计算
    ↓
rowIdx % 2 != 0：反向扫描（nTileIdx = nTileNum - 1 - nTileIdx）
```

## 特殊约束

### DP+SK 混合策略
StreamK 调度器将 tile 分为两种模式：
- **DP（Data Parallel）模式**：完整 tile，每个核独立处理完整的 (m, n) tile，结果直接输出到 GM
- **SK（StreamK）模式**：K 轴切分 tile，多个核协同处理一个 (m, n) tile 的不同 K 切分，结果输出到 workspace

### DP 模式 tile
- **tile 数量**：`totalMNTileNumInDP_ = mTileNum_ × nTileNum_ - tailMNTileNum`
- **每个核处理**：完整的 (m, n) tile，K 轴不切分
- **输出目标**：GM（通过 BlockMmad 输出）

### SK 模式 tile
- **tile 数量**：`tailMNTileNum × skKTileNum`
- **每个核处理**：一个 (m, n) tile 的部分 K 切分
- **K 切分数量**：`skKTileNum_ = CeilDiv(k_, skSingleCoreK_)`
- **输出目标**：workspace（通过 BlockMmad 输出）

### tile 索引分配
```
tileIdx 判断：
DP 模式：CeilDiv((tileIdx + 1), usedCoreNum_) < CeilDiv(tileNum_, usedCoreNum_)
SK 模式：CeilDiv((tileIdx + 1), usedCoreNum_) == CeilDiv(tileNum_, usedCoreNum_)
```

## 性能优化建议

### usedCoreNum 配置
- **建议值**：根据实际 AIC 核数量设置（如 8、16）
- **SK 模式比例**：`tailMNTileNum = (mTileNum × nTileNum) % usedCoreNum`
- **优化**：调整 usedCoreNum 以减少 SK 模式 tile 数量

### singleCoreK 配置
- **建议值**：约为 `k_ / 4`，平衡 K 轴切分数量
- **SK tile 数量**：`skKTileNum = CeilDiv(k_, singleCoreK)`
- **优化**：调整 singleCoreK 以减少 K 轴切分数量

### DP+SK 比例配置
- **DP 模式**：`totalMNTileNumInDP_ = mTileNum × nTileNum - tailMNTileNum`
- **SK 模式**：`tailMNTileNum × skKTileNum`
- **优化**：调整 mTileNum, nTileNum, usedCoreNum 以增加 DP 模式比例

### tile 形状配置
- **mL1 = baseM**：L1 M 维度等于 L0 base（如 256）
- **nL1 = baseN**：L1 N 维度等于 L0 base（如 256）
- **kL1**：L1 K 维度（如 baseK 或更大）
- **优化**：使用性能最优的 tile 形状

### HF32 模式配置
- **isHf32 = 1**：启用 HF32 计算模式 0=关闭, 1=开启
- **适用场景**：需要高精度计算的 FP32 场景

### L2 Cache 配置
- **L2_CACHE_DEFAULT**：L2 Cache 使能（默认）
- **A_L2_CACHE_DISABLE**：禁用 A 矩阵 L2 Cache
- **B_L2_CACHE_DISABLE**：禁用 B 矩阵 L2 Cache
- **ALL_L2_CACHE_DISABLE**：禁用所有 L2 Cache
- **适用场景**：大矩阵场景建议禁用 L2 Cache 避免缓存污染

### 适用场景
- **StreamK Kernel**：AIC + AIV 双核协同
- **大矩阵场景**：(m × n × k) 较大，需要多核并行
- **K 轴切分场景**：K 维度远大于 M/N
- **高并行度场景**：需要充分利用 AIC 和 AIV 双核