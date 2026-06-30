# Block Scheduler StreamK
> [代码位置](../../../../include/blaze/gemm/block/block_scheduler_matmul_streamk.h)

## 功能说明
StreamK 调度器，支持 DP+SK 混合策略。将问题规模切分为 DP（Data Parallel）模式和 SK（StreamK）模式的 tile，适用于 StreamK Kernel 的 AIC+AIV 双核协同计算场景。

**继承自**：[Block Scheduler 公共框架](./block_scheduler.md)

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

### Batch 支持
支持 Batch 场景：
- **tile 数量**：`tileNum_ × batch_`
- **Batch 索引**：在 tile 循环中由 Kernel 层处理

### Z 型扫描
使用 Z 型扫描策略：
- **正向扫描**：偶数行（rowIdx % 2 == 0）
- **反向扫描**：奇数行（rowIdx % 2 != 0）

### HF32 模式
支持 HF32 计算模式：
- **isHf32_**：HF32 标志（uint8_t，从 params 传入）
- **GetHf32Flag()**：返回 HF32 标志

### L2 Cache 配置
支持 L2 Cache 配置：
- **l2CacheMode_**：L2 Cache 模式（从 params 传入）
- **GetL2CacheMode()**：返回 L2 Cache 模式

## 特殊静态常量

| 常量 | 说明 |
|------|------|
| WINDOW_LEN | Z 型扫描窗口长度（4） |

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
    uint8_t isHf32{0};                                        // HF32 模式标志
    uint32_t l2CacheMode = L2_CACHE_DEFAULT; // L2 Cache 配置
};
```

说明：
- `usedCoreNum`：参与计算的 AIC 核数量
- `baseM` / `baseN` / `baseK`：L0 base 形状
- `singleCoreK`：SK 模式下单核处理的 K 维大小（用于 K 轴切分）
- `kL1`：L1 K 维度大小
- `baseK`：固定为 32，需根据 baseM, baseN, L0 调整
- `isHf32`：HF32 模式标志（uint8_t 类型）
- `l2CacheMode`：L2 Cache 配置模式

## 特殊成员变量

| 变量 | 说明 |
|------|------|
| usedCoreNum_ | 使用的核数 |
| skKTileNum_ | SK 模式 K 轴 tile 数量 |
| tileNum_ | 总 tile 数量（DP tile + SK tile） |
| totalMNTileNumInDP_ | DP 模式 tile 数量 |
| mTileIdx_ | 当前 M 轴 tile 索引 |
| nTileIdx_ | 当前 N 轴 tile 索引 |
| kTileIdx_ | 当前 K 轴切分索引（SK 模式） |
| curKTileNum_ | 当前 K 轴 tile 数量（DP=1, SK=skKTileNum_） |
| skSingleCoreK_ | SK 模式单核 K 大小 |
| isHf32_ | HF32 模式标志（uint8_t） |
| l2CacheMode_ | L2 Cache 配置模式 |

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
2. 设置 L1/L0 形状：`mL1_ = baseM_`, `nL1_ = baseN_`, `skSingleCoreK_ = singleCoreK`, `kL1_`, `baseK_`
3. 计算 tile 数量：`mTileNum_ = CeilDiv(m_, mL1_)`, `nTileNum_ = CeilDiv(n_, nL1_)`, `skKTileNum_ = CeilDiv(k_, skSingleCoreK_)`
4. 计算 DP+SK tile：
   - `tailMNTileNum = (mTileNum_ × nTileNum_) % usedCoreNum_`（SK 模式 tile 数量）
   - `totalMNTileNumInDP_ = mTileNum_ × nTileNum_ - tailMNTileNum`（DP 模式 tile 数量）
   - `tileNum_ = totalMNTileNumInDP_ + tailMNTileNum × skKTileNum_`（总 tile 数量）
5. 设置 HF32 和 L2 Cache 模式：`isHf32_`, `l2CacheMode_`

### GetTileNum
```
__aicore__ inline int64_t GetTileNum()
```
功能：返回总 tile 数量（`tileNum_ × batch_`）。

### GetHf32Flag
```
__aicore__ inline uint8_t GetHf32Flag()
```
功能：返回 HF32 模式标志（`isHf32_`）。

### GetL2CacheMode
```
__aicore__ inline uint32_t GetL2CacheMode()
```
功能：返回 L2 Cache 配置模式（`l2CacheMode_`）。


### GetMNKTileNum
```
__aicore__ inline Shape<int64_t, int64_t, int64_t, int64_t> GetMNKTileNum()
```
功能：返回 M/N/K tile 数量 `{mTileNum_, nTileNum_, skKTileNum_, 1}`。

### GetBlockNum
```
__aicore__ inline int64_t GetBlockNum(int64_t blockNum)
```
功能：返回实际使用的 Block 数量（不超过 tile 总数）。

### GetCurKSingleCore
```
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

### GetBlockShape
```
__aicore__ inline BlockShape GetBlockShape(int64_t tileIdx)
```
功能：返回当前 tile 的单核形状。
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
```
__aicore__ inline BlockCoord GetBlockCoord(int64_t tileIdx)
```
功能：返回当前 tile 的单核坐标。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| tileIdx | int64_t | tile 索引 |

返回值：`BlockCoord {mTileIdx_, nTileIdx_, kTileIdx_, 0}`
说明：K 轴索引 `kTileIdx_` 仅在 SK 模式有效（DP 模式为 0）。

### CheckIsSkScene
```
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
```
int64_t tileNum = scheduler.GetTileNum();
int64_t blockNum = scheduler.GetBlockNum(GetBlockNum());
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
```
int64_t curK = scheduler.GetCurKSingleCore(tileIdx);
// DP 模式：curK = k_
// SK 模式：curK = skSingleCoreK_
```

### 获取配置
```
uint8_t hf32Flag = scheduler.GetHf32Flag();
uint32_t l2CacheMode = scheduler.GetL2CacheMode();
auto mnkTileNum = scheduler.GetMNKTileNum();
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
- **isHf32 = 1**：启用 HF32 计算模式
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