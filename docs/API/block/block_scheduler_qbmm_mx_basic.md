# Block Scheduler Quant Batch Matmul
> [代码位置](../../../include/blaze/block/block_scheduler_qbmm.h)

## 功能说明
量化 Batch Matmul 调度器，支持多 Batch 维度切分、尾块切分、负载均衡、Z 型扫描。适用于 Quant Batch Matmul MX Kernel 场景，支持 MxFP4/MxFP8 量化数据类型。

**继承自**：[Block Scheduler 公共框架](./block_scheduler.md)

### 模板参数
```
template <
    class ProblemShape_,     // 问题规模类型
    uint64_t FullLoadMode_,  // 全载模式（0=非全载，1=A全载）
    class LayoutA_,          // A 矩阵布局类型
    class LayoutB_,          // B 矩阵布局类型
    class AType_>            // A 矩阵数据类型（用于判断 C0_SIZE）
class BlockSchedulerQuantBatchMatmulV3;
```

### 全载模式
支持两种全载模式：
- **FullLoadMode_ = 0**：非全载模式（默认）
- **FullLoadMode_ = A_FULL_LOAD_MODE（1）**：A 矩阵全载模式

### C0_SIZE 计算
根据数据类型自动计算 C0 对齐大小：
- **FP4（fp4x2_e2m1_t, fp4x2_e1m2_t）**：C0_SIZE = 64
- **FP8（fp8_e5m2_t, fp8_e4m3fn_t）**：C0_SIZE = 32

### 转置判断
根据布局类型判断转置：
- **transA**：`IsTrans<LayoutA_>::value`
- **transB**：`IsTrans<LayoutB_>::value`

### 尾块切分
支持尾块 tile 切分：
- **mTailTile**：M 轴尾块切分数量
- **nTailTile**：N 轴尾块切分数量
- **totalTailTile**：总尾块切分数量（mTailTile × nTailTile）

### Z 型扫描
使用 Z 型扫描策略：
- **WINDOW_LEN = 4**：扫描窗口大小
- **正向扫描**：偶数行（rowIdx % 2 == 0）
- **反向扫描**：奇数行（rowIdx & 1）

### 负载均衡
支持负载均衡配置：
- **mBaseNormCnt**：M 轴正常 tile 数量
- **mBaseTailMain**：M 轴尾块主尺寸
- **nBaseNormCnt**：N 轴正常 tile 数量
- **nBaseTailMain**：N 轴尾块主尺寸

## 特殊静态常量

| 常量 | 说明 |
|------|------|
| C0_SIZE | C0 对齐大小（FP4: 64，FP8: 32） |
| WINDOW_LEN | Z 型扫描窗口长度（4） |
| transA | A 矩阵是否转置 |
| transB | B 矩阵是否转置 |

## 特殊类型别名

| 类型 | 说明 |
|------|------|
| BlockShape | Block 形状：`Shape<int64_t, int64_t, int64_t, int64_t>` |
| BlockCoord | Block 坐标：`Coord<int64_t, int64_t, int64_t, int64_t>` |
| ProblemShape | 问题规模类型（模板参数） |
| AType | A 矩阵数据类型（模板参数） |

## 特殊数据结构

### Params
```
struct Params {
    int64_t baseM;              // L0 M 维度 base 大小
    int64_t baseN;              // L0 N 维度 base 大小
    int64_t mTailTile;          // M 轴尾块切分数量
    int64_t nTailTile;          // N 轴尾块切分数量
    int64_t mBaseTailSplitCnt;  // M 轴尾块 L1 切分数量
    int64_t nBaseTailSplitCnt;  // N 轴尾块 L1 切分数量
    int64_t mTailMain;          // M 轴尾块主尺寸
    int64_t nTailMain;          // N 轴尾块主尺寸
};
```

## 特殊成员变量

| 变量 | 说明 |
|------|------|
| m_, n_, k_ | 问题规模 |
| baseM_, baseN_ | L0 base 形状 |
| mCnt_, nCnt_, totalCnt_ | tile 数量 |
| mBaseNormCnt_, nBaseNormCnt_ | 正常 tile 数量 |
| mBaseTailMain_, nBaseTailMain_ | 尾块主尺寸 |
| mBaseTailLast_, nBaseTailLast_ | 尾块最后尺寸 |
| mCoreNum_, mTailCoreNum_ | M 轴核心数量、尾核心数量 |
| blockIdx_, blockNum_ | 当前 Block 索引、总 Block 数量 |
| startBlockIdx_, endBlockIdx_ | 起始/结束 Block 索引 |
| roundIdx_, round_ | 当前轮次、总轮次 |
| mTailTile_, nTailTile_, totalTailTile_ | 尾块切分数量 |
| mSplitAddrOffset_, nSplitAddrOffset_ | 尾块切分偏移 |
| mainRow_ | 主行数 |

## 特殊成员方法

### 构造函数
```
__aicore__ inline BlockSchedulerQuantBatchMatmulV3(const ProblemShape& shape, const Params& params)
```
功能：初始化 BlockSchedulerQuantBatchMatmulV3，计算 tile 切分、尾块参数、轮次等。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| shape | ProblemShape | 问题规模 `(m, n, k)` |
| params | Params | 调度参数 |

执行流程：
1. 设置问题规模：`m_`, `n_`, `k_`, `baseM_`, `baseN_`
2. 计算 tile 数量：`mCnt_`, `nCnt_`, `totalCnt_`
3. 计算扫描窗口：`mCoreNum_`, `mainRow_`, `mTailCoreNum_`
4. 计算轮次：`endBlockIdx_`, `round_`
5. 计算尾块参数：根据 `transA` 和 `transB` 计算尾块尺寸

### UpdateTailTile
```
__aicore__ inline void UpdateTailTile(uint32_t mTailTile, uint32_t nTailTile)
```
功能：更新尾块切分数量，重新计算结束 Block 索引和轮次。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| mTailTile | uint32_t | M 轴尾块切分数量 |
| nTailTile | uint32_t | N 轴尾块切分数量 |

### GetTotalCnt
```
__aicore__ inline int64_t GetTotalCnt()
```
功能：返回总 tile 数量（`totalCnt_`）。

### GetEndBlockIdx
```
__aicore__ inline int64_t GetEndBlockIdx()
```
功能：返回结束 Block 索引（`endBlockIdx_`）。


### CalSingleCoreShapeByCoord
```
__aicore__ inline void CalSingleCoreShapeByCoord(int64_t& singleCoreM, int64_t& singleCoreN, const BlockCoord& blockCoord)
```
功能：根据 Block 坐标计算单核形状（处理尾块）。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| singleCoreM | int64_t& | 单核 M 维度（原地修改） |
| singleCoreN | int64_t& | 单核 N 维度（原地修改） |
| blockCoord | BlockCoord | Block 坐标 |

### GetBlockShape
```
template <QuantBatchMatmul::QuantMode aQuantMode, QuantBatchMatmul::QuantMode bQuantMode, bool weightNz = false>
__aicore__ inline BlockShape GetBlockShape(BlockCoord blockCoord)
```
功能：返回当前 Block 的形状，支持量化模式和 NZ 格式。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| aQuantMode | QuantMode | A 矩阵量化模式（PERGROUP/PERBLOCK） |
| bQuantMode | QuantMode | B 矩阵量化模式（PERGROUP/PERBLOCK） |
| weightNz | bool | B 矩阵是否为 NZ 格式 |
| blockCoord | BlockCoord | Block 坐标 |

返回值：`BlockShape {singleCoreM, singleCoreN, mSplitAddrOffset_, nSplitAddrOffset_}`

特殊逻辑：
- **尾块切分判断**：`totalTailTile_ > 1 && roundIdx_ == round_`
- **FP4 对齐**：FP4 + transA 时 M 对齐到 2，FP4 + !transB 时 N 对齐到 2
- **PERBLOCK 模式**：对齐到 PER_BLOCK_SIZE 或 2 的幂次方
- **NZ 格式对齐**：根据 transB 对齐到 C0_SIZE 或 BLOCK_CUBE

### GetLoadBalanceInfo
```
__aicore__ inline AscendC::Std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> GetLoadBalanceInfo()
```
功能：返回负载均衡信息 `{mBaseNormCnt_, mBaseTailMain_, nBaseNormCnt_, nBaseTailMain_}`。

### UpdateNextBatchBlockRoundParams
```
__aicore__ inline void UpdateNextBatchBlockRoundParams()
```
功能：更新下一 Batch 的 Block 轮次参数。
执行流程：
1. 更新 `startBlockIdx_` 和 `endBlockIdx_`
2. 重置 `roundIdx_ = 0`
3. 重新计算 `round_`

### GetTileIdx
```
__aicore__ inline bool GetTileIdx(BlockCoord& blockCoord)
```
功能：获取当前轮次的 tile 索引，更新 Block 坐标。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| blockCoord | BlockCoord& | Block 坐标（原地修改） |

返回值：
- **true**：当前轮次有效，返回 tile 坐标
- **false**：当前轮次结束

执行流程：
1. 判断轮次结束：`roundIdx_ >= round_`
2. 计算 tile 索引：根据全载模式计算
3. Z 型扫描：计算 `blockCoordM`, `blockCoordN`
4. 反向扫描：奇数行反向（`blockCoordN = nCnt_ - 1 - blockCoordN`）
5. 更新 `roundIdx_++`

### GetTileCoord
```
__aicore__ inline void GetTileCoord(const BlockCoord& blockCoord, int64_t& mPos, int64_t& nPos)
```
功能：根据 Block 坐标计算 GM 地址偏移。
参数说明：
| 参数 | 类型 | 说明 |
|------|------|------|
| blockCoord | BlockCoord | Block 坐标 |
| mPos | int64_t& | M 轴 GM 偏移（原地修改） |
| nPos | int64_t& | N 轴 GM 偏移（原地修改） |

## 调用示例

### 组件组装
```
using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t>;
using LayoutA = AscendC::Te::NZLayoutPtn;
using LayoutB = AscendC::Te::NZLayoutPtn;
using AType = fp4x2_e2m1_t;
constexpr uint64_t FULL_LOAD_MODE = 0;

using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
    ProblemShape, FULL_LOAD_MODE, LayoutA, LayoutB, AType>;
```

### 参数准备
```
BlockScheduler::Params params = {
    baseM,              // L0 M 维度 base（如 128）
    baseN,              // L0 N 维度 base（如 128）
    mTailTile,          // M 轴尾块切分数量（如 1）
    nTailTile,          // N 轴尾块切分数量（如 1）
    mBaseTailSplitCnt,  // M 轴尾块 L1 切分数量（如 1）
    nBaseTailSplitCnt,  // N 轴尾块 L1 切分数量（如 1）
    mTailMain,          // M 轴尾块主尺寸（如 1）
    nTailMain           // N 轴尾块主尺寸（如 1）
};
```

### 组件初始化
```
ProblemShape shape{m, n, k};
BlockScheduler scheduler(shape, params);
```

### 更新尾块切分
```
scheduler.UpdateTailTile(mTailTile, nTailTile);
```

### 获取 tile 数量
```
int64_t totalCnt = scheduler.GetTotalCnt();
```

### Tile 循环处理
```
BlockCoord blockCoord{0, 0, 0, 0};
while (scheduler.GetTileIdx(blockCoord)) {
    // 获取 Block 形状
    constexpr auto aQuantMode = QuantBatchMatmul::QuantMode::PERGROUP_MODE;
    constexpr auto bQuantMode = QuantBatchMatmul::QuantMode::PERGROUP_MODE;
    constexpr bool weightNz = true;
    auto blockShape = scheduler.GetBlockShape<aQuantMode, bQuantMode, weightNz>(blockCoord);
    
    int64_t singleCoreM = Get<0>(blockShape);
    int64_t singleCoreN = Get<1>(blockShape);
    int64_t mSplitOffset = Get<2>(blockShape);
    int64_t nSplitOffset = Get<3>(blockShape);
    
    // 获取 GM 地址偏移
    int64_t mPos, nPos;
    scheduler.GetTileCoord(blockCoord, mPos, nPos);
    
    // 执行 BlockMmadMX 计算
    // ...
}
```

### 更新下一 Batch
```
scheduler.UpdateNextBatchBlockRoundParams();
```

### 获取负载均衡信息
```
auto loadBalanceInfo = scheduler.GetLoadBalanceInfo();
uint32_t mBaseNormCnt = std::get<0>(loadBalanceInfo);
uint32_t mBaseTailMain = std::get<1>(loadBalanceInfo);
uint32_t nBaseNormCnt = std::get<2>(loadBalanceInfo);
uint32_t nBaseTailMain = std::get<3>(loadBalanceInfo);
```

## 数据流

### Tile 切分流程
```
问题规模 (m, n, k)
    ↓
L0 tile 切分 (baseM, baseN)
    ↓
tile 数量计算 (mCnt × nCnt)
    ↓
尾块参数计算 (mBaseNormCnt, mBaseTailMain, nBaseNormCnt, nBaseTailMain)
    ↓
轮次计算 (round, startBlockIdx, endBlockIdx)
    ↓
Z 型扫描 (mCoreNum, mainRow, mTailCoreNum)
    ↓
Block 形状/坐标 (singleCoreM, singleCoreN, mPos, nPos)
```

### 量化模式对齐流程
```
GetBlockShape<QuantMode, QuantMode, weightNz>
    ↓
尾块切分判断 (totalTailTile > 1 && roundIdx == round)
    ↓
FP4 对齐：transA → M 对齐到 2，!transB → N 对齐到 2
    ↓
PERBLOCK 模式：对齐到 PER_BLOCK_SIZE 或 2 的幂次方
    ↓
NZ 格式对齐：!transB → C0_SIZE，transB → BLOCK_CUBE
    ↓
返回 BlockShape {singleCoreM, singleCoreN, mSplitAddrOffset, nSplitAddrOffset}
```

### Z 型扫描流程
```
tileIdx 计算
    ↓
rowIdx = tileIdx / nCnt / mCoreNum
    ↓
rowIdx < mainRow：blockCoordM = rowIdx × mCoreNum + tileIdx % mCoreNum
    ↓
rowIdx == mainRow：尾窗口计算
    ↓
rowIdx & 1：反向扫描（blockCoordN = nCnt - 1 - blockCoordN）
```

## 性能优化建议

### baseM/baseN 配置
- **建议值**：根据量化数据类型选择（如 128）
- **C0 对齐**：确保 baseM/baseN 对齐到 C0_SIZE（FP4: 64，FP8: 32）

### 尾块切分配置
- **mTailTile/nTailTile**：建议尾块切分数量不超过 4
- **mBaseTailSplitCnt/nBaseTailSplitCnt**：建议 L1 尾块切分数量为 1（不切分）

### 全载模式选择
- **非全载模式（FullLoadMode = 0）**：适用于一般场景
- **A 全载模式（FullLoadMode = A_FULL_LOAD_MODE）**：适用于大 K、小 M 场景

### 量化模式选择
- **PERGROUP_MODE**：per-group 量化，适用于小规模量化
- **PERBLOCK_MODE**：per-block 量化，对齐要求更高

### NZ 格式优化
- **weightNz = true**：B 矩阵 NZ 格式，提升 L1/L0 搬运效率
- 对齐要求：根据 transB 选择 C0_SIZE 或 BLOCK_CUBE

### 适用场景
- **Quant Batch Matmul MX Kernel**：量化 Batch Matmul
- **MxFP4/MxFP8 量化**：支持 FP4 E2M1/E1M2 和 FP8 E5M2/E4M3FN
- **多 Batch 维度**：支持 4 维 Batch（batchA/A2/A3/A4）
- **负载均衡**：动态调整 tile 分配