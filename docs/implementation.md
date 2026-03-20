# ops-tensor API 实现状态

## 版本信息
- **当前版本**: v1.0.0
- **更新日期**: 2026-03-20
- **开发阶段**: Phase 1 - Elementwise Binary (Add)

---

## ✅ 已实现接口 (Phase 1)

### 1. 句柄管理
- `acltensorCreate` - 创建库句柄
- `acltensorDestroy` - 销毁库句柄

### 2. 张量描述符
- `acltensorCreateTensorDescriptor` - 创建张量描述符
- `acltensorDestroyTensorDescriptor` - 销毁张量描述符

### 3. 操作描述符
- `acltensorCreateElementwiseBinary` - 创建二元元素操作描述符
- `acltensorDestroyOperationDescriptor` - 销毁操作描述符

### 4. Plan 管理
- `acltensorCreatePlanPreference` - 创建 Plan 偏好
- `acltensorDestroyPlanPreference` - 销毁 Plan 偏好
- `acltensorCreatePlan` - 创建执行 Plan
- `acltensorDestroyPlan` - 销毁 Plan

### 5. 执行函数
- `acltensorElementwiseBinaryExecute` - 执行二元元素操作

### 6. 辅助工具
- `acltensorGetErrorString` - 获取错误字符串
- `acltensorGetVersion` - 获取库版本号

### 7. 已实现算子
- **Add** (张量加法) - `src/add/`

### 8. 支持范围
- **数据类型**: FP32 (`ACLTENSOR_R_32F`)
- **一元操作符**: IDENTITY (`ACLTENSOR_OP_IDENTITY`)
- **二元操作符**: ADD (`ACLTENSOR_OP_ADD`)
- **算法**: DEFAULT (`ACLTENSOR_ALGO_DEFAULT`)

---

## ⏸️ 待实现接口

### Phase 2 - Elementwise 扩展

**必须实现**：
- `acltensorCreateElementwiseTrinary` - 三元元素操作描述符
- `acltensorElementwiseTrinaryExecute` - 执行三元元素操作

**重要功能**：
- `acltensorEstimateWorkspaceSize` - 估算工作空间大小
- `acltensorOperationDescriptorSetAttribute` - 设置操作描述符属性
- `acltensorOperationDescriptorGetAttribute` - 获取操作描述符属性

**可选功能**：
- `acltensorHandleResizePlanCache` - 调整 Plan 缓存
- `acltensorHandleWritePlanCacheToFile` - 写入缓存到文件
- `acltensorHandleReadPlanCacheFromFile` - 从文件读取缓存
- `acltensorPlanPreferenceSetAttribute` - 设置 Plan 偏好属性
- `acltensorPlanGetAttribute` - 获取 Plan 属性

### Phase 3 - Contraction & Reduction

**必须实现**：
- `acltensorCreateContraction` - 创建张量收缩描述符
- `acltensorContract` - 执行张量收缩 (D = αA·B + βC)
- `acltensorCreateReduction` - 创建张量归约描述符
- `acltensorReduce` - 执行张量归约

### Phase 4 - Permutation

**可选功能**：
- `acltensorCreatePermutation` - 创建张量排列描述符
- `acltensorPermute` - 执行张量排列/转置

---

## ⏸️ 待实现操作符

### 一元操作符 (21个)

**必须实现**：
- SQRT, RELU, SIGMOID, TANH, EXP, LOG, ABS, GELU, SILU

**重要功能**：
- CONJ, RCP, NEG

**可选功能**：
- SIN, COS, TAN, SINH, COSH, ASIN, ACOS, ATAN, CEIL, FLOOR

### 二元操作符 (6个)

**必须实现**：
- MUL, MAX, MIN

**重要功能**：
- SUB, DIV

**可选功能**：
- POW

---

## ⏸️ 待支持数据类型

### 必须实现
- FP16 (`ACLTENSOR_R_16F`)
- BF16 (`ACLTENSOR_R_16BF`)

### 重要功能
- FP64 (`ACLTENSOR_R_64F`)

### 可选功能
- INT8, UINT8, INT32, UINT32
- Complex FP32 (`ACLTENSOR_C_32F`)
- Complex FP64 (`ACLTENSOR_C_64F`)

---

## 📊 实现进度统计

| 类别 | 已实现 | 待实现 | 总计 |
|------|--------|--------|------|
| API 接口 | 15 | 19 | 34 |
| 操作符 | 2 | 26 | 28 |
| 数据类型 | 1 | 9 | 10 |
| 算子 | 1 | - | - |

**总体完成度**: Phase 1 (Elementwise Binary - Add) ✅

---

## 📝 开发路线图

### Phase 1: Elementwise Binary - Add ✅ (已完成)
- [x] 基础架构搭建
- [x] 句柄管理
- [x] 张量描述符（FP32）
- [x] Add 算子实现
- [x] Add 算子测试
- [x] 打包流程

### Phase 2: Elementwise 扩展 (下一步)
- [ ] 多数据类型支持 (FP16/BF16/FP64)
- [ ] 更多一元操作符（SQRT, RELU, SIGMOID, TANH, EXP, LOG, ABS, GELU, SILU）
- [ ] 更多二元操作符（MUL, MAX, MIN）
- [ ] Elementwise Trinary
- [ ] Plan 缓存机制
- [ ] 工作空间估算
- [ ] 属性查询接口

### Phase 3: Contraction & Reduction
- [ ] Contraction 接口与实现
- [ ] Reduction 接口与实现
- [ ] 复数数据类型支持
- [ ] 性能优化

### Phase 4: Permutation & 高级特性
- [ ] Permutation 接口与实现
- [ ] JIT 编译支持
- [ ] Auto-tune 机制
- [ ] 多 GPU 支持

---

## 📌 当前限制

1. **数据类型**: 仅支持 FP32
2. **操作符**: 仅支持 IDENTITY 和 ADD
3. **操作类型**: 仅支持 Elementwise Binary
4. **算子**: 仅实现 Add 算子
5. **不支持**: workspace 复用、Plan 缓存、属性查询

---

## 🔧 设计参考

- **API 设计**: 参考 hiptensor/cuTensor
- **算子注册机制**: 参考 rocFFT
- **测试框架**: 自研轻量级测试框架

---

**文档维护**: 请在实现新接口时及时更新本文档
