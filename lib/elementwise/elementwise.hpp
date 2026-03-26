/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file elementwise.hpp
 * \brief Elementwise 解决方案注册表
 *        支持 Binary 和 Trinary 操作
 */

#ifndef ACLTENSOR_LIB_ELEMENTWISE_ELEMENTWISE_HPP
#define ACLTENSOR_LIB_ELEMENTWISE_ELEMENTWISE_HPP

#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>
#include <cstdint>
#include <atomic>

#include "acl/acl.h"

#include "cann_ops_tensor_types.h"
#include "core/tensor_descriptor.hpp"
#include "core/operation_descriptor.hpp"

namespace acltensor {

/**
 * @brief 解决方案唯一标识符
 */
struct SolutionUid
{
    acltensorOperator_t   op;            // 操作符
    acltensorDataType_t   dataType;      // 数据类型
    uint32_t              numModes;      // 维度数（0 表示通用）
    OperationType         operationType; // 操作类型（Binary/Trinary）

    bool operator==(SolutionUid const& other) const
    {
        return op == other.op &&
               dataType == other.dataType &&
               numModes == other.numModes &&
               operationType == other.operationType;
    }
};

/**
 * @brief SolutionUid 哈希函数
 * 使用质数乘法降低碰撞概率
 */
struct SolutionUidHash
{
    size_t operator()(SolutionUid const& uid) const
    {
        size_t h1 = static_cast<size_t>(uid.op);
        size_t h2 = static_cast<size_t>(uid.dataType);
        size_t h3 = static_cast<size_t>(uid.numModes);
        size_t h4 = static_cast<size_t>(uid.operationType);

        // 使用质数乘法混合哈希值，降低碰撞概率
        // 31, 37, 41, 43 是质数，与2的幂次无公因数，能更好地分散哈希值
        return (((h1 * 31) ^ h2) * 37 + h3 * 41) ^ h4 * 43;
    }
};

/**
 * @brief Elementwise 执行参数（区分输入输出 buffer）
 */
struct ElementwiseArgs
{
    // ========== 输入张量（A, B, C） ==========
    const void* bufferA = nullptr;         // 输入 A：设备指针（只读）
    const int64_t* lengthsA = nullptr;      // A 的形状
    const int64_t* stridesA = nullptr;      // A 的步长
    const int32_t* modesA = nullptr;        // A 的维度标签
    uint32_t numModesA = 0;                // A 的维度数
    acltensorDataType_t dataTypeA = ACLTENSOR_R_32F;  // A 的数据类型

    const void* bufferB = nullptr;         // 输入 B：设备指针（只读，Trinary 用）
    const int64_t* lengthsB = nullptr;
    const int64_t* stridesB = nullptr;
    const int32_t* modesB = nullptr;
    uint32_t numModesB = 0;
    acltensorDataType_t dataTypeB = ACLTENSOR_R_32F;

    const void* bufferC = nullptr;         // 输入 C：设备指针（只读）
    const int64_t* lengthsC = nullptr;      // C 的形状
    const int64_t* stridesC = nullptr;      // C 的步长
    const int32_t* modesC = nullptr;        // C 的维度标签
    uint32_t numModesC = 0;
    acltensorDataType_t dataTypeC = ACLTENSOR_R_32F;

    // ========== 输出张量（D） ==========
    void* bufferD = nullptr;               // 输出 D：设备指针（可写）
    const int64_t* lengthsD = nullptr;      // D 的形状
    const int64_t* stridesD = nullptr;      // D 的步长
    const int32_t* modesD = nullptr;        // D 的维度标签
    uint32_t numModesD = 0;
    acltensorDataType_t dataTypeD = ACLTENSOR_R_32F;

    // ========== 标量 ==========
    const void* alpha = nullptr;   // A 的系数
    const void* beta = nullptr;    // (A*B) 或 C 的系数
    const void* gamma = nullptr;   // 位移或 C 的系数

    // ========== 操作符 ==========
    acltensorOperator_t opAC = ACLTENSOR_OP_ADD;   // Binary: C = A opAC C
    acltensorOperator_t opAB = ACLTENSOR_OP_IDENTITY;   // Trinary: A opAB B
    acltensorOperator_t opABC = ACLTENSOR_OP_IDENTITY;  // Trinary: (A opAB B) opABC C

    // ========== 执行配置 ==========
    aclrtStream stream = nullptr;
    void* workspace = nullptr;
};

/**
 * @brief 创建 Elementwise Binary 执行参数
 * @param opDesc   操作描述符
 * @param alpha    标量 alpha
 * @param A        输入张量 A 的设备指针
 * @param gamma    标量 gamma
 * @param C        输入张量 C 的设备指针
 * @param D        输出张量 D 的设备指针
 * @param stream   ACL stream
 * @return 构造好的 ElementwiseArgs
 */
inline ElementwiseArgs CreateElementwiseBinaryArgs(
    const acltensorOperationDescriptor_t opDesc,
    const void* alpha,
    const void* A,
    const void* gamma,
    const void* C,
    void* D,
    aclrtStream stream)
{
    ElementwiseArgs args;

    // 填充输入 A
    args.bufferA = A;
    args.lengthsA = opDesc->descA ? opDesc->descA->lens.data() : nullptr;
    args.stridesA = opDesc->descA ? opDesc->descA->strides.data() : nullptr;
    args.modesA = opDesc->modeA.data();
    args.numModesA = opDesc->descA ? opDesc->descA->numModes : 0;
    args.dataTypeA = opDesc->descA ? opDesc->descA->dataType : ACLTENSOR_R_32F;

    // 填充输入 C
    args.bufferC = C;
    args.lengthsC = opDesc->descC ? opDesc->descC->lens.data() : nullptr;
    args.stridesC = opDesc->descC ? opDesc->descC->strides.data() : nullptr;
    args.modesC = opDesc->modeC.data();
    args.numModesC = opDesc->descC ? opDesc->descC->numModes : 0;
    args.dataTypeC = opDesc->descC ? opDesc->descC->dataType : ACLTENSOR_R_32F;

    // 填充输出 D
    args.bufferD = D;
    args.lengthsD = opDesc->descD ? opDesc->descD->lens.data() : nullptr;
    args.stridesD = opDesc->descD ? opDesc->descD->strides.data() : nullptr;
    args.modesD = opDesc->modeD.data();
    args.numModesD = opDesc->descD ? opDesc->descD->numModes : 0;
    args.dataTypeD = opDesc->descD ? opDesc->descD->dataType : ACLTENSOR_R_32F;

    // Binary 不使用的 B
    args.bufferB = nullptr;
    args.lengthsB = nullptr;
    args.stridesB = nullptr;
    args.modesB = nullptr;
    args.numModesB = 0;
    args.dataTypeB = ACLTENSOR_R_32F;

    // 填充标量
    args.alpha = alpha;
    args.gamma = gamma;
    args.beta = nullptr;

    // 填充操作符
    args.opAC = opDesc->opAC;
    args.opAB = ACLTENSOR_OP_IDENTITY;
    args.opABC = ACLTENSOR_OP_IDENTITY;

    // 填充执行配置
    args.stream = stream;
    args.workspace = nullptr;

    return args;
}

/**
 * @brief Elementwise Binary 内核执行函数类型
 */
using ElementwiseBinaryExecuteFunc = acltensorStatus_t (*)(
    const ElementwiseArgs& args);

/**
 * @brief Elementwise 解决方案基类
 *        支持 Binary 和 Trinary 操作
 */
class ElementwiseSolution
{
public:
    ElementwiseSolution(SolutionUid const& uid, ElementwiseBinaryExecuteFunc executeFunc)
        : uid_(uid), executeFunc_(executeFunc) {}

    virtual ~ElementwiseSolution() = default;

    SolutionUid const& getUid() const { return uid_; }
    uint64_t getSolutionId() const { return solutionId_; }
    size_t getWorkspaceSize() const { return workspaceSize_; }

    /**
     * @brief 统一执行接口（支持 Binary 和 Trinary）
     */
    virtual acltensorStatus_t execute(const ElementwiseArgs& args) const
    {
        if (executeFunc_ == nullptr)
        {
            return ACLTENSOR_STATUS_INTERNAL_ERROR;
        }
        return executeFunc_(args);
    }

protected:
    SolutionUid                    uid_;
    ElementwiseBinaryExecuteFunc   executeFunc_ = nullptr;
    uint64_t                       solutionId_ = 0;
    size_t                         workspaceSize_ = 0;
};

/**
 * @brief Elementwise 解决方案注册表（单例）
 *        采用延迟初始化模式，首次使用时自动注册所有解决方案
 *        支持 Binary 和 Trinary 操作
 */
class ElementwiseSolutionRegistry
{
public:
    /**
     * @brief 获取单例实例
     *        解决方案通过各算子的自动注册宏进行注册
     */
    static ElementwiseSolutionRegistry& instance()
    {
        static ElementwiseSolutionRegistry registry;
        return registry;
    }

    /**
     * @brief 注册解决方案
     */
    void registerSolution(std::shared_ptr<ElementwiseSolution> solution)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (solution)
        {
            SolutionUid uid = solution->getUid();
            solutions_[uid] = solution;  // shared_ptr 支持拷贝，无需移动
        }
    }

    /**
     * @brief 批量注册解决方案
     */
    void registerSolutions(
        std::unordered_map<SolutionUid, std::shared_ptr<ElementwiseSolution>, SolutionUidHash>&& solutions)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& pair : solutions)
        {
            solutions_[pair.first] = pair.second;
        }
    }

    /**
     * @brief 查询匹配的解决方案
     * @param op            操作符
     * @param dataType      数据类型
     * @param numModes      维度数
     * @param operationType 操作类型（Binary/Trinary）
     * @return 匹配的解决方案列表
     */
    std::vector<std::shared_ptr<ElementwiseSolution>> getSolutions(
        acltensorOperator_t op,
        acltensorDataType_t dataType,
        uint32_t numModes,
        OperationType operationType) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<std::shared_ptr<ElementwiseSolution>> result;

        // 1. 精确匹配（包含 operationType）
        SolutionUid targetUid{op, dataType, numModes, operationType};
        auto it = solutions_.find(targetUid);
        if (it != solutions_.end())
        {
            result.push_back(it->second);
        }

        // 2. 如果没有精确匹配，查找通用解决方案（numModes = 0，相同 operationType）
        if (result.empty())
        {
            SolutionUid genericUid{op, dataType, 0, operationType};
            auto genericIt = solutions_.find(genericUid);
            if (genericIt != solutions_.end())
            {
                result.push_back(genericIt->second);
            }
        }

        return result;
    }

    /**
     * @brief 获取所有解决方案
     */
    std::vector<std::shared_ptr<ElementwiseSolution>> getAllSolutions() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::shared_ptr<ElementwiseSolution>> result;
        for (const auto& pair : solutions_)
        {
            result.push_back(pair.second);
        }
        return result;
    }

    /**
     * @brief 清空注册表
     */
    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        solutions_.clear();
    }

    /**
     * @brief 获取解决方案数量
     */
    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return solutions_.size();
    }

private:
    ElementwiseSolutionRegistry() = default;
    ~ElementwiseSolutionRegistry() = default;
    ElementwiseSolutionRegistry(ElementwiseSolutionRegistry const&) = delete;
    ElementwiseSolutionRegistry& operator=(ElementwiseSolutionRegistry const&) = delete;

    mutable std::mutex mutex_;
    std::unordered_map<SolutionUid, std::shared_ptr<ElementwiseSolution>, SolutionUidHash> solutions_;
};

// 兼容性别名（当前版本保留，后续可能会移除）
using ElementwiseBinarySolution = ElementwiseSolution;
using ElementwiseBinarySolutionRegistry = ElementwiseSolutionRegistry;

/**
 * @brief 自动注册宏 - Elementwise 解决方案
 *        使用静态初始化在程序启动时自动注册解决方案到注册表
 * @param OP_NAME    操作符名称（如 ADD, SUB, MUL, DIV）
 * @param DATA_TYPE  数据类型（如 R_32F, R_16F, R_16BF）
 * @param NUM_MODES  维度数（0 表示通用解决方案，支持任意维度）
 * @param OP_TYPE    操作类型（BINARY 或 TRINARY）
 * @param EXECUTE_FUNC 执行函数指针
 *
 * 使用示例：
 * REGISTER_ELEMENTWISE_SOLUTION(ADD, R_32F, 0, BINARY, AddF32Execute);
 */
#define REGISTER_ELEMENTWISE_SOLUTION(OP_NAME, DATA_TYPE, NUM_MODES, OP_TYPE, EXECUTE_FUNC) \
    namespace { \
        struct OP_NAME##_##DATA_TYPE##_##OP_TYPE##_Registrar { \
            OP_NAME##_##DATA_TYPE##_##OP_TYPE##_Registrar() { \
                auto& registry = acltensor::ElementwiseSolutionRegistry::instance(); \
                acltensor::SolutionUid uid{ \
                    ACLTENSOR_OP_##OP_NAME, \
                    ACLTENSOR_##DATA_TYPE, \
                    NUM_MODES, \
                    acltensor::OperationType::ELEMENTWISE_##OP_TYPE \
                }; \
                auto solution = std::make_shared<acltensor::ElementwiseSolution>(uid, EXECUTE_FUNC); \
                registry.registerSolution(solution); \
            } \
        }; \
        static OP_NAME##_##DATA_TYPE##_##OP_TYPE##_Registrar g_registrar_##OP_NAME##_##DATA_TYPE##_##OP_TYPE; \
    }

} // namespace acltensor

#endif // ACLTENSOR_LIB_ELEMENTWISE_ELEMENTWISE_HPP
