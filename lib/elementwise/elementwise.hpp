/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file elementwise.hpp
 * \brief Elementwise 解决方案注册表
 *        参考 hiptensor ElementwiseSolutionRegistry 设计
 *        支持 Binary 和 Trinary 操作
 */

#ifndef ACLTENSOR_LIB_ELEMENTWISE_ELEMENTWISE_HPP
#define ACLTENSOR_LIB_ELEMENTWISE_ELEMENTWISE_HPP

#include "cann_ops_tensor_types.h"
#include "acl/acl.h"
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>
#include <cstdint>

namespace acltensor {

/**
 * @brief 解决方案唯一标识符
 */
struct SolutionUid
{
    acltensorOperator_t   op;         // 操作符
    acltensorDataType_t   dataType;   // 数据类型
    uint32_t              numModes;   // 维度数

    bool operator==(SolutionUid const& other) const
    {
        return op == other.op && dataType == other.dataType && numModes == other.numModes;
    }
};

/**
 * @brief SolutionUid 哈希函数
 */
struct SolutionUidHash
{
    size_t operator()(SolutionUid const& uid) const
    {
        size_t h1 = static_cast<size_t>(uid.op);
        size_t h2 = static_cast<size_t>(uid.dataType);
        size_t h3 = static_cast<size_t>(uid.numModes);
        return h1 ^ (h2 << 8) ^ (h3 << 16);
    }
};

/**
 * @brief Elementwise Binary 内核执行函数类型
 */
using ElementwiseBinaryExecuteFunc = acltensorStatus_t (*)(
    const void* A, const void* C, void* D,
    int64_t elemNum,
    const void* alpha, const void* gamma,
    aclrtStream stream);

/**
 * @brief Elementwise 解决方案基类
 *        支持 Binary 和 Trinary 操作
 */
class ElementwiseSolution
{
public:
    ElementwiseSolution(SolutionUid const& uid, ElementwiseBinaryExecuteFunc executeFunc)
        : uid_(uid), binaryExecuteFunc_(executeFunc) {}

    virtual ~ElementwiseSolution() = default;

    SolutionUid const& getUid() const { return uid_; }
    uint64_t getSolutionId() const { return solutionId_; }
    size_t getWorkspaceSize() const { return workspaceSize_; }

    /**
     * @brief 执行 Binary 内核
     */
    acltensorStatus_t executeBinary(
        const void* alpha, const void* A,
        const void* gamma, const void* C,
        void* D, int64_t elemNum,
        aclrtStream stream) const
    {
        if (binaryExecuteFunc_ == nullptr)
        {
            return ACLTENSOR_STATUS_INTERNAL_ERROR;
        }
        return binaryExecuteFunc_(A, C, D, elemNum, alpha, gamma, stream);
    }

    /* TODO: Phase 2 - Trinary 支持 */
    // using ElementwiseTrinaryExecuteFunc = acltensorStatus_t (*)(...);
    // acltensorStatus_t executeTrinary(...) const;

protected:
    SolutionUid                    uid_;
    ElementwiseBinaryExecuteFunc   binaryExecuteFunc_ = nullptr;
    uint64_t                       solutionId_ = 0;
    size_t                         workspaceSize_ = 0;

    /* TODO: Phase 2 */
    // ElementwiseTrinaryExecuteFunc trinaryExecuteFunc_ = nullptr;
};

/**
 * @brief Elementwise 解决方案注册表（单例）
 *        参考 hiptensor ElementwiseSolutionRegistry 设计
 *        采用延迟初始化模式，首次使用时自动注册所有解决方案
 *        支持 Binary 和 Trinary 操作
 */
class ElementwiseSolutionRegistry
{
public:
    /**
     * @brief 获取单例实例（延迟初始化）
     *        首次调用时自动注册所有解决方案
     */
    static ElementwiseSolutionRegistry& instance()
    {
        static ElementwiseSolutionRegistry registry;
        static bool initialized = false;
        if (!initialized)
        {
            registry.registerAllSolutions();
            initialized = true;
        }
        return registry;
    }

    /**
     * @brief 注册解决方案
     */
    void registerSolution(std::unique_ptr<ElementwiseSolution> solution)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (solution)
        {
            SolutionUid uid = solution->getUid();
            solutions_[uid] = std::move(solution);
        }
    }

    /**
     * @brief 批量注册解决方案
     */
    void registerSolutions(
        std::unordered_map<SolutionUid, std::unique_ptr<ElementwiseSolution>, SolutionUidHash>&& solutions)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& pair : solutions)
        {
            solutions_[pair.first] = std::move(pair.second);
        }
    }

    /**
     * @brief 查询匹配的解决方案
     * @param op        操作符
     * @param dataType  数据类型
     * @param numModes  维度数
     * @return 匹配的解决方案列表
     */
    std::vector<ElementwiseSolution*> getSolutions(
        acltensorOperator_t op,
        acltensorDataType_t dataType,
        uint32_t numModes) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<ElementwiseSolution*> result;

        SolutionUid targetUid{op, dataType, numModes};

        // 精确匹配
        auto it = solutions_.find(targetUid);
        if (it != solutions_.end())
        {
            result.push_back(it->second.get());
        }

        // 如果没有精确匹配，查找通用解决方案（numModes = 0 表示任意维度）
        if (result.empty())
        {
            SolutionUid genericUid{op, dataType, 0};
            auto genericIt = solutions_.find(genericUid);
            if (genericIt != solutions_.end())
            {
                result.push_back(genericIt->second.get());
            }
        }

        return result;
    }

    /**
     * @brief 获取所有解决方案
     */
    std::vector<ElementwiseSolution*> getAllSolutions() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<ElementwiseSolution*> result;
        for (const auto& pair : solutions_)
        {
            result.push_back(pair.second.get());
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

    /**
     * @brief 注册所有解决方案（延迟初始化时调用）
     */
    void registerAllSolutions();

    mutable std::mutex mutex_;
    std::unordered_map<SolutionUid, std::unique_ptr<ElementwiseSolution>, SolutionUidHash> solutions_;
};

// 兼容性别名（Phase 1 使用，后续可移除）
using ElementwiseBinarySolution = ElementwiseSolution;
using ElementwiseBinarySolutionRegistry = ElementwiseSolutionRegistry;

} // namespace acltensor

#endif // ACLTENSOR_LIB_ELEMENTWISE_ELEMENTWISE_HPP
