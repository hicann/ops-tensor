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
 * \file dispatch_policy.h
 * \brief
 */
#pragma once

#include "blaze/gemm/utils/common_utils.h"

namespace Blaze {
namespace Gemm {
/* block schedule policies */
struct KernelMmadWithScaleMx {};   // Multi-block with Mx scale
struct KernelGroupedMmadWithScaleMx {}; // Grouped multi-block with Mx scale
struct KernelMmadWithScaleMxWithoutBatch {}; // Multi-block with Mx scale, without batch broadcast
struct KernelMmadWithScaleFixpipeQuant {}; // Multi-block with fixpipe quant scale (A8W8 fixpipe)
struct KernelMultiBlockStreamK {}; // Multi-tile transfer with K-axis spliting and caching
struct KernelQbmmMultiBlockStreamK {}; // QBMM MX StreamK schedule
struct KernelMmadMultiBlockBasic {}; // Multi-tile basic
struct KernelIterBatchBroadcast {}; // Multi-tile batchMatmul broadcast + iterbatch
struct KernelMmadMultiBlockBmmBroadcast {}; // Multi-tile batchMatmul broadcast
struct KernelMmadMultiBlockAFullLoad {};     // Multi-tile aFullLoad
enum class MatMulL0C2Out : std::uint8_t
{
    ON_THE_FLY = 0,
    ND_FIXPIPE_1_1 = 1,
    ND_FIXPIPE_1_2 = 2
};

/**
 * @struct MatmulWithScaleFixpipeQuant
 * @brief Quantized fixpipe matmul with scale and fixpipe dequant (Tensor API / Blaze)
 * @param [in] FULL_LOAD_MODE_: full-load mode, 0 = none, A_FULL_LOAD_MODE = A full load
 * @param [in] ATOMIC_ADD_: whether to enable atomic add on output
 */
template <uint64_t FULL_LOAD_MODE_ = 0, bool ATOMIC_ADD_ = false>
struct MatmulWithScaleFixpipeQuant {
    using ScheduleType = KernelMmadWithScaleFixpipeQuant;
    constexpr static uint64_t fullLoadMode = FULL_LOAD_MODE_;
    constexpr static bool isAtomicAdd = ATOMIC_ADD_;
};

/**
 * @struct MatmulWithScaleMx
 * @brief Mx Matrix multiplication with scaleA and scaleB
 */
template <uint64_t FULL_LOAD_MODE_ = 0, bool ATOMIC_ADD_ = false, class ScheduleType_ = KernelMmadWithScaleMx>
struct MatmulWithScaleMx {
    using ScheduleType = ScheduleType_;
    constexpr static uint64_t fullLoadMode = FULL_LOAD_MODE_;
    constexpr static bool isAtomicAdd = ATOMIC_ADD_;
};

/**
 * @struct GroupedMatmulWithScaleMx
 * @brief Grouped Mx matrix multiplication with scaleA and scaleB
 */
template <uint64_t FULL_LOAD_MODE_ = 0, bool ATOMIC_ADD = false,
          class ScheduleType_ = KernelGroupedMmadWithScaleMx>
struct GroupedMatmulWithScaleMx {
    using ScheduleType = ScheduleType_;
    constexpr static uint64_t fullLoadMode = FULL_LOAD_MODE_;
    constexpr static bool isAtomicAdd = ATOMIC_ADD;
};

/**
 * @struct MatmulWithScaleMxL0CPingpong
 * @brief Mx matrix multiplication with L0C ping-pong perf schedule.
 */
template <uint64_t FullLoadMode_ = 0, bool AtomicAdd_ = false, class ScheduleType_ = KernelMmadWithScaleMx>
struct MatmulWithScaleMxL0CPingpong {
    using ScheduleType = ScheduleType_;
    constexpr static uint64_t fullLoadMode = FullLoadMode_;
    constexpr static bool isAtomicAdd = AtomicAdd_;
};

/**
 * @struct MatmulMultiBlockWithStreamK
 * @brief Matrix multiplication split k axis processing structure, no quant, no bias, implemented base on layout
 * @param [in] FixpOpti_: enum, judge if enabling fixp align optimize, default is ON_THE_FLY
 * @param [in] FUSED_OP_TYPE_: execute fusion after mmad , default is 0
 */
template <
    MatMulL0C2Out FixpOpti_ = MatMulL0C2Out::ON_THE_FLY, uint64_t FUSED_OP_TYPE_ = 0,
    class KernelSchedule_ = KernelMultiBlockStreamK>
struct MatmulMultiBlockWithStreamK {
    using ScheduleType = KernelSchedule_;
    constexpr static bool enableInputDataLenCheck = false;
    constexpr static uint64_t fusedOpType = FUSED_OP_TYPE_;
    constexpr static MatMulL0C2Out fixpOpti = FixpOpti_;
};

/**
 * @struct MatmulMultiBlockWithStreamKSplitK
 * @brief Matrix multiplication split k axis processing structure, no quant, no bias, implemented base on layout
 * @param [in] FixpOpti_: enum, judge if enabling fixp align optimize, default is ON_THE_FLY
 * @param [in] IsSplitSinglecoreK_: indicate whether splited singlecorek is enabled，default is true(split single
 * core k)
 */
template <
    MatMulL0C2Out FixpOpti_ = MatMulL0C2Out::ON_THE_FLY, bool IsSplitSinglecoreK_ = true,
    class KernelSchedule_ = KernelMultiBlockStreamK>
struct MatmulMultiBlockWithStreamKSplitK {
    using ScheduleType = KernelSchedule_;
    static constexpr MatMulL0C2Out fixpOpti = FixpOpti_;
    static constexpr bool isSplitSingleCoreK = IsSplitSinglecoreK_;
};

/**
 * @struct MatmulMultiBlockBasic
 * @brief Matrix multiplication multi-block structure, no quant, implemented based on Layout
 * @param [in] FULL_LOAD_MODE_: mode of full load, default is 0(no full load)
 * @param [in] FUSED_OP_TYPE_: execute fusion after mmad , default is 0
 * @param [in] KernelSchedule_: mmad dispatch policy
 * @param [in] NonContiguousType_: matmul support non-contiguous scene such as: slice, transpose
 */
template <
    uint64_t FULL_LOAD_MODE_ = 0, uint64_t FUSED_OP_TYPE_ = 0, class KernelSchedule_ = KernelMmadMultiBlockBasic,
    uint64_t NonContiguousType_ = 0>
struct MatmulMultiBlockBasic {
    using ScheduleType = KernelSchedule_;
    constexpr static uint64_t fullLoadMode = FULL_LOAD_MODE_;
    constexpr static uint64_t fusedOpType = FUSED_OP_TYPE_;
    constexpr static uint64_t nonContiguousType = NonContiguousType_;
};

/**
 * @struct BatchMatmulIterbatchBroadcast
 * @brief Matrix multiplication with batch broadcast, no quant, implemented based on Layout
 * @param [in] ABroadcast: A tensor needs broadcast
 * @param [in] BBroadcast: B tensor needs broadcast
 */
template <bool ABroadcast_, bool BBroadcast_>
struct MatmulIterBatchBroadcast {
    using ScheduleType = KernelIterBatchBroadcast;
    static constexpr bool aBroadcast = ABroadcast_;
    static constexpr bool bBroadcast = BBroadcast_;
};

/**
 * @struct MatmulMultiBlockBasicSplitK
 * @brief Matrix multiplication multi-block structure, no quant, implemented based on Layout
 * @param [in] FullLoadMode_: mode of full load, default is 0(no full load)
 * @param [in] IsSplitSinglecoreK_: indicate whether splited singlecorek is enabled，default is true(split single
 * core k)
 * @param [in] KernelSchedule_: mmad dispatch policy
 * @param [in] NonContiguousType_: 0 indicates support for continuity
 */
template <
    uint64_t FullLoadMode_ = 0, bool IsSplitSinglecoreK_ = true,
    class KernelSchedule_ = KernelMmadMultiBlockBasic, uint64_t NonContiguousType_ = 0>
struct MatmulMultiBlockBasicSplitK {
    using ScheduleType = KernelSchedule_;
    static constexpr uint64_t fullLoadMode = FullLoadMode_;
    static constexpr bool isSplitSingleCoreK = IsSplitSinglecoreK_;
    static constexpr uint64_t nonContiguousType = NonContiguousType_;
};

/**
 * @struct MatmulMultiBlockAFullLoad
 * @brief Matrix multiplication multi-block structure, no quant, implemented based on Layout
 * @param [in] FullLoadMode_: mode of full load, default is 0(no full load)
 * @param [in] FusedOpType_: execute fusion after mmad , default is 0
 */
template <
    uint64_t FullLoadMode_ = A_FULL_LOAD_MODE, uint64_t FusedOpType_ = 0,
    class KernelSchedule_ = KernelMmadMultiBlockAFullLoad>
struct MatmulMultiBlockAFullLoad {
    using ScheduleType = KernelSchedule_;
    static constexpr uint64_t FULL_LOAD_MODE = FullLoadMode_;
    static constexpr uint64_t FUSED_OP_TYPE = FusedOpType_;
};

} // namespace Gemm
} // namespace Blaze
