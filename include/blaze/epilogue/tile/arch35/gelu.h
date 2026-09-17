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
 * \file gelu.h
 * \brief Tile-level GELU epilogue: Tanh approximation and Erf exact.
 *
 * Design:
 *   - Public  (__aicore__): accepts make_tensor-created UB tensors, extracts
 *     raw __ubuf__ pointers via .data().get(), and delegates to Vf.
 *     For GeluErf, also constructs LocalTensor from byte offsets for
 *     high-level AscendC API calls (Erf, Muls, Cast).
 *   - Private (__simd_vf__): Reg API register-level computation.
 *     Pure vector instructions (DataCopy, Mul, Axpy, Exp, Div, Cast, Store).
 */

#pragma once

#include "tensor_api/tensor.h"
#include "kernel_operator.h"
#include "blaze/gemm/utils/common_utils.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

// ---------------------------------------------------------------------------
// Gelu tile: reusable activation epilogue
//   Any input/output combination of float / bfloat16_t / half:
//     - non-fp32 input  : widened to fp32 (CT_16F_TO_32F) before computing
//     - fp32 output     : stored directly, no narrowing cast
//     - bf16 output     : narrowed with CT_32F_TO_16F (NO_SAT)
//     - half output     : narrowed with CT_32F_TO_16F_SAT (clamps at 65504)
// ---------------------------------------------------------------------------
template <typename DataTypeOut_, typename DataTypeIn_>
class Gelu {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;

    __aicore__ inline Gelu() = default;

    /*!
     * Tanh-approximation GELU: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
     * Equivalent sigmoid form: x / (1 + exp(-2*sqrt(2/pi)*(x+0.044715*x^3)))
     * Accepts make_tensor-created UB tensors, extracts __ubuf__ for Vf.
     */
    template <typename SrcTensor, typename DstTensor>
    __aicore__ inline void GeluTanh(const SrcTensor& srcTensor, const DstTensor& dstTensor, uint16_t mSize,
                                    uint16_t nSize);

    /*!
     * Erf-based GELU: 0.5*x*(1+erf(x/sqrt(2)))
     * Accepts make_tensor-created UB tensors for src/dst and temp buffers.
     * High-level: per-row AscendC::Erf (polynomial approximation) via LocalTensor.
     * Reg API  : per-row vfBlock assembly of (1+erf)*(0.5*x) -> DataTypeOut.
     */
    template <typename SrcTensor, typename DstTensor, typename ErfTensor, typename Fp32Tensor, typename GeluFp32Tensor>
    __aicore__ inline void GeluErf(const SrcTensor& srcTensor, const DstTensor& dstTensor, const ErfTensor& erfTensor,
                                   const Fp32Tensor& fp32Tensor, const GeluFp32Tensor& geluFp32Tensor, uint16_t mSize,
                                   uint16_t nSize);

private:
    // Constants
    static constexpr float TANH_APPROX_FACTOR = 1.0f / 0.044715f;
    static constexpr float NEG_SQRT_EIGHT_OVER_PI = -1.595769121f * 0.044715f;
    static constexpr float ONE_OVER_SQRT_TWO = 0.707106781f;
    static constexpr float GELU_HALF = 0.5f;
    static constexpr float GELU_ONE = 1.0f;

#ifdef __CCE_AICORE__
    // Cast traits (compile-time constexpr, shared between high-level and Vf).
    // Named by bit-width: 16F covers both half and bfloat16_t.
    // Narrowing fp32 -> 16F, no saturation: safe for bf16 (shares fp32's exponent range).
    static constexpr AscendC::Reg::CastTrait CT_32F_TO_16F = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

    // Narrowing fp32 -> 16F with saturation: for half (max 65504), clamps instead of inf.
    static constexpr AscendC::Reg::CastTrait CT_32F_TO_16F_SAT = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

    // Widening 16F -> fp32 (half / bfloat16_t); Sat/Round are ignored for widening.
    static constexpr AscendC::Reg::CastTrait CT_16F_TO_32F = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};

    // Per-instantiation narrowing trait selected by output type:
    // half -> SAT (avoid inf beyond 65504); bf16 -> NO_SAT (no overflow possible);
    // float output skips the narrowing cast entirely (if constexpr at call sites).
    static constexpr AscendC::Reg::CastTrait CT_32F_TO_OUT = AscendC::IsSameType<DataTypeOut, half>::value ?
                                                                 CT_32F_TO_16F_SAT :
                                                                 CT_32F_TO_16F;

    static constexpr AscendC::Reg::DivSpecificMode GELU_DIV_MODE = {
        AscendC::Reg::MaskMergeMode::ZEROING,
        true,
    };
#endif // __CCE_AICORE__

    static constexpr AscendC::ErfConfig GELU_ERF_CONFIG = {AscendC::ErfAlgo::SUBSECTION_POLYNOMIAL_APPROXIMATION};

    // Vf parameter structs: raw __ubuf__ pointers + shape info for Reg API
    template <typename DataTypeOut, typename DataTypeIn>
    struct GeluTanhVfParams {
        __ubuf__ DataTypeOut* dstAddr;
        __ubuf__ DataTypeIn* srcAddr;
        uint16_t mSize;
        uint16_t nSize;
        uint16_t sizePerRepeat;
        uint16_t oneRowRepeatTimes;
        uint32_t nAligned;
    };

    template <typename DataTypeOut>
    struct GeluErfVfParams {
        __ubuf__ DataTypeOut* dstRowAddr;
        __ubuf__ float* erfRowAddr;
        __ubuf__ float* srcRowAddr;
        uint32_t nSize;
        uint16_t sizePerRepeat;
        uint16_t oneRowRepeatTimes;
    };

    /*!
     * Callee: fp32 input register → gelu-tanh math → narrow to DataTypeOut → store.
     * Shared by GeluTanhVf (fp32 input loaded directly) and GeluTanhCastInVf
     * (16F input widened to fp32 before the call). Mirrors the
     * __simd_callee__ composition pattern of gemm/tile/arch35/shift_w4_to_w8.h.
     */
    static __simd_callee__ inline void GeluTanhCoreCallee(AscendC::Reg::RegTensor<float>& vregInput,
                                                          AscendC::Reg::MaskReg& mask, __ubuf__ DataTypeOut* dstAddr,
                                                          uint32_t offset)
    {
        AscendC::Reg::RegTensor<float> vregInputSqr;
        AscendC::Reg::RegTensor<float> vregInputCub;
        AscendC::Reg::RegTensor<float> vregOutput;
        AscendC::Reg::Mul(vregInputSqr, vregInput, vregInput, mask);
        AscendC::Reg::Mul(vregInputCub, vregInputSqr, vregInput, mask);
        AscendC::Reg::Axpy(vregInputCub, vregInput, TANH_APPROX_FACTOR, mask);
        AscendC::Reg::Muls(vregInputCub, vregInputCub, NEG_SQRT_EIGHT_OVER_PI, mask);
        AscendC::Reg::Exp(vregInputCub, vregInputCub, mask);
        AscendC::Reg::Adds(vregInputCub, vregInputCub, GELU_ONE, mask);
        AscendC::Reg::Div<float, &GELU_DIV_MODE>(vregOutput, vregInput, vregInputCub, mask);
        if constexpr (AscendC::IsSameType<DataTypeOut, float>::value) {
            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(dstAddr + offset, vregOutput, mask);
        } else {
            AscendC::Reg::RegTensor<DataTypeOut> vregOutputOut;
            AscendC::Reg::Cast<DataTypeOut, float, CT_32F_TO_OUT>(vregOutputOut, vregOutput, mask);
            AscendC::Reg::DataCopy<DataTypeOut, AscendC::Reg::StoreDist::DIST_PACK_B32>(dstAddr + offset, vregOutputOut,
                                                                                        mask);
        }
    }

    /*!
     * VF: fp32 input → load → delegate to GeluTanhCoreCallee.
     */
    static __simd_vf__ inline void GeluTanhVf(GeluTanhVfParams<DataTypeOut, DataTypeIn> params)
    {
        AscendC::Reg::RegTensor<float> vregInput;
        AscendC::Reg::MaskReg mask;

        __ubuf__ float* src = reinterpret_cast<__ubuf__ float*>(params.srcAddr);
        for (uint16_t mIdx = 0; mIdx < params.mSize; mIdx++) {
            uint32_t count = params.nSize;
            for (uint16_t vfIdx = 0; vfIdx < params.oneRowRepeatTimes; vfIdx++) {
                mask = AscendC::Reg::UpdateMask<float>(count);
                uint32_t offset = mIdx * params.nAligned + vfIdx * params.sizePerRepeat;
                AscendC::Reg::DataCopy(vregInput, src + offset);
                GeluTanhCoreCallee(vregInput, mask, params.dstAddr, offset);
            }
        }
    }

    /*!
     * VF: 16F input → unpack + widen to fp32 → delegate to GeluTanhCoreCallee.
     */
    static __simd_vf__ inline void GeluTanhCastInVf(GeluTanhVfParams<DataTypeOut, DataTypeIn> params)
    {
        AscendC::Reg::RegTensor<float> vregInput;
        AscendC::Reg::RegTensor<DataTypeIn> vregInput16;
        AscendC::Reg::MaskReg mask;

        for (uint16_t mIdx = 0; mIdx < params.mSize; mIdx++) {
            uint32_t count = params.nSize;
            for (uint16_t vfIdx = 0; vfIdx < params.oneRowRepeatTimes; vfIdx++) {
                mask = AscendC::Reg::UpdateMask<float>(count);
                uint32_t offset = mIdx * params.nAligned + vfIdx * params.sizePerRepeat;
                AscendC::Reg::DataCopy<DataTypeIn, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregInput16,
                                                                                            params.srcAddr + offset);
                AscendC::Reg::Cast<float, DataTypeIn, CT_16F_TO_32F>(vregInput, vregInput16, mask);
                GeluTanhCoreCallee(vregInput, mask, params.dstAddr, offset);
            }
        }
    }

    /*!
     * VF: per-row Erf GELU assembly.
     * Computes (1+erf(x/sqrt(2))) * 0.5*x, casts to DataTypeOut, stores.
     */
    static __simd_vf__ inline void GeluErfVf(GeluErfVfParams<DataTypeOut> params)
    {
        AscendC::Reg::RegTensor<float> vregErf;
        AscendC::Reg::RegTensor<float> vregInput;
        AscendC::Reg::RegTensor<float> vregAdds;
        AscendC::Reg::RegTensor<float> vregMuls;
        AscendC::Reg::RegTensor<float> vregOutput;
        AscendC::Reg::MaskReg mask;

        for (uint16_t vfIdx = 0; vfIdx < params.oneRowRepeatTimes; vfIdx++) {
            mask = AscendC::Reg::UpdateMask<float>(params.nSize);
            uint32_t offset = vfIdx * params.sizePerRepeat;
            AscendC::Reg::DataCopy(vregErf, params.erfRowAddr + offset);
            AscendC::Reg::DataCopy(vregInput, params.srcRowAddr + offset);
            AscendC::Reg::Adds(vregAdds, vregErf, GELU_ONE, mask);
            AscendC::Reg::Muls(vregMuls, vregInput, GELU_HALF, mask);
            AscendC::Reg::Mul(vregOutput, vregAdds, vregMuls, mask);
            if constexpr (AscendC::IsSameType<DataTypeOut, float>::value) {
                AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(params.dstRowAddr + offset,
                                                                                      vregOutput, mask);
            } else {
                AscendC::Reg::RegTensor<DataTypeOut> vregOutputOut;
                AscendC::Reg::Cast<DataTypeOut, float, CT_32F_TO_OUT>(vregOutputOut, vregOutput, mask);
                AscendC::Reg::DataCopy<DataTypeOut, AscendC::Reg::StoreDist::DIST_PACK_B32>(params.dstRowAddr + offset,
                                                                                            vregOutputOut, mask);
            }
        }
    }

    /*!
     * Resolve UB byte offset from a make_tensor-created tensor's raw pointer.
     */
    template <typename Tensor>
    __aicore__ inline static uint32_t GetUbByteOffset(const Tensor& tensor)
    {
        return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(tensor.data().get()) - asc_get_phy_buf_addr(0));
    }
};

// ===========================================================================
// GeluTanh – public high-level
// ===========================================================================
template <typename DataTypeOut_, typename DataTypeIn_>
template <typename SrcTensor, typename DstTensor>
__aicore__ inline void Gelu<DataTypeOut_, DataTypeIn_>::GeluTanh(const SrcTensor& srcTensor, const DstTensor& dstTensor,
                                                                 uint16_t mSize, uint16_t nSize)
{
    using SrcElementType = asc::te::get_attribute_element_type<typename SrcTensor::element_type*>;
    using DstElementType = asc::te::get_attribute_element_type<typename DstTensor::element_type*>;
    using SrcLayoutPattern = asc::te::get_layout_pattern<typename SrcTensor::layout_type>;
    using DstLayoutPattern = asc::te::get_layout_pattern<typename DstTensor::layout_type>;
    static_assert(AscendC::Std::is_same_v<DataTypeIn, float> || AscendC::Std::is_same_v<DataTypeIn, bfloat16_t> ||
                      AscendC::Std::is_same_v<DataTypeIn, half>,
                  "GeluTanh input must be float, bfloat16_t or half.");
    static_assert(AscendC::Std::is_same_v<DataTypeOut, float> || AscendC::Std::is_same_v<DataTypeOut, bfloat16_t> ||
                      AscendC::Std::is_same_v<DataTypeOut, half>,
                  "GeluTanh output must be float, bfloat16_t or half.");
    static_assert(
        AscendC::Std::is_same_v<SrcElementType, DataTypeIn> && AscendC::Std::is_same_v<DstElementType, DataTypeOut>,
        "GeluTanh tensor element types must match DataTypeIn/DataTypeOut.");
    static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<SrcTensor>, asc::te::location::ub> &&
                      AscendC::Std::is_same_v<asc::te::get_mem_location<DstTensor>, asc::te::location::ub>,
                  "GeluTanh only supports UB tensors.");
    static_assert(AscendC::Std::is_same_v<SrcLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<DstLayoutPattern, asc::te::nd_ext_layout_ptn>,
                  "GeluTanh only supports NDExt tensor layouts.");
    if ASCEND_IS_AIC {
        return;
    }
    uint16_t sizePerRepeat = static_cast<uint16_t>(AscendC::VECTOR_REG_WIDTH / sizeof(float));
    uint16_t oneRowRepeatTimes = Gemm::CeilDiv(static_cast<uint32_t>(nSize), static_cast<uint32_t>(sizePerRepeat));
    uint32_t nAligned = Gemm::Align32(static_cast<uint32_t>(nSize));
    GeluTanhVfParams<DataTypeOut, DataTypeIn> params{reinterpret_cast<__ubuf__ DataTypeOut*>(dstTensor.data().get()),
                                                     reinterpret_cast<__ubuf__ DataTypeIn*>(srcTensor.data().get()),
                                                     mSize,
                                                     nSize,
                                                     sizePerRepeat,
                                                     oneRowRepeatTimes,
                                                     nAligned};
    if constexpr (AscendC::IsSameType<DataTypeIn, float>::value) {
        asc_vf_call<GeluTanhVf>(params);
    } else {
        asc_vf_call<GeluTanhCastInVf>(params);
    }
}

// ===========================================================================
// GeluErf – public high-level
// ===========================================================================
template <typename DataTypeOut_, typename DataTypeIn_>
template <typename SrcTensor, typename DstTensor, typename ErfTensor, typename Fp32Tensor, typename GeluFp32Tensor>
__aicore__ inline void Gelu<DataTypeOut_, DataTypeIn_>::GeluErf(const SrcTensor& srcTensor, const DstTensor& dstTensor,
                                                                const ErfTensor& erfTensor,
                                                                const Fp32Tensor& fp32Tensor,
                                                                const GeluFp32Tensor& geluFp32Tensor, uint16_t mSize,
                                                                uint16_t nSize)
{
    using SrcElementType = asc::te::get_attribute_element_type<typename SrcTensor::element_type*>;
    using DstElementType = asc::te::get_attribute_element_type<typename DstTensor::element_type*>;
    using ErfElementType = asc::te::get_attribute_element_type<typename ErfTensor::element_type*>;
    using Fp32ElementType = asc::te::get_attribute_element_type<typename Fp32Tensor::element_type*>;
    using GeluFp32ElementType = asc::te::get_attribute_element_type<typename GeluFp32Tensor::element_type*>;
    using SrcLayoutPattern = asc::te::get_layout_pattern<typename SrcTensor::layout_type>;
    using DstLayoutPattern = asc::te::get_layout_pattern<typename DstTensor::layout_type>;
    using ErfLayoutPattern = asc::te::get_layout_pattern<typename ErfTensor::layout_type>;
    using Fp32LayoutPattern = asc::te::get_layout_pattern<typename Fp32Tensor::layout_type>;
    using GeluFp32LayoutPattern = asc::te::get_layout_pattern<typename GeluFp32Tensor::layout_type>;
    static_assert(AscendC::Std::is_same_v<DataTypeIn, float> || AscendC::Std::is_same_v<DataTypeIn, bfloat16_t> ||
                      AscendC::Std::is_same_v<DataTypeIn, half>,
                  "GeluErf input must be float, bfloat16_t or half.");
    static_assert(AscendC::Std::is_same_v<DataTypeOut, float> || AscendC::Std::is_same_v<DataTypeOut, bfloat16_t> ||
                      AscendC::Std::is_same_v<DataTypeOut, half>,
                  "GeluErf output must be float, bfloat16_t or half.");
    static_assert(
        AscendC::Std::is_same_v<SrcElementType, DataTypeIn> && AscendC::Std::is_same_v<DstElementType, DataTypeOut> &&
            AscendC::Std::is_same_v<ErfElementType, float> && AscendC::Std::is_same_v<Fp32ElementType, float> &&
            AscendC::Std::is_same_v<GeluFp32ElementType, float>,
        "GeluErf tensor element types must match DataTypeIn/DataTypeOut (temp buffers must be float).");
    static_assert(AscendC::Std::is_same_v<asc::te::get_mem_location<SrcTensor>, asc::te::location::ub> &&
                      AscendC::Std::is_same_v<asc::te::get_mem_location<DstTensor>, asc::te::location::ub> &&
                      AscendC::Std::is_same_v<asc::te::get_mem_location<ErfTensor>, asc::te::location::ub> &&
                      AscendC::Std::is_same_v<asc::te::get_mem_location<Fp32Tensor>, asc::te::location::ub> &&
                      AscendC::Std::is_same_v<asc::te::get_mem_location<GeluFp32Tensor>, asc::te::location::ub>,
                  "GeluErf only supports UB tensors.");
    static_assert(AscendC::Std::is_same_v<SrcLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<DstLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<ErfLayoutPattern, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<Fp32LayoutPattern, asc::te::nd_ext_layout_ptn> &&
                      AscendC::Std::is_same_v<GeluFp32LayoutPattern, asc::te::nd_ext_layout_ptn>,
                  "GeluErf only supports NDExt tensor layouts.");
    if ASCEND_IS_AIC {
        return;
    }

    uint32_t nAligned = Gemm::Align32(static_cast<uint32_t>(nSize));
    uint16_t sizePerRepeat = static_cast<uint16_t>(AscendC::VECTOR_REG_WIDTH / sizeof(float));
    uint16_t oneRowRepeatTimes = Gemm::CeilDiv(static_cast<uint32_t>(nSize), static_cast<uint32_t>(sizePerRepeat));

    uint32_t srcUbOffset = GetUbByteOffset(srcTensor);
    uint32_t erfUbOffset = GetUbByteOffset(erfTensor);
    uint32_t fp32UbOffset = GetUbByteOffset(fp32Tensor);
    uint32_t geluFp32UbOffset = GetUbByteOffset(geluFp32Tensor);

    AscendC::LocalTensor<DataTypeIn> srcLocal{AscendC::TPosition::VECIN, srcUbOffset, nAligned};
    AscendC::LocalTensor<float> erfLocal{AscendC::TPosition::VECCALC, erfUbOffset, nAligned};
    AscendC::LocalTensor<float> geluFp32Local{AscendC::TPosition::VECCALC, geluFp32UbOffset, nAligned};
    AscendC::LocalTensor<float> fp32Local{AscendC::TPosition::VECCALC, fp32UbOffset, nAligned};

    __ubuf__ float* erfAddr = reinterpret_cast<__ubuf__ float*>(erfTensor.data().get());
    __ubuf__ DataTypeOut* dstAddr = reinterpret_cast<__ubuf__ DataTypeOut*>(dstTensor.data().get());

    if constexpr (AscendC::IsSameType<DataTypeIn, float>::value) {
        __ubuf__ float* src = reinterpret_cast<__ubuf__ float*>(srcTensor.data().get());
        for (uint32_t mIdx = 0; mIdx < mSize; mIdx++) {
            AscendC::Muls(geluFp32Local, srcLocal[mIdx * nAligned], ONE_OVER_SQRT_TWO, nSize);
            AscendC::Erf<float, false, GELU_ERF_CONFIG>(erfLocal, geluFp32Local, nSize);

            GeluErfVfParams<DataTypeOut> params{
                dstAddr + mIdx * nAligned, erfAddr, src + mIdx * nAligned, nSize, sizePerRepeat, oneRowRepeatTimes};
            asc_vf_call<GeluErfVf>(params);
        }
    } else {
        __ubuf__ float* fp32Addr = reinterpret_cast<__ubuf__ float*>(fp32Tensor.data().get());
        for (uint32_t mIdx = 0; mIdx < mSize; mIdx++) {
            AscendC::Cast(fp32Local, srcLocal[mIdx * nAligned], AscendC::RoundMode::CAST_NONE, nSize);
            AscendC::Muls(geluFp32Local, fp32Local, ONE_OVER_SQRT_TWO, nSize);
            AscendC::Erf<float, false, GELU_ERF_CONFIG>(erfLocal, geluFp32Local, nSize);

            GeluErfVfParams<DataTypeOut> params{dstAddr + mIdx * nAligned, erfAddr, fp32Addr, nSize, sizePerRepeat,
                                                oneRowRepeatTimes};
            asc_vf_call<GeluErfVf>(params);
        }
    }
}

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
