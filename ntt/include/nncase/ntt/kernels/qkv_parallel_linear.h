/* Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "binary.h"
#include "matmul.h"
#include "packed_matmul.h"
#include <algorithm>
#include <cmath>
#include <type_traits>

namespace nncase::ntt {
enum class qkv_quantization_mode : uint8_t {
    none,
    static_tensor,
    dynamic_tensor,
};

namespace detail {
template <class TBias, class TOutput>
constexpr void add_optional_bias(const TBias &bias, TOutput &output) {
    if constexpr (!std::is_same_v<std::remove_cvref_t<TBias>, std::nullptr_t>) {
        binary<ops::add>(output, bias, output);
    }
}

template <qkv_quantization_mode QuantizationMode, class TScale, class TInput>
constexpr float qkv_input_scale(const TScale &scale, const TInput &input,
                                dim_t row) {
    if constexpr (QuantizationMode == qkv_quantization_mode::static_tensor) {
        static_assert(!std::is_same_v<std::remove_cvref_t<TScale>,
                                      std::nullptr_t>,
                      "Static-tensor QKV requires an input scale");
        return ntt::cast_elem<float>(scale.elements()[0]);
    } else {
        static_assert(QuantizationMode ==
                          qkv_quantization_mode::dynamic_tensor,
                      "Unquantized QKV must not evaluate an input scale");
        static_assert(std::is_same_v<std::remove_cvref_t<TScale>,
                                     std::nullptr_t>,
                      "Dynamic-tensor QKV computes the input scale per row");
        float max_abs = 0.0f;
        for (dim_t k = 0; k < input.shape()[1_dim]; ++k) {
            max_abs = std::max(
                max_abs,
                std::abs(ntt::cast_elem<float>(input(row, k))));
        }

        return std::max(max_abs, 1.0e-12f) / 448.0f;
    }
}

template <qkv_quantization_mode QuantizationMode, class TScale>
constexpr float qkv_weight_scale(const TScale &scale, dim_t column) {
    static_assert(!std::is_same_v<std::remove_cvref_t<TScale>,
                                  std::nullptr_t>,
                  "Quantized QKV requires a weight scale");
    if constexpr (QuantizationMode == qkv_quantization_mode::static_tensor) {
        return ntt::cast_elem<float>(scale.elements()[0]);
    } else {
        static_assert(QuantizationMode ==
                          qkv_quantization_mode::dynamic_tensor,
                      "Unquantized QKV must not evaluate a weight scale");
        static_assert(TScale::rank() == 1,
                      "Dynamic-tensor QKV weight scale must have shape [N]");
        return ntt::cast_elem<float>(scale(column));
    }
}

template <qkv_quantization_mode QuantizationMode, class TInput, class TWeight,
          class TBias, class TInputScale, class TWeightScale, class TOutput>
constexpr void qkv_scaled_projection(const TInput &input,
                                     const TWeight &weight,
                                     const TBias &bias,
                                     const TInputScale &input_scale,
                                     const TWeightScale &weight_scale,
                                     TOutput &output) {
    static_assert(TInput::rank() == 2 && TWeight::rank() == 2 &&
                      TOutput::rank() == 2,
                  "scaled QKV projection currently requires rank-2 tensors");
    for (dim_t row = 0; row < output.shape()[0_dim]; ++row) {
        const float activation_scale =
            qkv_input_scale<QuantizationMode>(input_scale, input, row);
        for (dim_t column = 0; column < output.shape()[1_dim]; ++column) {
            float accumulator = 0.0f;
            for (dim_t k = 0; k < input.shape()[1_dim]; ++k) {
                const auto quantized_input = nncase::float_e4m3_t(
                    ntt::cast_elem<float>(input(row, k)) / activation_scale);
                accumulator += ntt::cast_elem<float>(quantized_input) *
                               ntt::cast_elem<float>(weight(k, column));
            }

            accumulator *= activation_scale *
                           qkv_weight_scale<QuantizationMode>(weight_scale,
                                                              column);
            if constexpr (!std::is_same_v<std::remove_cvref_t<TBias>,
                                          std::nullptr_t>) {
                accumulator += ntt::cast_elem<float>(bias(column));
            }

            output(row, column) =
                ntt::cast_elem<typename TOutput::value_type>(accumulator);
        }
    }
}
} // namespace detail

template <qkv_quantization_mode QuantizationMode, class TInput,
          class TQWeight, class TKWeight, class TVWeight,
          class TQBias, class TKBias, class TVBias, class TQInputScale,
          class TKInputScale, class TVInputScale, class TQWeightScale,
          class TKWeightScale, class TVWeightScale, class TQOutput,
          class TKOutput, class TVOutput>
constexpr void qkv_parallel_linear(
    const TInput &input, const TQWeight &q_weight, const TKWeight &k_weight,
    const TVWeight &v_weight, const TQBias &q_bias, const TKBias &k_bias,
    const TVBias &v_bias, const TQInputScale &q_input_scale,
    const TKInputScale &k_input_scale, const TVInputScale &v_input_scale,
    const TQWeightScale &q_weight_scale,
    const TKWeightScale &k_weight_scale,
    const TVWeightScale &v_weight_scale, TQOutput &q_output,
    TKOutput &k_output, TVOutput &v_output) {
    if constexpr (QuantizationMode != qkv_quantization_mode::none) {
        detail::qkv_scaled_projection<QuantizationMode>(
            input, q_weight, q_bias, q_input_scale, q_weight_scale, q_output);
        detail::qkv_scaled_projection<QuantizationMode>(
            input, k_weight, k_bias, k_input_scale, k_weight_scale, k_output);
        detail::qkv_scaled_projection<QuantizationMode>(
            input, v_weight, v_bias, v_input_scale, v_weight_scale, v_output);
    } else {
        static_assert(
            std::is_same_v<std::remove_cvref_t<TQInputScale>,
                           std::nullptr_t> &&
                std::is_same_v<std::remove_cvref_t<TKInputScale>,
                               std::nullptr_t> &&
                std::is_same_v<std::remove_cvref_t<TVInputScale>,
                               std::nullptr_t> &&
                std::is_same_v<std::remove_cvref_t<TQWeightScale>,
                               std::nullptr_t> &&
                std::is_same_v<std::remove_cvref_t<TKWeightScale>,
                               std::nullptr_t> &&
                std::is_same_v<std::remove_cvref_t<TVWeightScale>,
                               std::nullptr_t>,
            "Unquantized QKV must not receive scale operands");
        matmul<false, false, false>(input, q_weight, q_output, nullptr,
                                   fixed_shape_v<>, fixed_shape_v<>,
                                   fixed_shape_v<>, fixed_shape_v<>);
        detail::add_optional_bias(q_bias, q_output);
        matmul<false, false, false>(input, k_weight, k_output, nullptr,
                                   fixed_shape_v<>, fixed_shape_v<>,
                                   fixed_shape_v<>, fixed_shape_v<>);
        detail::add_optional_bias(k_bias, k_output);
        matmul<false, false, false>(input, v_weight, v_output, nullptr,
                                   fixed_shape_v<>, fixed_shape_v<>,
                                   fixed_shape_v<>, fixed_shape_v<>);
        detail::add_optional_bias(v_bias, v_output);
    }
}

template <qkv_quantization_mode QuantizationMode, class TInput,
          class TQWeight, class TKWeight, class TVWeight,
          class TQBias, class TKBias, class TVBias, class TQInputScale,
          class TKInputScale, class TVInputScale, class TQWeightScale,
          class TKWeightScale, class TVWeightScale, class TQOutput,
          class TKOutput, class TVOutput>
constexpr void packed_qkv_parallel_linear(
    const TInput &input, const TQWeight &q_weight, const TKWeight &k_weight,
    const TVWeight &v_weight, const TQBias &q_bias, const TKBias &k_bias,
    const TVBias &v_bias, const TQInputScale &q_input_scale,
    const TKInputScale &k_input_scale, const TVInputScale &v_input_scale,
    const TQWeightScale &q_weight_scale,
    const TKWeightScale &k_weight_scale,
    const TVWeightScale &v_weight_scale, TQOutput &q_output,
    TKOutput &k_output, TVOutput &v_output) {
    static_assert(
        QuantizationMode == qkv_quantization_mode::none &&
        std::is_same_v<std::remove_cvref_t<TQInputScale>, std::nullptr_t> &&
            std::is_same_v<std::remove_cvref_t<TKInputScale>, std::nullptr_t> &&
            std::is_same_v<std::remove_cvref_t<TVInputScale>, std::nullptr_t> &&
            std::is_same_v<std::remove_cvref_t<TQWeightScale>, std::nullptr_t> &&
            std::is_same_v<std::remove_cvref_t<TKWeightScale>, std::nullptr_t> &&
            std::is_same_v<std::remove_cvref_t<TVWeightScale>, std::nullptr_t>,
        "packed CPU QKV currently requires unscaled operands");
    (void)q_input_scale;
    (void)k_input_scale;
    (void)v_input_scale;
    (void)q_weight_scale;
    (void)k_weight_scale;
    (void)v_weight_scale;
    packed_matmul<false>(input, q_weight, q_output, 1.0f);
    detail::add_optional_bias(q_bias, q_output);
    packed_matmul<false>(input, k_weight, k_output, 1.0f);
    detail::add_optional_bias(k_bias, k_output);
    packed_matmul<false>(input, v_weight, v_output, 1.0f);
    detail::add_optional_bias(v_bias, v_output);
}
} // namespace nncase::ntt
