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
#include "unary.h"
#include <type_traits>

namespace nncase::ntt {
namespace detail {
template <class TBias, class TOutput>
constexpr void add_matmul_glu_bias(const TBias &bias, TOutput &output) {
    if constexpr (!std::is_same_v<std::remove_cvref_t<TBias>, std::nullptr_t>) {
        binary<ops::add>(output, bias, output);
    }
}
} // namespace detail

template <class TInput, class TGateWeight, class TUpWeight, class TGateBias,
          class TUpBias, class TOutput>
constexpr void matmul_swiglu(const TInput &input,
                             const TGateWeight &gate_weight,
                             const TUpWeight &up_weight,
                             const TGateBias &gate_bias,
                             const TUpBias &up_bias, TOutput &output) {
    auto up_output =
        make_tensor<typename std::decay_t<TOutput>::element_type>(output.shape());
    matmul<false, false, false>(input, gate_weight, output, nullptr,
                               fixed_shape_v<>, fixed_shape_v<>,
                               fixed_shape_v<>, fixed_shape_v<>);
    detail::add_matmul_glu_bias(gate_bias, output);
    matmul<false, false, false>(input, up_weight, up_output, nullptr,
                               fixed_shape_v<>, fixed_shape_v<>,
                               fixed_shape_v<>, fixed_shape_v<>);
    detail::add_matmul_glu_bias(up_bias, up_output);
    unary<ops::swish>(output, output);
    binary<ops::mul>(output, up_output, output);
}

template <class TInput, class TGateWeight, class TUpWeight, class TGateBias,
          class TUpBias, class TOutput>
constexpr void packed_matmul_swiglu(const TInput &input,
                                    const TGateWeight &gate_weight,
                                    const TUpWeight &up_weight,
                                    const TGateBias &gate_bias,
                                    const TUpBias &up_bias, TOutput &output) {
    auto up_output =
        make_tensor<typename std::decay_t<TOutput>::element_type>(output.shape());
    packed_matmul<false>(input, gate_weight, output, 1.0f);
    detail::add_matmul_glu_bias(gate_bias, output);
    packed_matmul<false>(input, up_weight, up_output, 1.0f);
    detail::add_matmul_glu_bias(up_bias, up_output);
    unary<ops::swish>(output, output);
    binary<ops::mul>(output, up_output, output);
}
} // namespace nncase::ntt
