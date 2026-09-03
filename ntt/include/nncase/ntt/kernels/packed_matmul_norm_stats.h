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
#include "cast.h"
#include "packed_matmul.h"
#include "reduce.h"

namespace nncase::ntt {
template <bool UseMean, dim_t Axis, Tensor TLhs, Tensor TRhs, class TOut,
          class TStats, class TLoadC, class TScale, class TAddend>
NTT_ALWAYS_INLINE constexpr void packed_matmul_norm_stats(
    const TLhs &lhs, const TRhs &rhs, TOut &output, TStats &stats,
    const TLoadC &load_c, const TScale &scale, const TAddend &addend) {
    packed_matmul_kernel(lhs, rhs, output, load_c, scale, addend);

    using output_f32_element_t = replace_element_t<
        typename std::decay_t<TOut>::element_type, float>;
    auto output_f32 = make_tensor<output_f32_element_t>(output.shape());
    auto output_square = make_tensor<output_f32_element_t>(output.shape());
    cast(output, output_f32, fixed_shape_v<>);
    if constexpr (UseMean) {
        auto stats_sum = stats.view(0_dim);
        reduce_sum<false>(output_f32, stats_sum, fixed_shape_v<Axis>,
                          fixed_shape_v<Axis>);
    }

    binary<ops::mul>(output_f32, output_f32, output_square);
    auto stats_square = stats.view(fixed_dim_v<UseMean ? 1 : 0>);
    reduce_sum<false>(output_square, stats_square, fixed_shape_v<Axis>,
                      fixed_shape_v<Axis>);
}
} // namespace nncase::ntt
