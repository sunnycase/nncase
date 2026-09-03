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

#include "../tensor.h"
#include <algorithm>
#include <cmath>
#include <type_traits>

namespace nncase::ntt {

namespace block_scaled_matmul_detail {

template <class T>
constexpr decltype(auto) local_tensor(T &tensor) noexcept {
    if constexpr (ShardedTensor<T>) {
        return tensor.local();
    } else {
        return (tensor);
    }
}

template <class T>
constexpr decltype(auto) local_tensor(const T &tensor) noexcept {
    if constexpr (ShardedTensor<T>) {
        return tensor.local();
    } else {
        return (tensor);
    }
}

template <class T>
constexpr auto global_offset(const T &tensor) noexcept {
    if constexpr (ShardedTensor<T>) {
        using mesh_t = typename std::decay_t<T>::mesh_type;
        return tensor.sharding().global_offset(tensor.shape(),
                                               mesh_t::local_index());
    } else {
        return make_zeros_shape<std::decay_t<T>::rank()>();
    }
}

template <size_t WeightBlockN, size_t WeightBlockK, Tensor TLhs, Tensor TRhs,
          Tensor TRhsScale, Tensor TOutput>
constexpr void block_scaled_matmul_local(const TLhs &lhs, const TRhs &rhs,
                                         const TRhsScale &rhs_scale,
                                         TOutput &output,
                                         dim_t rhs_reduction_offset,
                                         dim_t rhs_column_offset) {
    static_assert(TLhs::rank() == 2 && TRhs::rank() == 2 &&
                      TRhsScale::rank() == 2 && TOutput::rank() == 2,
                  "block_scaled_matmul currently requires rank-2 tensors");
    static_assert(WeightBlockN > 0 && WeightBlockK > 0,
                  "block dimensions must be positive");

    const auto rows = output.shape()[0_dim];
    const auto columns = output.shape()[1_dim];
    const auto reduction = lhs.shape()[1_dim];
    for (dim_t row = 0; row < rows; ++row) {
        for (dim_t column = 0; column < columns; ++column) {
            float accumulator = 0.0f;
            for (dim_t block_start = 0; block_start < reduction;
                 block_start += WeightBlockK) {
                const auto block_end =
                    std::min<dim_t>(reduction, block_start + WeightBlockK);
                float max_abs = 0.0f;
                for (dim_t k = block_start; k < block_end; ++k) {
                    max_abs = std::max(
                        max_abs,
                        std::abs(ntt::cast_elem<float>(lhs(row, k))));
                }

                const float lhs_scale =
                    std::max(max_abs, 1.0e-12f) / 448.0f;
                const float weight_scale = ntt::cast_elem<float>(rhs_scale(
                    (rhs_column_offset + column) / WeightBlockN,
                    (rhs_reduction_offset + block_start) / WeightBlockK));
                for (dim_t k = block_start; k < block_end; ++k) {
                    const auto quantized_lhs = nncase::float_e4m3_t(
                        ntt::cast_elem<float>(lhs(row, k)) / lhs_scale);
                    accumulator += ntt::cast_elem<float>(quantized_lhs) *
                                   ntt::cast_elem<float>(rhs(k, column)) *
                                   lhs_scale * weight_scale;
                }
            }

            output(row, column) = ntt::cast_elem<typename TOutput::value_type>(
                accumulator);
        }
    }
}

} // namespace block_scaled_matmul_detail

template <size_t WeightBlockN, size_t WeightBlockK, class TLhs, class TRhs,
          class TRhsScale, class TOutput>
    requires((Tensor<TLhs> || ShardedTensor<TLhs>) &&
             (Tensor<TRhs> || ShardedTensor<TRhs>) &&
             (Tensor<TRhsScale> || ShardedTensor<TRhsScale>) &&
             (Tensor<TOutput> || ShardedTensor<TOutput>))
constexpr void block_scaled_matmul(const TLhs &lhs, const TRhs &rhs,
                                   const TRhsScale &rhs_scale,
                                   TOutput &output) {
    constexpr auto rhs_rank = std::decay_t<TRhs>::rank();
    static_assert(rhs_rank >= 2,
                  "block_scaled_matmul requires a rank-2 or higher rhs");
    const auto rhs_offset = block_scaled_matmul_detail::global_offset(rhs);
    block_scaled_matmul_detail::block_scaled_matmul_local<WeightBlockN,
                                                          WeightBlockK>(
        block_scaled_matmul_detail::local_tensor(lhs),
        block_scaled_matmul_detail::local_tensor(rhs),
        block_scaled_matmul_detail::local_tensor(rhs_scale),
        block_scaled_matmul_detail::local_tensor(output),
        rhs_offset[rhs_rank - 2], rhs_offset[rhs_rank - 1]);
}
} // namespace nncase::ntt
