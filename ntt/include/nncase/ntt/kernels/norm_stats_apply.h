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

#include "../apply.h"
#include "../primitive_ops.h"
#include "../tensor_traits.h"
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>

namespace nncase::ntt {
namespace norm_stats_apply_detail {

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

template <ScalarOrVector T>
constexpr decltype(auto) scalar_at(T &value, size_t index) noexcept {
    if constexpr (Scalar<T>) {
        return (value);
    } else if constexpr (T::rank() == 1) {
        return value(make_shape(index));
    } else {
        constexpr auto inner_size = T::shape().template slice<1>().length();
        auto &&inner = value(make_shape(index / inner_size));
        return scalar_at(inner, index % inner_size);
    }
}

template <ScalarOrVector T>
constexpr decltype(auto) scalar_at(const T &value, size_t index) noexcept {
    if constexpr (Scalar<T>) {
        return (value);
    } else if constexpr (T::rank() == 1) {
        return value(make_shape(index));
    } else {
        constexpr auto inner_size = T::shape().template slice<1>().length();
        const auto &inner = value(make_shape(index / inner_size));
        return scalar_at(inner, index % inner_size);
    }
}

template <ScalarOrVector T>
constexpr float horizontal_sum(const T &value) noexcept {
    if constexpr (Scalar<T>) {
        return ntt::cast_elem<float>(value);
    } else {
        float result = 0.0f;
        for (size_t lane = 0; lane < element_scalar_count_v<T>; ++lane) {
            result += ntt::cast_elem<float>(scalar_at(value, lane));
        }

        return result;
    }
}

template <Tensor TParameter, Dimensions TInnerIndex>
constexpr auto parameter_index(const TParameter &parameter,
                               const TInnerIndex &inner_index) noexcept {
    static_assert(TParameter::rank() == TInnerIndex::rank(),
                  "NormApply parameter rank must match the normalized suffix rank");
    return generate_shape<TParameter::rank()>([&](auto axis) {
        return parameter.shape()[axis] == 1_dim ? 0_dim : inner_index[axis];
    });
}

} // namespace norm_stats_apply_detail

template <bool UseMean, dim_t Axis, class TInput, class TStats>
NTT_ALWAYS_INLINE constexpr void norm_stats(const TInput &input,
                                            TStats &stats) noexcept {
    using namespace norm_stats_apply_detail;
    const auto &input_local = local_tensor(input);
    auto &stats_local = local_tensor(stats);
    using input_local_t = std::decay_t<decltype(input_local)>;
    using input_element_t = typename input_local_t::element_type;
    using accumulator_t =
        decltype(ntt::cast_elem<float>(std::declval<input_element_t>()));
    using stats_element_t =
        typename std::decay_t<decltype(stats_local)>::element_type;
    static_assert(std::is_same_v<std::remove_cv_t<stats_element_t>, float>,
                  "NormStats output must use float32 elements");

    constexpr auto normalized_axis =
        positive_index(Axis, input_local_t::rank());
    const auto outer_shape =
        input_local.shape().template slice<0, normalized_axis>();
    const auto inner_shape =
        input_local.shape().template slice<normalized_axis>();
    constexpr auto reduced_rank = input_local_t::rank() - normalized_axis;

    ntt::apply(outer_shape, [&](auto outer_index) {
        accumulator_t sum{};
        accumulator_t square_sum{};
        ntt::apply(inner_shape, [&](auto inner_index) {
            const auto input_index = outer_index.concat(inner_index);
            const auto value =
                ntt::cast_elem<float>(input_local(input_index));
            if constexpr (UseMean) {
                sum += value;
            }

            square_sum += ntt::square(value);
        });

        const auto reduced_index =
            outer_index.concat(make_zeros_shape<reduced_rank>());
        if constexpr (UseMean) {
            stats_local(make_shape(0_dim).concat(reduced_index)) =
                horizontal_sum(sum);
            stats_local(make_shape(1_dim).concat(reduced_index)) =
                horizontal_sum(square_sum);
        } else {
            stats_local(make_shape(0_dim).concat(reduced_index)) =
                horizontal_sum(square_sum);
        }
    });
}

template <bool UseMean, dim_t Axis, class TInput, class TStats, class TScale,
          class TBias, class TOutput>
NTT_ALWAYS_INLINE constexpr void norm_apply(
    const TInput &input, const TStats &stats, const TScale &scale,
    const TBias &bias, TOutput &output, float epsilon) noexcept {
    using namespace norm_stats_apply_detail;
    const auto &input_local = local_tensor(input);
    const auto &stats_local = local_tensor(stats);
    const auto &scale_local = local_tensor(scale);
    const auto &bias_local = local_tensor(bias);
    auto &output_local = local_tensor(output);
    using input_local_t = std::decay_t<decltype(input_local)>;
    using output_element_t =
        typename std::decay_t<decltype(output_local)>::element_type;
    using output_scalar_t = element_or_scalar_t<output_element_t>;

    constexpr auto normalized_axis =
        positive_index(Axis, input_local_t::rank());
    const auto outer_shape =
        input_local.shape().template slice<0, normalized_axis>();
    const auto inner_shape =
        input_local.shape().template slice<normalized_axis>();
    constexpr auto reduced_rank = input_local_t::rank() - normalized_axis;
    const float normalization_size =
        (float)(input.shape().template slice<normalized_axis>().length() *
                element_scalar_count_v<typename input_local_t::element_type>);

    ntt::apply(outer_shape, [&](auto outer_index) {
        const auto reduced_index =
            outer_index.concat(make_zeros_shape<reduced_rank>());
        float mean = 0.0f;
        float square_sum;
        if constexpr (UseMean) {
            mean = ntt::cast_elem<float>(
                       stats_local(make_shape(0_dim).concat(reduced_index))) /
                   normalization_size;
            square_sum = ntt::cast_elem<float>(
                stats_local(make_shape(1_dim).concat(reduced_index)));
        } else {
            square_sum = ntt::cast_elem<float>(
                stats_local(make_shape(0_dim).concat(reduced_index)));
        }

        float variance = square_sum / normalization_size;
        if constexpr (UseMean) {
            variance -= mean * mean;
        }

        const float inv_std =
            1.0f / std::sqrt(std::max(variance, 0.0f) + epsilon);
        ntt::apply(inner_shape, [&](auto inner_index) {
            const auto input_index = outer_index.concat(inner_index);
            const auto scale_index = parameter_index(scale_local, inner_index);
            const auto bias_index = parameter_index(bias_local, inner_index);
            const auto input_value =
                ntt::cast_elem<float>(input_local(input_index));
            const auto scale_value =
                ntt::cast_elem<float>(scale_local(scale_index));
            const auto bias_value =
                ntt::cast_elem<float>(bias_local(bias_index));
            const auto centered = UseMean ? input_value - mean : input_value;
            const auto result =
                ((centered * inv_std) * scale_value) + bias_value;
            output_local(input_index) =
                ntt::cast_elem<output_scalar_t>(result);
        });
    });
}

} // namespace nncase::ntt
