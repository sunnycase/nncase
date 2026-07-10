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
#include "../tensor_traits.h"
#include "../utility.h"
#include <type_traits>
#include <utility>

namespace nncase::ntt {
namespace detail {
template <class TValue>
constexpr decltype(auto) bitcast_scalar_at(TValue &&value, dim_t lane) {
    using value_type = std::remove_cvref_t<TValue>;
    if constexpr (Vector<value_type>) {
        return std::forward<TValue>(value)(
            unravel_index(lane, value.shape()));
    } else {
        return std::forward<TValue>(value);
    }
}

template <Tensor TIn, Tensor TOut> class bitcast_impl {
  public:
    using input_element_type = typename TIn::element_type;
    using output_element_type = typename TOut::element_type;
    using input_scalar_type = element_or_scalar_t<input_element_type>;
    using output_scalar_type = element_or_scalar_t<output_element_type>;

    static_assert(
        std::is_same_v<std::remove_cv_t<input_scalar_type>,
                       std::remove_cv_t<output_scalar_type>>,
        "NTT bitcast only supports lane reinterpretation of one scalar type");

    constexpr void operator()(const TIn &input, TOut &output) {
        constexpr auto input_lanes =
            element_scalar_count_v<input_element_type>;
        constexpr auto output_lanes =
            element_scalar_count_v<output_element_type>;
        const auto scalar_size = input.size() * input_lanes;

        for (dim_t linear = 0; linear < scalar_size; linear++) {
            const auto input_index =
                unravel_index(linear / input_lanes, input.shape());
            const auto output_index =
                unravel_index(linear / output_lanes, output.shape());
            auto &&input_value = input(input_index);
            auto &&output_value = output(output_index);
            bitcast_scalar_at(output_value, linear % output_lanes) =
                bitcast_scalar_at(input_value, linear % input_lanes);
        }
    }
};
} // namespace detail

template <Tensor TIn, class TOut>
constexpr void bitcast(const TIn &input, TOut &&output) {
    detail::bitcast_impl<TIn, std::decay_t<TOut>>()(input, output);
}
} // namespace nncase::ntt
