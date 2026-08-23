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
#include "../shape.h"
#include "../tensor.h"
#include "../tensor_traits.h"
#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>
#include "../caching.h"
#include "binary.h"
#include "matmul.h"
#include "nncase/ntt/dimension.h"
#include "nncase/ntt/shape.h"
#include "nncase/ntt/tensor.h"
#include "nncase/ntt/tensor_traits.h"
#include "reduce.h"
#include "unary.h"
#include <type_traits>

// This version follows the Evaluator logic in src/Nncase.Evaluator/NN/SparseExperts.cs
//
// Shapes:
//  input q:                     [seq_len, hidden_size]
//  router_expert_ids:           [seq_len, num_top_k] - already selected topk expert indices
//  router_expert_weights:       [seq_len, num_top_k] - router weights for selected experts
//  moeExpertGateProjW:          [num_expert, moe_intermediate_size, hidden_size]
//  moeExpertGateProjScale:      [num_expert, moe_intermediate_size, 1] or [num_expert, 1]
//  moeExpertGateInputScale:     [num_expert, 1] (optional, can be empty/null)
//  moeExpertUpProjW:            [num_expert, moe_intermediate_size, hidden_size]
//  moeExpertUpProjScale:        [num_expert, moe_intermediate_size, 1] or [num_expert, 1]
//  moeExpertUpInputScale:       [num_expert, 1] (optional, can be empty/null)
//  moeExpertDownProjW:          [num_expert, hidden_size, moe_intermediate_size]
//  moeExpertDownProjScale:      [num_expert, hidden_size, 1] or [num_expert, 1]
//  moeExpertDownInputScale:     [num_expert, 1] (optional, can be empty/null)
//  output:                      [seq_len, hidden_size]
//
// Processing: Loop over experts (not tokens), gather tokens assigned to each expert.

namespace nncase::ntt {

namespace sparse_experts_detail {

template <Tensor T, Dimension... TPrefix>
NTT_ALWAYS_INLINE constexpr auto
read_last_axis_scalar(const T &tensor, size_t logical_index,
                      const TPrefix &...prefix) noexcept {
    using element_type = typename T::element_type;
    if constexpr (Vector<element_type>) {
        constexpr auto lane_count = element_scalar_count_v<element_type>;
        const auto value = tensor(prefix..., logical_index / lane_count);
        return unwrap_proxy(value(unravel_index(logical_index % lane_count,
                                                element_type::shape())));
    } else {
        return unwrap_proxy(tensor(prefix..., logical_index));
    }
}

} // namespace sparse_experts_detail

template <Tensor TQ, Tensor TRouterIds, Tensor TGateInputScale,
          Tensor TGateProjW, Tensor TGateProjScale, Tensor TUpInputScale,
          Tensor TUpProjW, Tensor TUpProjScale, class TOut>
void sparse_experts_gate_up(
    const TQ &q, const TRouterIds &topk_indices,
    const TGateInputScale &gate_input_scales,
    const TGateProjW &gate_weights, const TGateProjScale &gate_scales,
    const TUpInputScale &up_input_scales, const TUpProjW &up_weights,
    const TUpProjScale &up_scales, TOut &&output, size_t /* hidden_size */,
    size_t /* moe_intermediate_size */, size_t /* num_expert */,
    size_t /* num_top_k */, size_t /* chunk_size */) noexcept {
    using output_type = typename std::remove_reference_t<TOut>::element_type;
    using output_scalar_type = element_or_scalar_t<output_type>;
    using input_type = typename TQ::element_type;
    constexpr auto input_lane_count = element_scalar_count_v<input_type>;
    constexpr auto output_lane_count = element_scalar_count_v<output_type>;
    const auto tokens = q.shape()[0_dim];
    const auto hidden = q.shape()[1_dim] * input_lane_count;
    const auto top_k = topk_indices.shape()[1_dim];
    for (size_t token = 0; token < tokens; token++) {
        for (size_t topk_index = 0; topk_index < top_k; topk_index++) {
            const auto expert = static_cast<size_t>(topk_indices(token, topk_index));
            const auto gate_input_scale = static_cast<float>(gate_input_scales(expert, 0));
            const auto gate_scale = static_cast<float>(gate_scales(expert, 0));
            const auto up_input_scale = static_cast<float>(up_input_scales(expert, 0));
            const auto up_scale = static_cast<float>(up_scales(expert, 0));
            const auto compute = [&](size_t logical_intermediate) {
                float gate = 0.f;
                float up = 0.f;
                for (size_t h = 0; h < hidden; h++) {
                    const auto input = static_cast<float>(
                        sparse_experts_detail::read_last_axis_scalar(q, h,
                                                                    token));
                    gate +=
                        (input / gate_input_scale) * static_cast<float>(
                            gate_weights(expert, logical_intermediate, h));
                    up += (input / up_input_scale) * static_cast<float>(
                        up_weights(expert, logical_intermediate, h));
                }

                gate *= gate_input_scale * gate_scale;
                up *= up_input_scale * up_scale;
                return (gate * sigmoid(gate)) * up;
            };
            for (size_t d = 0; d < output.shape()[2_dim]; d++) {
                if constexpr (Vector<output_type>) {
                    output_type values{};
                    for (size_t lane = 0; lane < output_lane_count; lane++) {
                        const auto lane_index =
                            unravel_index(lane, output_type::shape());
                        values(lane_index) = static_cast<output_scalar_type>(
                            compute(d * output_lane_count + lane));
                    }

                    output(token, topk_index, d) = values;
                } else {
                    output(token, topk_index, d) =
                        static_cast<output_type>(compute(d));
                }
            }
        }
    }
}

template <Tensor TActivations, Tensor TRouterIds, Tensor TRouterWeights,
          Tensor TDownInputScale, Tensor TDownProjW, Tensor TDownProjScale,
          class TOut>
void sparse_experts_down(
    const TActivations &activations, const TRouterIds &topk_indices,
    const TRouterWeights &topk_probs,
    const TDownInputScale &down_input_scales,
    const TDownProjW &down_weights, const TDownProjScale &down_scales,
    TOut &&output, size_t /* hidden_size */,
    size_t /* moe_intermediate_size */, size_t /* num_expert */,
    size_t /* num_top_k */, size_t /* chunk_size */) noexcept {
    using output_type = typename std::remove_reference_t<TOut>::element_type;
    using output_scalar_type = element_or_scalar_t<output_type>;
    using activation_type = typename TActivations::element_type;
    constexpr auto activation_lane_count =
        element_scalar_count_v<activation_type>;
    constexpr auto output_lane_count = element_scalar_count_v<output_type>;
    const auto tokens = activations.shape()[0_dim];
    const auto top_k = activations.shape()[1_dim];
    const auto intermediate =
        activations.shape()[2_dim] * activation_lane_count;
    for (size_t token = 0; token < tokens; token++) {
        const auto compute = [&](size_t logical_hidden) {
            auto result = 0.f;
            for (size_t topk_index = 0; topk_index < top_k; topk_index++) {
                const auto expert = static_cast<size_t>(topk_indices(token, topk_index));
                const auto input_scale = static_cast<float>(down_input_scales(expert, 0));
                const auto down_scale = static_cast<float>(down_scales(expert, 0));
                float expert_result = 0.f;
                for (size_t d = 0; d < intermediate; d++) {
                    const auto activation = static_cast<float>(
                        sparse_experts_detail::read_last_axis_scalar(
                            activations, d, token, topk_index));
                    expert_result += (activation / input_scale) *
                                     static_cast<float>(down_weights(
                                         expert, logical_hidden, d));
                }

                result += static_cast<float>(topk_probs(token, topk_index)) *
                          expert_result * input_scale * down_scale;
            }

            return result;
        };
        for (size_t h = 0; h < output.shape()[1_dim]; h++) {
            if constexpr (Vector<output_type>) {
                output_type values{};
                for (size_t lane = 0; lane < output_lane_count; lane++) {
                    const auto lane_index =
                        unravel_index(lane, output_type::shape());
                    values(lane_index) = static_cast<output_scalar_type>(
                        compute(h * output_lane_count + lane));
                }

                output(token, h) = values;
            } else {
                output(token, h) = static_cast<output_type>(compute(h));
            }
        }
    }
}

namespace detail {

template <Tensor TQ, Tensor TRouterIds, Tensor TRouterWeights,
          Tensor TGateInputScale, Tensor TGateProjW, Tensor TGateProjScale,
          Tensor TUpInputScale, Tensor TUpProjW, Tensor TUpProjScale,
          Tensor TDownInputScale, Tensor TDownProjW, Tensor TDownProjScale,
          class TOut>
void sparse_experts_impl(const TQ &q,
                         const TRouterIds &topk_indices,
                         const TRouterWeights &topk_probs,
                         const TGateInputScale &moeExpertGateInputScale,
                         const TGateProjW &moeExpertGateProjW,
                         const TGateProjScale &moeExpertGateProjScale,
                         const TDownInputScale &moeExpertDownInputScale,
                         const TDownProjW &moeExpertDownProjW,
                         const TDownProjScale &moeExpertDownProjScale,
                         const TUpInputScale &moeExpertUpInputScale,
                         const TUpProjW &moeExpertUpProjW,
                         const TUpProjScale &moeExpertUpProjScale,
                         size_t /* hidden_size */,
                         size_t /* moe_intermediate_size */,
                         size_t /* num_expert */,
                         size_t /* num_top_k */,
                         size_t /* chunk_size */,
                         TOut &output) {
    using ElemType = typename TQ::element_type;
    const auto seq_len = q.shape()[0_dim];
    const auto hidden_size = q.shape()[1_dim];
    const auto moe_intermediate_size = moeExpertGateProjW.shape()[1_dim];
    const auto output_hidden_size = output.shape()[1_dim];
    const auto num_top_k = topk_indices.shape()[1_dim];

    // Initialize output to zero
    for (size_t i = 0; i < seq_len; i++)
        for (size_t h = 0; h < output_hidden_size; h++) output(i, h) = (ElemType)0;

    // For each token, accumulate expert contributions.
    for (size_t i = 0; i < seq_len; i++) {
        // take input vector
        // For each top expert
        for (size_t k = 0; k < num_top_k; k++) {
            int32_t expert = topk_indices(i, k);
            if (expert < 0) continue;
            auto prob = topk_probs(i, k);
            // --- MLP ---
            // gate/up: [moe_intermediate_size, hidden_size]
            // down:    [hidden_size, moe_intermediate_size]
            // scales: match output dim of corresponding matmul

            // gate
            std::vector<float> gate(moe_intermediate_size);
            constexpr bool gate_scale_is_2d = moeExpertGateProjScale.shape().rank() == 2;
            float gate_scale_val = 1.f;
            if constexpr(gate_scale_is_2d)
            {
                gate_scale_val = (float)moeExpertGateProjScale(expert, 0);
            }
            
            // Check if gate input scale exists (not empty)
            constexpr bool gate_input_scale_is_2d = moeExpertGateInputScale.rank() == 2;
            float gate_input_scale_val = 1.f;
            if constexpr(gate_input_scale_is_2d)
            {
                gate_input_scale_val = (float)moeExpertGateInputScale(expert, 0);
            }
            
            for (size_t d = 0; d < moe_intermediate_size; d++) {
                float acc = 0.f;
                for (size_t h = 0; h < hidden_size; h++) {
                    auto input_val = (float)q(i, h);
                    // Apply input scaling if available

                    if constexpr(gate_input_scale_is_2d) {
                        input_val /= gate_input_scale_val;
                    } else {
                        input_val /= (float)moeExpertGateInputScale(expert, h, 0);
                    }

                    acc += input_val * (float)moeExpertGateProjW(expert, d, h);
                }
                if constexpr(gate_scale_is_2d)
                {
                    acc *= (gate_scale_val * gate_input_scale_val);
                }
                else
                {
                    acc *= ((float)moeExpertGateProjScale(expert, d, 0) * gate_input_scale_val);
                }

                // silu
                auto sig = sigmoid(acc);
                gate[d] = sig * acc; // silu(x) = sigmoid(x) * x
            }
            // up
            std::vector<float> up(moe_intermediate_size);
            // Check if up proj scale is 2D [num_expert, 1] or 3D [num_expert, moe_intermediate_size, 1]
            constexpr bool up_scale_is_2d = (moeExpertUpProjScale.rank() == 2);

            float up_scale_val = 1.f;
            if constexpr(up_scale_is_2d)
            {
                up_scale_val = (float)moeExpertUpProjScale(expert, 0);
            }

            constexpr bool up_input_scale_is_2d = moeExpertUpInputScale.rank() == 2;
            float up_input_scale_val = 1.f;
            if constexpr(up_input_scale_is_2d)
            {
                up_input_scale_val = (float)moeExpertUpInputScale(expert, 0);
            }

            for (size_t d = 0; d < moe_intermediate_size; d++) {
                float acc = 0.f;
                for (size_t h = 0; h < hidden_size; h++) {
                    float input_val = (float)q(i, h);
                    if constexpr(up_input_scale_is_2d) {
                        input_val /= up_input_scale_val;
                    } else {
                        input_val /= (float)moeExpertUpInputScale(expert, h, 0);
                    }

                    acc += input_val * (float)moeExpertUpProjW(expert, d, h);
                }
                if constexpr(up_scale_is_2d)
                {
                    acc *= (up_scale_val * up_input_scale_val);
                }
                else
                {
                    acc *= ((float)moeExpertUpProjScale(expert, d, 0) * up_input_scale_val);
                }

                up[d] = acc;
            }
            // down input = gate * up (elementwise)
            // down: (gate*up)[moe_intermediate_size] @ downW[hidden_size, moe_intermediate_size]
            // Check if down proj scale is 2D [num_expert, 1] or 3D [num_expert, hidden_size, 1]
            constexpr bool down_scale_is_2d = (moeExpertDownProjScale.rank() == 2);
            float down_scale_val = 1.f;
            if constexpr(down_scale_is_2d)
            {
                down_scale_val = (float)moeExpertDownProjScale(expert, 0);
            }
            // Check if down input scale exists (not empty)
            constexpr bool down_input_scale_is_2d = moeExpertDownInputScale.rank() == 2;
            float down_input_scale_val = 1.f;
            if constexpr(down_input_scale_is_2d)
            {
                down_input_scale_val = (float)moeExpertDownInputScale(expert, 0);
            }
            
            for (size_t h = 0; h < output_hidden_size; h++) {
                float acc = 0.f;
                for (size_t d = 0; d < moe_intermediate_size; d++) {
                    float down_in = gate[d] * up[d];
                    // Apply input scaling if available
                    if constexpr(down_input_scale_is_2d) {
                        down_in /= down_input_scale_val;
                    } else {
                        down_in /= (float)moeExpertDownInputScale(expert, d, 0);
                    }

                    acc += down_in * (float)moeExpertDownProjW(expert, h, d);
                }
                if constexpr(down_scale_is_2d)
                {
                    acc *= (down_scale_val * down_input_scale_val);
                }
                else
                {
                    acc *= ((float)moeExpertDownProjScale(expert, h, 0) * down_input_scale_val);
                }
                output(i, h) += (ElemType)(prob * acc); // accumulate
            }
        }
    }
}

} // namespace detail

template <Tensor TQ, Tensor TRouterIds, Tensor TRouterWeights,
          Tensor TGateInputScale, Tensor TGateProjW, Tensor TGateProjScale,
          Tensor TDownInputScale, Tensor TDownProjW, Tensor TDownProjScale,
          Tensor TUpInputScale, Tensor TUpProjW, Tensor TUpProjScale,
          class TOut>
void sparse_experts(const TQ &q,
                   const TRouterIds &router_expert_ids,
                   const TRouterWeights &router_expert_weights,
                   const TGateInputScale &moeExpertGateInputScale,
                   const TGateProjW &moeExpertGateProjW,
                   const TGateProjScale &moeExpertGateProjScale,
                   const TDownInputScale &moeExpertDownInputScale,
                   const TDownProjW &moeExpertDownProjW,
                   const TDownProjScale &moeExpertDownProjScale,
                   const TUpInputScale &moeExpertUpInputScale,
                   const TUpProjW &moeExpertUpProjW,
                   const TUpProjScale &moeExpertUpProjScale,
                   TOut &&output,
                   size_t hidden_size,
                   size_t moe_intermediate_size,
                   size_t num_expert,
                   size_t num_top_k,
                   size_t chunk_size) noexcept {
    detail::sparse_experts_impl(q, router_expert_ids, router_expert_weights,
                           moeExpertGateInputScale, moeExpertGateProjW, moeExpertGateProjScale,
                           moeExpertDownInputScale, moeExpertDownProjW, moeExpertDownProjScale,
                           moeExpertUpInputScale, moeExpertUpProjW, moeExpertUpProjScale,
                           hidden_size, moe_intermediate_size,
                           num_expert, num_top_k, chunk_size, output);

}

} // namespace nncase::ntt
