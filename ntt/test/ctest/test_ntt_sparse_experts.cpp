/* Copyright 2019-2026 Canaan Inc.
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
#include "ntt_test.h"
#include <gtest/gtest.h>
#include <nncase/ntt/ntt.h>

using namespace nncase;

TEST(SparseExpertsTest, PackedStagesMatchScalarStages) {
    constexpr size_t hidden_size = 4;
    constexpr size_t intermediate_size = 4;
    constexpr size_t num_experts = 2;
    constexpr size_t num_top_k = 2;
    constexpr size_t chunk_size = 1;
    constexpr size_t lanes = 2;

    auto q = ntt::make_tensor<float>(
        ntt::fixed_shape_v<chunk_size, hidden_size>);
    auto ids = ntt::make_tensor<int64_t>(
        ntt::fixed_shape_v<chunk_size, num_top_k>);
    auto probabilities = ntt::make_tensor<float>(
        ntt::fixed_shape_v<chunk_size, num_top_k>);
    auto scales =
        ntt::make_tensor<float>(ntt::fixed_shape_v<num_experts, 1>);
    auto gate_weights = ntt::make_tensor<float>(
        ntt::fixed_shape_v<num_experts, intermediate_size, hidden_size>);
    auto up_weights = ntt::make_tensor<float>(
        ntt::fixed_shape_v<num_experts, intermediate_size, hidden_size>);
    auto down_weights = ntt::make_tensor<float>(
        ntt::fixed_shape_v<num_experts, hidden_size, intermediate_size>);

    for (size_t hidden = 0; hidden < hidden_size; hidden++) {
        q(0, hidden) = static_cast<float>(hidden + 1) * 0.125f;
    }

    ids(0, 0) = 0;
    ids(0, 1) = 1;
    probabilities(0, 0) = 0.25f;
    probabilities(0, 1) = 0.75f;
    for (size_t expert = 0; expert < num_experts; expert++) {
        scales(expert, 0) = 1.f;
        for (size_t intermediate = 0; intermediate < intermediate_size;
             intermediate++) {
            for (size_t hidden = 0; hidden < hidden_size; hidden++) {
                gate_weights(expert, intermediate, hidden) =
                    0.03125f * static_cast<float>(1 + expert + intermediate + hidden);
                up_weights(expert, intermediate, hidden) =
                    0.0625f * static_cast<float>(1 + expert + intermediate + hidden);
                down_weights(expert, hidden, intermediate) =
                    0.015625f *
                    static_cast<float>(1 + expert + hidden + intermediate);
            }
        }
    }

    auto scalar_activations = ntt::make_tensor<float>(
        ntt::fixed_shape_v<chunk_size, num_top_k, intermediate_size>);
    auto scalar_output = ntt::make_tensor<float>(
        ntt::fixed_shape_v<chunk_size, hidden_size>);
    ntt::sparse_experts_gate_up(
        q, ids, scales, gate_weights, scales, scales, up_weights, scales,
        scalar_activations, hidden_size, intermediate_size, num_experts,
        num_top_k, chunk_size);
    ntt::sparse_experts_down(
        scalar_activations, ids, probabilities, scales, down_weights, scales,
        scalar_output, hidden_size, intermediate_size, num_experts, num_top_k,
        chunk_size);

    auto packed_q = ntt::make_tensor<ntt::vector<float, lanes>>(
        ntt::fixed_shape_v<chunk_size, hidden_size / lanes>);
    auto packed_activations =
        ntt::make_tensor<ntt::vector<float, lanes>>(
            ntt::fixed_shape_v<chunk_size, num_top_k,
                               intermediate_size / lanes>);
    auto packed_output = ntt::make_tensor<ntt::vector<float, lanes>>(
        ntt::fixed_shape_v<chunk_size, hidden_size / lanes>);
    ntt::pack(q, packed_q, ntt::fixed_shape_v<1>);
    ntt::sparse_experts_gate_up(
        packed_q, ids, scales, gate_weights, scales, scales, up_weights, scales,
        packed_activations, hidden_size, intermediate_size, num_experts,
        num_top_k, chunk_size);
    ntt::sparse_experts_down(
        packed_activations, ids, probabilities, scales, down_weights, scales,
        packed_output, hidden_size, intermediate_size, num_experts, num_top_k,
        chunk_size);

    auto unpacked_activations = ntt::make_tensor<float>(
        ntt::fixed_shape_v<chunk_size, num_top_k, intermediate_size>);
    auto unpacked_output = ntt::make_tensor<float>(
        ntt::fixed_shape_v<chunk_size, hidden_size>);
    ntt::unpack(packed_activations, unpacked_activations,
                ntt::fixed_shape_v<2>);
    ntt::unpack(packed_output, unpacked_output, ntt::fixed_shape_v<1>);

    for (size_t top_k = 0; top_k < num_top_k; top_k++) {
        for (size_t intermediate = 0; intermediate < intermediate_size;
             intermediate++) {
            EXPECT_NEAR(scalar_activations(0, top_k, intermediate),
                        unpacked_activations(0, top_k, intermediate), 1e-6f);
        }
    }

    for (size_t hidden = 0; hidden < hidden_size; hidden++) {
        EXPECT_NEAR(scalar_output(0, hidden), unpacked_output(0, hidden),
                    1e-6f);
    }
}
