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

#include "../caching.h"
#include "../primitive_ops.h"
#include "../tensor.h"
#include "../tensor_traits.h"
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>

namespace nncase::ntt {
namespace qkv_rope_with_cache_detail {

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
        using mesh_t = typename T::mesh_type;
        return tensor.sharding().global_offset(
            tensor.shape(), mesh_t::local_index());
    } else {
        return make_zeros_shape<T::rank()>();
    }
}

template <class TTensor, FixedDimensions TLayout>
constexpr float read_scalar(const TTensor &tensor, const TLayout &layout,
                            dim_t seq, dim_t head, dim_t scalar_dim) noexcept {
    constexpr auto seq_axis = TLayout{}.index_of(
        fixed_dim_v<(dim_t)caching::attention_dim_kind::seq>);
    constexpr auto head_axis = TLayout{}.index_of(
        fixed_dim_v<(dim_t)caching::attention_dim_kind::head>);
    constexpr auto dim_axis = TLayout{}.index_of(
        fixed_dim_v<(dim_t)caching::attention_dim_kind::dim>);
    const auto &local = local_tensor(tensor);
    const auto offset = global_offset(tensor);
    constexpr size_t lanes =
        element_scalar_count_v<typename std::decay_t<decltype(local)>::element_type>;
    const dim_t physical_dim = scalar_dim / (dim_t)lanes;
    const auto index = make_zeros_shape<3>()
                           .template replace_at<seq_axis>(
                               local.shape()[seq_axis] == 1_dim
                                   ? 0_dim
                                   : seq - offset[seq_axis])
                           .template replace_at<head_axis>(
                               local.shape()[head_axis] == 1_dim
                                   ? 0_dim
                                   : head - offset[head_axis])
                           .template replace_at<dim_axis>(physical_dim -
                                                          offset[dim_axis]);
    const auto &value = local(index);
    return ntt::cast_elem<float>(scalar_at(value, (size_t)(scalar_dim % lanes)));
}

template <class TTensor>
constexpr float read_norm_parameter(const TTensor &tensor,
                                    dim_t scalar_dim) noexcept {
    const auto &local = local_tensor(tensor);
    const auto offset = global_offset(tensor);
    static_assert(std::decay_t<decltype(local)>::rank() == 1,
                  "QKVRoPEWithCache norm parameters must have rank one");
    constexpr size_t lanes =
        element_scalar_count_v<typename std::decay_t<decltype(local)>::element_type>;
    const dim_t physical_dim = scalar_dim / (dim_t)lanes;
    const auto &value = local(physical_dim - offset[0_dim]);
    return ntt::cast_elem<float>(scalar_at(value, (size_t)(scalar_dim % lanes)));
}

template <class TTensor, FixedDimensions TLayout>
constexpr void write_scalar(TTensor &tensor, const TLayout &layout, dim_t seq,
                            dim_t head, dim_t scalar_dim,
                            float value) noexcept {
    constexpr auto seq_axis = TLayout{}.index_of(
        fixed_dim_v<(dim_t)caching::attention_dim_kind::seq>);
    constexpr auto head_axis = TLayout{}.index_of(
        fixed_dim_v<(dim_t)caching::attention_dim_kind::head>);
    constexpr auto dim_axis = TLayout{}.index_of(
        fixed_dim_v<(dim_t)caching::attention_dim_kind::dim>);
    auto &local = local_tensor(tensor);
    const auto offset = global_offset(tensor);
    using element_t = typename std::decay_t<decltype(local)>::element_type;
    using scalar_t = element_or_scalar_t<element_t>;
    constexpr size_t lanes = element_scalar_count_v<element_t>;
    const dim_t physical_dim = scalar_dim / (dim_t)lanes;
    const auto index = make_zeros_shape<3>()
                           .template replace_at<seq_axis>(seq -
                                                          offset[seq_axis])
                           .template replace_at<head_axis>(head -
                                                           offset[head_axis])
                           .template replace_at<dim_axis>(physical_dim -
                                                          offset[dim_axis]);
    auto &packed = local(index);
    scalar_at(packed, (size_t)(scalar_dim % lanes)) =
        ntt::cast_elem<scalar_t>(value);
}

template <bool UseMean, class TInput, class TScale, class TBias,
          FixedDimensions TLayout>
constexpr float normalized_scalar(const TInput &input, const TScale &scale,
                                  const TBias &bias, const TLayout &layout,
                                  dim_t seq, dim_t head, dim_t scalar_dim,
                                  float mean, float inv_std) noexcept {
    const float value = read_scalar(input, layout, seq, head, scalar_dim);
    const float scale_value = read_norm_parameter(scale, scalar_dim);
    const float bias_value = read_norm_parameter(bias, scalar_dim);
    return ((value - (UseMean ? mean : 0.0f)) * inv_std * scale_value) +
           bias_value;
}

template <bool UseMean, class TInput, FixedDimensions TLayout>
constexpr auto norm_statistics(const TInput &input, const TLayout &layout,
                               dim_t seq, dim_t head,
                               dim_t head_dim) noexcept {
    float sum = 0.0f;
    float square_sum = 0.0f;
    for (dim_t dim = 0; dim < head_dim; ++dim) {
        const float value = read_scalar(input, layout, seq, head, dim);
        if constexpr (UseMean) {
            sum += value;
        }
        square_sum += value * value;
    }

    const float reciprocal = 1.0f / (float)head_dim;
    const float mean = UseMean ? sum * reciprocal : 0.0f;
    const float variance =
        std::max(square_sum * reciprocal - mean * mean, 0.0f);
    return std::pair{mean, variance};
}

template <class TCache, class TSlots>
constexpr dim_t cache_local_head(const TSlots &slots, dim_t global_head) noexcept {
    using cache_t = std::decay_t<TCache>;
    using config_t = typename cache_t::config_t;
    if constexpr (ShardedTensor<TSlots>) {
        using mesh_t = typename TSlots::mesh_type;
        const auto shard_index = mesh_t::local_index();
        const auto policy = config_t::template axis_policy<
            caching::paged_kvcache_dim_kind::num_kv_heads>();
        const auto local_head_dim = policy.template shard_dim<mesh_t>(
            config_t::num_kv_heads, shard_index);
        return global_head % local_head_dim;
    } else {
        return global_head;
    }
}

template <caching::attention_cache_kind Kind, class TSlot>
constexpr void write_cache_scalar(TSlot &slot, dim_t scalar_dim,
                                  float value) noexcept {
    using element_t = typename std::decay_t<TSlot>::element_type;
    using scalar_t = element_or_scalar_t<element_t>;
    constexpr size_t lanes = element_scalar_count_v<element_t>;
    auto &packed = slot(scalar_dim / (dim_t)lanes);
    scalar_at(packed, (size_t)(scalar_dim % lanes)) =
        ntt::cast_elem<scalar_t>(value);
}

} // namespace qkv_rope_with_cache_detail

template <bool QUseMean, bool KUseMean, class TQ, class TK, class TV,
          class TQScale, class TKScale, class TQBias, class TKBias, class TCos,
          class TSin, Tensor TKVCache, class TQOutput,
          FixedDimensions TQKVLayout, FixedDimensions TAttentionLayout>
constexpr void qkv_rope_with_cache(
    const TQ &q, const TK &k, const TV &v, const TQScale &q_scale,
    const TKScale &k_scale, const TQBias &q_bias, const TKBias &k_bias,
    const TCos &cos, const TSin &sin, TKVCache &kv_cache_tensor,
    dim_t layer_id, TQOutput &q_output, float q_epsilon, float k_epsilon,
    const TQKVLayout &qkv_layout,
    const TAttentionLayout &attention_layout) noexcept {
    using namespace qkv_rope_with_cache_detail;
    constexpr auto seq_axis = TQKVLayout{}.index_of(
        fixed_dim_v<(dim_t)caching::attention_dim_kind::seq>);
    constexpr auto head_axis = TQKVLayout{}.index_of(
        fixed_dim_v<(dim_t)caching::attention_dim_kind::head>);
    constexpr auto dim_axis = TQKVLayout{}.index_of(
        fixed_dim_v<(dim_t)caching::attention_dim_kind::dim>);

    const auto &q_local = local_tensor(q);
    const auto q_offset = global_offset(q);
    constexpr size_t q_lanes =
        element_scalar_count_v<typename std::decay_t<decltype(q_local)>::element_type>;
    const dim_t q_head_dim = q.shape()[dim_axis] * (dim_t)q_lanes;
    const auto &cos_local = local_tensor(cos);
    constexpr size_t cos_lanes =
        element_scalar_count_v<typename std::decay_t<decltype(cos_local)>::element_type>;
    const dim_t rotary_extent = cos.shape()[dim_axis] * (dim_t)cos_lanes;
    const dim_t rotary_half = rotary_extent / 2_dim;

    for (dim_t local_seq = 0; local_seq < q_local.shape()[seq_axis];
         ++local_seq) {
        const dim_t seq = q_offset[seq_axis] + local_seq;
        for (dim_t local_head = 0; local_head < q_local.shape()[head_axis];
             ++local_head) {
            const dim_t head = q_offset[head_axis] + local_head;
            const auto [mean, variance] =
                norm_statistics<QUseMean>(q, qkv_layout, seq, head,
                                           q_head_dim);
            const float inv_std = 1.0f / std::sqrt(variance + q_epsilon);
            for (dim_t dim = 0; dim < q_head_dim; ++dim) {
                const float normalized = normalized_scalar<QUseMean>(
                    q, q_scale, q_bias, qkv_layout, seq, head, dim, mean,
                    inv_std);
                float result = normalized;
                if (dim < rotary_extent) {
                    const dim_t paired_dim =
                        dim < rotary_half ? dim + rotary_half
                                          : dim - rotary_half;
                    const float paired = normalized_scalar<QUseMean>(
                        q, q_scale, q_bias, qkv_layout, seq, head, paired_dim,
                        mean, inv_std);
                    const float cos_value = read_scalar(
                        cos, qkv_layout, seq, 0_dim, dim);
                    const float sin_value = read_scalar(
                        sin, qkv_layout, seq, 0_dim, dim);
                    result = normalized * cos_value +
                             (dim < rotary_half ? -paired : paired) * sin_value;
                }

                write_scalar(q_output, attention_layout, seq, head, dim,
                             result);
            }
        }
    }

    auto &kv_cache = kv_cache_tensor(fixed_shape_v<>);
    const auto &k_local = local_tensor(k);
    const auto k_offset = global_offset(k);
    constexpr size_t k_lanes =
        element_scalar_count_v<typename std::decay_t<decltype(k_local)>::element_type>;
    const dim_t k_head_dim = k.shape()[dim_axis] * (dim_t)k_lanes;
    for (dim_t local_seq = 0; local_seq < k_local.shape()[seq_axis];
         ++local_seq) {
        const dim_t seq = k_offset[seq_axis] + local_seq;
        if ((size_t)seq >= kv_cache.num_tokens()) {
            continue;
        }

        const auto slot_id = kv_cache.get_slot_id(seq);
        for (dim_t local_head = 0; local_head < k_local.shape()[head_axis];
             ++local_head) {
            const dim_t head = k_offset[head_axis] + local_head;
            const dim_t cache_head = cache_local_head<decltype(kv_cache)>(k, head);
            auto cache_slot =
                kv_cache.template get_slot<caching::attention_cache_kind::key>(
                    layer_id, cache_head, slot_id);
            const auto [mean, variance] =
                norm_statistics<KUseMean>(k, qkv_layout, seq, head,
                                           k_head_dim);
            const float inv_std = 1.0f / std::sqrt(variance + k_epsilon);
            for (dim_t dim = 0; dim < k_head_dim; ++dim) {
                const float normalized = normalized_scalar<KUseMean>(
                    k, k_scale, k_bias, qkv_layout, seq, head, dim, mean,
                    inv_std);
                float result = normalized;
                if (dim < rotary_extent) {
                    const dim_t paired_dim =
                        dim < rotary_half ? dim + rotary_half
                                          : dim - rotary_half;
                    const float paired = normalized_scalar<KUseMean>(
                        k, k_scale, k_bias, qkv_layout, seq, head, paired_dim,
                        mean, inv_std);
                    const float cos_value = read_scalar(
                        cos, qkv_layout, seq, 0_dim, dim);
                    const float sin_value = read_scalar(
                        sin, qkv_layout, seq, 0_dim, dim);
                    result = normalized * cos_value +
                             (dim < rotary_half ? -paired : paired) * sin_value;
                }

                write_cache_scalar<caching::attention_cache_kind::key>(
                    cache_slot, dim, result);
            }
        }
    }

    const auto &v_local = local_tensor(v);
    const auto v_offset = global_offset(v);
    constexpr size_t v_lanes =
        element_scalar_count_v<typename std::decay_t<decltype(v_local)>::element_type>;
    const dim_t v_head_dim = v.shape()[dim_axis] * (dim_t)v_lanes;
    for (dim_t local_seq = 0; local_seq < v_local.shape()[seq_axis];
         ++local_seq) {
        const dim_t seq = v_offset[seq_axis] + local_seq;
        if ((size_t)seq >= kv_cache.num_tokens()) {
            continue;
        }

        const auto slot_id = kv_cache.get_slot_id(seq);
        for (dim_t local_head = 0; local_head < v_local.shape()[head_axis];
             ++local_head) {
            const dim_t head = v_offset[head_axis] + local_head;
            const dim_t cache_head = cache_local_head<decltype(kv_cache)>(v, head);
            auto cache_slot =
                kv_cache.template get_slot<caching::attention_cache_kind::value>(
                    layer_id, cache_head, slot_id);
            for (dim_t dim = 0; dim < v_head_dim; ++dim) {
                write_cache_scalar<caching::attention_cache_kind::value>(
                    cache_slot, dim,
                    read_scalar(v, qkv_layout, seq, head, dim));
            }
        }
    }
}

} // namespace nncase::ntt
