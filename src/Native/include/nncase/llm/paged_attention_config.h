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
#include "attention_config.h"
#include "nncase/runtime/datatypes.h"
#include "paged_attention_enums.h"
#include <nncase/object.h>
#include <vector>

namespace nncase::llm {
using paged_kvcache_axes_t = itlib::small_vector<paged_kvcache_dim_kind, 8>;

class paged_attention_config_node : public attention_config_node {
    DEFINE_OBJECT_KIND(attention_config_node, object_paged_attention_config);

  public:
    paged_attention_config_node(
        size_t num_layers, size_t num_kv_heads, size_t head_dim,
        typecode_t kv_type, size_t block_size,
        const std::array<paged_kvcache_dim_kind, 6> &key_cache_layout,
        const std::array<paged_kvcache_dim_kind, 6> &value_cache_layout,
        const std::vector<paged_kvcache_dim_kind> &key_vectorized_axes,
        const std::vector<paged_kvcache_dim_kind> &value_vectorized_axes,
        const dims_t &key_lanes, const dims_t &value_lanes,
        const std::vector<paged_kvcache_dim_kind> &sharding_axes,
        const std::vector<dims_t> &axis_policies) noexcept
        : attention_config_node(num_layers, num_kv_heads, head_dim, kv_type),
          block_size_(block_size),
          key_cache_layout_(key_cache_layout),
          value_cache_layout_(value_cache_layout),
          key_vectorized_axes_(key_vectorized_axes.begin(),
                               key_vectorized_axes.end()),
          value_vectorized_axes_(value_vectorized_axes.begin(),
                                 value_vectorized_axes.end()),
          key_lanes_(key_lanes),
          value_lanes_(value_lanes),
          sharding_axes_(sharding_axes.begin(), sharding_axes.end()),
          axis_policies_(axis_policies.begin(), axis_policies.end()) {}

    size_t block_size() const noexcept { return block_size_; }

    void block_size(size_t block_size) noexcept { block_size_ = block_size; }

    const std::array<paged_kvcache_dim_kind, 6> &
    key_cache_layout() const noexcept {
        return key_cache_layout_;
    }

    void key_cache_layout(
        const std::array<paged_kvcache_dim_kind, 6> &cache_layout) noexcept {
        key_cache_layout_ = cache_layout;
    }

    const std::array<paged_kvcache_dim_kind, 6> &
    value_cache_layout() const noexcept {
        return value_cache_layout_;
    }

    void value_cache_layout(
        const std::array<paged_kvcache_dim_kind, 6> &cache_layout) noexcept {
        value_cache_layout_ = cache_layout;
    }

    const std::array<paged_kvcache_dim_kind, 2>
    key_block_layout() const noexcept {
        return make_block_layout(key_cache_layout_);
    }

    const std::array<paged_kvcache_dim_kind, 2>
    value_block_layout() const noexcept {
        return make_block_layout(value_cache_layout_);
    }

    const auto &key_vectorized_axes() const noexcept {
        return key_vectorized_axes_;
    }

    void key_vectorized_axes(
        const std::vector<paged_kvcache_dim_kind> &vectorized_axes) noexcept {
        key_vectorized_axes_.clear();
        key_vectorized_axes_.assign(vectorized_axes.begin(),
                                    vectorized_axes.end());
    }

    const auto &value_vectorized_axes() const noexcept {
        return value_vectorized_axes_;
    }

    void value_vectorized_axes(
        const std::vector<paged_kvcache_dim_kind> &vectorized_axes) noexcept {
        value_vectorized_axes_.clear();
        value_vectorized_axes_.assign(vectorized_axes.begin(),
                                      vectorized_axes.end());
    }

    const dims_t &key_lanes() const noexcept { return key_lanes_; }

    void key_lanes(const std::vector<int> &lanes) noexcept {
        key_lanes_.clear();
        key_lanes_.assign(lanes.begin(), lanes.end());
    }

    const dims_t &value_lanes() const noexcept { return value_lanes_; }

    void value_lanes(const std::vector<int> &lanes) noexcept {
        value_lanes_.clear();
        value_lanes_.assign(lanes.begin(), lanes.end());
    }

    const auto &sharding_axes() const noexcept { return sharding_axes_; }

    void sharding_axes(
        const std::vector<paged_kvcache_dim_kind> &sharding_axes) noexcept {
        sharding_axes_.clear();
        sharding_axes_.assign(sharding_axes.begin(), sharding_axes.end());
    }

    const auto &axis_policies() const noexcept { return axis_policies_; }

    void axis_policies(const std::vector<dims_t> &axis_policies) noexcept {
        axis_policies_.clear();
        axis_policies_.assign(axis_policies.begin(), axis_policies.end());
    }

    void axis_policies(int32_t i, const dims_t axis_policy) noexcept {
        axis_policies_[i] = axis_policy;
    }

    datatype_t key_type() const noexcept {
        return make_kv_type(key_lanes_);
    }

    datatype_t value_type() const noexcept {
        return make_kv_type(value_lanes_);
    }

    std::vector<size_t>
    get_default_dimensions(size_t num_blocks) const noexcept {
        return {num_blocks,  num_layers(),   2,
                block_size_, num_kv_heads(), head_dim()};
    }

    std::vector<size_t>
    get_dimensions(size_t num_blocks, attention_cache_kind kind) const
        noexcept {
        auto default_dims = get_default_dimensions(num_blocks);
        const auto &layout = cache_layout(kind);
        std::vector<size_t> dims;
        dims.reserve(layout.size());
        for (auto item : layout) {
            dims.push_back(default_dims[static_cast<size_t>(item)]);
        }
        return dims;
    }

    dims_t get_block_table_dimensions(size_t num_seqs,
                                      size_t max_seq_len) const noexcept {
        size_t blocks_per_seq =
            (max_seq_len + block_size_ - 1) / block_size_; // ceil division
        return {num_seqs, blocks_per_seq, sharding_axes_.size() + 1};
    }

    dims_t get_slot_mapping_dimensions(size_t num_tokens) const noexcept {
        return {num_tokens, sharding_axes_.size() + 1};
    }

    dims_t get_logical_shard_dimensions(size_t num_blocks, dims_t hierarchy,
                                        attention_cache_kind kind) const
        noexcept {
        auto dims = get_default_dimensions(num_blocks);

        // 1. process vectorized axes
        const auto &axes = vectorized_axes(kind);
        const auto &lane_values = lanes(kind);
        for (size_t i = 0; i < axes.size() && i < lane_values.size(); i++) {
            auto axis = static_cast<size_t>(axes[i]);
            dims[axis] /= lane_values[i];
        }

        // 2. process sharding axes
        std::vector<size_t> sharding_dims(sharding_axes_.size(), 1);
        for (size_t i = 0; i < sharding_axes_.size(); i++) {
            auto axis = static_cast<size_t>(sharding_axes_[i]);
            const auto &policy = axis_policies_[i];
            for (size_t j = 0; j < policy.size(); j++) {
                dims[axis] /= hierarchy[policy[j]];
                sharding_dims[i] *= hierarchy[policy[j]];
            }
        }

        // 3. reorder dims according to cache layout
        const auto &layout = cache_layout(kind);
        std::vector<size_t> cache_dims;
        cache_dims.reserve(layout.size());
        for (auto item : layout) {
            cache_dims.push_back(dims[static_cast<size_t>(item)]);
        }

        // 4. concatenate sharding dims and cache dims
        dims_t result;
        for (auto d : sharding_dims) {
            result.push_back(d);
        }
        for (auto d : cache_dims) {
            result.push_back(d);
        }

        return result;
    }

  private:
    const std::array<paged_kvcache_dim_kind, 6> &
    cache_layout(attention_cache_kind kind) const noexcept {
        return kind == attention_cache_kind::key ? key_cache_layout_
                                                 : value_cache_layout_;
    }

    const paged_kvcache_axes_t &
    vectorized_axes(attention_cache_kind kind) const noexcept {
        return kind == attention_cache_kind::key ? key_vectorized_axes_
                                                 : value_vectorized_axes_;
    }

    const dims_t &lanes(attention_cache_kind kind) const noexcept {
        return kind == attention_cache_kind::key ? key_lanes_ : value_lanes_;
    }

    datatype_t make_kv_type(const dims_t &lane_values) const noexcept {
        return lane_values.size() == 0
                   ? datatype_t(prim_type_t(
                         std::in_place, attention_config_node::kv_prim_type()))
                   : datatype_t(vector_type_t(
                         std::in_place, attention_config_node::kv_prim_type(),
                         lane_values));
    }

    static std::array<paged_kvcache_dim_kind, 2> make_block_layout(
        const std::array<paged_kvcache_dim_kind, 6> &cache_layout) noexcept {
        std::array<paged_kvcache_dim_kind, 2> block_layout;
        size_t j = 0;
        for (size_t i = 0; i < 6; i++) {
            auto dim = cache_layout[i];
            if ((dim == paged_kvcache_dim_kind::head_dim) ||
                (dim == paged_kvcache_dim_kind::block_size)) {
                block_layout[j++] = dim;
            }
        }
        return block_layout;
    }
    size_t block_size_;
    std::array<paged_kvcache_dim_kind, 6> key_cache_layout_;
    std::array<paged_kvcache_dim_kind, 6> value_cache_layout_;
    paged_kvcache_axes_t key_vectorized_axes_;
    paged_kvcache_axes_t value_vectorized_axes_;
    dims_t key_lanes_;
    dims_t value_lanes_;
    paged_kvcache_axes_t sharding_axes_;
    itlib::small_vector<dims_t, 8> axis_policies_;
};

using paged_attention_config = object_t<paged_attention_config_node>;
} // namespace nncase::llm
