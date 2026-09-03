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
#include "../distributed/sharded_tensor.h"
#include "../distributed/topology.h"
#include "../runtime.h"
#include "../tensor_traits.h"
#include <atomic>
#include <cstddef>
#include <type_traits>

namespace nncase::ntt::runtime {
namespace detail {
inline bool pipeline_worker_leader() noexcept {
    return distributed::program_id<distributed::topology::chip>() == 0 &&
           distributed::program_id<distributed::topology::block>() == 0;
}

template <ntt::ShardedTensor TTensor>
inline bool pipeline_shard_owner(const TTensor &tensor) noexcept {
    using tensor_type = std::decay_t<TTensor>;
    using sharding_type = typename tensor_type::sharding_type;
    using mesh_type = typename tensor_type::mesh_type;
    const auto shard_index = mesh_type::local_index();
    constexpr auto replication_axes =
        distributed::detail::mesh_axes_of_non_split_shard_policies<
            sharding_type>();
    return replication_axes.aggregate(
        true, [&](bool owner, auto mesh_axis, auto) {
            return owner && shard_index[mesh_axis] == 0;
        });
}

template <ntt::ShardedTensor TTensor>
inline void copy_local_to_channel(pipeline_channel *channel,
                                  const TTensor &tensor) noexcept {
    using tensor_type = std::decay_t<TTensor>;
    using value_type = typename tensor_type::value_type;
    if (!pipeline_shard_owner(tensor)) {
        return;
    }

    using mesh_type = typename tensor_type::mesh_type;
    const auto shard_index = mesh_type::local_index();
    const auto global_offset =
        tensor.sharding().global_offset(tensor.shape(), shard_index);
    const auto global_strides = ntt::default_strides(tensor.shape());
    const auto &local = tensor.local();
    auto *payload = reinterpret_cast<value_type *>(
        reinterpret_cast<std::byte *>(channel) +
        pipeline_channel_header_bytes);
    const auto payload_offset = ntt::linear_offset(global_offset, global_strides);
    ntt::apply(local.shape(), [&](auto local_index) {
        const auto destination_index =
            ntt::generate_shape<tensor_type::rank()>([&](auto axis) {
                return global_offset[axis] + local_index[axis];
            });
        payload[ntt::linear_offset(destination_index, global_strides)] =
            local(local_index);
    });
}

template <ntt::Tensor TTensor>
inline void copy_local_to_channel(pipeline_channel *channel,
                                  const TTensor &tensor) noexcept {
    using tensor_type = std::decay_t<TTensor>;
    using value_type = typename tensor_type::value_type;
    if (!pipeline_worker_leader()) {
        return;
    }

    auto *payload = reinterpret_cast<value_type *>(
        reinterpret_cast<std::byte *>(channel) +
        pipeline_channel_header_bytes);
    const auto global_strides = ntt::default_strides(tensor.shape());
    ntt::apply(tensor.shape(), [&](auto index) {
        payload[ntt::linear_offset(index, global_strides)] = tensor(index);
    });
}

template <ntt::ShardedTensor TTensor>
inline void copy_channel_to_local(const pipeline_channel *channel,
                                  TTensor &tensor) noexcept {
    using tensor_type = std::decay_t<TTensor>;
    using value_type = typename tensor_type::value_type;
    using mesh_type = typename tensor_type::mesh_type;
    const auto shard_index = mesh_type::local_index();
    const auto global_offset =
        tensor.sharding().global_offset(tensor.shape(), shard_index);
    const auto global_strides = ntt::default_strides(tensor.shape());
    auto &local = tensor.local();
    const auto *payload = reinterpret_cast<const value_type *>(
        reinterpret_cast<const std::byte *>(channel) +
        pipeline_channel_header_bytes);
    const auto payload_offset = ntt::linear_offset(global_offset, global_strides);
    ntt::apply(local.shape(), [&](auto local_index) {
        const auto source_index =
            ntt::generate_shape<tensor_type::rank()>([&](auto axis) {
                return global_offset[axis] + local_index[axis];
            });
        local(local_index) =
            payload[ntt::linear_offset(source_index, global_strides)];
    });
}

template <ntt::Tensor TTensor>
inline void copy_channel_to_local(const pipeline_channel *channel,
                                  TTensor &tensor) noexcept {
    using tensor_type = std::decay_t<TTensor>;
    using value_type = typename tensor_type::value_type;
    const auto *payload = reinterpret_cast<const value_type *>(
        reinterpret_cast<const std::byte *>(channel) +
        pipeline_channel_header_bytes);
    const auto global_strides = ntt::default_strides(tensor.shape());
    ntt::apply(tensor.shape(), [&](auto index) {
        tensor(index) = payload[ntt::linear_offset(index, global_strides)];
    });
}
} // namespace detail

template <class TTensor>
inline void pipeline_channel_produce(pipeline_channel *channel,
                                     const TTensor &tensor,
                                     uint32_t phase) noexcept {
    detail::copy_local_to_channel(channel, tensor);

    // RMWs on one atomic form a release sequence. The last worker therefore
    // acquires every preceding payload write before publishing the phase to
    // the heterogeneous consumer.
    auto arrivals = std::atomic_ref(channel->producer_arrivals);
    const auto arrival = arrivals.fetch_add(1, std::memory_order_acq_rel) + 1;
    constexpr auto worker_count =
        distributed::program_dim<distributed::topology::chip>() *
        distributed::program_dim<distributed::topology::block>();
    if (arrival == worker_count) {
        arrivals.store(0, std::memory_order_relaxed);
        std::atomic_ref(channel->produced_phase)
            .store(phase, std::memory_order_release);
    }
}

template <class TTensor>
inline void pipeline_channel_consume(pipeline_channel *channel,
                                     TTensor &tensor,
                                     uint32_t phase) noexcept {
    auto produced = std::atomic_ref(channel->produced_phase);
    while (produced.load(std::memory_order_acquire) < phase) {
    }

    detail::copy_channel_to_local(channel, tensor);
}
} // namespace nncase::ntt::runtime
