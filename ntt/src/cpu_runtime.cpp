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
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstddef>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <nncase/ntt/arch/cpu/distributed.h>
#include <nncase/ntt/arch/cpu/runtime.h>
#include <nncase/ntt/distributed.h>
#include <nncase/ntt/profiling.h>
#include <nncase/ntt/shape.h>
#include <thread>
#include <string>
#include <vector>

#ifdef WIN32
#include <Windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/thread_policy.h>
#else
#include <pthread.h>
#endif

using namespace nncase;
using namespace nncase::ntt;
using namespace nncase::ntt::runtime;

namespace {
class block_worker_pool {
  public:
    explicit block_worker_pool(size_t worker_count) {
        workers_.reserve(worker_count);
        for (size_t worker_id = 0; worker_id < worker_count; worker_id++) {
            workers_.emplace_back([this, worker_id] { worker_loop(worker_id); });
        }
    }

    ~block_worker_pool() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
            generation_++;
        }
        task_cv_.notify_all();
        for (auto &worker : workers_) {
            worker.join();
        }
    }

    template <class F> void run(size_t task_count, F &&task) noexcept {
        using task_t = std::remove_reference_t<F>;
        std::unique_lock lock(mutex_);
        task_context_ = &task;
        task_invoker_ = [](void *context, size_t worker_id) noexcept {
            (*static_cast<task_t *>(context))(worker_id);
        };
        task_count_ = task_count;
        remaining_ = task_count;
        generation_++;
        task_cv_.notify_all();
        done_cv_.wait(lock, [this] { return remaining_ == 0; });
        task_context_ = nullptr;
        task_invoker_ = nullptr;
    }

  private:
    using task_invoker_t = void (*)(void *, size_t) noexcept;

    static void bind_worker(size_t worker_id) noexcept {
#if WIN32
        constexpr auto mask_bits = sizeof(DWORD_PTR) * 8;
        SetThreadAffinityMask(GetCurrentThread(),
                              (DWORD_PTR)1 << (worker_id % mask_bits));
#elif defined(__APPLE__)
        thread_affinity_policy_data_t policy = {
            static_cast<integer_t>(worker_id)};
        thread_policy_set(pthread_mach_thread_np(pthread_self()),
                          THREAD_AFFINITY_POLICY,
                          reinterpret_cast<thread_policy_t>(&policy),
                          THREAD_AFFINITY_POLICY_COUNT);
#else
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t),
                                   &allowed) != 0) {
            return;
        }

        size_t allowed_count = 0;
        for (size_t cpu = 0; cpu < CPU_SETSIZE; cpu++) {
            allowed_count += CPU_ISSET(cpu, &allowed) ? 1 : 0;
        }
        if (allowed_count == 0) {
            return;
        }

        const auto target_index = worker_id % allowed_count;
        size_t current_index = 0;
        for (size_t cpu = 0; cpu < CPU_SETSIZE; cpu++) {
            if (!CPU_ISSET(cpu, &allowed)) {
                continue;
            }

            if (current_index++ == target_index) {
                cpu_set_t target;
                CPU_ZERO(&target);
                CPU_SET(cpu, &target);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
                                       &target);
                return;
            }
        }
#endif
    }

    void worker_loop(size_t worker_id) noexcept {
        bind_worker(worker_id);
        size_t seen_generation = 0;
        while (true) {
            void *task_context;
            task_invoker_t task_invoker;
            bool active;
            {
                std::unique_lock lock(mutex_);
                task_cv_.wait(lock, [this, seen_generation] {
                    return stopping_ || generation_ != seen_generation;
                });
                if (stopping_) {
                    return;
                }

                seen_generation = generation_;
                active = worker_id < task_count_;
                task_context = task_context_;
                task_invoker = task_invoker_;
            }

            if (active) {
                task_invoker(task_context, worker_id);
                std::lock_guard lock(mutex_);
                if (--remaining_ == 0) {
                    done_cv_.notify_one();
                }
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable task_cv_;
    std::condition_variable done_cv_;
    void *task_context_ = nullptr;
    task_invoker_t task_invoker_ = nullptr;
    size_t task_count_ = 0;
    size_t remaining_ = 0;
    size_t generation_ = 0;
    bool stopping_ = false;
};

thread_local std::string direct_last_error;
std::mutex direct_run_mutex;
std::unique_ptr<block_worker_pool> direct_pool;
size_t direct_pool_size = 0;

void set_direct_error(std::string message) { direct_last_error = std::move(message); }

bool checked_product(size_t lhs, size_t rhs, size_t &result) {
    if (rhs != 0 && lhs > SIZE_MAX / rhs) {
        return false;
    }

    result = lhs * rhs;
    return true;
}

struct block_rdata_range {
    size_t offset;
    size_t size;
};
} // namespace

decltype(nncase::ntt::make_tensor<nncase::ntt::vector<uintptr_t, 2>>(
    nncase::ntt::distributed::topology_shape))
    nncase::ntt::distributed::detail::global_local_data_ptr =
        nncase::ntt::make_tensor<nncase::ntt::vector<uintptr_t, 2>>(
            nncase::ntt::distributed::topology_shape);

decltype(nncase::ntt::make_tensor<nncase::ntt::vector<uintptr_t, 2>>(
    nncase::ntt::distributed::topology_shape))
    nncase::ntt::distributed::detail::global_block_local_data_ptr =
        nncase::ntt::make_tensor<nncase::ntt::vector<uintptr_t, 2>>(
            nncase::ntt::distributed::topology_shape);

decltype(nncase::ntt::make_tensor<nncase::ntt::vector<uintptr_t, 2>>(
    nncase::ntt::distributed::topology_shape))
    nncase::ntt::distributed::detail::global_block_local_rdata_ptr =
        nncase::ntt::make_tensor<nncase::ntt::vector<uintptr_t, 2>>(
            nncase::ntt::distributed::topology_shape);

namespace nncase::ntt::runtime {
#ifdef __APPLE__
pthread_key_t cpu_thread_context_key;
#else
thread_local cpu_thread_context_t cpu_thread_context;
#endif

void *thread_alloc(size_t bytes, size_t alignment) {
#ifdef WIN32
    return _aligned_malloc(bytes, alignment);
#else
    size_t mask = alignment - 1;
    size_t aligned_bytes = bytes + (-bytes & mask);
    auto ptr = aligned_alloc(alignment, aligned_bytes);
    if (!ptr) {
        std::terminate();
    }
    return ptr;
#endif
}

void thread_free(void *ptr) {
#ifdef WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

bool is_profiling_enabled() noexcept {
    return cpu_thread_context_t::current().enable_profiling;
}

uint64_t get_profile_time() noexcept {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

void record_profile(profile_level level,
                    const profile_record &record) noexcept {
    // Other levels are not supported yet.
    if (level == profile_level::kernel) {
        auto &ctx = cpu_thread_context_t::current();
        auto idx = ctx.profile_record_counts[0]++;
        ctx.profile_records[idx] = record;
    }
}
} // namespace nncase::ntt::runtime

cpu_thread_context_t &cpu_thread_context_t::current() noexcept {
#ifndef __APPLE__
    return cpu_thread_context;
#else
    return *reinterpret_cast<cpu_thread_context_t *>(
        pthread_getspecific(cpu_thread_context_key));
#endif
}

extern "C" void block_entry(const cpu_block_entry_params_t &params) {
#ifdef __APPLE__
    cpu_thread_context_key = params.cpu_thread_context_key;
#endif

    auto profile_records = params.profile_records;
#ifdef __APPLE__
    pthread_setspecific(
        cpu_thread_context_key, new cpu_thread_context_t
#else
    cpu_thread_context_t::current() =
#endif
        {.bid = params.bid,
         .cid = params.cid,
         .enable_profiling = params.enable_profiling,
         .profile_records = profile_records,
         .profile_record_counts = params.profile_record_counts}
#ifdef __APPLE__
    );
#else
    ;
#endif

    const auto program_ids = make_shape(params.cid, params.bid);

    auto data = params.data;
    auto block_local_data = params.block_local_data;

    ntt::distributed::detail::global_local_data_ptr(program_ids)(
        0_dim) = (uintptr_t)data.data();
    ntt::distributed::detail::global_local_data_ptr(program_ids)(1_dim) =
        (uintptr_t)(data.data() + data.size_bytes());
    ntt::distributed::detail::global_block_local_data_ptr(program_ids)(0_dim) =
        (uintptr_t)block_local_data.data();
    ntt::distributed::detail::global_block_local_data_ptr(program_ids)(1_dim) =
        (uintptr_t)(block_local_data.data() + block_local_data.size_bytes());
    ntt::distributed::detail::global_block_local_rdata_ptr(program_ids)(0_dim) =
        (uintptr_t)params.block_local_rdata.data();
    ntt::distributed::detail::global_block_local_rdata_ptr(program_ids)(1_dim) =
        (uintptr_t)(params.block_local_rdata.data() +
                    params.block_local_rdata.size_bytes());

    block_main(params.function_id, params.input_descs, params.output_descs,
               params.rdata.data(), params.block_local_rdata.data(),
               data.data(), block_local_data.data(), params.output);
}

extern "C" int32_t nncase_ntt_cpu_run(const ntt_cpu_run_params_t *params) {
    direct_last_error.clear();
    if (!params || (params->input_count != 0 && !params->inputs) ||
        (params->output_count != 0 && !params->outputs) ||
        (params->output_size != 0 && !params->output) ||
        (params->rdata_size != 0 && !params->rdata) ||
        !params->block_local_rdata ||
        (params->data_bytes_per_block != 0 && !params->data) ||
        (params->block_local_data_bytes_per_block != 0 &&
         !params->block_local_data)) {
        set_direct_error("CPU NTT run received a null required argument.");
        return -1;
    }

    size_t block_count;
    if (!checked_product(params->bdim, params->cdim, block_count) ||
        block_count == 0) {
        set_direct_error("CPU NTT run has an invalid block topology.");
        return -1;
    }

    const size_t block_rdata_header_size =
        block_count * 2 * sizeof(uint64_t);
    if (params->block_local_rdata_size < block_rdata_header_size) {
        set_direct_error("CPU NTT block-local rdata header is truncated.");
        return -1;
    }

    const auto *block_rdata_bytes =
        static_cast<const std::byte *>(params->block_local_rdata);
    const auto *block_rdata_content =
        block_rdata_bytes + block_rdata_header_size;
    const auto block_rdata_content_size =
        params->block_local_rdata_size - block_rdata_header_size;
    std::vector<block_rdata_range> block_rdata_ranges(block_count);
    for (size_t i = 0; i < block_count; i++) {
        uint64_t offset;
        uint64_t size;
        std::memcpy(&offset, block_rdata_bytes + i * 2 * sizeof(uint64_t),
                    sizeof(offset));
        std::memcpy(&size,
                    block_rdata_bytes + (i * 2 + 1) * sizeof(uint64_t),
                    sizeof(size));
        if (offset > block_rdata_content_size ||
            size > block_rdata_content_size - offset) {
            set_direct_error("CPU NTT block-local rdata range is invalid.");
            return -1;
        }

        block_rdata_ranges[i] = {
            .offset = static_cast<size_t>(offset),
            .size = static_cast<size_t>(size),
        };
    }

    std::vector<block_inout_desc> input_descs(params->input_count);
    std::vector<block_inout_desc> output_descs(params->output_count);
    for (size_t i = 0; i < params->input_count; i++) {
        const auto &source = params->inputs[i];
        if ((source.rank != 0 && (!source.shape || !source.strides)) ||
            (source.size != 0 && !source.data)) {
            set_direct_error("CPU NTT input descriptor is incomplete.");
            return -1;
        }

        input_descs[i] = {
            .data = static_cast<std::byte *>(source.data),
            .size = source.size,
            .shape = const_cast<size_t *>(source.shape),
            .strides = const_cast<size_t *>(source.strides),
            .rank = source.rank,
        };
    }

    for (size_t i = 0; i < params->output_count; i++) {
        const auto &source = params->outputs[i];
        if ((source.rank != 0 && (!source.shape || !source.strides)) ||
            (source.size != 0 && !source.data)) {
            set_direct_error("CPU NTT output descriptor is incomplete.");
            return -1;
        }

        output_descs[i] = {
            .data = static_cast<std::byte *>(source.data),
            .size = source.size,
            .shape = const_cast<size_t *>(source.shape),
            .strides = const_cast<size_t *>(source.strides),
            .rank = source.rank,
        };
    }

    auto *output_bytes = static_cast<std::byte *>(params->output);
    auto *data_bytes = static_cast<std::byte *>(params->data);
    auto *block_local_data_bytes =
        static_cast<std::byte *>(params->block_local_data);
    auto rdata = std::span<const std::byte>(
        static_cast<const std::byte *>(params->rdata), params->rdata_size);

    std::lock_guard run_lock(direct_run_mutex);
    if (!direct_pool || direct_pool_size != block_count) {
        direct_pool = std::make_unique<block_worker_pool>(block_count);
        direct_pool_size = block_count;
    }

    direct_pool->run(block_count, [&](size_t linear_bid) noexcept {
        const auto rdata_range = block_rdata_ranges[linear_bid];
        const auto cid = linear_bid / params->bdim;
        const auto bid = linear_bid % params->bdim;
        cpu_block_entry_params_t block_params{
                .function_id = params->function_id,
                .bdim = params->bdim,
                .cdim = params->cdim,
                .bid = bid,
                .cid = cid,
                .enable_profiling = 0,
                .input_descs = input_descs.data(),
                .output_descs = output_descs.data(),
                .rdata = rdata,
                .output = output_bytes,
                .block_local_rdata = std::span<const std::byte>(
                    block_rdata_content + rdata_range.offset,
                    rdata_range.size),
                .data = std::span<std::byte>(
                    data_bytes + linear_bid * params->data_bytes_per_block,
                    params->data_bytes_per_block),
                .block_local_data = std::span<std::byte>(
                    block_local_data_bytes +
                        linear_bid * params->block_local_data_bytes_per_block,
                    params->block_local_data_bytes_per_block),
                .profile_records = {},
                .profile_record_counts = nullptr,
#ifdef __APPLE__
                .cpu_thread_context_key = cpu_thread_context_key,
#endif
        };
        block_entry(block_params);
    });

    for (size_t i = 0; i < params->output_count; i++) {
        params->outputs[i] = {
            .data = output_descs[i].data,
            .size = output_descs[i].size,
            .shape = output_descs[i].shape,
            .strides = output_descs[i].strides,
            .rank = output_descs[i].rank,
        };
    }

    return 0;
}

extern "C" const char *nncase_ntt_cpu_last_error() {
    return direct_last_error.c_str();
}
