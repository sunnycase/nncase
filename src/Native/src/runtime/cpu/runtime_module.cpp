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
#include "runtime_module.h"
#include "runtime_function.h"
#include <algorithm>
#include <nncase/ntt/arch/cpu/runtime.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_loader.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <new>
#include <string_view>

#ifdef WIN32
#include <Windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/thread_policy.h>
#else
#include <pthread.h>
#endif

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cpu;
using namespace nncase::ntt::runtime;

typedef struct {
    uint32_t bdim;
    uint32_t cdim;
} module_desc_header;

cpu_core_thread_pool::cpu_core_thread_pool(size_t worker_count) {
    workers_.reserve(worker_count);
    for (size_t worker_id = 0; worker_id < worker_count; worker_id++) {
        workers_.emplace_back([this, worker_id] { worker_loop(worker_id); });
    }
}

cpu_core_thread_pool::~cpu_core_thread_pool() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
        generation_++;
    }

    task_cv_.notify_all();
    for (auto &worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void cpu_core_thread_pool::run_core(size_t task_count, void *task_context,
                                    task_invoker_t task_invoker) noexcept {
    if (task_count == 0) {
        return;
    }

    std::unique_lock lock(mutex_);
    task_context_ = task_context;
    task_invoker_ = task_invoker;
    task_count_ = std::min(task_count, workers_.size());
    remaining_ = task_count_;
    generation_++;
    task_cv_.notify_all();
    done_cv_.wait(lock, [this] { return remaining_ == 0; });
    task_context_ = nullptr;
    task_invoker_ = nullptr;
    task_count_ = 0;
}

void cpu_core_thread_pool::worker_loop(size_t worker_id) noexcept {
    bind_worker(worker_id);

    size_t seen_generation = 0;
    while (true) {
        void *task_context = nullptr;
        task_invoker_t task_invoker = nullptr;
        bool active = false;

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

        if (active && task_invoker) {
            task_invoker(task_context, worker_id);
            std::lock_guard lock(mutex_);
            if (--remaining_ == 0) {
                done_cv_.notify_one();
            }
        }
    }
}

void cpu_core_thread_pool::bind_worker(size_t worker_id) noexcept {
#if WIN32
    constexpr auto mask_bits = sizeof(DWORD_PTR) * 8;
    SetThreadAffinityMask(GetCurrentThread(),
                          (DWORD_PTR)1 << (worker_id % mask_bits));
#elif defined(__APPLE__)
    thread_affinity_policy_data_t policy = {(int)worker_id};
    thread_policy_set(pthread_mach_thread_np(pthread_self()),
                      THREAD_AFFINITY_POLICY, (thread_policy_t)&policy,
                      THREAD_AFFINITY_POLICY_COUNT);
#else
#ifdef _POSIX_PRIORITY_SCHEDULING
    if (worker_id < CPU_SETSIZE) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(worker_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }
#endif
#endif
}

cpu_runtime_module::cpu_runtime_module() noexcept
    : bdim_(0), cdim_(0) {
#ifdef __APPLE__
    pthread_key_create(&cpu_thread_context_key_,
                       [](void *ptr) { delete (cpu_thread_context_t *)ptr; });
#endif
}

cpu_runtime_module::~cpu_runtime_module() {
#ifdef __APPLE__
    pthread_key_delete(cpu_thread_context_key_);
#endif
}

result<void> cpu_runtime_module::initialize_before_functions(
    runtime_module_init_context &context) noexcept {
    try_(context.read_section(
        ".desc", [this](auto reader, size_t) -> result<void> {
            auto header = reader.template read<module_desc_header>();
            this->bdim_ = header.bdim;
            this->cdim_ = header.cdim;
            return ok();
        }));
    try {
        thread_pool_ =
            std::make_unique<cpu_core_thread_pool>(std::max<uint64_t>(
                1, static_cast<uint64_t>(bdim_) * static_cast<uint64_t>(cdim_)));
    } catch (...) {
        return err(std::errc::not_enough_memory);
    }

    try_(initialize_text(context));
    try_set(rdata_,
            context.get_or_read_section(".rdata", rdata_storage_, false));
    try_set(block_local_rdata_,
            context.get_or_read_section(".block_local_rdata",
                                        block_local_rdata_storage_, false));
    return ok();
}
result<void> cpu_runtime_module::initialize_text(
    runtime_module_init_context &context) noexcept {
    auto cpu_external_module_path =
        context.interp().options().get<std::string>("cpu_external_module_path");
    if (cpu_external_module_path.is_ok() &&
        !cpu_external_module_path.unwrap().empty()) {
        loader_.load_from_file(cpu_external_module_path.unwrap());
    } else {
        try_set(text_,
                context.get_or_read_section(".text", text_storage_, false));
        loader_.load(text_);
    }

    return ok();
}

result<uintptr_t>
cpu_runtime_module::native_handle(uint32_t flags) const noexcept {
    CHECK_WITH_ERR(flags == 0, std::errc::invalid_argument);
    return ok(loader_.handle());
}

result<block_entry_t> cpu_runtime_module::block_entry() const noexcept {
    return ok((block_entry_t)loader_.entry());
}

result<std::unique_ptr<runtime_function>>
cpu_runtime_module::create_function() noexcept {
    std::unique_ptr<runtime_function> mod(new (std::nothrow)
                                              cpu_runtime_function(*this));
    if (mod)
        return ok(std::move(mod));
    return err(std::errc::not_enough_memory);
}

result<std::unique_ptr<runtime_module>> cpu::create_cpu_runtime_module() {
    std::unique_ptr<runtime_module> mod(new (std::nothrow)
                                            cpu_runtime_module());
    if (mod)
        return ok(std::move(mod));
    return err(std::errc::not_enough_memory);
}
