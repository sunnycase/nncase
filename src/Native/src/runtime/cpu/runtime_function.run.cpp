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
#include "runtime_function.h"
#include <nncase/ntt/arch/cpu/runtime.h>
#include <nncase/runtime/dbg.h>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_op_utility.h>
#include <nncase/runtime/type_serializer.h>

using namespace nncase;
using namespace nncase::runtime;
using namespace nncase::runtime::cpu;
using namespace nncase::ntt::runtime;

result<void> cpu_runtime_function::run(std::byte *output_data) noexcept {
    try_var(enable_profiling,
            module().interp().options().get_scalar_opt<uint8_t>(
                "enable_profiling"));
    auto blocks_count = module().cdim() * module().bdim();
    module().thread_pool().run(blocks_count, [&](size_t linear_bid) noexcept {
        auto cid = linear_bid / module().bdim();
        auto bid = linear_bid % module().bdim();
        auto block_local_rdata_offset =
            module().block_local_rdata_header(linear_bid)[0];
        auto block_local_rdata_size =
            module().block_local_rdata_header(linear_bid)[1];
        auto block_local_rdata = module().block_local_rdata_content().subspan(
            block_local_rdata_offset, block_local_rdata_size);
        cpu_block_entry_params_t block_entry_params{
            .bdim = module().bdim(),
            .cdim = module().cdim(),
            .bid = bid,
            .cid = cid,
            .enable_profiling = enable_profiling,
            .input_descs = this->input_descs_.data(),
            .output_descs = this->output_descs_.data(),
            .rdata = module().rdata(),
            .output = output_data,
            .block_local_rdata = block_local_rdata,
            .data = data(linear_bid),
            .block_local_data = block_local_data(linear_bid),
            .profile_records =
                enable_profiling ? profile_records(linear_bid)
                                 : std::span<ntt::runtime::profile_record>{},
            .profile_record_counts =
                enable_profiling
                    ? profile_record_counts(linear_bid).data()
                    : nullptr,
#ifdef __APPLE__
            .cpu_thread_context_key = module().cpu_thread_context_key(),
#endif
        };

        block_entry_(block_entry_params);
    });

    return ok();
}
