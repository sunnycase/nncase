// Raw CUDA SM120 benchmark for the Qwen3 one-layer decode projections.
// It mirrors PyNTT's sharding, packed BF16 weights, N64/K128/S4 shared arena,
// NVMMAShared 128-byte swizzle, and producer/consumer role split, then replaces
// TMA with legacy cp.async for a controlled implementation comparison. It keeps
// the original one-producer-warp path as a baseline and adds an SM120
// warpgroup-aligned path with four producer warps and eight consumer warps.
// CUDA's pipeline cursor cannot hold both GLU slots concurrently, so two
// lockstep depth-2 rings emulate PyNTT's interleaved depth-4 gate/up ring while
// preserving its wait-both, load-input-once, accumulate-both consumer schedule.

#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cg = cooperative_groups;

using bf16 = __nv_bfloat16;

namespace {

constexpr int kGrid = 32;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerWarpGroup = 4;
constexpr int kBlockThreads = 384;
constexpr int kRoleSplitProducerWarps = 1;
constexpr int kWarpGroupProducerWarps = kWarpsPerWarpGroup;
constexpr int kConsumerWarps = 8;
constexpr int kConsumerThreads = 256;
constexpr int kProducerRegisterBudget = 64;
constexpr int kConsumerRegisterBudget = 216;
constexpr int kRegisterSpecializedRegisters =
    (kWarpGroupProducerWarps * kProducerRegisterBudget +
     kConsumerWarps * kConsumerRegisterBudget) *
    kWarpSize;
constexpr int kBlockN = 64;
constexpr int kBlockK = 128;
constexpr int kStageElements = kBlockN * kBlockK;
constexpr int kStageBytes = kStageElements * sizeof(bf16);
constexpr int kSharedStages = 4;
static_assert(kBlockThreads ==
                  (kWarpGroupProducerWarps + kConsumerWarps) * kWarpSize,
              "warpgroup-specialized block geometry must use four producer and "
              "eight consumer warps");
static_assert(kProducerRegisterBudget >= 24 && kProducerRegisterBudget <= 256 &&
                  kProducerRegisterBudget % 8 == 0,
              "producer setmaxnreg budget is invalid");
static_assert(kConsumerRegisterBudget >= 24 && kConsumerRegisterBudget <= 256 &&
                  kConsumerRegisterBudget % 8 == 0,
              "consumer setmaxnreg budget is invalid");
static_assert(kRegisterSpecializedRegisters <= 65536,
              "warpgroup register budgets exceed the SM120 register file");

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

using QkvPipelineState =
    cuda::pipeline_shared_state<cuda::thread_scope_block, 4>;
using GluPipelineState =
    cuda::pipeline_shared_state<cuda::thread_scope_block, 2>;
constexpr std::size_t kQkvStageArenaOffset =
    align_up(sizeof(QkvPipelineState), 16);
constexpr std::size_t kGluUpStateOffset =
    align_up(sizeof(GluPipelineState), alignof(GluPipelineState));
constexpr std::size_t kGluStageArenaOffset =
    align_up(kGluUpStateOffset + sizeof(GluPipelineState), 16);
constexpr int kQkvDynamicSharedBytes =
    static_cast<int>(kQkvStageArenaOffset + kSharedStages * kStageBytes);
constexpr int kGluDynamicSharedBytes =
    static_cast<int>(kGluStageArenaOffset + kSharedStages * kStageBytes);

constexpr int kQkvK = 1024;
constexpr int kQGlobalN = 2048;
constexpr int kKGlobalN = 1024;
constexpr int kVGlobalN = 1024;
constexpr int kQLocalN = 64;
constexpr int kKLocalN = 32;
constexpr int kVLocalN = 32;

constexpr int kGluK = 1024;
constexpr int kGluGlobalN = 3072;
constexpr int kGluLocalN = 96;

constexpr int kOK = 2048;
constexpr int kOGlobalN = 1024;
constexpr int kOLocalN = 32;

constexpr int kVectorElements = 8;
constexpr int kVectorBytes = kVectorElements * sizeof(bf16);

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error_ = (call);                                           \
        if (error_ != cudaSuccess) {                                           \
            std::cerr << "CUDA failure at " << __FILE__ << ':' << __LINE__     \
                      << ": " << cudaGetErrorString(error_) << std::endl;      \
            std::exit(1);                                                      \
        }                                                                      \
    } while (false)

__host__ __device__ inline std::size_t packed_offset(int n, int k,
                                                     int total_n) {
    const int k_outer = k >> 4;
    const int n_outer = n >> 3;
    const int n_lane = n & 7;
    const int k_pack = (k & 15) >> 3;
    return (
        ((((static_cast<std::size_t>(k_outer) * (total_n >> 3) + n_outer) * 8 +
           n_lane) *
              2 +
          k_pack) *
         8));
}

__device__ __forceinline__ const bf16 *packed_vec8(const bf16 *base, int n,
                                                   int k, int total_n) {
    return base + packed_offset(n, k, total_n);
}

// Match the current Qwen PyNTT nv_mma_shared_layout=True allocation. The
// logical 4-D stage is [8, 8, 2, 64], collapsed to 128 rows of 64 BF16.
// NVMMAShared applies a 128-byte XOR swizzle: vectorSize=8 BF16,
// perPhase=1, maxPhase=8. QKV is N-major; GLU is K-major.
__device__ __forceinline__ int nv_mma_swizzled_offset(int outer0, int outer1,
                                                      int payload) {
    const int row = (outer0 * 8 + outer1) * 2 + (payload >> 6);
    const int column = payload & 63;
    return row * 64 + (column ^ ((row & 7) * 8));
}

__device__ __forceinline__ int qkv_stage_offset(int n, int k) {
    const int payload = (n & 7) * 16 + (k & 15);
    return nv_mma_swizzled_offset(n >> 3, k >> 4, payload);
}

__device__ __forceinline__ int glu_stage_offset(int n, int k) {
    const int payload = (n & 7) * 16 + (k & 15);
    return nv_mma_swizzled_offset(k >> 4, n >> 3, payload);
}

__device__ __forceinline__ float dot8(const bf16 *weight, const bf16 *input) {
    const auto *weight2 = reinterpret_cast<const __nv_bfloat162 *>(weight);
    const auto *input2 = reinterpret_cast<const __nv_bfloat162 *>(input);
    float result = 0.0f;
#pragma unroll
    for (int index = 0; index < 4; ++index) {
        const float2 weight_value = __bfloat1622float2(weight2[index]);
        const float2 input_value = __bfloat1622float2(input2[index]);
        result = fmaf(weight_value.x, input_value.x, result);
        result = fmaf(weight_value.y, input_value.y, result);
    }
    return result;
}

__device__ __forceinline__ void dot8_pair(const bf16 *gate, const bf16 *up,
                                          const bf16 *input, float &gate_acc,
                                          float &up_acc) {
    const auto *gate2 = reinterpret_cast<const __nv_bfloat162 *>(gate);
    const auto *up2 = reinterpret_cast<const __nv_bfloat162 *>(up);
    const auto *input2 = reinterpret_cast<const __nv_bfloat162 *>(input);
#pragma unroll
    for (int index = 0; index < 4; ++index) {
        const float2 gate_value = __bfloat1622float2(gate2[index]);
        const float2 up_value = __bfloat1622float2(up2[index]);
        const float2 input_value = __bfloat1622float2(input2[index]);
        gate_acc = fmaf(gate_value.x, input_value.x, gate_acc);
        gate_acc = fmaf(gate_value.y, input_value.y, gate_acc);
        up_acc = fmaf(up_value.x, input_value.x, up_acc);
        up_acc = fmaf(up_value.y, input_value.y, up_acc);
    }
}

__device__ __forceinline__ float reduce_width4(float value) {
    value += __shfl_down_sync(0xffffffffu, value, 2, 4);
    value += __shfl_down_sync(0xffffffffu, value, 1, 4);
    return value;
}

__device__ __forceinline__ float reduce_width8(float value) {
    value += __shfl_down_sync(0xffffffffu, value, 4, 8);
    value += __shfl_down_sync(0xffffffffu, value, 2, 8);
    value += __shfl_down_sync(0xffffffffu, value, 1, 8);
    return value;
}

template <int RegisterCount>
__device__ __forceinline__ void warpgroup_deallocate_registers() {
    static_assert(RegisterCount >= 24 && RegisterCount <= 256);
    static_assert(RegisterCount % 8 == 0);
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n"
                 :
                 : "n"(RegisterCount));
}

template <int RegisterCount>
__device__ __forceinline__ void warpgroup_allocate_registers() {
    static_assert(RegisterCount >= 24 && RegisterCount <= 256);
    static_assert(RegisterCount % 8 == 0);
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n"
                 :
                 : "n"(RegisterCount));
}

__device__ __forceinline__ std::uint64_t make_evict_first_policy() {
    std::uint64_t policy;
    const float fraction = 1.0f;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, %1;\n"
                 : "=l"(policy)
                 : "f"(fraction));
    return policy;
}

template <bool EvictFirst, typename Pipeline>
__device__ __forceinline__ void
issue_cp_async(Pipeline &pipeline, bf16 *destination, const bf16 *source,
               std::uint64_t cache_policy) {
    if constexpr (EvictFirst) {
        const auto shared_address =
            static_cast<std::uint32_t>(__cvta_generic_to_shared(destination));
        const auto global_address =
            static_cast<std::uint64_t>(__cvta_generic_to_global(source));
        asm volatile("cp.async.cg.shared.global.L2::cache_hint "
                     "[%0], [%1], 16, 16, %2;\n"
                     :
                     : "r"(shared_address), "l"(global_address),
                       "l"(cache_policy)
                     : "memory");
    } else {
        cuda::memcpy_async(destination, source,
                           cuda::aligned_size_t<kVectorBytes>(kVectorBytes),
                           pipeline);
    }
}

template <int ProducerThreads, bool EvictFirst, typename Pipeline>
__device__ __forceinline__ void
copy_qkv_stage(Pipeline &pipeline, bf16 *stage, const bf16 *q_weight,
               const bf16 *k_weight, const bf16 *v_weight, int shard,
               int n_tile, int k_tile, int producer_thread,
               std::uint64_t cache_policy) {
    constexpr int chunks_per_row = kBlockK / kVectorElements;
    constexpr int chunks_per_stage = kBlockN * chunks_per_row;
    static_assert(ProducerThreads == kWarpSize ||
                      ProducerThreads == kWarpsPerWarpGroup * kWarpSize,
                  "producer group must be one warp or one warpgroup");
    for (int chunk = producer_thread; chunk < chunks_per_stage;
         chunk += ProducerThreads) {
        const int local_n = chunk / chunks_per_row;
        const int k_chunk = chunk % chunks_per_row;
        const int concatenated_n = n_tile * kBlockN + local_n;
        const int global_k = k_tile * kBlockK + k_chunk * kVectorElements;
        const bf16 *source = nullptr;
        if (concatenated_n < kQLocalN) {
            source = packed_vec8(q_weight, shard * kQLocalN + concatenated_n,
                                 global_k, kQGlobalN);
        } else if (concatenated_n < kQLocalN + kKLocalN) {
            source = packed_vec8(k_weight,
                                 shard * kKLocalN + concatenated_n - kQLocalN,
                                 global_k, kKGlobalN);
        } else {
            source = packed_vec8(v_weight,
                                 shard * kVLocalN + concatenated_n - kQLocalN -
                                     kKLocalN,
                                 global_k, kVGlobalN);
        }
        bf16 *destination =
            stage + qkv_stage_offset(local_n, k_chunk * kVectorElements);
        issue_cp_async<EvictFirst>(pipeline, destination, source, cache_policy);
    }
}

template <int ProducerThreads, bool EvictFirst, typename Pipeline>
__device__ __forceinline__ void
copy_glu_stage(Pipeline &pipeline, bf16 *stage, const bf16 *weight,
               const bf16 *zero_chunk, int shard, int n_tile, int k_tile,
               int producer_thread, std::uint64_t cache_policy) {
    constexpr int chunks_per_row = kBlockK / kVectorElements;
    constexpr int chunks_per_stage = kBlockN * chunks_per_row;
    static_assert(ProducerThreads == kWarpSize ||
                      ProducerThreads == kWarpsPerWarpGroup * kWarpSize,
                  "producer group must be one warp or one warpgroup");
    for (int chunk = producer_thread; chunk < chunks_per_stage;
         chunk += ProducerThreads) {
        const int local_n = chunk / chunks_per_row;
        const int k_chunk = chunk % chunks_per_row;
        const int global_n = shard * kGluLocalN + n_tile * kBlockN + local_n;
        const int global_k = k_tile * kBlockK + k_chunk * kVectorElements;
        const bf16 *source =
            global_n < kGluGlobalN
                ? packed_vec8(weight, global_n, global_k, kGluGlobalN)
                : zero_chunk;
        bf16 *destination =
            stage + glu_stage_offset(local_n, k_chunk * kVectorElements);
        issue_cp_async<EvictFirst>(pipeline, destination, source, cache_policy);
    }
}

template <int ProducerWarps, bool DynamicRegisterSpecialization,
          bool EvictFirst>
__global__ __launch_bounds__(kBlockThreads, 1) void qkv_cp_async_kernel(
    const bf16 *__restrict__ input, const bf16 *__restrict__ q_weight,
    const bf16 *__restrict__ k_weight, const bf16 *__restrict__ v_weight,
    bf16 *__restrict__ q_output, bf16 *__restrict__ k_output,
    bf16 *__restrict__ v_output) {
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    const auto block = cg::this_thread_block();
    auto *pipeline_state = reinterpret_cast<QkvPipelineState *>(dynamic_shared);
    if (threadIdx.x == 0) {
        ::new (pipeline_state) QkvPipelineState;
    }
    block.sync();
    auto *stages =
        reinterpret_cast<bf16 *>(dynamic_shared + kQkvStageArenaOffset);
    constexpr int producer_threads = ProducerWarps * kWarpSize;
    constexpr int consumer_begin = producer_threads;
    constexpr int consumer_end = consumer_begin + kConsumerThreads;
    static_assert(ProducerWarps == kRoleSplitProducerWarps ||
                      ProducerWarps == kWarpGroupProducerWarps,
                  "unsupported producer geometry");
    if constexpr (DynamicRegisterSpecialization) {
        static_assert(
            ProducerWarps == kWarpGroupProducerWarps,
            "dynamic register specialization requires warpgroup-aligned roles");
        if (threadIdx.x < producer_threads) {
            warpgroup_deallocate_registers<kProducerRegisterBudget>();
        } else {
            warpgroup_allocate_registers<kConsumerRegisterBudget>();
        }
    }
    auto pipeline = cuda::make_pipeline(
        block, pipeline_state, static_cast<std::size_t>(producer_threads));
    const int shard = static_cast<int>(blockIdx.x);

    if (threadIdx.x < producer_threads) {
        std::uint64_t cache_policy = 0;
        if constexpr (EvictFirst) {
            cache_policy = make_evict_first_policy();
        }
#pragma unroll 1
        for (int n_tile = 0; n_tile < 2; ++n_tile) {
#pragma unroll 1
            for (int k_tile = 0; k_tile < kQkvK / kBlockK; ++k_tile) {
                const int sequence = n_tile * (kQkvK / kBlockK) + k_tile;
                pipeline.producer_acquire();
                copy_qkv_stage<producer_threads, EvictFirst>(
                    pipeline, stages + (sequence & 3) * kStageElements,
                    q_weight, k_weight, v_weight, shard, n_tile, k_tile,
                    static_cast<int>(threadIdx.x), cache_policy);
                pipeline.producer_commit();
            }
        }
        return;
    }

    const bool computes =
        threadIdx.x >= consumer_begin && threadIdx.x < consumer_end;
    const int consumer_thread = static_cast<int>(threadIdx.x) - consumer_begin;
    const int consumer_warp = consumer_thread >> 5;
    const int lane = consumer_thread & 31;
    const int n_in_tile = consumer_warp * 8 + (lane >> 2);
    const int k_subgroup = lane & 3;
    const bf16 *shard_input = input + shard * kQkvK;

#pragma unroll 1
    for (int n_tile = 0; n_tile < 2; ++n_tile) {
        float accumulator = 0.0f;
#pragma unroll 1
        for (int k_tile = 0; k_tile < kQkvK / kBlockK; ++k_tile) {
            const int sequence = n_tile * (kQkvK / kBlockK) + k_tile;
            pipeline.consumer_wait();
            if (computes) {
                const bf16 *stage = stages + (sequence & 3) * kStageElements;
#pragma unroll
                for (int k_group = 0; k_group < kBlockK / 32; ++k_group) {
                    const int local_k = k_group * 32 + k_subgroup * 8;
                    accumulator +=
                        dot8(stage + qkv_stage_offset(n_in_tile, local_k),
                             shard_input + k_tile * kBlockK + local_k);
                }
            }
            pipeline.consumer_release();
        }
        if (computes) {
            accumulator = reduce_width4(accumulator);
            if (k_subgroup == 0) {
                const int concatenated_n = n_tile * kBlockN + n_in_tile;
                if (concatenated_n < kQLocalN) {
                    q_output[shard * kQLocalN + concatenated_n] =
                        __float2bfloat16_rn(accumulator);
                } else if (concatenated_n < kQLocalN + kKLocalN) {
                    k_output[shard * kKLocalN + concatenated_n - kQLocalN] =
                        __float2bfloat16_rn(accumulator);
                } else {
                    v_output[shard * kVLocalN + concatenated_n - kQLocalN -
                             kKLocalN] = __float2bfloat16_rn(accumulator);
                }
            }
        }
    }
}

__global__ __launch_bounds__(kBlockThreads, 1) void qkv_direct_kernel(
    const bf16 *__restrict__ input, const bf16 *__restrict__ q_weight,
    const bf16 *__restrict__ k_weight, const bf16 *__restrict__ v_weight,
    bf16 *__restrict__ q_output, bf16 *__restrict__ k_output,
    bf16 *__restrict__ v_output) {
    if (threadIdx.x >= kConsumerThreads) {
        return;
    }
    const int shard = static_cast<int>(blockIdx.x);
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int n_in_tile = warp * 8 + (lane >> 2);
    const int k_subgroup = lane & 3;
    const bf16 *shard_input = input + shard * kQkvK;

#pragma unroll 1
    for (int n_tile = 0; n_tile < 2; ++n_tile) {
        const int concatenated_n = n_tile * kBlockN + n_in_tile;
        const bf16 *weight = nullptr;
        int global_n = 0;
        int total_n = 0;
        if (concatenated_n < kQLocalN) {
            weight = q_weight;
            global_n = shard * kQLocalN + concatenated_n;
            total_n = kQGlobalN;
        } else if (concatenated_n < kQLocalN + kKLocalN) {
            weight = k_weight;
            global_n = shard * kKLocalN + concatenated_n - kQLocalN;
            total_n = kKGlobalN;
        } else {
            weight = v_weight;
            global_n = shard * kVLocalN + concatenated_n - kQLocalN - kKLocalN;
            total_n = kVGlobalN;
        }
        float accumulator = 0.0f;
#pragma unroll 1
        for (int k_base = 0; k_base < kQkvK; k_base += 32) {
            const int global_k = k_base + k_subgroup * 8;
            accumulator +=
                dot8(packed_vec8(weight, global_n, global_k, total_n),
                     shard_input + global_k);
        }
        accumulator = reduce_width4(accumulator);
        if (k_subgroup == 0) {
            if (concatenated_n < kQLocalN) {
                q_output[shard * kQLocalN + concatenated_n] =
                    __float2bfloat16_rn(accumulator);
            } else if (concatenated_n < kQLocalN + kKLocalN) {
                k_output[shard * kKLocalN + concatenated_n - kQLocalN] =
                    __float2bfloat16_rn(accumulator);
            } else {
                v_output[shard * kVLocalN + concatenated_n - kQLocalN -
                         kKLocalN] = __float2bfloat16_rn(accumulator);
            }
        }
    }
}

template <int ProducerWarps, bool DynamicRegisterSpecialization,
          bool EvictFirst>
__global__ __launch_bounds__(kBlockThreads, 1) void glu_cp_async_kernel(
    const bf16 *__restrict__ input, const bf16 *__restrict__ gate_weight,
    const bf16 *__restrict__ up_weight, const bf16 *__restrict__ zero_chunk,
    bf16 *__restrict__ output) {
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    const auto block = cg::this_thread_block();
    auto *gate_pipeline_state =
        reinterpret_cast<GluPipelineState *>(dynamic_shared);
    auto *up_pipeline_state = reinterpret_cast<GluPipelineState *>(
        dynamic_shared + kGluUpStateOffset);
    if (threadIdx.x == 0) {
        ::new (gate_pipeline_state) GluPipelineState;
        ::new (up_pipeline_state) GluPipelineState;
    }
    block.sync();
    auto *stages =
        reinterpret_cast<bf16 *>(dynamic_shared + kGluStageArenaOffset);
    bf16 *gate_stages = stages;
    bf16 *up_stages = stages + 2 * kStageElements;
    constexpr int producer_threads = ProducerWarps * kWarpSize;
    constexpr int consumer_begin = producer_threads;
    constexpr int consumer_end = consumer_begin + kConsumerThreads;
    static_assert(ProducerWarps == kRoleSplitProducerWarps ||
                      ProducerWarps == kWarpGroupProducerWarps,
                  "unsupported producer geometry");
    if constexpr (DynamicRegisterSpecialization) {
        static_assert(
            ProducerWarps == kWarpGroupProducerWarps,
            "dynamic register specialization requires warpgroup-aligned roles");
        if (threadIdx.x < producer_threads) {
            warpgroup_deallocate_registers<kProducerRegisterBudget>();
        } else {
            warpgroup_allocate_registers<kConsumerRegisterBudget>();
        }
    }
    auto gate_pipeline = cuda::make_pipeline(
        block, gate_pipeline_state, static_cast<std::size_t>(producer_threads));
    auto up_pipeline = cuda::make_pipeline(
        block, up_pipeline_state, static_cast<std::size_t>(producer_threads));
    const int shard = static_cast<int>(blockIdx.x);

    if (threadIdx.x < producer_threads) {
        std::uint64_t cache_policy = 0;
        if constexpr (EvictFirst) {
            cache_policy = make_evict_first_policy();
        }
#pragma unroll 1
        for (int n_tile = 0; n_tile < 2; ++n_tile) {
#pragma unroll 1
            for (int k_tile = 0; k_tile < kGluK / kBlockK; ++k_tile) {
                const int sequence = n_tile * (kGluK / kBlockK) + k_tile;
                gate_pipeline.producer_acquire();
                copy_glu_stage<producer_threads, EvictFirst>(
                    gate_pipeline,
                    gate_stages + (sequence & 1) * kStageElements, gate_weight,
                    zero_chunk, shard, n_tile, k_tile,
                    static_cast<int>(threadIdx.x), cache_policy);
                gate_pipeline.producer_commit();

                up_pipeline.producer_acquire();
                copy_glu_stage<producer_threads, EvictFirst>(
                    up_pipeline, up_stages + (sequence & 1) * kStageElements,
                    up_weight, zero_chunk, shard, n_tile, k_tile,
                    static_cast<int>(threadIdx.x), cache_policy);
                up_pipeline.producer_commit();
            }
        }
        return;
    }

    const bool computes =
        threadIdx.x >= consumer_begin && threadIdx.x < consumer_end;
    const int consumer_thread = static_cast<int>(threadIdx.x) - consumer_begin;
    const int consumer_warp = consumer_thread >> 5;
    const int lane = consumer_thread & 31;
    const int n_in_tile = consumer_warp * 8 + (lane >> 2);
    const int k_subgroup = lane & 3;
    const bf16 *shard_input = input + shard * kGluK;

#pragma unroll 1
    for (int n_tile = 0; n_tile < 2; ++n_tile) {
        float gate_accumulator = 0.0f;
        float up_accumulator = 0.0f;
#pragma unroll 1
        for (int k_tile = 0; k_tile < kGluK / kBlockK; ++k_tile) {
            const int sequence = n_tile * (kGluK / kBlockK) + k_tile;
            gate_pipeline.consumer_wait();
            up_pipeline.consumer_wait();
            if (computes) {
                const bf16 *gate_stage =
                    gate_stages + (sequence & 1) * kStageElements;
                const bf16 *up_stage =
                    up_stages + (sequence & 1) * kStageElements;
#pragma unroll
                for (int k_group = 0; k_group < kBlockK / 32; ++k_group) {
                    const int local_k = k_group * 32 + k_subgroup * 8;
                    dot8_pair(gate_stage + glu_stage_offset(n_in_tile, local_k),
                              up_stage + glu_stage_offset(n_in_tile, local_k),
                              shard_input + k_tile * kBlockK + local_k,
                              gate_accumulator, up_accumulator);
                }
            }
            gate_pipeline.consumer_release();
            up_pipeline.consumer_release();
        }
        if (computes) {
            gate_accumulator = reduce_width4(gate_accumulator);
            up_accumulator = reduce_width4(up_accumulator);
            if (k_subgroup == 0) {
                const int local_n = n_tile * kBlockN + n_in_tile;
                if (local_n < kGluLocalN) {
                    const float silu =
                        gate_accumulator / (1.0f + __expf(-gate_accumulator));
                    output[shard * kGluLocalN + local_n] =
                        __float2bfloat16_rn(silu * up_accumulator);
                }
            }
        }
    }
}

__global__ __launch_bounds__(kBlockThreads, 1) void glu_direct_kernel(
    const bf16 *__restrict__ input, const bf16 *__restrict__ gate_weight,
    const bf16 *__restrict__ up_weight, bf16 *__restrict__ output) {
    if (threadIdx.x >= kConsumerThreads) {
        return;
    }
    const int shard = static_cast<int>(blockIdx.x);
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int n_in_tile = warp * 8 + (lane >> 2);
    const int k_subgroup = lane & 3;
    const bf16 *shard_input = input + shard * kGluK;

#pragma unroll 1
    for (int n_tile = 0; n_tile < 2; ++n_tile) {
        const int local_n = n_tile * kBlockN + n_in_tile;
        const int global_n = shard * kGluLocalN + local_n;
        float gate_accumulator = 0.0f;
        float up_accumulator = 0.0f;
        if (global_n < kGluGlobalN) {
#pragma unroll 1
            for (int k_base = 0; k_base < kGluK; k_base += 32) {
                const int global_k = k_base + k_subgroup * 8;
                dot8_pair(
                    packed_vec8(gate_weight, global_n, global_k, kGluGlobalN),
                    packed_vec8(up_weight, global_n, global_k, kGluGlobalN),
                    shard_input + global_k, gate_accumulator, up_accumulator);
            }
        }
        gate_accumulator = reduce_width4(gate_accumulator);
        up_accumulator = reduce_width4(up_accumulator);
        if (k_subgroup == 0 && local_n < kGluLocalN) {
            const float silu =
                gate_accumulator / (1.0f + __expf(-gate_accumulator));
            output[shard * kGluLocalN + local_n] =
                __float2bfloat16_rn(silu * up_accumulator);
        }
    }
}

__global__ __launch_bounds__(kBlockThreads, 1) void o_direct_kernel(
    const bf16 *__restrict__ input, const bf16 *__restrict__ weight,
    bf16 *__restrict__ output) {
    if (threadIdx.x >= kConsumerThreads) {
        return;
    }
    const int shard = static_cast<int>(blockIdx.x);
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int n_in_shard = warp * 4 + (lane >> 3);
    const int k_subgroup = lane & 7;
    const int global_n = shard * kOLocalN + n_in_shard;
    const bf16 *shard_input = input + shard * kOK;
    float accumulator = 0.0f;
#pragma unroll 1
    for (int k_base = 0; k_base < kOK; k_base += 64) {
        const int global_k = k_base + k_subgroup * 8;
        accumulator += dot8(packed_vec8(weight, global_n, global_k, kOGlobalN),
                            shard_input + global_k);
    }
    accumulator = reduce_width8(accumulator);
    if (k_subgroup == 0) {
        output[global_n] = __float2bfloat16_rn(accumulator);
    }
}

__global__ void flush_cache_kernel(uint4 *buffer, std::size_t count,
                                   uint32_t seed) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const uint32_t value = static_cast<uint32_t>(index) ^ seed;
        buffer[index] = make_uint4(value, value * 1664525u + 1013904223u,
                                   value ^ 0x9e3779b9u, value + 0x7f4a7c15u);
    }
}

struct Summary {
    std::vector<float> samples_us;
    double median_us = 0.0;
    double p10_us = 0.0;
    double p90_us = 0.0;
    double mad_us = 0.0;
    double minimum_us = 0.0;
    double maximum_us = 0.0;
};

double percentile(const std::vector<float> &sorted, double quantile) {
    if (sorted.empty()) {
        return 0.0;
    }
    const double position = (sorted.size() - 1) * quantile;
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(lower + 1, sorted.size() - 1);
    const double fraction = position - lower;
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

Summary summarize(std::vector<float> samples_us) {
    std::vector<float> sorted = samples_us;
    std::sort(sorted.begin(), sorted.end());
    Summary result;
    result.samples_us = std::move(samples_us);
    result.median_us = percentile(sorted, 0.5);
    result.p10_us = percentile(sorted, 0.1);
    result.p90_us = percentile(sorted, 0.9);
    result.minimum_us = sorted.front();
    result.maximum_us = sorted.back();
    std::vector<float> deviations;
    deviations.reserve(sorted.size());
    for (float value : sorted) {
        deviations.push_back(
            static_cast<float>(std::abs(value - result.median_us)));
    }
    std::sort(deviations.begin(), deviations.end());
    result.mad_us = percentile(deviations, 0.5);
    return result;
}

template <typename Launch>
Summary measure_warm(Launch launch, int warmup, int repeats, int iterations) {
    for (int index = 0; index < warmup; ++index) {
        launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> samples;
    samples.reserve(repeats);
    for (int repeat = 0; repeat < repeats; ++repeat) {
        cudaEvent_t start;
        cudaEvent_t end;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));
        CUDA_CHECK(cudaEventRecord(start));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            launch();
        }
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, end));
        samples.push_back(elapsed_ms * 1000.0f / iterations);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(end));
    }
    return summarize(std::move(samples));
}

template <typename Launch>
Summary measure_cold(Launch launch, uint4 *cache_buffer,
                     std::size_t cache_count, int warmup, int repeats) {
    for (int index = 0; index < warmup; ++index) {
        launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<cudaEvent_t> starts(repeats);
    std::vector<cudaEvent_t> ends(repeats);
    for (int index = 0; index < repeats; ++index) {
        CUDA_CHECK(cudaEventCreate(&starts[index]));
        CUDA_CHECK(cudaEventCreate(&ends[index]));
        const int blocks = static_cast<int>((cache_count + 255) / 256);
        flush_cache_kernel<<<blocks, 256>>>(cache_buffer, cache_count,
                                            static_cast<uint32_t>(index + 1));
        CUDA_CHECK(cudaEventRecord(starts[index]));
        launch();
        CUDA_CHECK(cudaEventRecord(ends[index]));
    }
    CUDA_CHECK(cudaEventSynchronize(ends.back()));
    std::vector<float> samples;
    samples.reserve(repeats);
    for (int index = 0; index < repeats; ++index) {
        float elapsed_ms = 0.0f;
        CUDA_CHECK(
            cudaEventElapsedTime(&elapsed_ms, starts[index], ends[index]));
        samples.push_back(elapsed_ms * 1000.0f);
        CUDA_CHECK(cudaEventDestroy(starts[index]));
        CUDA_CHECK(cudaEventDestroy(ends[index]));
    }
    return summarize(std::move(samples));
}

std::vector<bf16> random_bf16(std::size_t count, float magnitude,
                              uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-magnitude, magnitude);
    std::vector<bf16> result(count);
    for (bf16 &value : result) {
        value = __float2bfloat16_rn(distribution(generator));
    }
    return result;
}

std::vector<bf16> pack_k_major_bf16(const std::vector<bf16> &logical,
                                    int total_n, int k_extent) {
    if (total_n % 8 != 0 || k_extent % 16 != 0 ||
        logical.size() != static_cast<std::size_t>(total_n) * k_extent) {
        throw std::invalid_argument(
            "invalid logical K-major BF16 weight shape");
    }
    std::vector<bf16> packed(logical.size());
    for (int k_outer = 0; k_outer < k_extent / 16; ++k_outer) {
        for (int n_outer = 0; n_outer < total_n / 8; ++n_outer) {
            for (int n_lane = 0; n_lane < 8; ++n_lane) {
                for (int k_pack = 0; k_pack < 2; ++k_pack) {
                    for (int k_lane = 0; k_lane < 8; ++k_lane) {
                        const int n = n_outer * 8 + n_lane;
                        const int k = k_outer * 16 + k_pack * 8 + k_lane;
                        const std::size_t packed_index =
                            (((((static_cast<std::size_t>(k_outer) *
                                     (total_n / 8) +
                                 n_outer) *
                                    8 +
                                n_lane) *
                                   2 +
                               k_pack) *
                              8) +
                             k_lane);
                        packed[packed_index] =
                            logical[static_cast<std::size_t>(n) * k_extent + k];
                    }
                }
            }
        }
    }
    return packed;
}

float host_linear(const std::vector<bf16> &weight, int n, const bf16 *input,
                  int k_extent) {
    float result = 0.0f;
    for (int k = 0; k < k_extent; ++k) {
        result =
            std::fma(__bfloat162float(
                         weight[static_cast<std::size_t>(n) * k_extent + k]),
                     __bfloat162float(input[k]), result);
    }
    return result;
}

struct QkvReference {
    std::vector<bf16> q;
    std::vector<bf16> k;
    std::vector<bf16> v;
};

QkvReference reference_qkv(const std::vector<bf16> &input,
                           const std::vector<bf16> &q_weight,
                           const std::vector<bf16> &k_weight,
                           const std::vector<bf16> &v_weight) {
    QkvReference result{
        std::vector<bf16>(kQGlobalN),
        std::vector<bf16>(kKGlobalN),
        std::vector<bf16>(kVGlobalN),
    };
    for (int shard = 0; shard < kGrid; ++shard) {
        const bf16 *shard_input = input.data() + shard * kQkvK;
        for (int local_n = 0; local_n < kQLocalN; ++local_n) {
            const int global_n = shard * kQLocalN + local_n;
            result.q[global_n] = __float2bfloat16_rn(
                host_linear(q_weight, global_n, shard_input, kQkvK));
        }
        for (int local_n = 0; local_n < kKLocalN; ++local_n) {
            const int global_n = shard * kKLocalN + local_n;
            result.k[global_n] = __float2bfloat16_rn(
                host_linear(k_weight, global_n, shard_input, kQkvK));
            result.v[global_n] = __float2bfloat16_rn(
                host_linear(v_weight, global_n, shard_input, kQkvK));
        }
    }
    return result;
}

std::vector<bf16> reference_glu(const std::vector<bf16> &input,
                                const std::vector<bf16> &gate_weight,
                                const std::vector<bf16> &up_weight) {
    std::vector<bf16> result(kGluGlobalN);
    for (int shard = 0; shard < kGrid; ++shard) {
        const bf16 *shard_input = input.data() + shard * kGluK;
        for (int local_n = 0; local_n < kGluLocalN; ++local_n) {
            const int global_n = shard * kGluLocalN + local_n;
            const float gate =
                host_linear(gate_weight, global_n, shard_input, kGluK);
            const float up =
                host_linear(up_weight, global_n, shard_input, kGluK);
            const float silu = gate / (1.0f + std::exp(-gate));
            result[global_n] = __float2bfloat16_rn(silu * up);
        }
    }
    return result;
}

std::vector<bf16> reference_o(const std::vector<bf16> &input,
                              const std::vector<bf16> &weight) {
    std::vector<bf16> result(kOGlobalN);
    for (int shard = 0; shard < kGrid; ++shard) {
        const bf16 *shard_input = input.data() + shard * kOK;
        for (int local_n = 0; local_n < kOLocalN; ++local_n) {
            const int global_n = shard * kOLocalN + local_n;
            result[global_n] = __float2bfloat16_rn(
                host_linear(weight, global_n, shard_input, kOK));
        }
    }
    return result;
}

template <typename T> T *copy_to_device(const std::vector<T> &host) {
    T *device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    return device;
}

template <typename T> T *allocate_device(std::size_t count) {
    T *device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, count * sizeof(T)));
    return device;
}

std::vector<bf16> copy_from_device(const bf16 *device, std::size_t count) {
    std::vector<bf16> host(count);
    CUDA_CHECK(cudaMemcpy(host.data(), device, count * sizeof(bf16),
                          cudaMemcpyDeviceToHost));
    return host;
}

struct ErrorStats {
    double maximum_abs = 0.0;
    double mean_abs = 0.0;
};

ErrorStats compare(const std::vector<bf16> &lhs, const std::vector<bf16> &rhs) {
    if (lhs.size() != rhs.size()) {
        std::cerr << "comparison size mismatch" << std::endl;
        std::exit(1);
    }
    ErrorStats result;
    double sum = 0.0;
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        const double difference =
            std::abs(static_cast<double>(__bfloat162float(lhs[index])) -
                     static_cast<double>(__bfloat162float(rhs[index])));
        result.maximum_abs = std::max(result.maximum_abs, difference);
        sum += difference;
    }
    result.mean_abs = sum / lhs.size();
    return result;
}

struct KernelMeta {
    int registers_per_thread = 0;
    int static_shared_bytes = 0;
    int local_bytes = 0;
    int maximum_threads = 0;
    int active_blocks_per_sm = 0;
    int block_threads = 0;
    int dynamic_shared_bytes = 0;
};

template <typename Kernel>
KernelMeta kernel_meta(Kernel kernel, int block_threads, int dynamic_shared) {
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, kernel, block_threads, dynamic_shared));
    return {
        attributes.numRegs,
        static_cast<int>(attributes.sharedSizeBytes),
        static_cast<int>(attributes.localSizeBytes),
        attributes.maxThreadsPerBlock,
        active_blocks,
        block_threads,
        dynamic_shared,
    };
}

struct Result {
    std::string name;
    Summary cold;
    Summary warm;
    KernelMeta meta;
    std::size_t logical_weight_bytes = 0;
    std::size_t requested_weight_bytes = 0;
};

void print_summary(const Summary &summary) {
    std::cout << "{\"median_us\":" << summary.median_us
              << ",\"p10_us\":" << summary.p10_us
              << ",\"p90_us\":" << summary.p90_us
              << ",\"mad_us\":" << summary.mad_us
              << ",\"minimum_us\":" << summary.minimum_us
              << ",\"maximum_us\":" << summary.maximum_us
              << ",\"sample_count\":" << summary.samples_us.size() << '}';
}

void print_result(const Result &result) {
    const double cold_logical_gbps =
        result.logical_weight_bytes / (result.cold.median_us * 1000.0);
    const double warm_logical_gbps =
        result.logical_weight_bytes / (result.warm.median_us * 1000.0);
    const double cold_requested_gbps =
        result.requested_weight_bytes / (result.cold.median_us * 1000.0);
    const double warm_requested_gbps =
        result.requested_weight_bytes / (result.warm.median_us * 1000.0);
    std::cout << "\"" << result.name << "\":{";
    std::cout << "\"cold\":";
    print_summary(result.cold);
    std::cout << ",\"warm\":";
    print_summary(result.warm);
    std::cout << ",\"logical_weight_gbps\":{"
              << "\"cold\":" << cold_logical_gbps
              << ",\"warm\":" << warm_logical_gbps << '}';
    std::cout << ",\"requested_weight_gbps\":{"
              << "\"cold\":" << cold_requested_gbps
              << ",\"warm\":" << warm_requested_gbps << '}';
    std::cout << ",\"resources\":{"
              << "\"registers_per_thread\":" << result.meta.registers_per_thread
              << ",\"static_shared_bytes\":" << result.meta.static_shared_bytes
              << ",\"dynamic_shared_bytes\":"
              << result.meta.dynamic_shared_bytes
              << ",\"local_bytes\":" << result.meta.local_bytes
              << ",\"block_threads\":" << result.meta.block_threads
              << ",\"active_blocks_per_sm\":"
              << result.meta.active_blocks_per_sm << "}}";
}

int parse_integer(const char *text, const char *name) {
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 ||
        value > std::numeric_limits<int>::max()) {
        std::cerr << "invalid " << name << ": " << text << std::endl;
        std::exit(2);
    }
    return static_cast<int>(value);
}

} // namespace

int main(int argc, char **argv) {
    int cold_repeats = 200;
    int warm_repeats = 7;
    int warm_iterations = 200;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--cold-repeats" && index + 1 < argc) {
            cold_repeats = parse_integer(argv[++index], "cold repeats");
        } else if (argument == "--warm-repeats" && index + 1 < argc) {
            warm_repeats = parse_integer(argv[++index], "warm repeats");
        } else if (argument == "--warm-iterations" && index + 1 < argc) {
            warm_iterations = parse_integer(argv[++index], "warm iterations");
        } else {
            std::cerr << "unknown argument: " << argument << std::endl;
            return 2;
        }
    }

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 12 || properties.minor != 0) {
        std::cerr << "this calibrated benchmark requires SM120, got SM"
                  << properties.major << properties.minor << std::endl;
        return 2;
    }
    if (properties.regsPerBlock < kRegisterSpecializedRegisters) {
        std::cerr << "warpgroup register budgets require "
                  << kRegisterSpecializedRegisters
                  << " registers per block, but this device exposes "
                  << properties.regsPerBlock << std::endl;
        return 2;
    }

    CUDA_CHECK(cudaFuncSetAttribute(
        qkv_cp_async_kernel<kRoleSplitProducerWarps, false, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kQkvDynamicSharedBytes));
    CUDA_CHECK(cudaFuncSetAttribute(
        qkv_cp_async_kernel<kWarpGroupProducerWarps, false, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kQkvDynamicSharedBytes));
    CUDA_CHECK(cudaFuncSetAttribute(
        qkv_cp_async_kernel<kWarpGroupProducerWarps, false, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kQkvDynamicSharedBytes));
    CUDA_CHECK(cudaFuncSetAttribute(
        qkv_cp_async_kernel<kWarpGroupProducerWarps, true, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kQkvDynamicSharedBytes));
    CUDA_CHECK(cudaFuncSetAttribute(
        glu_cp_async_kernel<kRoleSplitProducerWarps, false, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kGluDynamicSharedBytes));
    CUDA_CHECK(cudaFuncSetAttribute(
        glu_cp_async_kernel<kWarpGroupProducerWarps, false, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kGluDynamicSharedBytes));
    CUDA_CHECK(cudaFuncSetAttribute(
        glu_cp_async_kernel<kWarpGroupProducerWarps, false, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kGluDynamicSharedBytes));
    CUDA_CHECK(cudaFuncSetAttribute(
        glu_cp_async_kernel<kWarpGroupProducerWarps, true, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kGluDynamicSharedBytes));
    CUDA_CHECK(cudaFuncSetAttribute(
        qkv_cp_async_kernel<kRoleSplitProducerWarps, false, false>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CUDA_CHECK(cudaFuncSetAttribute(
        qkv_cp_async_kernel<kWarpGroupProducerWarps, false, false>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CUDA_CHECK(cudaFuncSetAttribute(
        qkv_cp_async_kernel<kWarpGroupProducerWarps, false, true>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CUDA_CHECK(cudaFuncSetAttribute(
        qkv_cp_async_kernel<kWarpGroupProducerWarps, true, false>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CUDA_CHECK(cudaFuncSetAttribute(
        glu_cp_async_kernel<kRoleSplitProducerWarps, false, false>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CUDA_CHECK(cudaFuncSetAttribute(
        glu_cp_async_kernel<kWarpGroupProducerWarps, false, false>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CUDA_CHECK(cudaFuncSetAttribute(
        glu_cp_async_kernel<kWarpGroupProducerWarps, false, true>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CUDA_CHECK(cudaFuncSetAttribute(
        glu_cp_async_kernel<kWarpGroupProducerWarps, true, false>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));

    const auto h_qkv_input = random_bf16(kGrid * kQkvK, 1.0f, 101);
    const auto h_glu_input = random_bf16(kGrid * kGluK, 1.0f, 102);
    const auto h_o_input = random_bf16(kGrid * kOK, 1.0f, 103);
    const auto h_q_weight_logical = random_bf16(kQGlobalN * kQkvK, 0.02f, 201);
    const auto h_k_weight_logical = random_bf16(kKGlobalN * kQkvK, 0.02f, 202);
    const auto h_v_weight_logical = random_bf16(kVGlobalN * kQkvK, 0.02f, 203);
    const auto h_gate_weight_logical =
        random_bf16(kGluGlobalN * kGluK, 0.02f, 204);
    const auto h_up_weight_logical =
        random_bf16(kGluGlobalN * kGluK, 0.02f, 205);
    const auto h_o_weight_logical = random_bf16(kOGlobalN * kOK, 0.02f, 206);
    const auto h_q_weight =
        pack_k_major_bf16(h_q_weight_logical, kQGlobalN, kQkvK);
    const auto h_k_weight =
        pack_k_major_bf16(h_k_weight_logical, kKGlobalN, kQkvK);
    const auto h_v_weight =
        pack_k_major_bf16(h_v_weight_logical, kVGlobalN, kQkvK);
    const auto h_gate_weight =
        pack_k_major_bf16(h_gate_weight_logical, kGluGlobalN, kGluK);
    const auto h_up_weight =
        pack_k_major_bf16(h_up_weight_logical, kGluGlobalN, kGluK);
    const auto h_o_weight =
        pack_k_major_bf16(h_o_weight_logical, kOGlobalN, kOK);
    const std::vector<bf16> h_zero_chunk(kVectorElements,
                                         __float2bfloat16_rn(0.0f));

    bf16 *d_qkv_input = copy_to_device(h_qkv_input);
    bf16 *d_glu_input = copy_to_device(h_glu_input);
    bf16 *d_o_input = copy_to_device(h_o_input);
    bf16 *d_q_weight = copy_to_device(h_q_weight);
    bf16 *d_k_weight = copy_to_device(h_k_weight);
    bf16 *d_v_weight = copy_to_device(h_v_weight);
    bf16 *d_gate_weight = copy_to_device(h_gate_weight);
    bf16 *d_up_weight = copy_to_device(h_up_weight);
    bf16 *d_o_weight = copy_to_device(h_o_weight);
    bf16 *d_zero_chunk = copy_to_device(h_zero_chunk);

    bf16 *d_q_cp = allocate_device<bf16>(kQGlobalN);
    bf16 *d_k_cp = allocate_device<bf16>(kKGlobalN);
    bf16 *d_v_cp = allocate_device<bf16>(kVGlobalN);
    bf16 *d_q_ws = allocate_device<bf16>(kQGlobalN);
    bf16 *d_k_ws = allocate_device<bf16>(kKGlobalN);
    bf16 *d_v_ws = allocate_device<bf16>(kVGlobalN);
    bf16 *d_q_evict_first = allocate_device<bf16>(kQGlobalN);
    bf16 *d_k_evict_first = allocate_device<bf16>(kKGlobalN);
    bf16 *d_v_evict_first = allocate_device<bf16>(kVGlobalN);
    bf16 *d_q_reg_ws = allocate_device<bf16>(kQGlobalN);
    bf16 *d_k_reg_ws = allocate_device<bf16>(kKGlobalN);
    bf16 *d_v_reg_ws = allocate_device<bf16>(kVGlobalN);
    bf16 *d_q_direct = allocate_device<bf16>(kQGlobalN);
    bf16 *d_k_direct = allocate_device<bf16>(kKGlobalN);
    bf16 *d_v_direct = allocate_device<bf16>(kVGlobalN);
    bf16 *d_glu_cp = allocate_device<bf16>(kGluGlobalN);
    bf16 *d_glu_ws = allocate_device<bf16>(kGluGlobalN);
    bf16 *d_glu_evict_first = allocate_device<bf16>(kGluGlobalN);
    bf16 *d_glu_reg_ws = allocate_device<bf16>(kGluGlobalN);
    bf16 *d_glu_direct = allocate_device<bf16>(kGluGlobalN);
    bf16 *d_o = allocate_device<bf16>(kOGlobalN);

    const std::size_t cache_bytes = std::max<std::size_t>(
        static_cast<std::size_t>(properties.l2CacheSize) * 2,
        256ull * 1024 * 1024);
    const std::size_t cache_count = cache_bytes / sizeof(uint4);
    uint4 *d_cache = allocate_device<uint4>(cache_count);

    auto launch_qkv_cp = [&]() {
        qkv_cp_async_kernel<kRoleSplitProducerWarps, false, false>
            <<<kGrid, kBlockThreads, kQkvDynamicSharedBytes>>>(
                d_qkv_input, d_q_weight, d_k_weight, d_v_weight, d_q_cp, d_k_cp,
                d_v_cp);
    };
    auto launch_qkv_ws = [&]() {
        qkv_cp_async_kernel<kWarpGroupProducerWarps, false, false>
            <<<kGrid, kBlockThreads, kQkvDynamicSharedBytes>>>(
                d_qkv_input, d_q_weight, d_k_weight, d_v_weight, d_q_ws, d_k_ws,
                d_v_ws);
    };
    auto launch_qkv_evict_first = [&]() {
        qkv_cp_async_kernel<kWarpGroupProducerWarps, false, true>
            <<<kGrid, kBlockThreads, kQkvDynamicSharedBytes>>>(
                d_qkv_input, d_q_weight, d_k_weight, d_v_weight,
                d_q_evict_first, d_k_evict_first, d_v_evict_first);
    };
    auto launch_qkv_reg_ws = [&]() {
        qkv_cp_async_kernel<kWarpGroupProducerWarps, true, false>
            <<<kGrid, kBlockThreads, kQkvDynamicSharedBytes>>>(
                d_qkv_input, d_q_weight, d_k_weight, d_v_weight, d_q_reg_ws,
                d_k_reg_ws, d_v_reg_ws);
    };
    auto launch_qkv_direct = [&]() {
        qkv_direct_kernel<<<kGrid, kBlockThreads>>>(
            d_qkv_input, d_q_weight, d_k_weight, d_v_weight, d_q_direct,
            d_k_direct, d_v_direct);
    };
    auto launch_glu_cp = [&]() {
        glu_cp_async_kernel<kRoleSplitProducerWarps, false, false>
            <<<kGrid, kBlockThreads, kGluDynamicSharedBytes>>>(
                d_glu_input, d_gate_weight, d_up_weight, d_zero_chunk,
                d_glu_cp);
    };
    auto launch_glu_ws = [&]() {
        glu_cp_async_kernel<kWarpGroupProducerWarps, false, false>
            <<<kGrid, kBlockThreads, kGluDynamicSharedBytes>>>(
                d_glu_input, d_gate_weight, d_up_weight, d_zero_chunk,
                d_glu_ws);
    };
    auto launch_glu_evict_first = [&]() {
        glu_cp_async_kernel<kWarpGroupProducerWarps, false, true>
            <<<kGrid, kBlockThreads, kGluDynamicSharedBytes>>>(
                d_glu_input, d_gate_weight, d_up_weight, d_zero_chunk,
                d_glu_evict_first);
    };
    auto launch_glu_reg_ws = [&]() {
        glu_cp_async_kernel<kWarpGroupProducerWarps, true, false>
            <<<kGrid, kBlockThreads, kGluDynamicSharedBytes>>>(
                d_glu_input, d_gate_weight, d_up_weight, d_zero_chunk,
                d_glu_reg_ws);
    };
    auto launch_glu_direct = [&]() {
        glu_direct_kernel<<<kGrid, kBlockThreads>>>(d_glu_input, d_gate_weight,
                                                    d_up_weight, d_glu_direct);
    };
    auto launch_o = [&]() {
        o_direct_kernel<<<kGrid, kBlockThreads>>>(d_o_input, d_o_weight, d_o);
    };

    launch_qkv_cp();
    launch_qkv_ws();
    launch_qkv_evict_first();
    launch_qkv_reg_ws();
    launch_qkv_direct();
    launch_glu_cp();
    launch_glu_ws();
    launch_glu_evict_first();
    launch_glu_reg_ws();
    launch_glu_direct();
    launch_o();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto h_q_cp = copy_from_device(d_q_cp, kQGlobalN);
    const auto h_k_cp = copy_from_device(d_k_cp, kKGlobalN);
    const auto h_v_cp = copy_from_device(d_v_cp, kVGlobalN);
    const auto h_q_ws = copy_from_device(d_q_ws, kQGlobalN);
    const auto h_k_ws = copy_from_device(d_k_ws, kKGlobalN);
    const auto h_v_ws = copy_from_device(d_v_ws, kVGlobalN);
    const auto h_q_evict_first = copy_from_device(d_q_evict_first, kQGlobalN);
    const auto h_k_evict_first = copy_from_device(d_k_evict_first, kKGlobalN);
    const auto h_v_evict_first = copy_from_device(d_v_evict_first, kVGlobalN);
    const auto h_q_reg_ws = copy_from_device(d_q_reg_ws, kQGlobalN);
    const auto h_k_reg_ws = copy_from_device(d_k_reg_ws, kKGlobalN);
    const auto h_v_reg_ws = copy_from_device(d_v_reg_ws, kVGlobalN);
    const auto h_q_direct = copy_from_device(d_q_direct, kQGlobalN);
    const auto h_k_direct = copy_from_device(d_k_direct, kKGlobalN);
    const auto h_v_direct = copy_from_device(d_v_direct, kVGlobalN);
    const auto h_glu_cp = copy_from_device(d_glu_cp, kGluGlobalN);
    const auto h_glu_ws = copy_from_device(d_glu_ws, kGluGlobalN);
    const auto h_glu_evict_first =
        copy_from_device(d_glu_evict_first, kGluGlobalN);
    const auto h_glu_reg_ws = copy_from_device(d_glu_reg_ws, kGluGlobalN);
    const auto h_glu_direct = copy_from_device(d_glu_direct, kGluGlobalN);
    const auto h_o = copy_from_device(d_o, kOGlobalN);

    const auto q_error = compare(h_q_cp, h_q_direct);
    const auto k_error = compare(h_k_cp, h_k_direct);
    const auto v_error = compare(h_v_cp, h_v_direct);
    const auto glu_error = compare(h_glu_cp, h_glu_direct);
    const auto q_ws_error = compare(h_q_ws, h_q_direct);
    const auto k_ws_error = compare(h_k_ws, h_k_direct);
    const auto v_ws_error = compare(h_v_ws, h_v_direct);
    const auto glu_ws_error = compare(h_glu_ws, h_glu_direct);
    const auto q_evict_first_error = compare(h_q_evict_first, h_q_direct);
    const auto k_evict_first_error = compare(h_k_evict_first, h_k_direct);
    const auto v_evict_first_error = compare(h_v_evict_first, h_v_direct);
    const auto glu_evict_first_error = compare(h_glu_evict_first, h_glu_direct);
    const auto q_reg_ws_error = compare(h_q_reg_ws, h_q_direct);
    const auto k_reg_ws_error = compare(h_k_reg_ws, h_k_direct);
    const auto v_reg_ws_error = compare(h_v_reg_ws, h_v_direct);
    const auto glu_reg_ws_error = compare(h_glu_reg_ws, h_glu_direct);
    const double qkv_max_error = std::max({
        q_error.maximum_abs,
        k_error.maximum_abs,
        v_error.maximum_abs,
    });
    const double qkv_ws_max_error = std::max({
        q_ws_error.maximum_abs,
        k_ws_error.maximum_abs,
        v_ws_error.maximum_abs,
    });
    const double qkv_evict_first_max_error = std::max({
        q_evict_first_error.maximum_abs,
        k_evict_first_error.maximum_abs,
        v_evict_first_error.maximum_abs,
    });
    const double qkv_reg_ws_max_error = std::max({
        q_reg_ws_error.maximum_abs,
        k_reg_ws_error.maximum_abs,
        v_reg_ws_error.maximum_abs,
    });

    const auto qkv_reference =
        reference_qkv(h_qkv_input, h_q_weight_logical, h_k_weight_logical,
                      h_v_weight_logical);
    const auto glu_reference =
        reference_glu(h_glu_input, h_gate_weight_logical, h_up_weight_logical);
    const auto o_reference = reference_o(h_o_input, h_o_weight_logical);
    const auto q_reference_error = compare(h_q_direct, qkv_reference.q);
    const auto k_reference_error = compare(h_k_direct, qkv_reference.k);
    const auto v_reference_error = compare(h_v_direct, qkv_reference.v);
    const auto q_cp_reference_error = compare(h_q_cp, qkv_reference.q);
    const auto k_cp_reference_error = compare(h_k_cp, qkv_reference.k);
    const auto v_cp_reference_error = compare(h_v_cp, qkv_reference.v);
    const auto q_ws_reference_error = compare(h_q_ws, qkv_reference.q);
    const auto k_ws_reference_error = compare(h_k_ws, qkv_reference.k);
    const auto v_ws_reference_error = compare(h_v_ws, qkv_reference.v);
    const auto q_evict_first_reference_error =
        compare(h_q_evict_first, qkv_reference.q);
    const auto k_evict_first_reference_error =
        compare(h_k_evict_first, qkv_reference.k);
    const auto v_evict_first_reference_error =
        compare(h_v_evict_first, qkv_reference.v);
    const auto q_reg_ws_reference_error = compare(h_q_reg_ws, qkv_reference.q);
    const auto k_reg_ws_reference_error = compare(h_k_reg_ws, qkv_reference.k);
    const auto v_reg_ws_reference_error = compare(h_v_reg_ws, qkv_reference.v);
    const auto glu_reference_error = compare(h_glu_direct, glu_reference);
    const auto glu_cp_reference_error = compare(h_glu_cp, glu_reference);
    const auto glu_ws_reference_error = compare(h_glu_ws, glu_reference);
    const auto glu_evict_first_reference_error =
        compare(h_glu_evict_first, glu_reference);
    const auto glu_reg_ws_reference_error =
        compare(h_glu_reg_ws, glu_reference);
    const auto o_reference_error = compare(h_o, o_reference);
    const double qkv_reference_max_error = std::max({
        q_reference_error.maximum_abs,
        k_reference_error.maximum_abs,
        v_reference_error.maximum_abs,
    });
    const double qkv_cp_reference_max_error = std::max({
        q_cp_reference_error.maximum_abs,
        k_cp_reference_error.maximum_abs,
        v_cp_reference_error.maximum_abs,
    });
    const double qkv_ws_reference_max_error = std::max({
        q_ws_reference_error.maximum_abs,
        k_ws_reference_error.maximum_abs,
        v_ws_reference_error.maximum_abs,
    });
    const double qkv_evict_first_reference_max_error = std::max({
        q_evict_first_reference_error.maximum_abs,
        k_evict_first_reference_error.maximum_abs,
        v_evict_first_reference_error.maximum_abs,
    });
    const double qkv_reg_ws_reference_max_error = std::max({
        q_reg_ws_reference_error.maximum_abs,
        k_reg_ws_reference_error.maximum_abs,
        v_reg_ws_reference_error.maximum_abs,
    });
    if (qkv_max_error > 0.02 || qkv_ws_max_error > 0.02 ||
        qkv_evict_first_max_error > 0.02 || qkv_reg_ws_max_error > 0.02 ||
        glu_error.maximum_abs > 0.03 || glu_ws_error.maximum_abs > 0.03 ||
        glu_evict_first_error.maximum_abs > 0.03 ||
        glu_reg_ws_error.maximum_abs > 0.03 || qkv_reference_max_error > 0.02 ||
        qkv_cp_reference_max_error > 0.02 ||
        qkv_ws_reference_max_error > 0.02 ||
        qkv_evict_first_reference_max_error > 0.02 ||
        qkv_reg_ws_reference_max_error > 0.02 ||
        glu_reference_error.maximum_abs > 0.03 ||
        glu_cp_reference_error.maximum_abs > 0.03 ||
        glu_ws_reference_error.maximum_abs > 0.03 ||
        glu_evict_first_reference_error.maximum_abs > 0.03 ||
        glu_reg_ws_reference_error.maximum_abs > 0.03 ||
        o_reference_error.maximum_abs > 0.02) {
        std::cerr << "correctness check failed: cp-vs-direct qkv="
                  << qkv_max_error << ", ws-vs-direct qkv=" << qkv_ws_max_error
                  << ", evict-first-vs-direct qkv=" << qkv_evict_first_max_error
                  << ", reg-ws-vs-direct qkv=" << qkv_reg_ws_max_error
                  << ", cp-vs-direct glu=" << glu_error.maximum_abs
                  << ", ws-vs-direct glu=" << glu_ws_error.maximum_abs
                  << ", evict-first-vs-direct glu="
                  << glu_evict_first_error.maximum_abs
                  << ", reg-ws-vs-direct glu=" << glu_reg_ws_error.maximum_abs
                  << ", reference qkv=" << qkv_reference_max_error
                  << ", cp reference qkv=" << qkv_cp_reference_max_error
                  << ", ws reference qkv=" << qkv_ws_reference_max_error
                  << ", evict-first reference qkv="
                  << qkv_evict_first_reference_max_error
                  << ", reg-ws reference qkv=" << qkv_reg_ws_reference_max_error
                  << ", reference glu=" << glu_reference_error.maximum_abs
                  << ", cp reference glu=" << glu_cp_reference_error.maximum_abs
                  << ", ws reference glu=" << glu_ws_reference_error.maximum_abs
                  << ", evict-first reference glu="
                  << glu_evict_first_reference_error.maximum_abs
                  << ", reg-ws reference glu="
                  << glu_reg_ws_reference_error.maximum_abs
                  << ", reference o=" << o_reference_error.maximum_abs
                  << std::endl;
        return 1;
    }

    constexpr std::size_t qkv_weight_bytes =
        static_cast<std::size_t>(kQGlobalN + kKGlobalN + kVGlobalN) * kQkvK *
        sizeof(bf16);
    constexpr std::size_t glu_logical_weight_bytes =
        static_cast<std::size_t>(kGluGlobalN) * kGluK * sizeof(bf16) * 2;
    constexpr std::size_t glu_requested_weight_bytes =
        static_cast<std::size_t>(kGrid) * 2 * kBlockN * kGluK * sizeof(bf16) *
        2;
    constexpr std::size_t o_weight_bytes =
        static_cast<std::size_t>(kOGlobalN) * kOK * sizeof(bf16);

    std::vector<Result> results;
    results.push_back({
        "qkv_cp_async_1p8c_role_split",
        measure_cold(launch_qkv_cp, d_cache, cache_count, 25, cold_repeats),
        measure_warm(launch_qkv_cp, 25, warm_repeats, warm_iterations),
        kernel_meta(qkv_cp_async_kernel<kRoleSplitProducerWarps, false, false>,
                    kBlockThreads, kQkvDynamicSharedBytes),
        qkv_weight_bytes,
        qkv_weight_bytes,
    });
    results.push_back({
        "qkv_cp_async_4p8c_warpgroup",
        measure_cold(launch_qkv_ws, d_cache, cache_count, 25, cold_repeats),
        measure_warm(launch_qkv_ws, 25, warm_repeats, warm_iterations),
        kernel_meta(qkv_cp_async_kernel<kWarpGroupProducerWarps, false, false>,
                    kBlockThreads, kQkvDynamicSharedBytes),
        qkv_weight_bytes,
        qkv_weight_bytes,
    });
    results.push_back({
        "qkv_cp_async_4p8c_evict_first",
        measure_cold(launch_qkv_evict_first, d_cache, cache_count, 25,
                     cold_repeats),
        measure_warm(launch_qkv_evict_first, 25, warm_repeats, warm_iterations),
        kernel_meta(qkv_cp_async_kernel<kWarpGroupProducerWarps, false, true>,
                    kBlockThreads, kQkvDynamicSharedBytes),
        qkv_weight_bytes,
        qkv_weight_bytes,
    });
    results.push_back({
        "qkv_cp_async_4p8c_reg_specialized",
        measure_cold(launch_qkv_reg_ws, d_cache, cache_count, 25, cold_repeats),
        measure_warm(launch_qkv_reg_ws, 25, warm_repeats, warm_iterations),
        kernel_meta(qkv_cp_async_kernel<kWarpGroupProducerWarps, true, false>,
                    kBlockThreads, kQkvDynamicSharedBytes),
        qkv_weight_bytes,
        qkv_weight_bytes,
    });
    results.push_back({
        "qkv_direct",
        measure_cold(launch_qkv_direct, d_cache, cache_count, 25, cold_repeats),
        measure_warm(launch_qkv_direct, 25, warm_repeats, warm_iterations),
        kernel_meta(qkv_direct_kernel, kBlockThreads, 0),
        qkv_weight_bytes,
        qkv_weight_bytes,
    });
    results.push_back({
        "glu_cp_async_1p8c_role_split",
        measure_cold(launch_glu_cp, d_cache, cache_count, 25, cold_repeats),
        measure_warm(launch_glu_cp, 25, warm_repeats, warm_iterations),
        kernel_meta(glu_cp_async_kernel<kRoleSplitProducerWarps, false, false>,
                    kBlockThreads, kGluDynamicSharedBytes),
        glu_logical_weight_bytes,
        glu_requested_weight_bytes,
    });
    results.push_back({
        "glu_cp_async_4p8c_warpgroup",
        measure_cold(launch_glu_ws, d_cache, cache_count, 25, cold_repeats),
        measure_warm(launch_glu_ws, 25, warm_repeats, warm_iterations),
        kernel_meta(glu_cp_async_kernel<kWarpGroupProducerWarps, false, false>,
                    kBlockThreads, kGluDynamicSharedBytes),
        glu_logical_weight_bytes,
        glu_requested_weight_bytes,
    });
    results.push_back({
        "glu_cp_async_4p8c_evict_first",
        measure_cold(launch_glu_evict_first, d_cache, cache_count, 25,
                     cold_repeats),
        measure_warm(launch_glu_evict_first, 25, warm_repeats, warm_iterations),
        kernel_meta(glu_cp_async_kernel<kWarpGroupProducerWarps, false, true>,
                    kBlockThreads, kGluDynamicSharedBytes),
        glu_logical_weight_bytes,
        glu_requested_weight_bytes,
    });
    results.push_back({
        "glu_cp_async_4p8c_reg_specialized",
        measure_cold(launch_glu_reg_ws, d_cache, cache_count, 25, cold_repeats),
        measure_warm(launch_glu_reg_ws, 25, warm_repeats, warm_iterations),
        kernel_meta(glu_cp_async_kernel<kWarpGroupProducerWarps, true, false>,
                    kBlockThreads, kGluDynamicSharedBytes),
        glu_logical_weight_bytes,
        glu_requested_weight_bytes,
    });
    results.push_back({
        "glu_direct",
        measure_cold(launch_glu_direct, d_cache, cache_count, 25, cold_repeats),
        measure_warm(launch_glu_direct, 25, warm_repeats, warm_iterations),
        kernel_meta(glu_direct_kernel, kBlockThreads, 0),
        glu_logical_weight_bytes,
        glu_requested_weight_bytes,
    });
    results.push_back({
        "o_direct",
        measure_cold(launch_o, d_cache, cache_count, 25, cold_repeats),
        measure_warm(launch_o, 25, warm_repeats, warm_iterations),
        kernel_meta(o_direct_kernel, kBlockThreads, 0),
        o_weight_bytes,
        o_weight_bytes,
    });
    CUDA_CHECK(cudaGetLastError());

    std::cout << std::fixed << std::setprecision(6);
    std::cout << '{';
    std::cout << "\"device\":{"
              << "\"name\":\"" << properties.name << "\""
              << ",\"compute_capability\":\"" << properties.major << '.'
              << properties.minor << "\""
              << ",\"sm_count\":" << properties.multiProcessorCount
              << ",\"l2_bytes\":" << properties.l2CacheSize
              << ",\"registers_per_block\":" << properties.regsPerBlock
              << ",\"registers_per_sm\":" << properties.regsPerMultiprocessor
              << '}';
    std::cout << ",\"configuration\":{"
              << "\"grid\":" << kGrid << ",\"block_threads\":" << kBlockThreads
              << ",\"role_split_producer_warps\":" << kRoleSplitProducerWarps
              << ",\"warpgroup_producer_warps\":" << kWarpGroupProducerWarps
              << ",\"consumer_warps\":" << kConsumerWarps
              << ",\"role_split_padding_warps\":"
              << (kBlockThreads - kRoleSplitProducerWarps * kWarpSize -
                  kConsumerThreads) /
                     kWarpSize
              << ",\"warpgroup_aligned\":true"
              << ",\"producer_register_budget\":" << kProducerRegisterBudget
              << ",\"consumer_register_budget\":" << kConsumerRegisterBudget
              << ",\"block_n\":" << kBlockN << ",\"block_k\":" << kBlockK
              << ",\"stages\":" << kSharedStages
              << ",\"copy_primitive\":\"legacy_cp_async\""
              << ",\"cache_policy_ab\":"
                 "\"default_vs_fractional_l2_evict_first\""
              << ",\"shared_layout\":\"nv_mma_128b_xor\""
              << ",\"qkv_pipe\":\"depth4\""
              << ",\"glu_pipe\":\"paired_depth2_cuda_analogue\""
              << ",\"cold_repeats\":" << cold_repeats
              << ",\"warm_repeats\":" << warm_repeats
              << ",\"warm_iterations\":" << warm_iterations
              << ",\"cache_flush_bytes\":" << cache_bytes << '}';
    std::cout
        << ",\"correctness\":{"
        << "\"cp_vs_direct_qkv_max_abs\":" << qkv_max_error
        << ",\"cp_vs_direct_qkv_mean_abs\":"
        << (q_error.mean_abs + k_error.mean_abs + v_error.mean_abs) / 3.0
        << ",\"ws_vs_direct_qkv_max_abs\":" << qkv_ws_max_error
        << ",\"ws_vs_direct_qkv_mean_abs\":"
        << (q_ws_error.mean_abs + k_ws_error.mean_abs + v_ws_error.mean_abs) /
               3.0
        << ",\"evict_first_vs_direct_qkv_max_abs\":"
        << qkv_evict_first_max_error
        << ",\"evict_first_vs_direct_qkv_mean_abs\":"
        << (q_evict_first_error.mean_abs + k_evict_first_error.mean_abs +
            v_evict_first_error.mean_abs) /
               3.0
        << ",\"reg_ws_vs_direct_qkv_max_abs\":" << qkv_reg_ws_max_error
        << ",\"reg_ws_vs_direct_qkv_mean_abs\":"
        << (q_reg_ws_error.mean_abs + k_reg_ws_error.mean_abs +
            v_reg_ws_error.mean_abs) /
               3.0
        << ",\"cp_vs_direct_glu_max_abs\":" << glu_error.maximum_abs
        << ",\"cp_vs_direct_glu_mean_abs\":" << glu_error.mean_abs
        << ",\"ws_vs_direct_glu_max_abs\":" << glu_ws_error.maximum_abs
        << ",\"ws_vs_direct_glu_mean_abs\":" << glu_ws_error.mean_abs
        << ",\"evict_first_vs_direct_glu_max_abs\":"
        << glu_evict_first_error.maximum_abs
        << ",\"evict_first_vs_direct_glu_mean_abs\":"
        << glu_evict_first_error.mean_abs
        << ",\"reg_ws_vs_direct_glu_max_abs\":" << glu_reg_ws_error.maximum_abs
        << ",\"reg_ws_vs_direct_glu_mean_abs\":" << glu_reg_ws_error.mean_abs
        << ",\"reference_qkv_max_abs\":" << qkv_reference_max_error
        << ",\"reference_qkv_cp_max_abs\":" << qkv_cp_reference_max_error
        << ",\"reference_qkv_ws_max_abs\":" << qkv_ws_reference_max_error
        << ",\"reference_qkv_evict_first_max_abs\":"
        << qkv_evict_first_reference_max_error
        << ",\"reference_qkv_reg_ws_max_abs\":"
        << qkv_reg_ws_reference_max_error
        << ",\"reference_glu_max_abs\":" << glu_reference_error.maximum_abs
        << ",\"reference_glu_cp_max_abs\":"
        << glu_cp_reference_error.maximum_abs
        << ",\"reference_glu_ws_max_abs\":"
        << glu_ws_reference_error.maximum_abs
        << ",\"reference_glu_evict_first_max_abs\":"
        << glu_evict_first_reference_error.maximum_abs
        << ",\"reference_glu_reg_ws_max_abs\":"
        << glu_reg_ws_reference_error.maximum_abs
        << ",\"reference_o_max_abs\":" << o_reference_error.maximum_abs << '}';
    std::cout << ",\"results\":{";
    for (std::size_t index = 0; index < results.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        print_result(results[index]);
    }
    std::cout << "}}" << std::endl;

    CUDA_CHECK(cudaFree(d_cache));
    CUDA_CHECK(cudaFree(d_o));
    CUDA_CHECK(cudaFree(d_glu_direct));
    CUDA_CHECK(cudaFree(d_glu_reg_ws));
    CUDA_CHECK(cudaFree(d_glu_evict_first));
    CUDA_CHECK(cudaFree(d_glu_ws));
    CUDA_CHECK(cudaFree(d_glu_cp));
    CUDA_CHECK(cudaFree(d_v_direct));
    CUDA_CHECK(cudaFree(d_k_direct));
    CUDA_CHECK(cudaFree(d_q_direct));
    CUDA_CHECK(cudaFree(d_v_reg_ws));
    CUDA_CHECK(cudaFree(d_k_reg_ws));
    CUDA_CHECK(cudaFree(d_q_reg_ws));
    CUDA_CHECK(cudaFree(d_v_evict_first));
    CUDA_CHECK(cudaFree(d_k_evict_first));
    CUDA_CHECK(cudaFree(d_q_evict_first));
    CUDA_CHECK(cudaFree(d_v_ws));
    CUDA_CHECK(cudaFree(d_k_ws));
    CUDA_CHECK(cudaFree(d_q_ws));
    CUDA_CHECK(cudaFree(d_v_cp));
    CUDA_CHECK(cudaFree(d_k_cp));
    CUDA_CHECK(cudaFree(d_q_cp));
    CUDA_CHECK(cudaFree(d_zero_chunk));
    CUDA_CHECK(cudaFree(d_o_weight));
    CUDA_CHECK(cudaFree(d_up_weight));
    CUDA_CHECK(cudaFree(d_gate_weight));
    CUDA_CHECK(cudaFree(d_v_weight));
    CUDA_CHECK(cudaFree(d_k_weight));
    CUDA_CHECK(cudaFree(d_q_weight));
    CUDA_CHECK(cudaFree(d_o_input));
    CUDA_CHECK(cudaFree(d_glu_input));
    CUDA_CHECK(cudaFree(d_qkv_input));
    return 0;
}
