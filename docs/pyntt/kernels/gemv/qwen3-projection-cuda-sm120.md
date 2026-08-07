# Qwen3 projection CUDA warp specialization and L2 eviction on SM120

This benchmark measures the Qwen3 one-layer decode QKV, GLU, and O projection
shapes with raw CUDA. It preserves the generated PyNTT sharding, K-major packed
BF16 weights, N64/K128 staging, four shared stages, and 128-byte NVMMAShared XOR
layout. The QKV and GLU candidates use legacy `cp.async` with distinct producer
and consumer warp groups. A same-binary A/B changes only the `cp.async` L2
eviction priority; custom same-sharding direct-load kernels remain as mechanism
baselines.

The source revision measured here is
`e2dd82a941160bf028e54dd027fd98c65e2a3eb6ef5c5ab364cb9261386f48b8`:
[benchmark_qwen3_projection_cuda.cu](benchmark_qwen3_projection_cuda.cu).

## Variants

- `1p8c_role_split` keeps the previous one-producer/eight-consumer baseline in
  a 12-warp block. Its three padding warps still participate in the libcu++
  pipeline protocol.
- `4p8c_warpgroup` assigns one complete four-warp group to `cp.async` and two
  complete four-warp groups to compute. There are no padding warps.
- `4p8c_evict_first` is otherwise identical to `4p8c_warpgroup`. Each producer
  thread creates one 100%-fractional `L2::evict_first` policy outside the
  pipeline loops and attaches it to every weight `cp.async`. This lowers the
  retention priority of the fetched L2 lines; it does not bypass L2.
- `4p8c_reg_specialized` adds warp-group-uniform
  `setmaxnreg.dec.sync.aligned.u32 64` for producers and
  `setmaxnreg.inc.sync.aligned.u32 216` for consumers. The weighted request is
  63,488 registers, within the SM120 65,536-register CTA pool.
- The O projection stays direct because it has no independent, profitable
  global-to-shared transfer phase in this implementation.

The Python/Triton examples in this directory already call
`tle.gpu.warp_specialize`; the new A/B variants make the raw-CUDA analogue
physically warp-group aligned and expose dynamic register specialization
separately.

## Build and run

`setmaxnreg` is an architecture-specific SM120a feature. Generate only the
SM120a cubin: the `-arch=sm_120a` shorthand also asks nvcc for generic
`compute_120` PTX, which cannot assemble this instruction.

```sh
mkdir -p build/cuda_qwen3_projection_benchmark/final_evict_ab_keep

/usr/local/cuda/bin/nvcc \
  -std=c++17 -O3 -lineinfo \
  -gencode arch=compute_120a,code=sm_120a \
  --resource-usage \
  -Xptxas=-v,--warn-on-spills,--warn-on-local-memory-usage \
  -keep \
  -keep-dir build/cuda_qwen3_projection_benchmark/final_evict_ab_keep \
  -o build/cuda_qwen3_projection_benchmark/qwen3_projection_warp_specialized_evict_ab \
  docs/pyntt/kernels/gemv/benchmark_qwen3_projection_cuda.cu

build/cuda_qwen3_projection_benchmark/qwen3_projection_warp_specialized_evict_ab \
  --cold-repeats 200 \
  --warm-repeats 7 \
  --warm-iterations 200 \
  > build/cuda_qwen3_projection_benchmark/qwen3_projection_evict_ab_run_1.json
```

## Measurement

- GPU: NVIDIA GeForce RTX 5060 Ti, SM120, 36 SMs, 32-MiB L2.
- Cold-L2: clear a 256-MiB buffer before every individually timed launch, then
  report the median of 200 CUDA-event samples.
- Warm-L2: 25 warm-up launches followed by seven event batches of 200 launches;
  report the median batch mean.
- Correctness: create row-major logical weights, independently pack them, and
  compare every CUDA path against an FP32-accumulating CPU reference. QKV and O
  are exact after BF16 rounding; GLU maximum absolute error is 0.000122.
- The table uses run 1 of three consecutive full runs. Across the three runs,
  default/`evict_first` QKV warm time was 12.297--12.305/24.097--24.182 us;
  default/`evict_first` GLU warm time was
  24.599--24.620/43.904--44.507 us.

## Results

RTX 5060 Ti, 384 threads/block, microseconds:

| Projection/path | Cold p50 | Warm p50 | Registers/thread | Active blocks/SM |
| --- | ---: | ---: | ---: | ---: |
| QKV `cp.async`, 1P/8C role split | 49.152 | 32.795 | 52 | 1 |
| QKV `cp.async`, 4P/8C warp groups | 47.104 | 12.305 | 52 | 1 |
| QKV `cp.async`, 4P/8C + `evict_first` | 46.560 | 24.159 | 52 | 1 |
| QKV `cp.async`, 4P/8C + `setmaxnreg` | 46.560 | 12.297 | 168 | 1 |
| QKV direct | 53.248 | 12.296 | 29 | 4 |
| GLU `cp.async`, 1P/8C role split | 67.584 | 26.615 | 56 | 1 |
| GLU `cp.async`, 4P/8C warp groups | 67.584 | 24.610 | 56 | 1 |
| GLU `cp.async`, 4P/8C + `evict_first` | 75.776 | 44.269 | 56 | 1 |
| GLU `cp.async`, 4P/8C + `setmaxnreg` | 67.584 | 24.609 | 168 | 1 |
| GLU direct | 63.488 | 14.345 | 40 | 4 |
| O direct | 32.768 | 6.672 | 21 | 4 |

Moving from one to four producer warps makes warm QKV 2.67x faster and warm
GLU 1.08x faster. QKV reaches the direct-load warm result. Cold QKV improves
about 1.04x while cold GLU is unchanged, so the large QKV warm gain is a
strong indication of a producer issue-rate bottleneck rather than a reduction
in first-touch memory latency.

Adding `evict_first` is neutral within the coarse cold-event granularity for
QKV (0.99x the default time), but makes warm QKV 1.96x slower. It makes cold
GLU 1.12x and warm GLU 1.80x slower. The warm result is expected for this
isolated repeat-launch benchmark: both projection weights benefit from L2
residency, while the hint explicitly makes those lines the first eviction
candidates. The cold GLU penalty also shows lost reuse among duplicate weight
requests during one launch; it is not policy-creation startup overhead, since
the policy is created only once per producer thread and QKV cold time does not
regress.

At the start of this experiment, the selected PyNTT SIMT shared-memory
pipeline templates for QKV, GLU, and matmul all requested `evict_first`, so the
hinted row was the closer cache-policy analogue. The generated follow-ups below
remove the QKV hint and change GLU to `evict_last`; matmul remains unchanged.
This does not mean either hint is globally desirable: a full decode layer also
has activations, residuals, KV metadata, and following-operation weights
competing for L2. Other PyNTT algorithms, including the MMA templates, must be
assessed separately.

Dynamic register specialization is neutral for these kernels. The current
consumer keeps only one QKV accumulator or one gate/up accumulator pair, so a
216-register ceiling does not create more instruction-level parallelism or a
larger partial layout. It also cannot improve occupancy: the 65,616-byte shared
arena already limits each `cp.async` variant to one block per SM.

## Generated PyNTT QKV follow-up

The generated QKV TMA kernel uses one concatenated logical Q/K/V N stream,
N64/K128, four shared stages, grid 32, eight consumer warps, and one producer
warp. The original common-N32 schedule filled both N64 tiles with two copies.
The retained schedule instead fills the Q tile with one N64 copy and the second
tile with exact K and V N32 copies. It therefore reduces TMA requests from 32
to 24 per CTA (1,024 to 768 over grid 32) while keeping exact traffic at
256 KiB per CTA, or 8 MiB per launch. No QKV copy has an L2 eviction hint.

| Generated QKV variant | Cold p50 (us) | Warm p50 (us) |
| --- | ---: | ---: |
| Common N32 copies, `evict_first`, run 1 | 43.552 | 24.694 |
| Common N32 copies, no hint, persistence-reset A/B | 45.056 | 18.457 |
| Mixed Q64/K32/V32 copies, no hint, persistence-reset A/B | 27.200 | 17.751 |
| Mixed Q64/K32/V32 copies, no hint, final | 27.168 | 17.276 |
| Mixed copies, `evict_last`, persistence reset first | 27.168 | 17.443 |
| FlagGems concatenated raw GEMV, same final run | 26.080 | 11.706 |
| FlagGems separate three raw GEMVs, same final run | 32.768 | 38.586 |

Under the same persistence-reset methodology, the mixed schedule is 1.656x
faster cold and 1.040x faster warm than the common-N32 no-hint schedule. In the
final standard run it is 1.21x/2.23x faster than three separate FlagGems GEMVs
cold/warm. The single concatenated FlagGems GEMV remains 1.04x/1.48x faster
cold/warm. Every measurement passed the complete Q/K/V output check.

A rank-5 per-CTA TensorMap was also compiled and tested with shape
`[32,4,64,2,64]`, block `[1,8,8,2,64]`, and a nonzero-primed shared slot. Its
valid N32 half was exact, its out-of-bounds N32 half was overwritten with zero,
and PTX used `cp.async.bulk.tensor.5d` with no spills. This proves that a CTA
dimension can create a local zero-fill boundary. It is not retained for this
QKV shape: Q64/K32/V32 are already exact. Uniform padded N64 K and V copies
cannot share the second N64 slot because each copy's zero-filled half would
overwrite the other's valid half; placing them in separate slots would increase
shared writes and consumer FMA width from 128N to 192N. Rank-5 descriptors with
the exact Q64/K32/V32 blocks would have the same requests and bytes as the simpler
rank-4 descriptors.

`triton.testing.do_bench` pressures L2 by zeroing a 256-MiB buffer, but that is
not an invalidation. `evict_last` weight lines can survive that normal-priority
traffic and produced misleading roughly 18-us "cold" samples. Calling
`cudaCtxResetPersistingL2Cache()` before the same normal flush restored the
mixed `evict_last` result to 27.168 us, equal to no hint. The implementation
therefore uses no QKV eviction hint and does not obtain its result by retaining
projection weights over the cache flush.

The final artifact has Q/K/V descriptor blocks `[8,8,2,64]`, `[4,8,2,64]`,
and `[4,8,2,64]`. PTX contains 24 `cp.async.bulk.tensor.4d` sites and no L2
cache-hint form. The kernel uses 65,740 bytes of shared memory, an 8-byte stack
frame, zero global scratch, and zero spill loads/stores.

## Generated PyNTT GLU follow-up

The generated TMA kernel was measured with the first Qwen3-0.6B layer, BF16
input/weights, grid 32, eight consumer warps, one producer warp, N64/K128, and
four shared stages. A strict reset-cold sample calls
`cudaCtxResetPersistingL2Cache()` and then writes a 256-MiB normal-priority
flush buffer before every timed launch. Warm samples use nine batches of 400
launches. JIT, descriptor construction, and rdata upload are excluded.

The local output N is 96, so the second N64 transfer of each Gate/Up projection
has only 32 valid values. The retained rank-4 descriptor can read the following
CTA's 32 values (the final grid boundary zero-fills instead) and therefore
schedules 16 MiB over grid 32 for 12 MiB of unique weights. A tested per-CTA
rank-5 descriptor instead used:

```text
shape       = [32, 64, 12, 2, 64]
strides     = [1536, 49152, 128, 64, 1]
block_shape = [1, 8, 8, 2, 64]
```

The leading coordinate selects the CTA's exact 12-element packed-N shard. Its
second N64 transfer reads the four valid packed-N rows and TMA zero-fills the
remaining four, so no neighboring CTA weight is fetched. It keeps the same
1,024 dynamic TMA requests and 16 MiB of TMA-to-Shared writes, but reduces
in-bounds global/L2 weight reads to exactly 12 MiB. A nonzero-primed Shared
probe confirmed that every out-of-bounds value was overwritten with zero.

| Strict reset-cold GLU variant | Cold p50 (us) | Warm p50 (us) |
| --- | ---: | ---: |
| Rank-4 N64, `evict_last` | 38.912 | 16.013 |
| Rank-4 N64, no hint control | 39.280 | 16.119 |
| Per-CTA rank-5 N64, no hint | 38.912 | 16.412 |

The rank-5 result is cold-neutral and 2.5% slower warm than the retained
rank-4 kernel. It is numerically correct (`max_abs_error=8.50e-5`) and uses
`cp.async.bulk.tensor.5d` without an L2 policy, global scratch, or spills. Both
valid variants use 12 physical warps, 65,740 bytes of Shared, and an 8-byte
stack frame.

The earlier exact-N32 subcopy timing is invalid and must not be used for a
performance conclusion. A stricter comparison found `max_abs_error=0.017329`
and mean absolute error `0.001256`; one failing element was zero instead of
`-0.017371`. The six rank-4 TMA sites did not map the smaller copies into the
parent N64 NVMMAShared layout correctly. It happened to pass the old absolute
tolerance because the test values were too small.

`triton.testing.do_bench` with only its normal-priority flush reported
33.312/16.056 us for rank-4 and 38.400/16.622 us for rank-5. The first number
is not a true cold-L2 comparison: `evict_last` lines can survive that flush.
The persistence reset above removes this artifact. In the same standard run,
FlagGems measured 36.192/11.858 us for raw concatenated GEMV and
41.536/41.101 us including SwiGLU; only the latter has the same fused semantic
boundary as PyNTT.

A full one-layer decode A/B loaded both generated packages in one process,
bound them to one byte-identical device rdata allocation, and alternated AB/BA
order. Across 128 reset-cold samples per variant, both complete mega-kernels
measured 882.688 us. Across 200 warm samples, rank-4 measured 951.440 us and
rank-5 measured 958.480 us. Balancing the large second-launch order effect gave
a rank-5 penalty of 6.363 us, or 0.67%, with a 95% confidence interval of
5.203--7.524 us. Full 151,936-element logits were bitwise identical before and
after the run.

The retained implementation remains rank-4 N64 with `evict_last`: rank-5 proves
that a per-CTA TensorMap can eliminate the nominal 4-MiB cross-CTA tail traffic
without an L2 hint, but it gives no reset-cold gain and loses 2.5% in the
isolated warm kernel. The final artifact therefore keeps two `[8,8,2,64]`
descriptor blocks; ptxas reports zero spill loads/stores and zero global
scratch.

## Instruction and safety checks

The retained PTX contains:

```text
cp.async.cg.shared.global [...], [...], 16, 16;
createpolicy.fractional.L2::evict_first.b64 policy, fraction;
cp.async.cg.shared.global.L2::cache_hint [...], [...], 16, 16, policy;
setmaxnreg.dec.sync.aligned.u32 64;
setmaxnreg.inc.sync.aligned.u32 216;
```

SASS contains `LDGSTS.E.BYPASS.128`,
`USETMAXREG.DEALLOC.CTAPOOL`, and `USETMAXREG.TRY_ALLOC.CTAPOOL`. The retained
PTX has exactly two `createpolicy` sites, one in each hinted QKV/GLU kernel,
while the default kernels retain unhinted `cp.async`. The SASS copy mnemonic is
unchanged because the cache policy is carried by the load descriptor. There is
no `cp.async.bulk` or tensor-memory instruction in this raw-CUDA benchmark.

All variants compile with zero stack bytes, local bytes, spill stores, and spill
loads. Reduced `compute-sanitizer --tool memcheck` and `--tool synccheck` runs
both report zero errors.

## Interpretation limits

This remains a structural CUDA analogue rather than a transport-only clone of
generated PyNTT. PyNTT has a larger consumer partial layout. The CUDA GLU
implementation uses two lockstep depth-2 rings to emulate the four interleaved
gate/up slots while retaining the wait-both, one-input-load consumer schedule.
Runtime data-pool strides and the O-projection wrapper also differ. The direct
paths are custom packed kernels, not FlagGems.

Warm-L2 methodology matches the PyNTT benchmark. Cold-L2 has the same event
range and 256-MiB cache clear, but uses a fixed sample count instead of Triton's
duration-adaptive sample count. Cross-harness cold ratios should therefore be
treated as directional evidence.
