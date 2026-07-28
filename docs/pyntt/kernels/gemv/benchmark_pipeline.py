#!/usr/bin/env python3
"""Benchmark the SM120 warp-specialized K-major BF16 GEMV pipeline."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


WEIGHT_LAYOUT = tl.constexpr(
    tle.gpu.BlockEncoding(
        [1, 8],
        [8, 4],
        [8, 1],
        [1, 0],
    )
)
X_LAYOUT = tl.constexpr(
    tle.gpu.SlicedEncoding(
        0,
        WEIGHT_LAYOUT.value,
    )
)
OUTPUT_LAYOUT = tl.constexpr(
    tle.gpu.SlicedEncoding(
        1,
        WEIGHT_LAYOUT.value,
    )
)

N_VECTOR = 8
K_PACK = 2
K_VECTOR = 8
K_ATOM = K_PACK * K_VECTOR
REDUCTION_GROUP = 32
N_VECTOR_TL = tl.constexpr(N_VECTOR)
K_PACK_TL = tl.constexpr(K_PACK)
K_VECTOR_TL = tl.constexpr(K_VECTOR)
K_ATOM_TL = tl.constexpr(K_ATOM)
REDUCTION_GROUP_TL = tl.constexpr(REDUCTION_GROUP)


@triton.jit
def _producer(
    writer,
    weight,
    k: tl.constexpr,
    n_groups: tl.constexpr,
    groups: tl.constexpr,
    block_k: tl.constexpr,
):
    pid = tl.program_id(0)
    grid = tl.num_programs(0)
    num_k_tiles: tl.constexpr = k // block_k
    block_n: tl.constexpr = groups * 32
    packed_k_outer: tl.constexpr = block_k // K_ATOM_TL
    packed_n_outer: tl.constexpr = block_n // N_VECTOR_TL
    shared_rows: tl.constexpr = packed_k_outer * packed_n_outer
    shared_cols: tl.constexpr = N_VECTOR_TL * K_ATOM_TL
    local_row = tl.arange(0, shared_rows)[:, None]
    local_col = tl.arange(0, shared_cols)[None, :]
    for work_index in tl.range(pid, n_groups // groups, grid):
        n_start = work_index * block_n
        local_work_index = (work_index - pid) // grid
        for k_tile in tl.range(0, num_k_tiles):
            sequence = local_work_index * num_k_tiles + k_tile
            slot = writer.acquire(sequence)
            k_outer = (
                k_tile * packed_k_outer
                + local_row // packed_n_outer
            )
            n_outer = (
                n_start // N_VECTOR_TL
                + local_row % packed_n_outer
            )
            n_lane = local_col // K_ATOM_TL
            k_inner = local_col % K_ATOM_TL
            k_pack = k_inner // K_VECTOR_TL
            k_lane = k_inner % K_VECTOR_TL
            weight_offset = (
                (
                    (
                        (
                            k_outer * (n_groups * 32 // N_VECTOR_TL)
                            + n_outer
                        )
                        * N_VECTOR_TL
                        + n_lane
                    )
                    * K_PACK_TL
                    + k_pack
                )
                * K_VECTOR_TL
                + k_lane
            )
            tle.gpu.copy(
                tl.max_contiguous(
                    weight + weight_offset,
                    [1, shared_cols],
                ),
                slot.weight,
                [shared_rows, shared_cols],
            )
            writer.commit(sequence)


@triton.jit
def _consumer(
    reader,
    x,
    output,
    k: tl.constexpr,
    n_groups: tl.constexpr,
    groups: tl.constexpr,
    block_k: tl.constexpr,
):
    pid = tl.program_id(0)
    grid = tl.num_programs(0)
    num_k_tiles: tl.constexpr = k // block_k
    block_n: tl.constexpr = groups * 32
    packed_n_outer: tl.constexpr = block_n // N_VECTOR_TL
    local_n = tle.encoding(
        tle.encoding(tl.arange(0, block_n), OUTPUT_LAYOUT)[:, None],
        WEIGHT_LAYOUT,
    )
    local_k_vector = tle.encoding(
        tl.arange(0, REDUCTION_GROUP_TL),
        X_LAYOUT,
    )
    local_k = tle.encoding(
        local_k_vector[None, :],
        WEIGHT_LAYOUT,
    )
    for work_index in tl.range(pid, n_groups // groups, grid):
        n_start = work_index * block_n
        local_work_index = (work_index - pid) // grid
        partial = tle.encoding(
            tl.zeros((block_n, REDUCTION_GROUP_TL), tl.float32),
            WEIGHT_LAYOUT,
        )
        for k_tile in tl.range(0, num_k_tiles):
            sequence = local_work_index * num_k_tiles + k_tile
            ready = reader.wait(sequence)
            for k_group in tl.static_range(
                0,
                block_k // REDUCTION_GROUP_TL,
            ):
                shared_k = k_group * REDUCTION_GROUP_TL + local_k
                shared_row = (
                    shared_k // K_ATOM_TL * packed_n_outer
                    + local_n // N_VECTOR_TL
                )
                shared_col = (
                    local_n % N_VECTOR_TL * K_ATOM_TL
                    + shared_k % K_ATOM_TL
                )
                weight_ptr = tle.gpu.local_ptr(
                    ready.slot.weight,
                    (shared_row, shared_col),
                    shape=(block_n, REDUCTION_GROUP_TL),
                )
                weight_ptr = tl.max_contiguous(weight_ptr, [1, 8])
                weight_ptr = tl.multiple_of(weight_ptr, [1, 16])
                x_offset = (
                    k_tile * block_k
                    + k_group * REDUCTION_GROUP_TL
                    + local_k_vector
                )
                weight_value = tle.encoding(
                    tl.load(weight_ptr),
                    WEIGHT_LAYOUT,
                )
                x_value = tle.encoding(
                    tl.load(x + x_offset),
                    X_LAYOUT,
                )
                x_value = tle.encoding(x_value[None, :], WEIGHT_LAYOUT)
                partial += (
                    weight_value.to(tl.float32)
                    * x_value.to(tl.float32)
                )
            reader.release(sequence)

        result = tl.sum(partial, axis=1)
        output_n = n_start + tle.encoding(
            tl.arange(0, block_n),
            OUTPUT_LAYOUT,
        )
        tl.store(output + output_n, result)


@triton.jit
def packed_gemv_pipeline(
    x,
    weight,
    output,
    k: tl.constexpr,
    n_groups: tl.constexpr,
    groups: tl.constexpr,
    block_k: tl.constexpr,
    stages: tl.constexpr,
):
    block_n: tl.constexpr = groups * 32
    packed_k_outer: tl.constexpr = block_k // K_ATOM_TL
    packed_n_outer: tl.constexpr = block_n // N_VECTOR_TL
    shared_rows: tl.constexpr = packed_k_outer * packed_n_outer
    shared_cols: tl.constexpr = N_VECTOR_TL * K_ATOM_TL
    weight_shared_layout: tl.constexpr = tle.gpu.swizzled_shared_layout(
        vectorSize=8,
        perPhase=2,
        maxPhase=8,
        order=[2, 1, 0],
        numCTAs=[1, 1, 1],
        numCTAsPerCGA=[1, 1, 1],
        numCTASplit=[1, 1, 1],
        numCTAOrder=[2, 1, 0],
    )
    weight_stages = tle.gpu.alloc(
        [stages, shared_rows, shared_cols],
        dtype=weight.dtype.element_ty,
        layout=weight_shared_layout,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    weight_pipe = tle.pipe(
        capacity=stages,
        scope="cta",
        name="packed_gemv_weight",
        weight=weight_stages,
    )
    tle.gpu.warp_specialize(
        [
            (
                _consumer,
                (
                    weight_pipe.reader(),
                    x,
                    output,
                    k,
                    n_groups,
                    groups,
                    block_k,
                ),
            ),
            (
                _producer,
                (
                    weight_pipe.writer(),
                    weight,
                    k,
                    n_groups,
                    groups,
                    block_k,
                ),
            ),
        ],
        [1],
        [48],
    )


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        k_text, n_text = value.lower().split("x", maxsplit=1)
        k, n = int(k_text), int(n_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            f"shape must be KxN, got {value!r}"
        ) from exc
    if k <= 0 or n <= 0 or k % 256 != 0 or n % 128 != 0:
        raise argparse.ArgumentTypeError(
            "K and N must be positive, with K divisible by 256 and N by 128"
        )
    return k, n


def _pack_weight(weight: torch.Tensor) -> torch.Tensor:
    k, n = weight.shape
    if k % K_ATOM or n % N_VECTOR:
        raise ValueError(
            f"K must be divisible by {K_ATOM} and N by {N_VECTOR}, "
            f"got K={k}, N={n}"
        )

    return (
        weight.reshape(
            k // K_ATOM,
            K_PACK,
            K_VECTOR,
            n // N_VECTOR,
            N_VECTOR,
        )
        .permute(0, 3, 4, 1, 2)
        .contiguous()
    )


def _summarize_samples(samples_ms: list[float]) -> dict[str, object]:
    median_ms = statistics.median(samples_ms)
    deviations = [abs(value - median_ms) for value in samples_ms]
    return {
        "sample_count": len(samples_ms),
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        "mad_ms": statistics.median(deviations),
        "minimum_ms": min(samples_ms),
        "maximum_ms": max(samples_ms),
    }


def _measure_warm_l2(
    fn: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
    iterations: int,
) -> dict[str, object]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end) / iterations)

    return _summarize_samples(samples_ms)


def _measure_cold_l2(
    fn: Callable[[], object],
    *,
    warmup_ms: int,
    measurement_ms: int,
) -> dict[str, object]:
    samples_ms = triton.testing.do_bench(
        fn,
        warmup=warmup_ms,
        rep=measurement_ms,
        return_mode="all",
    )
    return _summarize_samples(samples_ms)


def _query_nvidia_smi() -> dict[str, str]:
    fields = (
        "name,clocks.max.sm,clocks.max.memory,memory.total,power.limit"
    )
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return {}
    values = [part.strip() for part in output.split(",")]
    names = [
        "name",
        "max_sm_clock_mhz",
        "max_memory_clock_mhz",
        "memory_mib",
        "power_limit_w",
    ]
    return dict(zip(names, values, strict=True))


def _query_git_revision(repository: Path) -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repository), *args],
            text=True,
        ).strip()

    try:
        return {
            "path": str(repository),
            "branch": git("branch", "--show-current"),
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"path": str(repository), "unavailable": True}


def _kernel_evidence(compiled: object) -> dict[str, object]:
    metadata = compiled.metadata
    ptx = compiled.asm["ptx"]
    return {
        "compiled_num_warps": metadata.num_warps,
        "shared_bytes": metadata.shared,
        "stack_frame_bytes": metadata.ptxas_stack_frame_bytes,
        "spill_load_bytes": metadata.ptxas_spill_load_bytes,
        "spill_store_bytes": metadata.ptxas_spill_store_bytes,
        "cp_async_instructions": ptx.count("cp.async.cg.shared.global"),
        "shared_load_b16_instructions": ptx.count("ld.shared.b16"),
        "shared_load_b32_instructions": ptx.count("ld.shared.b32"),
        "shared_load_v2_b32_instructions": ptx.count("ld.shared.v2.b32"),
        "shared_load_v4_b32_instructions": ptx.count("ld.shared.v4.b32"),
        "mbarrier_waits": ptx.count("mbarrier.try_wait"),
        "cta_barriers": ptx.count("barrier.sync"),
    }


def _effective_bandwidth_gbps(k: int, n: int, median_ms: float) -> float:
    logical_bytes = (k * n + k + n) * torch.bfloat16.itemsize
    return logical_bytes / (median_ms * 1.0e6)


def _load_flaggems_mv(flaggems_root: Path) -> Callable[..., torch.Tensor]:
    sys.path.insert(0, str(flaggems_root / "src"))
    from flag_gems.ops.mv import mv

    return mv


def _candidate_space() -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    for groups, block_k, maximum_stages in (
        (2, 32, 16),
        (2, 64, 8),
        (2, 128, 6),
        (2, 256, 3),
    ):
        candidates.extend(
            (groups, block_k, stages)
            for stages in range(1, maximum_stages + 1)
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        dest="shapes",
        help="KxN shape; may be repeated",
    )
    parser.add_argument("--warmup-launches", type=int, default=25)
    parser.add_argument("--warm-repeats", type=int, default=7)
    parser.add_argument("--warm-iterations", type=int, default=200)
    parser.add_argument("--cold-warmup-ms", type=int, default=25)
    parser.add_argument("--cold-measurement-ms", type=int, default=100)
    parser.add_argument(
        "--flaggems-root",
        type=Path,
        default=Path("/mnt/home-nas/work/repo/FlagGems"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("results-sm120-rtx5060ti.json"),
    )
    args = parser.parse_args()
    shapes = args.shapes or [(4096, 4096), (4096, 12288)]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 0):
        raise RuntimeError(
            f"This calibration requires SM120, got {props.major}.{props.minor}"
        )
    measurement_values = (
        args.warmup_launches,
        args.warm_repeats,
        args.warm_iterations,
        args.cold_warmup_ms,
        args.cold_measurement_ms,
    )
    if any(value <= 0 for value in measurement_values):
        raise ValueError("all measurement counts and durations must be positive")

    flaggems_mv = _load_flaggems_mv(args.flaggems_root)
    flagtree_root = Path(triton.__file__).resolve().parents[2]
    nncase_root = Path(__file__).resolve().parents[4]
    torch.manual_seed(20260723)
    results: dict[str, object] = {
        "schema_version": 3,
        "timestamp_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "environment": {
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
            "device_name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "sm_count": props.multi_processor_count,
            "warp_size": props.warp_size,
            "l2_cache_bytes": props.L2_cache_size,
            "shared_memory_per_block_bytes": props.shared_memory_per_block,
            "shared_memory_per_block_optin_bytes": (
                props.shared_memory_per_block_optin
            ),
            "nvidia_smi": _query_nvidia_smi(),
            "logical_consumer_warps": 8,
            "logical_producer_warps": 1,
            "producer_registers": 48,
            "persistent_grid_ctas": props.multi_processor_count,
            "weight_layout": (
                "[K/16,N/8]<NVector=8,KPack=2,KVector=8>"
            ),
            "shared_stage_layout": (
                "[block_k/16*block_n/8,8*16] with 16-byte swizzle vectors"
            ),
            "source_revisions": {
                "nncase": _query_git_revision(nncase_root),
                "flagtree": _query_git_revision(flagtree_root),
                "flaggems": _query_git_revision(args.flaggems_root),
            },
        },
        "measurement": {
            "warm_l2": {
                "warmup_launches": args.warmup_launches,
                "repeats": args.warm_repeats,
                "launches_per_repeat": args.warm_iterations,
                "sample": "CUDA-event batch mean",
            },
            "cold_l2": {
                "warmup_ms": args.cold_warmup_ms,
                "measurement_ms": args.cold_measurement_ms,
                "cache_policy": "Triton benchmark cache clear before every timed launch",
                "sample": "one CUDA-event kernel duration",
            },
            "statistic": "median",
            "dispersion": "median absolute deviation",
        },
        "shapes": [],
    }

    for k, n in shapes:
        x = torch.randn((1, k), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
        packed_weight = _pack_weight(weight)
        output = torch.empty((1, n), device="cuda", dtype=torch.bfloat16)
        reference = x @ weight
        flaggems_weight = weight.T.contiguous()
        shape_result: dict[str, object] = {
            "m": 1,
            "k": k,
            "n": n,
            "logical_bytes": (k * n + k + n) * torch.bfloat16.itemsize,
            "baselines": {},
            "candidates": [],
        }

        def flaggems_launch() -> torch.Tensor:
            return flaggems_mv(flaggems_weight, x[0])

        flaggems_output = flaggems_launch()
        torch.testing.assert_close(
            flaggems_output,
            reference[0],
            rtol=2.0e-2,
            atol=3.0e-1,
        )
        flaggems_timing = {
            "cold_l2": _measure_cold_l2(
                flaggems_launch,
                warmup_ms=args.cold_warmup_ms,
                measurement_ms=args.cold_measurement_ms,
            ),
            "warm_l2": _measure_warm_l2(
                flaggems_launch,
                warmup=args.warmup_launches,
                repeats=args.warm_repeats,
                iterations=args.warm_iterations,
            ),
        }
        for cache_mode in ("cold_l2", "warm_l2"):
            timing_mode = flaggems_timing[cache_mode]
            timing_mode["effective_bandwidth_gbps"] = (
                _effective_bandwidth_gbps(
                    k,
                    n,
                    float(timing_mode["median_ms"]),
                )
            )
        shape_result["baselines"]["flaggems_mv"] = flaggems_timing

        for groups, block_k, stages in _candidate_space():
            if k % block_k != 0 or (n // 32) % groups != 0:
                continue
            def launch(
                groups: int = groups,
                block_k: int = block_k,
                stages: int = stages,
            ) -> object:
                return packed_gemv_pipeline[(props.multi_processor_count,)](
                    x,
                    packed_weight,
                    output,
                    k=k,
                    n_groups=n // 32,
                    groups=groups,
                    block_k=block_k,
                    stages=stages,
                    num_warps=8,
                )
            launch()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                output,
                reference,
                rtol=2.0e-2,
                atol=3.0e-1,
            )
            timing = {
                "cold_l2": _measure_cold_l2(
                    launch,
                    warmup_ms=args.cold_warmup_ms,
                    measurement_ms=args.cold_measurement_ms,
                ),
                "warm_l2": _measure_warm_l2(
                    launch,
                    warmup=args.warmup_launches,
                    repeats=args.warm_repeats,
                    iterations=args.warm_iterations,
                ),
            }
            compiled = packed_gemv_pipeline.warmup(
                x,
                packed_weight,
                output,
                k=k,
                n_groups=n // 32,
                groups=groups,
                block_k=block_k,
                stages=stages,
                num_warps=8,
                grid=(props.multi_processor_count,),
            )
            for cache_mode in ("cold_l2", "warm_l2"):
                timing_mode = timing[cache_mode]
                timing_mode["effective_bandwidth_gbps"] = (
                    _effective_bandwidth_gbps(
                        k,
                        n,
                        float(timing_mode["median_ms"]),
                    )
                )
            shape_result["candidates"].append(
                {
                    "groups": groups,
                    "tile_n": groups * 32,
                    "tile_k": block_k,
                    "stages": stages,
                    "stage_payload_bytes": (
                        groups
                        * block_k
                        * 32
                        * torch.bfloat16.itemsize
                    ),
                    "shared_stage_shape": [
                        block_k // K_ATOM * groups * 32 // N_VECTOR,
                        N_VECTOR * K_ATOM,
                    ],
                    "output_work_items": n // (groups * 32),
                    "k_iterations_per_work_item": k // block_k,
                    "timing": timing,
                    "compiled": _kernel_evidence(compiled),
                }
            )
            print(
                f"K={k} N={n} tile={groups * 32}x{block_k} "
                f"stages={stages}: "
                f"cold={timing['cold_l2']['median_ms']:.6f} ms, "
                f"warm={timing['warm_l2']['median_ms']:.6f} ms"
            )

        results["shapes"].append(shape_result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
