#!/usr/bin/env python3
"""Trace the SM120 packed GEMV producer/consumer pipeline."""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton import runtime

from benchmark_pipeline import (
    K_ATOM_TL,
    K_PACK_TL,
    K_VECTOR_TL,
    N_VECTOR_TL,
    OUTPUT_LAYOUT,
    REDUCTION_GROUP_TL,
    WEIGHT_LAYOUT,
    X_LAYOUT,
    _pack_weight,
)


PRODUCER_EVENT_NAMES = (
    "acquire_begin",
    "issue_begin",
    "commit_end",
)
CONSUMER_EVENT_NAMES = (
    "wait_begin",
    "consume_begin",
    "release_end",
)
PRODUCER_EVENTS = len(PRODUCER_EVENT_NAMES)
CONSUMER_EVENTS = len(CONSUMER_EVENT_NAMES)
PRODUCER_EVENTS_TL = tl.constexpr(PRODUCER_EVENTS)
CONSUMER_EVENTS_TL = tl.constexpr(CONSUMER_EVENTS)
CONSUMER_WARPS = 8
CONSUMER_WARPS_TL = tl.constexpr(CONSUMER_WARPS)
TRACE_SEQUENCES = 1
TRACE_VALUES = (
    PRODUCER_EVENTS * TRACE_SEQUENCES
    + CONSUMER_EVENTS * TRACE_SEQUENCES * CONSUMER_WARPS
)
# Experiment-integrity policy, not a hardware characteristic.
TRACE_PERTURBATION_LIMIT = 0.05


@triton.jit
def _clock64():
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %clock64;",
        "=l",
        [],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _thread_id():
    return tl.inline_asm_elementwise(
        "mov.u32 $0, %tid.x;",
        "=r",
        [],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _clock64_vector(dummy):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .u32 unused;
            mov.u32 unused, $1;
            mov.u64 $0, %clock64;
        }
        """,
        "=l,r",
        [dummy],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _trace_producer(
    writer,
    weight,
    trace,
    k: tl.constexpr,
    n_groups: tl.constexpr,
    groups: tl.constexpr,
    block_k: tl.constexpr,
    trace_enabled: tl.constexpr,
    trace_k_tile: tl.constexpr,
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
        if trace_enabled:
            leader = _thread_id() % 32 == 0
        for k_tile in tl.range(0, num_k_tiles):
            sequence = local_work_index * num_k_tiles + k_tile
            if trace_enabled:
                trace_hit0 = (
                    (pid == 0)
                    & (local_work_index == 0)
                    & (k_tile == trace_k_tile)
                )
                acquire_begin = tl.full((), 0, tl.int64)
                issue_begin = tl.full((), 0, tl.int64)
                if trace_hit0:
                    acquire_begin = _clock64()
            slot = writer.acquire(sequence)
            if trace_enabled:
                if trace_hit0:
                    issue_begin = _clock64()
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
            if trace_enabled:
                if trace_hit0:
                    commit_end = _clock64()
                    tl.store(trace + 0, acquire_begin, mask=leader)
                    tl.store(trace + 1, issue_begin, mask=leader)
                    tl.store(trace + 2, commit_end, mask=leader)


@triton.jit
def _trace_consumer(
    reader,
    x,
    output,
    trace,
    k: tl.constexpr,
    n_groups: tl.constexpr,
    groups: tl.constexpr,
    block_k: tl.constexpr,
    trace_enabled: tl.constexpr,
    trace_k_tile: tl.constexpr,
):
    pid = tl.program_id(0)
    grid = tl.num_programs(0)
    num_k_tiles: tl.constexpr = k // block_k
    block_n: tl.constexpr = groups * 32
    packed_n_outer: tl.constexpr = block_n // N_VECTOR_TL
    local_n = tle.encoding(
        tle.encoding(
            tl.arange(0, block_n),
            OUTPUT_LAYOUT,
        )[:, None],
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
        if trace_enabled:
            thread_slot = tl.arange(0, CONSUMER_WARPS_TL * 32)
            consumer_warp = thread_slot // 32
            warp_leader = thread_slot % 32 == 0
        for k_tile in tl.range(0, num_k_tiles):
            sequence = local_work_index * num_k_tiles + k_tile
            if trace_enabled:
                trace_hit0 = (
                    (pid == 0)
                    & (local_work_index == 0)
                    & (k_tile == trace_k_tile)
                )
                wait_begin = tl.full(
                    (CONSUMER_WARPS_TL * 32,),
                    0,
                    tl.int64,
                )
                consume_begin = tl.full(
                    (CONSUMER_WARPS_TL * 32,),
                    0,
                    tl.int64,
                )
                if trace_hit0:
                    wait_begin = _clock64_vector(thread_slot)
            ready = reader.wait(sequence)
            if trace_enabled:
                if trace_hit0:
                    consume_begin = _clock64_vector(thread_slot)
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
                x_value = tle.encoding(
                    x_value[None, :],
                    WEIGHT_LAYOUT,
                )
                partial += (
                    weight_value.to(tl.float32)
                    * x_value.to(tl.float32)
                )
            reader.release(sequence)
            if trace_enabled:
                if trace_hit0:
                    release_end = _clock64_vector(thread_slot)
                    consumer_base = (
                        PRODUCER_EVENTS_TL
                        + consumer_warp * CONSUMER_EVENTS_TL
                    )
                    tl.store(
                        trace + consumer_base,
                        wait_begin,
                        mask=warp_leader,
                    )
                    tl.store(
                        trace + consumer_base + 1,
                        consume_begin,
                        mask=warp_leader,
                    )
                    tl.store(
                        trace + consumer_base + 2,
                        release_end,
                        mask=warp_leader,
                    )

        result = tl.sum(partial, axis=1)
        output_n = n_start + tle.encoding(
            tl.arange(0, block_n),
            OUTPUT_LAYOUT,
        )
        tl.store(output + output_n, result)

@triton.jit
def packed_gemv_pipeline_trace(
    x,
    weight,
    output,
    trace,
    k: tl.constexpr,
    n_groups: tl.constexpr,
    groups: tl.constexpr,
    block_k: tl.constexpr,
    stages: tl.constexpr,
    trace_enabled: tl.constexpr,
    trace_k_tile: tl.constexpr,
):
    tl.static_assert(groups == 2)
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
        name="packed_gemv_weight_trace",
        weight=weight_stages,
    )
    tle.gpu.warp_specialize(
        [
            (
                _trace_consumer,
                (
                    weight_pipe.reader(),
                    x,
                    output,
                    trace,
                    k,
                    n_groups,
                    groups,
                    block_k,
                    trace_enabled,
                    trace_k_tile,
                ),
            ),
            (
                _trace_producer,
                (
                    weight_pipe.writer(),
                    weight,
                    trace,
                    k,
                    n_groups,
                    groups,
                    block_k,
                    trace_enabled,
                    trace_k_tile,
                ),
            ),
        ],
        [1],
        [48],
    )


@triton.jit
def clock64_pair_probe(output):
    tid = _thread_id()
    begin = _clock64()
    end = _clock64()
    tl.store(output, end - begin, mask=tid == 0)


@triton.jit
def barrier_component_probe(trace):
    thread_slot = tl.arange(0, 256)
    warp_leader = thread_slot % 32 == 0
    warp = thread_slot // 32
    tl.debug_barrier()
    begin = _clock64_vector(thread_slot)
    tl.debug_barrier()
    end = _clock64_vector(thread_slot)
    tl.store(trace + warp, begin, mask=warp_leader)
    tl.store(trace + 8 + warp, end, mask=warp_leader)


@triton.jit
def ffma_component_probe(a, x, output, trace, repeats):
    block_n: tl.constexpr = 64
    reduction: tl.constexpr = 32
    local_n = tle.encoding(
        tle.encoding(
            tl.arange(0, block_n),
            OUTPUT_LAYOUT,
        )[:, None],
        WEIGHT_LAYOUT,
    )
    local_k_vector = tle.encoding(
        tl.arange(0, reduction),
        X_LAYOUT,
    )
    local_k = tle.encoding(
        local_k_vector[None, :],
        WEIGHT_LAYOUT,
    )
    offset = local_n * reduction + local_k
    a_value = tle.encoding(
        tl.load(a + offset),
        WEIGHT_LAYOUT,
    ).to(tl.float32)
    x_value = tle.encoding(
        tl.load(x + local_k_vector),
        X_LAYOUT,
    ).to(tl.float32)
    x_broadcast = tle.encoding(
        x_value[None, :],
        WEIGHT_LAYOUT,
    )
    partial = tl.fma(
        a_value,
        x_broadcast,
        tle.encoding(
            tl.zeros((block_n, reduction), tl.float32),
            WEIGHT_LAYOUT,
        ),
    )
    tl.debug_barrier()

    thread_slot = tl.arange(0, 256)
    warp_leader = thread_slot % 32 == 0
    warp = thread_slot // 32
    begin = _clock64_vector(thread_slot)
    for _ in tl.range(0, repeats):
        partial = tl.fma(a_value, x_broadcast, partial)
    tl.debug_barrier()
    end = _clock64_vector(thread_slot)

    tl.store(output + offset, partial)
    tl.store(trace + warp, begin, mask=warp_leader)
    tl.store(trace + 8 + warp, end, mask=warp_leader)


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _mad(values: list[float]) -> float:
    median = _median(values)
    return float(statistics.median(abs(value - median) for value in values))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(
        ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    )


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "median": _median(values),
        "mad": _mad(values),
        "p10": _percentile(values, 0.10),
        "p90": _percentile(values, 0.90),
        "minimum": min(values),
        "maximum": max(values),
        "samples": values,
    }


def _error_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float]:
    error = (actual.to(torch.float32) - expected.to(torch.float32)).abs()
    return {
        "max_abs_error": float(error.max().item()),
        "mean_abs_error": float(error.mean().item()),
    }


def _kernel_evidence(compiled: Any) -> dict[str, Any]:
    ptx = compiled.asm["ptx"]
    metadata = compiled.metadata
    return {
        "num_warps": metadata.num_warps,
        "shared_bytes": metadata.shared,
        "stack_frame_bytes": metadata.ptxas_stack_frame_bytes,
        "spill_load_bytes": metadata.ptxas_spill_load_bytes,
        "spill_store_bytes": metadata.ptxas_spill_store_bytes,
        "cp_async": ptx.count("cp.async.cg.shared.global"),
        "cp_async_mbarrier_arrive": ptx.count(
            "cp.async.mbarrier.arrive.noinc"
        ),
        "shared_vector_loads": ptx.count("ld.shared.v4.b32"),
        "mbarrier_waits": ptx.count("mbarrier.try_wait"),
        "cta_barriers": ptx.count("barrier.sync"),
        "ffma_instructions": (
            ptx.count("fma.rn.f32")
            + ptx.count("fma.rn.ftz.f32")
        ),
        "clock64_reads": ptx.count("%clock64"),
        "globaltimer_reads": ptx.count("%globaltimer"),
    }


def _decode_trace(raw_trace: torch.Tensor) -> dict[str, Any]:
    values = [int(value) for value in raw_trace.cpu().tolist()]
    producer = [
        dict(
            zip(
                PRODUCER_EVENT_NAMES,
                values[
                    sequence * PRODUCER_EVENTS:
                    (sequence + 1) * PRODUCER_EVENTS
                ],
                strict=True,
            )
        )
        for sequence in range(TRACE_SEQUENCES)
    ]
    consumer_base = PRODUCER_EVENTS * TRACE_SEQUENCES
    consumers = [
        dict(
            zip(
                CONSUMER_EVENT_NAMES,
                values[
                    consumer_base + warp * CONSUMER_EVENTS:
                    consumer_base + (warp + 1) * CONSUMER_EVENTS
                ],
                strict=True,
            )
        )
        for warp in range(CONSUMER_WARPS)
    ]
    if any(value <= 0 for event in producer for value in event.values()):
        raise RuntimeError(f"producer trace is incomplete: {producer}")
    if any(
        event[name] <= 0
        for event in consumers
        for name in CONSUMER_EVENT_NAMES
    ):
        raise RuntimeError(f"consumer trace is incomplete: {consumers}")

    first_producer = producer[0]
    wait_begin = min(event["wait_begin"] for event in consumers)
    consume_begin = max(event["consume_begin"] for event in consumers)
    release_end = max(event["release_end"] for event in consumers)
    derived = {
        "producer_acquire_stall_cycles": (
            first_producer["issue_begin"]
            - first_producer["acquire_begin"]
        ),
        "producer_issue_commit_cycles": (
            first_producer["commit_end"]
            - first_producer["issue_begin"]
        ),
        "consumer_group_wait_cycles": (
            consume_begin - wait_begin
        ),
        "consumer_group_service_cycles": (
            release_end - consume_begin
        ),
        "issue_to_consume_cycles": (
            consume_begin - first_producer["issue_begin"]
        ),
        "slot_live_cycles": (
            release_end - first_producer["acquire_begin"]
        ),
    }
    if any(value < 0 for value in derived.values()):
        raise RuntimeError(
            f"trace clocks are inconsistent: {derived}, raw={values}"
        )
    return {
        "raw": values,
        "producer": producer,
        "consumers": consumers,
        "derived": derived,
    }


def _measure_launch_ms(
    launch: Callable[[], object],
    *,
    warmup: int = 25,
    repeats: int = 9,
    iterations: int = 100,
) -> dict[str, Any]:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(iterations):
            launch()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) / iterations)
    return _summary(samples)


def _measure_cold_launch_ms(
    launch: Callable[[], object],
    *,
    cache: Any,
    repeats: int,
) -> dict[str, Any]:
    samples = []
    for _ in range(repeats):
        runtime.driver.active.clear_cache(cache)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        launch()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return _summary(samples)


def _trace_samples(
    launch: Callable[[], object],
    trace: torch.Tensor,
    *,
    samples: int,
    cold: bool,
    cache: Any,
) -> list[dict[str, Any]]:
    records = []
    for _ in range(samples):
        trace.zero_()
        if cold:
            runtime.driver.active.clear_cache(cache)
        launch()
        torch.cuda.synchronize()
        records.append(_decode_trace(trace))
    return records


def _summarize_trace_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_names = records[0]["derived"].keys()
    metrics = {
        name: _summary(
            [float(record["derived"][name]) for record in records]
        )
        for name in metric_names
    }
    return {
        "metrics": metrics,
        "records": records,
    }


def _validate_evidence(
    baseline: dict[str, Any],
    traced: dict[str, Any],
    *,
    expected_cp_async: int,
) -> None:
    errors = []
    for label, evidence in (("baseline", baseline), ("traced", traced)):
        if evidence["cp_async"] != expected_cp_async:
            errors.append(
                f"{label} has {evidence['cp_async']} cp.async instructions; "
                f"expected {expected_cp_async}"
            )
        if evidence["cp_async_mbarrier_arrive"] < 1:
            errors.append(f"{label} has no cp.async mbarrier arrival")
        if evidence["shared_vector_loads"] < 1:
            errors.append(f"{label} has no vector shared load")
        if evidence["spill_load_bytes"] or evidence["spill_store_bytes"]:
            errors.append(f"{label} spills registers: {evidence}")
    if baseline["clock64_reads"] != 0:
        errors.append("trace-disabled variant still reads %clock64")
    if traced["clock64_reads"] < 6:
        errors.append("trace-enabled variant has too few %clock64 reads")
    if errors:
        raise RuntimeError("; ".join(errors))


def _timer_calibration(samples: int) -> dict[str, Any]:
    output = torch.zeros((), device="cuda", dtype=torch.int64)
    values = []
    for _ in range(samples):
        output.zero_()
        clock64_pair_probe[(1,)](output, num_warps=1)
        torch.cuda.synchronize()
        values.append(float(output.item()))
    compiled = clock64_pair_probe.warmup(
        output,
        num_warps=1,
        grid=(1,),
    )
    return {
        "back_to_back_read_delta_cycles": _summary(values),
        "compiled": _kernel_evidence(compiled),
    }


def _component_delta_samples(
    launch: Callable[[], object],
    trace: torch.Tensor,
    *,
    samples: int,
) -> list[float]:
    deltas = []
    for _ in range(samples):
        trace.zero_()
        launch()
        torch.cuda.synchronize()
        timestamps = [int(value) for value in trace.cpu().tolist()]
        begins = timestamps[:CONSUMER_WARPS]
        ends = timestamps[CONSUMER_WARPS:]
        if (
            any(value <= 0 for value in begins)
            or any(value <= 0 for value in ends)
            or max(ends) < min(begins)
        ):
            raise RuntimeError(
                f"invalid component timestamps: {timestamps}"
            )
        deltas.append(float(max(ends) - min(begins)))
    return deltas


def _run_component_probes(
    *,
    samples: int,
    ffma_repeats: int,
) -> dict[str, Any]:
    trace = torch.zeros(
        (CONSUMER_WARPS * 2,),
        device="cuda",
        dtype=torch.int64,
    )
    reduction_groups_per_stage = 4
    packed_elements = 64 * 32
    source = torch.randn(
        (packed_elements,),
        device="cuda",
        dtype=torch.bfloat16,
    )
    x = torch.randn((32,), device="cuda", dtype=torch.bfloat16)
    ffma_output = torch.empty(
        (packed_elements,),
        device="cuda",
        dtype=torch.float32,
    )

    def barrier_launch() -> object:
        return barrier_component_probe[(1,)](
            trace,
            num_warps=CONSUMER_WARPS,
        )

    def ffma_launch() -> object:
        return ffma_component_probe[(1,)](
            source,
            x,
            ffma_output,
            trace,
            repeats=ffma_repeats,
            num_warps=CONSUMER_WARPS,
        )

    ffma_launch()
    torch.cuda.synchronize()
    source_2d = source.reshape(64, 32).to(torch.float32)
    expected_ffma = (
        source_2d
        * x.to(torch.float32)[None, :]
        * float(ffma_repeats + 1)
    ).reshape(-1)
    torch.testing.assert_close(
        ffma_output,
        expected_ffma,
        rtol=2.0e-5,
        atol=2.0e-4,
    )

    barrier_samples = _component_delta_samples(
        barrier_launch,
        trace,
        samples=samples,
    )
    ffma_samples = _component_delta_samples(
        ffma_launch,
        trace,
        samples=samples,
    )
    barrier_median = _median(barrier_samples)
    ffma_group_cycles_per_iteration = [
        max(0.0, value - barrier_median) / ffma_repeats
        for value in ffma_samples
    ]
    ffma_stage_cycles_per_iteration = [
        value * reduction_groups_per_stage
        for value in ffma_group_cycles_per_iteration
    ]
    ffma_group_median = _median(ffma_group_cycles_per_iteration)

    barrier_compiled = barrier_component_probe.warmup(
        trace,
        num_warps=CONSUMER_WARPS,
        grid=(1,),
    )
    ffma_compiled = ffma_component_probe.warmup(
        source,
        x,
        ffma_output,
        trace,
        repeats=ffma_repeats,
        num_warps=CONSUMER_WARPS,
        grid=(1,),
    )
    return {
        "configuration": {
            "group_shape": [64, 32],
            "reduction_groups_per_stage": reduction_groups_per_stage,
            "group_payload_bytes": (
                packed_elements * torch.bfloat16.itemsize
            ),
            "stage_payload_bytes": (
                packed_elements
                * reduction_groups_per_stage
                * torch.bfloat16.itemsize
            ),
            "ffma_repeats": ffma_repeats,
            "scalar_ffmas_per_group": packed_elements,
            "scalar_ffmas_per_stage": (
                packed_elements * reduction_groups_per_stage
            ),
        },
        "barrier_cycles": _summary(barrier_samples),
        "smem_validation": {
            "method": "Nsight Compute on the unmodified pipeline kernel",
            "reason": (
                "An in-kernel timestamp cannot establish a dependency on a "
                "shared load without changing the measured instruction mix."
            ),
        },
        "ffma_cycles_raw": _summary(ffma_samples),
        "ffma_cycles_per_group_barrier_corrected": _summary(
            ffma_group_cycles_per_iteration
        ),
        "ffma_cycles_per_iteration_barrier_corrected": _summary(
            ffma_stage_cycles_per_iteration
        ),
        "ffma_warp_instructions_per_cycle": (
            (packed_elements // 32) / ffma_group_median
            if ffma_group_median > 0
            else None
        ),
        "compiled": {
            "barrier": _kernel_evidence(barrier_compiled),
            "ffma": _kernel_evidence(ffma_compiled),
        },
    }


def _analyze_results(results: dict[str, Any]) -> dict[str, Any]:
    stages = results["stages"]
    stage1 = next(
        (stage for stage in stages if stage["stages"] == 1),
        None,
    )
    if stage1 is None:
        raise RuntimeError("stage 1 is required for diagnostic comparison")
    stage1_metrics = stage1["trace"]["cold"]["metrics"]
    wait_cycles = stage1_metrics[
        "consumer_group_wait_cycles"
    ]["median"]
    ffma_cycles = results["component_probes"][
        "ffma_cycles_per_iteration_barrier_corrected"
    ]["median"]
    overheads = {
        stage["stages"]: stage["timing_ms"][
            "median_overhead_fraction"
        ]
        for stage in stages
    }
    return {
        "stage1_consumer_wait_cycles": wait_cycles,
        "stage1_producer_issue_commit_cycles": stage1_metrics[
            "producer_issue_commit_cycles"
        ]["median"],
        "stage1_consumer_service_cycles": stage1_metrics[
            "consumer_group_service_cycles"
        ]["median"],
        "ffma_cycles": ffma_cycles,
        "trace_overhead_fraction_by_stage": overheads,
        "maximum_absolute_trace_overhead_fraction": max(
            abs(value) for value in overheads.values()
        ),
        "verdicts": {
            "transport_structure": "confirmed",
            "timer_overhead": "confirmed",
            "stage_selection": "benchmark_sweep_required",
        },
        "stage_formula_note": (
            "Every cycle value in this report comes from this trace's "
            "%clock64 measurements. Spill-free instrumentation records "
            "producer acquire, issue/commit, consumer wait, and consumer "
            "service intervals. No external latency or throughput prior is "
            "injected; "
            "benchmark_pipeline.py owns the statistically sized stage sweep."
        ),
    }


def _make_svg(results: dict[str, Any]) -> str:
    width = 1120
    height = 690
    stage_rows = sorted(results["stages"], key=lambda row: row["stages"])
    stage1 = next(
        (row for row in stage_rows if row["stages"] == 1),
        stage_rows[0],
    )
    ready_cold = stage1["trace"]["cold"]["metrics"][
        "consumer_group_wait_cycles"
    ]["median"]
    ready_warm = stage1["trace"]["warm"]["metrics"][
        "consumer_group_wait_cycles"
    ]["median"]
    loader_measured = stage1["trace"]["cold"]["metrics"][
        "producer_issue_commit_cycles"
    ]["median"]
    consumer_measured = stage1["trace"]["cold"]["metrics"][
        "consumer_group_service_cycles"
    ]["median"]
    ffma_measured = results["component_probes"][
        "ffma_cycles_per_iteration_barrier_corrected"
    ]["median"]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>"
        "text{font-family:Inter,Arial,sans-serif;fill:#1f2933}"
        ".title{font-size:22px;font-weight:700}"
        ".section{font-size:14px;font-weight:700}"
        ".sub{font-size:12px;fill:#52606d}"
        ".label{font-size:12px}"
        ".grid{stroke:#d9e2ec;stroke-width:1}"
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="36" class="title">'
        "SM120 packed GEMV diagnostic trace"
        "</text>",
        '<text x="28" y="58" class="sub">'
        "Cycle values are raw %clock64 medians; no fitted correction."
        "</text>",
        '<text x="28" y="94" class="section">'
        "Measured S1 consumer wait"
        "</text>",
    ]

    bar_x = 190.0
    bar_width = 640.0
    ready_max = max(ready_cold, ready_warm) * 1.08
    ready_rows = (
        ("S1 wait, cold", ready_cold, "#2563eb"),
        ("S1 wait, warm", ready_warm, "#dc2626"),
    )
    for index, (label, value, color) in enumerate(ready_rows):
        y = 118 + index * 30
        value_width = bar_width * value / ready_max
        lines.extend(
            [
                f'<text x="{bar_x - 12}" y="{y + 13}" '
                f'text-anchor="end" class="label">{label}</text>',
                f'<rect x="{bar_x}" y="{y}" width="{value_width:.2f}" '
                f'height="18" fill="{color}"/>',
                f'<text x="{bar_x + value_width + 8:.2f}" y="{y + 13}" '
                f'class="label">{value:.1f}</text>',
            ]
        )
    lines.append(
        '<text x="850" y="137" class="sub">'
        "Both bars are raw trace medians from this run."
        "</text>"
    )
    lines.append(
        '<text x="850" y="154" class="sub">'
        "They are not interpreted as a complete copy latency."
        "</text>"
    )

    lines.extend(
        [
            '<text x="28" y="238" class="section">'
            "Measured S1 intervals"
            "</text>",
        ]
    )
    component_rows = (
        ("Loader issue + commit", loader_measured, "#b45309"),
        ("Consumer group", consumer_measured, "#2563eb"),
        ("FFMA-only probe", ffma_measured, "#0f766e"),
    )
    component_max = max(value for _, value, _ in component_rows) * 1.25
    for index, (label, value, color) in enumerate(component_rows):
        y = 258 + index * 30
        value_width = 300.0 * value / component_max
        lines.extend(
            [
                f'<text x="138" y="{y + 13}" text-anchor="end" '
                f'class="label">{label}</text>',
                f'<rect x="150" y="{y}" width="{value_width:.2f}" '
                f'height="18" fill="{color}"/>',
                f'<text x="{158 + value_width:.2f}" y="{y + 13}" '
                f'class="label">{value:.1f} cycles</text>',
            ]
        )

    plot_left = 92.0
    plot_top = 360.0
    plot_width = 980.0
    plot_height = 240.0
    lines.append(
        '<text x="28" y="338" class="section">'
        "Cold pipeline cadence and stalls by stage"
        "</text>"
    )
    y_ticks = (10.0, 100.0, 1000.0, 10000.0)
    y_log_min = math.log10(y_ticks[0])
    y_log_max = math.log10(y_ticks[-1])

    def plot_y(value: float) -> float:
        bounded = max(y_ticks[0], min(y_ticks[-1], value))
        ratio = (math.log10(bounded) - y_log_min) / (
            y_log_max - y_log_min
        )
        return plot_top + plot_height * (1.0 - ratio)

    for tick in y_ticks:
        y = plot_y(tick)
        lines.append(
            f'<line x1="{plot_left}" y1="{y:.2f}" '
            f'x2="{plot_left + plot_width}" y2="{y:.2f}" class="grid"/>'
        )
        lines.append(
            f'<text x="{plot_left - 10}" y="{y + 4:.2f}" '
            f'text-anchor="end" class="sub">{int(tick)}</text>'
        )

    x_step = plot_width / max(1, len(stage_rows) - 1)
    x_positions = {
        row["stages"]: plot_left + index * x_step
        for index, row in enumerate(stage_rows)
    }
    series = (
        ("Acquire stall", "producer_acquire_stall_cycles", "#7c3aed"),
        ("Issue + commit", "producer_issue_commit_cycles", "#b45309"),
        ("Consumer wait", "consumer_group_wait_cycles", "#dc2626"),
    )
    for label, metric_name, color in series:
        points = []
        for row in stage_rows:
            value = row["trace"]["cold"]["metrics"][metric_name]["median"]
            points.append(
                f"{x_positions[row['stages']]:.2f},{plot_y(value):.2f}"
            )
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{color}" stroke-width="2.5"/>'
        )
        for row in stage_rows:
            value = row["trace"]["cold"]["metrics"][metric_name]["median"]
            x_pos = x_positions[row["stages"]]
            y_pos = plot_y(value)
            lines.append(
                f'<circle cx="{x_pos:.2f}" cy="{y_pos:.2f}" r="4" '
                f'fill="{color}"/>'
            )
    for row in stage_rows:
        x_pos = x_positions[row["stages"]]
        latency = row["timing_ms"]["cold_baseline"]["median"]
        lines.append(
            f'<text x="{x_pos:.2f}" y="{plot_top + plot_height + 20}" '
            f'text-anchor="middle" class="label">S{row["stages"]}</text>'
        )
        lines.append(
            f'<text x="{x_pos:.2f}" y="{plot_top + plot_height + 37}" '
            f'text-anchor="middle" class="sub">{latency:.4f} ms</text>'
        )
    for index, (label, _, color) in enumerate(series):
        x_pos = 92 + index * 170
        lines.append(
            f'<line x1="{x_pos}" y1="662" x2="{x_pos + 24}" y2="662" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{x_pos + 32}" y="666" class="label">'
            f"{html.escape(label)}</text>"
        )
    lines.append(
        '<text x="1072" y="666" text-anchor="end" class="sub">'
        "Source: this trace JSON; cold latency is printed below each stage."
        "</text>"
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _make_markdown(results: dict[str, Any]) -> str:
    timer = results["timer_calibration"]["back_to_back_read_delta_cycles"]
    components = results["component_probes"]
    ffma_cycles = components[
        "ffma_cycles_per_iteration_barrier_corrected"
    ]["median"]
    scalar_ffmas_per_stage = components["configuration"][
        "scalar_ffmas_per_stage"
    ]
    analysis = results["analysis"]
    lines = [
        "# SM120 Packed GEMV Pipeline Trace",
        "",
        f"Generated: `{results['timestamp_utc']}`",
        "",
        "## Integrity",
        "",
        "| Variant | cp.async | shared vector loads | spills | clock64 reads |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    first_stage = results["stages"][0]
    for name in ("baseline", "traced"):
        evidence = first_stage["compiled"][name]
        spills = evidence["spill_load_bytes"] + evidence["spill_store_bytes"]
        lines.append(
            f"| {name} | {evidence['cp_async']} | "
            f"{evidence['shared_vector_loads']} | {spills} | "
            f"{evidence['clock64_reads']} |"
        )
    lines.extend(
        [
            "",
            "## Correctness",
            "",
            "| Stages | Baseline vs Torch max abs | "
            "Traced vs Torch max abs | Traced vs baseline max abs |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for stage in results["stages"]:
        correctness = stage["correctness"]
        lines.append(
            f"| {stage['stages']} | "
            f"{correctness['baseline']['max_abs_error']:.6f} | "
            f"{correctness['traced']['max_abs_error']:.6f} | "
            f"{correctness['traced_vs_baseline']['max_abs_error']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Every baseline and traced launch passes `torch.testing.assert_close` "
            "against the BF16 Torch reference. Traced output must also match "
            "the untraced output exactly.",
            "",
            "## Timer",
            "",
            "Back-to-back `%clock64` reads:",
            "",
            f"- median: `{timer['median']:.1f}` cycles",
            f"- MAD: `{timer['mad']:.1f}` cycles",
            "",
            "## Component Probes",
            "",
            "| Probe | Measured | Derivation |",
            "| --- | ---: | --- |",
            f"| CTA barrier | "
            f"{components['barrier_cycles']['median']:.1f} cycles | "
            "Raw `%clock64` interval |",
            f"| {scalar_ffmas_per_stage} scalar-FFMA stage equivalent | "
            f"{ffma_cycles:.1f} cycles | Barrier median subtracted; "
            f"{components['ffma_warp_instructions_per_cycle']:.3f} "
            "warp instructions/cycle |",
            "| Shared load | Not measured by this trace | "
            "Use the separate NCU artifact |",
            "",
            "The compiled pipeline's vector shared-load site count is recorded "
            "in the Integrity table. "
            "Shared throughput is intentionally not inferred from an "
            "instrumented load because enforcing the timestamp dependency "
            "changes the instruction mix. NCU collection is a separate "
            "profiling step.",
            "",
            "## Stage Trace",
            "",
            "| Stages | Mode | Acquire | Loader issue + commit | "
            "Consumer wait | Consumer service | Cold latency |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for stage in results["stages"]:
        for mode in ("cold", "warm"):
            metrics = stage["trace"][mode]["metrics"]
            cold_latency = stage["timing_ms"]["cold_baseline"]["median"]
            lines.append(
                f"| {stage['stages']} | {mode} | "
                f"{metrics['producer_acquire_stall_cycles']['median']:.1f} | "
                f"{metrics['producer_issue_commit_cycles']['median']:.1f} | "
                f"{metrics['consumer_group_wait_cycles']['median']:.1f} | "
                f"{metrics['consumer_group_service_cycles']['median']:.1f} | "
                f"{cold_latency:.6f} ms |"
            )
    lines.extend(
        [
            "",
            "## Measured Diagnostics",
            "",
            f"- Measured S1 consumer wait stall: "
            f"`{analysis['stage1_consumer_wait_cycles']:.1f}` cycles.",
            f"- Measured S1 loader issue plus commit service: "
            f"`{analysis['stage1_producer_issue_commit_cycles']:.1f}` cycles.",
            f"- Measured S1 consumer group service: "
            f"`{analysis['stage1_consumer_service_cycles']:.1f}` cycles.",
            f"- Measured FFMA service: `{analysis['ffma_cycles']:.1f}` "
            "cycles for the barrier-corrected probe.",
            f"- Maximum absolute trace timing perturbation: "
            f"`{analysis['maximum_absolute_trace_overhead_fraction'] * 100.0:.2f}%`.",
            "",
            "Verdict: the compiled transport structure is recorded and no "
            "ring depth is derived from this trace; use benchmark_pipeline.py "
            "for stage selection.",
            "",
            analysis["stage_formula_note"],
            "",
            "The raw JSON is authoritative. Intervals are not silently "
            "corrected for timer overhead.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace the SM120 packed GEMV pipeline"
    )
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=12288)
    parser.add_argument("--block-k", type=int, default=128)
    parser.add_argument("--trace-k-tile", type=int, default=8)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--timer-samples", type=int, default=101)
    parser.add_argument("--component-samples", type=int, default=101)
    parser.add_argument("--ffma-repeats", type=int, default=1024)
    parser.add_argument(
        "--stages",
        type=int,
        action="append",
        help="Pipeline stage count; may be repeated",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/pyntt_gemv_trace"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stages_to_test = args.stages or [1, 2, 3, 4, 5, 6]
    if 1 not in stages_to_test:
        raise ValueError("stage 1 is required for diagnostic comparison")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 0):
        raise RuntimeError(
            f"SM120 is required, got {props.major}.{props.minor}"
        )
    if args.k <= 0 or args.k % args.block_k != 0:
        raise ValueError("K must be positive and divisible by block-k")
    if args.n <= 0 or args.n % 64 != 0:
        raise ValueError("N must be positive and divisible by 64")
    num_k_tiles = args.k // args.block_k
    if not 0 <= args.trace_k_tile < num_k_tiles:
        raise ValueError(
            f"trace-k-tile must be in [0, {num_k_tiles}), "
            f"got {args.trace_k_tile}"
        )
    if (
        args.samples <= 0
        or args.timer_samples <= 0
        or args.component_samples <= 0
        or args.ffma_repeats <= 0
    ):
        raise ValueError("sample counts must be positive")
    if any(stage <= 0 for stage in stages_to_test):
        raise ValueError("stage counts must be positive")

    torch.manual_seed(20260723)
    k = args.k
    n = args.n
    groups = 2
    grid = (props.multi_processor_count,)
    x = torch.randn((1, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
    packed_weight = _pack_weight(weight)
    output = torch.empty((1, n), device="cuda", dtype=torch.bfloat16)
    trace = torch.zeros(
        (TRACE_VALUES,),
        device="cuda",
        dtype=torch.int64,
    )
    reference = x @ weight
    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results: dict[str, Any] = {
        "schema_version": 4,
        "timestamp_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "environment": {
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
            "device": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "sm_count": props.multi_processor_count,
            "warp_size": props.warp_size,
            "l2_cache_bytes": props.L2_cache_size,
        },
        "configuration": {
            "k": k,
            "n": n,
            "tile_n": groups * 32,
            "tile_k": args.block_k,
            "trace_k_tile": args.trace_k_tile,
            "grid": props.multi_processor_count,
            "consumer_warps": CONSUMER_WARPS,
            "producer_warps": 1,
            "samples_per_mode": args.samples,
        },
        "measurement_policy": {
            "external_hardware_priors": False,
            "trace_perturbation_limit_fraction": (
                TRACE_PERTURBATION_LIMIT
            ),
            "stage_selection": "benchmark_sweep",
        },
        "timer_calibration": _timer_calibration(args.timer_samples),
        "component_probes": _run_component_probes(
            samples=args.component_samples,
            ffma_repeats=args.ffma_repeats,
        ),
        "stages": [],
    }

    for stage in stages_to_test:
        def launch(trace_enabled: bool) -> object:
            return packed_gemv_pipeline_trace[grid](
                x,
                packed_weight,
                output,
                trace,
                k=k,
                n_groups=n // 32,
                groups=groups,
                block_k=args.block_k,
                stages=stage,
                trace_enabled=trace_enabled,
                trace_k_tile=args.trace_k_tile,
                num_warps=CONSUMER_WARPS,
            )

        launch(False)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output,
            reference,
            rtol=2.0e-2,
            atol=3.0e-1,
        )
        baseline_correctness = _error_metrics(output, reference)
        baseline_output = output.clone()
        launch(True)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output,
            reference,
            rtol=2.0e-2,
            atol=3.0e-1,
        )
        torch.testing.assert_close(
            output,
            baseline_output,
            rtol=0.0,
            atol=0.0,
        )
        traced_correctness = _error_metrics(output, reference)
        trace_delta = _error_metrics(output, baseline_output)
        baseline_compiled = packed_gemv_pipeline_trace.warmup(
            x,
            packed_weight,
            output,
            trace,
            k=k,
            n_groups=n // 32,
            groups=groups,
            block_k=args.block_k,
            stages=stage,
            trace_enabled=False,
            trace_k_tile=args.trace_k_tile,
            num_warps=CONSUMER_WARPS,
            grid=grid,
        )
        traced_compiled = packed_gemv_pipeline_trace.warmup(
            x,
            packed_weight,
            output,
            trace,
            k=k,
            n_groups=n // 32,
            groups=groups,
            block_k=args.block_k,
            stages=stage,
            trace_enabled=True,
            trace_k_tile=args.trace_k_tile,
            num_warps=CONSUMER_WARPS,
            grid=grid,
        )
        baseline_evidence = _kernel_evidence(baseline_compiled)
        traced_evidence = _kernel_evidence(traced_compiled)
        _validate_evidence(
            baseline_evidence,
            traced_evidence,
            expected_cp_async=args.block_k // 4,
        )

        baseline_timing = _measure_launch_ms(lambda: launch(False))
        traced_timing = _measure_launch_ms(lambda: launch(True))
        cold_baseline_timing = _measure_cold_launch_ms(
            lambda: launch(False),
            cache=cache,
            repeats=args.samples,
        )
        overhead = (
            traced_timing["median"] / baseline_timing["median"] - 1.0
        )
        cold_records = _trace_samples(
            lambda: launch(True),
            trace,
            samples=args.samples,
            cold=True,
            cache=cache,
        )
        for _ in range(5):
            launch(True)
        torch.cuda.synchronize()
        warm_records = _trace_samples(
            lambda: launch(True),
            trace,
            samples=args.samples,
            cold=False,
            cache=cache,
        )
        stage_result = {
            "stages": stage,
            "correctness": {
                "baseline": baseline_correctness,
                "traced": traced_correctness,
                "traced_vs_baseline": trace_delta,
            },
            "compiled": {
                "baseline": baseline_evidence,
                "traced": traced_evidence,
            },
            "timing_ms": {
                "baseline": baseline_timing,
                "traced": traced_timing,
                "cold_baseline": cold_baseline_timing,
                "median_overhead_fraction": overhead,
            },
            "trace": {
                "cold": _summarize_trace_records(cold_records),
                "warm": _summarize_trace_records(warm_records),
            },
        }
        results["stages"].append(stage_result)
        cold_metrics = stage_result["trace"]["cold"]["metrics"]
        print(
            f"S{stage}: acquire="
            f"{cold_metrics['producer_acquire_stall_cycles']['median']:.1f}, "
            "issue+commit="
            f"{cold_metrics['producer_issue_commit_cycles']['median']:.1f}, "
            "wait="
            f"{cold_metrics['consumer_group_wait_cycles']['median']:.1f}, "
            "consumer="
            f"{cold_metrics['consumer_group_service_cycles']['median']:.1f}, "
            f"trace overhead={overhead * 100.0:.2f}%"
        )

    results["analysis"] = _analyze_results(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"gemv_pipeline_trace_{timestamp}"
    json_path = args.output_dir / f"{stem}.json"
    svg_path = args.output_dir / f"{stem}.svg"
    markdown_path = args.output_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    svg_path.write_text(_make_svg(results), encoding="utf-8")
    markdown_path.write_text(_make_markdown(results), encoding="utf-8")
    print(json_path)
    print(svg_path)
    print(markdown_path)
    trace_perturbation = results["analysis"][
        "maximum_absolute_trace_overhead_fraction"
    ]
    if trace_perturbation > TRACE_PERTURBATION_LIMIT:
        raise RuntimeError(
            "trace perturbation exceeds the experiment limit: "
            f"{trace_perturbation * 100.0:.2f}% > "
            f"{TRACE_PERTURBATION_LIMIT * 100.0:.2f}%"
        )


if __name__ == "__main__":
    main()
