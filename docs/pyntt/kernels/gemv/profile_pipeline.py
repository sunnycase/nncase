#!/usr/bin/env python3
"""Launch one uninstrumented packed GEMV kernel for hardware profiling."""

from __future__ import annotations

import argparse

import torch

from benchmark_pipeline import _pack_weight, packed_gemv_pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the packed GEMV pipeline under NCU"
    )
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=12288)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--block-k", type=int, default=128)
    parser.add_argument("--stages", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.k <= 0 or args.n <= 0:
        raise ValueError("K and N must be positive")
    if args.groups != 2:
        raise ValueError("the calibrated K-major kernel requires groups=2")
    if args.block_k <= 0 or args.block_k & (args.block_k - 1):
        raise ValueError("block-k must be a positive power of two")
    if args.stages <= 0:
        raise ValueError("stages must be positive")
    if args.k % args.block_k:
        raise ValueError("K must be divisible by block-k")
    if args.n % (args.groups * 32):
        raise ValueError("N must be divisible by groups * 32")

    torch.manual_seed(20260723)
    properties = torch.cuda.get_device_properties(0)
    if properties.major != 12:
        raise RuntimeError(
            f"this profile is calibrated for SM120, got "
            f"SM{properties.major}{properties.minor}"
        )
    x = torch.randn(
        (1, args.k),
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = torch.randn(
        (args.k, args.n),
        device="cuda",
        dtype=torch.bfloat16,
    )
    packed_weight = _pack_weight(weight)
    output = torch.empty(
        (1, args.n),
        device="cuda",
        dtype=torch.bfloat16,
    )

    torch.cuda.synchronize()
    packed_gemv_pipeline[(properties.multi_processor_count,)](
        x,
        packed_weight,
        output,
        k=args.k,
        n_groups=args.n // 32,
        groups=args.groups,
        block_k=args.block_k,
        stages=args.stages,
        num_warps=8,
    )
    torch.cuda.synchronize()

    reference = x @ weight
    torch.testing.assert_close(
        output,
        reference,
        rtol=2.0e-2,
        atol=3.0e-1,
    )
    print(
        f"profiled K={args.k} N={args.n} "
        f"tile={args.groups * 32}x{args.block_k} "
        f"stages={args.stages} grid={properties.multi_processor_count}"
    )


if __name__ == "__main__":
    main()
