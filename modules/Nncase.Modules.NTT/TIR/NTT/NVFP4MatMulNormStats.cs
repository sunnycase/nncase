// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

/// <summary>
/// Direct TIR NVFP4 matrix multiplication with local normalization statistics.
/// </summary>
public sealed partial class NVFP4MatMulNormStats : NTTKernelOp
{
    public static readonly ParameterInfo Lhs = new(typeof(NVFP4MatMulNormStats), 0, "lhs", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo RhsPacked = new(typeof(NVFP4MatMulNormStats), 1, "rhs_packed", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo RhsScale = new(typeof(NVFP4MatMulNormStats), 2, "rhs_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo LhsGlobalScale = new(typeof(NVFP4MatMulNormStats), 3, "lhs_global_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo RhsGlobalScale = new(typeof(NVFP4MatMulNormStats), 4, "rhs_global_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Output = new(typeof(NVFP4MatMulNormStats), 5, "output", memoryEffect: MemoryEffect.ReductionWrite);

    public static readonly ParameterInfo Stats = new(typeof(NVFP4MatMulNormStats), 6, "stats", memoryEffect: MemoryEffect.Write);

    public static readonly ParameterInfo Addend = new(typeof(NVFP4MatMulNormStats), 7, "addend", memoryEffect: MemoryEffect.Read);

    public long GroupSize { get; }

    public int Axis { get; }

    public bool UseMean { get; }

    public override string DisplayProperty() =>
        $"GroupSize: {GroupSize}, Axis: {Axis}, UseMean: {UseMean}";
}
