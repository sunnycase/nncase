// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

/// <summary>
/// Target-packed block-scaled matrix multiplication with an additional local
/// additive normalization-statistics output.
/// </summary>
public sealed partial class PackedBlockScaledMatMulNormStats : NTTKernelOp
{
    public static readonly ParameterInfo Lhs = new(typeof(PackedBlockScaledMatMulNormStats), 0, "lhs", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Rhs = new(typeof(PackedBlockScaledMatMulNormStats), 1, "rhs", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo RhsScale = new(typeof(PackedBlockScaledMatMulNormStats), 2, "rhs_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Output = new(typeof(PackedBlockScaledMatMulNormStats), 3, "output", memoryEffect: MemoryEffect.ReductionWrite);

    public static readonly ParameterInfo Stats = new(typeof(PackedBlockScaledMatMulNormStats), 4, "stats", memoryEffect: MemoryEffect.Write);

    public static readonly ParameterInfo Addend = new(typeof(PackedBlockScaledMatMulNormStats), 5, "addend", memoryEffect: MemoryEffect.Read);

    public IR.NTT.PackedMatMulRhsLayout RhsLayout { get; }

    public int OutputNVectorLaneCount { get; }

    public long WeightBlockN { get; }

    public long WeightBlockK { get; }

    public int Axis { get; }

    public bool UseMean { get; }

    public override string DisplayProperty() =>
        $"RhsLayout: {RhsLayout}, OutputNVectorLaneCount: {OutputNVectorLaneCount}, " +
        $"WeightBlock: [{WeightBlockN}, {WeightBlockK}], Axis: {Axis}, UseMean: {UseMean}";
}
