// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

/// <summary>
/// Target-packed matrix multiplication with dynamic activation blocks and
/// block-scaled E4M3 weights.
/// </summary>
public sealed partial class PackedBlockScaledMatMul : NTTKernelOp
{
    public static readonly ParameterInfo Lhs = new(typeof(PackedBlockScaledMatMul), 0, "lhs", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Rhs = new(typeof(PackedBlockScaledMatMul), 1, "rhs", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo RhsScale = new(typeof(PackedBlockScaledMatMul), 2, "rhs_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Output = new(typeof(PackedBlockScaledMatMul), 3, "output", memoryEffect: MemoryEffect.ReductionWrite);

    public static readonly ParameterInfo Addend = new(typeof(PackedBlockScaledMatMul), 4, "addend", memoryEffect: MemoryEffect.Read);

    public IR.NTT.PackedMatMulRhsLayout RhsLayout { get; }

    public int OutputNVectorLaneCount { get; }

    public long WeightBlockN { get; }

    public long WeightBlockK { get; }

    public override string DisplayProperty() =>
        $"RhsLayout: {RhsLayout}, OutputNVectorLaneCount: {OutputNVectorLaneCount}, " +
        $"WeightBlock: [{WeightBlockN}, {WeightBlockK}]";
}
