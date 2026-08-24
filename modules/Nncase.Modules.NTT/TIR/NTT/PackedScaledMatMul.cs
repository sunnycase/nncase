// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

/// <summary>
/// Target packed matrix multiplication with explicit activation and weight scales.
/// </summary>
public sealed partial class PackedScaledMatMul : NTTKernelOp
{
    public static readonly ParameterInfo Lhs = new(typeof(PackedScaledMatMul), 0, "lhs", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Rhs = new(typeof(PackedScaledMatMul), 1, "rhs", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo LhsScale = new(typeof(PackedScaledMatMul), 2, "lhs_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo RhsScale = new(typeof(PackedScaledMatMul), 3, "rhs_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Output = new(typeof(PackedScaledMatMul), 4, "output", memoryEffect: MemoryEffect.ReductionWrite);

    public IR.NTT.PackedMatMulRhsLayout RhsLayout { get; }

    public override string DisplayProperty() => $"RhsLayout: {RhsLayout}";
}
