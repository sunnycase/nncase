// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

/// <summary>
/// Packed matmul with an additional block-local additive statistics output.
/// </summary>
public sealed partial class PackedMatMulNormStats : NTTKernelOp
{
    public static readonly ParameterInfo Lhs = new(typeof(PackedMatMulNormStats), 0, "lhs", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Rhs = new(typeof(PackedMatMulNormStats), 1, "rhs", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Output = new(typeof(PackedMatMulNormStats), 2, "output", memoryEffect: MemoryEffect.ReductionReadWrite);

    public static readonly ParameterInfo Stats = new(typeof(PackedMatMulNormStats), 3, "stats", memoryEffect: MemoryEffect.Write);

    public static readonly ParameterInfo LoadC = new(typeof(PackedMatMulNormStats), 4, "loadC", memoryEffect: MemoryEffect.None);

    public static readonly ParameterInfo Scale = new(typeof(PackedMatMulNormStats), 5, "scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Addend = new(typeof(PackedMatMulNormStats), 6, "addend", memoryEffect: MemoryEffect.Read);

    public IR.NTT.PackedMatMulRhsLayout RhsLayout { get; }

    public int Axis { get; }

    public bool UseMean { get; }

    public override string DisplayProperty() =>
        $"RhsLayout: {RhsLayout}, Axis: {Axis}, UseMean: {UseMean}";
}
