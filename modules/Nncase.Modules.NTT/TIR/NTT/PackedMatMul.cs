// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class PackedMatMul : NTTKernelOp
{
    public static readonly ParameterInfo Lhs = new(typeof(PackedMatMul), 0, "lhs", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Rhs = new(typeof(PackedMatMul), 1, "rhs", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Output = new(typeof(PackedMatMul), 2, "output", memoryEffect: MemoryEffect.ReductionReadWrite);

    public static readonly ParameterInfo LoadC = new(typeof(PackedMatMul), 3, "loadC", memoryEffect: MemoryEffect.None);

    public static readonly ParameterInfo Scale = new(typeof(PackedMatMul), 4, "scale", memoryEffect: MemoryEffect.Read);

    /// <summary>
    /// Gets the optional initial value used when <see cref="LoadC"/> is false.
    /// </summary>
    public static readonly ParameterInfo Addend = new(typeof(PackedMatMul), 5, "addend", memoryEffect: MemoryEffect.Read);

    public bool FusedReduce { get; }

    public IR.NTT.PackedMatMulRhsLayout RhsLayout { get; }

    public override string DisplayProperty() => $"FusedReduce: {FusedReduce}, RhsLayout: {RhsLayout}";
}
