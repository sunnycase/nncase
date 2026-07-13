// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class Matmul : NTTKernelOp
{
    public static readonly ParameterInfo Lhs = new(typeof(Matmul), 0, "lhs", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Rhs = new(typeof(Matmul), 1, "rhs", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Output = new(typeof(Matmul), 2, "output", memoryEffect: MemoryEffect.ReductionReadWrite);

    public static readonly ParameterInfo LoadC = new(typeof(Matmul), 3, "loadC", memoryEffect: MemoryEffect.None);

    public static readonly ParameterInfo Scale = new(typeof(Matmul), 4, "scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo ExtraWorkload = new(typeof(Matmul), 5, "extraWorkload", memoryEffect: MemoryEffect.Read);

    public IRArray<int> LhsVectorizedAxes { get; }

    public IRArray<int> RhsVectorizedAxes { get; }

    public bool TransposeA { get; }

    public bool TransposeB { get; }

    public bool FusedReduce { get; }

    public string CSourcePath { get; }

    public string FuncName { get; }

    public override string DisplayProperty() => $"LhsVectorizedAxes: {LhsVectorizedAxes}, RhsVectorizedAxes: {RhsVectorizedAxes}, TransposeA: {TransposeA}, TransposeB: {TransposeB}";
}
