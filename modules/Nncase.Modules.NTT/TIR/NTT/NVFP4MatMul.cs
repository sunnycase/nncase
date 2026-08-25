// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

/// <summary>
/// Direct TIR NVFP4 matrix multiplication.
/// </summary>
public sealed partial class NVFP4MatMul : NTTKernelOp
{
    public static readonly ParameterInfo Lhs = new(typeof(NVFP4MatMul), 0, "lhs", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo RhsPacked = new(typeof(NVFP4MatMul), 1, "rhs_packed", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo RhsScale = new(typeof(NVFP4MatMul), 2, "rhs_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo LhsGlobalScale = new(typeof(NVFP4MatMul), 3, "lhs_global_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo RhsGlobalScale = new(typeof(NVFP4MatMul), 4, "rhs_global_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Output = new(typeof(NVFP4MatMul), 5, "output", memoryEffect: MemoryEffect.ReductionWrite);

    public long GroupSize { get; }

    public override string DisplayProperty() => $"GroupSize: {GroupSize}";
}
