// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class BlockScaledMatMul : NTTKernelOp
{
    public static readonly ParameterInfo Lhs = new(typeof(BlockScaledMatMul), 0, "lhs", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Rhs = new(typeof(BlockScaledMatMul), 1, "rhs", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo RhsScale = new(typeof(BlockScaledMatMul), 2, "rhs_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Output = new(typeof(BlockScaledMatMul), 3, "output", memoryEffect: MemoryEffect.ReductionWrite);

    public DataType OutputDataType { get; }

    public long WeightBlockN { get; }

    public long WeightBlockK { get; }

    public override string DisplayProperty() =>
        $"OutputDataType: {OutputDataType}, WeightBlock: [{WeightBlockN}, {WeightBlockK}]";
}
