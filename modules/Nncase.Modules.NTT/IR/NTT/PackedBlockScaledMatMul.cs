// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

/// <summary>
/// Target-packed block-scaled low-precision matrix multiplication.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class PackedBlockScaledMatMul : Op
{
    public static readonly ParameterInfo Lhs = new(typeof(PackedBlockScaledMatMul), 0, "lhs", ParameterKind.Input);

    public static readonly ParameterInfo Rhs = new(typeof(PackedBlockScaledMatMul), 1, "rhs", ParameterKind.Input);

    public static readonly ParameterInfo RhsScale = new(typeof(PackedBlockScaledMatMul), 2, "rhs_scale", ParameterKind.Input);

    public static readonly ParameterInfo Addend = new(typeof(PackedBlockScaledMatMul), 3, "addend", ParameterKind.Input);

    public DataType OutputDataType { get; }

    public PackedMatMulRhsLayout RhsLayout { get; }

    public int OutputNVectorLaneCount { get; }

    public long WeightBlockN { get; }

    public long WeightBlockK { get; }

    public override string DisplayProperty() =>
        $"OutputDataType: {OutputDataType}, RhsLayout: {RhsLayout}, " +
        $"OutputNVectorLaneCount: {OutputNVectorLaneCount}, " +
        $"WeightBlock: [{WeightBlockN}, {WeightBlockK}]";
}
