// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

/// <summary>
/// Target-packed block-scaled matrix multiplication that also emits local
/// additive normalization statistics for the final output values.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class PackedBlockScaledMatMulNormStats : Op
{
    public static readonly ParameterInfo Lhs = new(typeof(PackedBlockScaledMatMulNormStats), 0, "lhs", ParameterKind.Input);

    public static readonly ParameterInfo Rhs = new(typeof(PackedBlockScaledMatMulNormStats), 1, "rhs", ParameterKind.Input);

    public static readonly ParameterInfo RhsScale = new(typeof(PackedBlockScaledMatMulNormStats), 2, "rhs_scale", ParameterKind.Input);

    public static readonly ParameterInfo Addend = new(typeof(PackedBlockScaledMatMulNormStats), 3, "addend", ParameterKind.Input);

    public DataType OutputDataType { get; }

    public PackedMatMulRhsLayout RhsLayout { get; }

    public int OutputNVectorLaneCount { get; }

    public long WeightBlockN { get; }

    public long WeightBlockK { get; }

    public int Axis { get; }

    public bool UseMean { get; }

    public override string DisplayProperty() =>
        $"OutputDataType: {OutputDataType}, RhsLayout: {RhsLayout}, " +
        $"OutputNVectorLaneCount: {OutputNVectorLaneCount}, " +
        $"WeightBlock: [{WeightBlockN}, {WeightBlockK}], Axis: {Axis}, UseMean: {UseMean}";
}
