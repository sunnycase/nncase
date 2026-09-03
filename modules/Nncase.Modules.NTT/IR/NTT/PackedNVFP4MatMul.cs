// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

/// <summary>
/// Target-packed NVFP4 matrix multiplication.
/// </summary>
/// <remarks>
/// The input K axis, packed-weight K-byte axis, and output N axis are represented
/// by vector lanes. The semantic NVFP4 nibble packing remains part of the U8
/// weight payload and is independent of these target vector lanes.
/// </remarks>
[PatternFunctionalGenerator]
public sealed partial class PackedNVFP4MatMul : Op
{
    public static readonly ParameterInfo Lhs = new(typeof(PackedNVFP4MatMul), 0, "lhs", ParameterKind.Input);

    public static readonly ParameterInfo RhsPacked = new(typeof(PackedNVFP4MatMul), 1, "rhs_packed", ParameterKind.Input);

    public static readonly ParameterInfo RhsScale = new(typeof(PackedNVFP4MatMul), 2, "rhs_scale", ParameterKind.Input);

    public static readonly ParameterInfo LhsGlobalScale = new(typeof(PackedNVFP4MatMul), 3, "lhs_global_scale", ParameterKind.Input);

    public static readonly ParameterInfo RhsGlobalScale = new(typeof(PackedNVFP4MatMul), 4, "rhs_global_scale", ParameterKind.Input);

    public static readonly ParameterInfo Addend = new(typeof(PackedNVFP4MatMul), 5, "addend", ParameterKind.Input);

    public DataType OutputDataType { get; }

    public long GroupSize { get; }

    public int InputKVectorLaneCount { get; }

    public int RhsKPackLaneCount { get; }

    public int RhsKVectorLaneCount { get; }

    public int OutputNVectorLaneCount { get; }

    public override string DisplayProperty() =>
        $"OutputDataType: {OutputDataType}, GroupSize: {GroupSize}, " +
        $"InputKLanes: {InputKVectorLaneCount}, " +
        $"RhsKLanes: [{RhsKPackLaneCount},{RhsKVectorLaneCount}], " +
        $"OutputNLanes: {OutputNVectorLaneCount}";
}
