// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.NN;
using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

/// <summary>
/// Target-packed fused gate/up NVFP4 projections followed by GLU.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class PackedNVFP4MatMulGlu : Op
{
    public static readonly ParameterInfo Input = new(typeof(PackedNVFP4MatMulGlu), 0, "input", ParameterKind.Input);

    public static readonly ParameterInfo GateWeightPacked = new(typeof(PackedNVFP4MatMulGlu), 1, "gate_weight_packed", ParameterKind.Input);

    public static readonly ParameterInfo UpWeightPacked = new(typeof(PackedNVFP4MatMulGlu), 2, "up_weight_packed", ParameterKind.Input);

    public static readonly ParameterInfo GateWeightScale = new(typeof(PackedNVFP4MatMulGlu), 3, "gate_weight_scale", ParameterKind.Input);

    public static readonly ParameterInfo UpWeightScale = new(typeof(PackedNVFP4MatMulGlu), 4, "up_weight_scale", ParameterKind.Input);

    public static readonly ParameterInfo GateInputGlobalScale = new(typeof(PackedNVFP4MatMulGlu), 5, "gate_input_global_scale", ParameterKind.Input);

    public static readonly ParameterInfo UpInputGlobalScale = new(typeof(PackedNVFP4MatMulGlu), 6, "up_input_global_scale", ParameterKind.Input);

    public static readonly ParameterInfo GateWeightGlobalScale = new(typeof(PackedNVFP4MatMulGlu), 7, "gate_weight_global_scale", ParameterKind.Input);

    public static readonly ParameterInfo UpWeightGlobalScale = new(typeof(PackedNVFP4MatMulGlu), 8, "up_weight_global_scale", ParameterKind.Input);

    public GluType GluType { get; }

    public DataType OutputDataType { get; }

    public long GroupSize { get; }

    public int InputKVectorLaneCount { get; }

    public int RhsKPackLaneCount { get; }

    public int RhsKVectorLaneCount { get; }

    public int OutputNVectorLaneCount { get; }

    public override string DisplayProperty() =>
        $"GluType: {GluType}, OutputDataType: {OutputDataType}, GroupSize: {GroupSize}, " +
        $"InputKLanes: {InputKVectorLaneCount}, " +
        $"RhsKLanes: [{RhsKPackLaneCount},{RhsKVectorLaneCount}], " +
        $"OutputNLanes: {OutputNVectorLaneCount}";
}
