// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.NN;
using Nncase.IR.Math;
using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

/// <summary>
/// Fused gate/up linear projection with packed RHS and output layout.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class PackedMatMulGlu : Op
{
    public static readonly ParameterInfo Input = new(typeof(PackedMatMulGlu), 0, "input", ParameterKind.Input);

    public static readonly ParameterInfo GateWeight = new(typeof(PackedMatMulGlu), 1, "gate_weight", ParameterKind.Input);

    public static readonly ParameterInfo UpWeight = new(typeof(PackedMatMulGlu), 2, "up_weight", ParameterKind.Input);

    public static readonly ParameterInfo GateBias = new(typeof(PackedMatMulGlu), 3, "gate_bias", ParameterKind.Input);

    public static readonly ParameterInfo UpBias = new(typeof(PackedMatMulGlu), 4, "up_bias", ParameterKind.Input);

    public static readonly ParameterInfo GateInputScale = new(typeof(PackedMatMulGlu), 5, "gate_input_scale", ParameterKind.Input);

    public static readonly ParameterInfo UpInputScale = new(typeof(PackedMatMulGlu), 6, "up_input_scale", ParameterKind.Input);

    public static readonly ParameterInfo GateWeightScale = new(typeof(PackedMatMulGlu), 7, "gate_weight_scale", ParameterKind.Input);

    public static readonly ParameterInfo UpWeightScale = new(typeof(PackedMatMulGlu), 8, "up_weight_scale", ParameterKind.Input);

    public GluType GluType { get; }

    public DataType OutputDataType { get; }

    public PackedMatMulRhsLayout RhsLayout { get; }

    public MatMulQuantizationMode QuantizationMode { get; }

    public long WeightBlockN { get; }

    public long WeightBlockK { get; }

    public override string DisplayProperty() =>
        $"GluType: {GluType}, OutputDataType: {OutputDataType}, RhsLayout: {RhsLayout}, " +
        $"QuantizationMode: {QuantizationMode}, WeightBlock: [{WeightBlockN}, {WeightBlockK}]";
}
