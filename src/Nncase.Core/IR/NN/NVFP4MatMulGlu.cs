// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.NN;

/// <summary>
/// Fused gate/up NVFP4 projections followed by a GLU activation.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class NVFP4MatMulGlu : Op
{
    public static readonly ParameterInfo Input = new(typeof(NVFP4MatMulGlu), 0, "input", ParameterKind.Input);

    public static readonly ParameterInfo GateWeightPacked = new(typeof(NVFP4MatMulGlu), 1, "gate_weight_packed", ParameterKind.Input);

    public static readonly ParameterInfo UpWeightPacked = new(typeof(NVFP4MatMulGlu), 2, "up_weight_packed", ParameterKind.Input);

    public static readonly ParameterInfo GateWeightScale = new(typeof(NVFP4MatMulGlu), 3, "gate_weight_scale", ParameterKind.Input);

    public static readonly ParameterInfo UpWeightScale = new(typeof(NVFP4MatMulGlu), 4, "up_weight_scale", ParameterKind.Input);

    public static readonly ParameterInfo GateInputGlobalScale = new(typeof(NVFP4MatMulGlu), 5, "gate_input_global_scale", ParameterKind.Input);

    public static readonly ParameterInfo UpInputGlobalScale = new(typeof(NVFP4MatMulGlu), 6, "up_input_global_scale", ParameterKind.Input);

    public static readonly ParameterInfo GateWeightGlobalScale = new(typeof(NVFP4MatMulGlu), 7, "gate_weight_global_scale", ParameterKind.Input);

    public static readonly ParameterInfo UpWeightGlobalScale = new(typeof(NVFP4MatMulGlu), 8, "up_weight_global_scale", ParameterKind.Input);

    public GluType GluType { get; }

    public DataType OutputDataType { get; }

    public long GroupSize { get; }

    public override string DisplayProperty() =>
        $"GluType: {GluType}, OutputDataType: {OutputDataType}, GroupSize: {GroupSize}";
}
