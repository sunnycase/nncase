// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.NN;

/// <summary>
/// GLU activation kind used by <see cref="MatMulGlu"/>.
/// </summary>
public enum GluType
{
    /// <summary>
    /// SwiGLU: silu(gate) * up.
    /// </summary>
    SwiGLU,
}

/// <summary>
/// Fused gate/up linear projection followed by GLU activation.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class MatMulGlu : Op
{
    public static readonly ParameterInfo Input = new(typeof(MatMulGlu), 0, "input", ParameterKind.Input);

    public static readonly ParameterInfo GateWeight = new(typeof(MatMulGlu), 1, "gate_weight", ParameterKind.Input);

    public static readonly ParameterInfo UpWeight = new(typeof(MatMulGlu), 2, "up_weight", ParameterKind.Input);

    public static readonly ParameterInfo GateBias = new(typeof(MatMulGlu), 3, "gate_bias", ParameterKind.Input);

    public static readonly ParameterInfo UpBias = new(typeof(MatMulGlu), 4, "up_bias", ParameterKind.Input);

    public static readonly ParameterInfo GateInputScale = new(typeof(MatMulGlu), 5, "gate_input_scale", ParameterKind.Input);

    public static readonly ParameterInfo UpInputScale = new(typeof(MatMulGlu), 6, "up_input_scale", ParameterKind.Input);

    public static readonly ParameterInfo GateWeightScale = new(typeof(MatMulGlu), 7, "gate_weight_scale", ParameterKind.Input);

    public static readonly ParameterInfo UpWeightScale = new(typeof(MatMulGlu), 8, "up_weight_scale", ParameterKind.Input);

    public GluType GluType { get; }

    public DataType OutputDataType { get; }

    public override string DisplayProperty() => $"GluType: {GluType}, OutputDataType: {OutputDataType}";
}
