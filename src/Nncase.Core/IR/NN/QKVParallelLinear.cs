// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.NN;

/// <summary>
/// Fused Q/K/V linear projection.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class QKVParallelLinear : Op
{
    public static readonly ParameterInfo Input = new(typeof(QKVParallelLinear), 0, "input", ParameterKind.Input);

    public static readonly ParameterInfo QWeight = new(typeof(QKVParallelLinear), 1, "q_weight", ParameterKind.Input);

    public static readonly ParameterInfo KWeight = new(typeof(QKVParallelLinear), 2, "k_weight", ParameterKind.Input);

    public static readonly ParameterInfo VWeight = new(typeof(QKVParallelLinear), 3, "v_weight", ParameterKind.Input);

    public static readonly ParameterInfo QBias = new(typeof(QKVParallelLinear), 4, "q_bias", ParameterKind.Input);

    public static readonly ParameterInfo KBias = new(typeof(QKVParallelLinear), 5, "k_bias", ParameterKind.Input);

    public static readonly ParameterInfo VBias = new(typeof(QKVParallelLinear), 6, "v_bias", ParameterKind.Input);

    public static readonly ParameterInfo QInputScale = new(typeof(QKVParallelLinear), 7, "q_input_scale", ParameterKind.Input);

    public static readonly ParameterInfo KInputScale = new(typeof(QKVParallelLinear), 8, "k_input_scale", ParameterKind.Input);

    public static readonly ParameterInfo VInputScale = new(typeof(QKVParallelLinear), 9, "v_input_scale", ParameterKind.Input);

    public static readonly ParameterInfo QWeightScale = new(typeof(QKVParallelLinear), 10, "q_weight_scale", ParameterKind.Input);

    public static readonly ParameterInfo KWeightScale = new(typeof(QKVParallelLinear), 11, "k_weight_scale", ParameterKind.Input);

    public static readonly ParameterInfo VWeightScale = new(typeof(QKVParallelLinear), 12, "v_weight_scale", ParameterKind.Input);

    public long NumHeads { get; }

    public long NumKvHeads { get; }

    public DataType OutputDataType { get; }

    public override string DisplayProperty() => $"NumHeads: {NumHeads}, NumKvHeads: {NumKvHeads}, OutputDataType: {OutputDataType}";
}
