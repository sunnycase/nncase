// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

/// <summary>
/// Fused Q/K/V linear projection with packed-N RHS and output layout.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class PackedQKVParallelLinear : Op
{
    public static readonly ParameterInfo Input = new(typeof(PackedQKVParallelLinear), 0, "input", ParameterKind.Input);

    public static readonly ParameterInfo QWeight = new(typeof(PackedQKVParallelLinear), 1, "q_weight", ParameterKind.Input);

    public static readonly ParameterInfo KWeight = new(typeof(PackedQKVParallelLinear), 2, "k_weight", ParameterKind.Input);

    public static readonly ParameterInfo VWeight = new(typeof(PackedQKVParallelLinear), 3, "v_weight", ParameterKind.Input);

    public static readonly ParameterInfo QBias = new(typeof(PackedQKVParallelLinear), 4, "q_bias", ParameterKind.Input);

    public static readonly ParameterInfo KBias = new(typeof(PackedQKVParallelLinear), 5, "k_bias", ParameterKind.Input);

    public static readonly ParameterInfo VBias = new(typeof(PackedQKVParallelLinear), 6, "v_bias", ParameterKind.Input);

    public static readonly ParameterInfo QInputScale = new(typeof(PackedQKVParallelLinear), 7, "q_input_scale", ParameterKind.Input);

    public static readonly ParameterInfo KInputScale = new(typeof(PackedQKVParallelLinear), 8, "k_input_scale", ParameterKind.Input);

    public static readonly ParameterInfo VInputScale = new(typeof(PackedQKVParallelLinear), 9, "v_input_scale", ParameterKind.Input);

    public static readonly ParameterInfo QWeightScale = new(typeof(PackedQKVParallelLinear), 10, "q_weight_scale", ParameterKind.Input);

    public static readonly ParameterInfo KWeightScale = new(typeof(PackedQKVParallelLinear), 11, "k_weight_scale", ParameterKind.Input);

    public static readonly ParameterInfo VWeightScale = new(typeof(PackedQKVParallelLinear), 12, "v_weight_scale", ParameterKind.Input);

    public long NumHeads { get; }

    public long NumKvHeads { get; }

    public DataType OutputDataType { get; }

    public override string DisplayProperty() => $"NumHeads: {NumHeads}, NumKvHeads: {NumKvHeads}, OutputDataType: {OutputDataType}";
}
