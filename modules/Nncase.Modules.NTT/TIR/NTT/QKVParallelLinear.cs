// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class QKVParallelLinear : NTTKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(QKVParallelLinear), 0, "input");

    public static readonly ParameterInfo QWeight = new(typeof(QKVParallelLinear), 1, "q_weight");

    public static readonly ParameterInfo KWeight = new(typeof(QKVParallelLinear), 2, "k_weight");

    public static readonly ParameterInfo VWeight = new(typeof(QKVParallelLinear), 3, "v_weight");

    public static readonly ParameterInfo QBias = new(typeof(QKVParallelLinear), 4, "q_bias");

    public static readonly ParameterInfo KBias = new(typeof(QKVParallelLinear), 5, "k_bias");

    public static readonly ParameterInfo VBias = new(typeof(QKVParallelLinear), 6, "v_bias");

    public static readonly ParameterInfo QInputScale = new(typeof(QKVParallelLinear), 7, "q_input_scale");

    public static readonly ParameterInfo KInputScale = new(typeof(QKVParallelLinear), 8, "k_input_scale");

    public static readonly ParameterInfo VInputScale = new(typeof(QKVParallelLinear), 9, "v_input_scale");

    public static readonly ParameterInfo QWeightScale = new(typeof(QKVParallelLinear), 10, "q_weight_scale");

    public static readonly ParameterInfo KWeightScale = new(typeof(QKVParallelLinear), 11, "k_weight_scale");

    public static readonly ParameterInfo VWeightScale = new(typeof(QKVParallelLinear), 12, "v_weight_scale");

    public static readonly ParameterInfo QOutput = new(typeof(QKVParallelLinear), 13, "q_output");

    public static readonly ParameterInfo KOutput = new(typeof(QKVParallelLinear), 14, "k_output");

    public static readonly ParameterInfo VOutput = new(typeof(QKVParallelLinear), 15, "v_output");

    public long NumHeads { get; }

    public long NumKvHeads { get; }

    public override string DisplayProperty() => $"NumHeads: {NumHeads}, NumKvHeads: {NumKvHeads}";
}
