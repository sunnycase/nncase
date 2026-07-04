// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

public sealed partial class PackedQKVParallelLinear : NTTKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(PackedQKVParallelLinear), 0, "input");

    public static readonly ParameterInfo QWeight = new(typeof(PackedQKVParallelLinear), 1, "q_weight");

    public static readonly ParameterInfo KWeight = new(typeof(PackedQKVParallelLinear), 2, "k_weight");

    public static readonly ParameterInfo VWeight = new(typeof(PackedQKVParallelLinear), 3, "v_weight");

    public static readonly ParameterInfo QBias = new(typeof(PackedQKVParallelLinear), 4, "q_bias");

    public static readonly ParameterInfo KBias = new(typeof(PackedQKVParallelLinear), 5, "k_bias");

    public static readonly ParameterInfo VBias = new(typeof(PackedQKVParallelLinear), 6, "v_bias");

    public static readonly ParameterInfo QInputScale = new(typeof(PackedQKVParallelLinear), 7, "q_input_scale");

    public static readonly ParameterInfo KInputScale = new(typeof(PackedQKVParallelLinear), 8, "k_input_scale");

    public static readonly ParameterInfo VInputScale = new(typeof(PackedQKVParallelLinear), 9, "v_input_scale");

    public static readonly ParameterInfo QWeightScale = new(typeof(PackedQKVParallelLinear), 10, "q_weight_scale");

    public static readonly ParameterInfo KWeightScale = new(typeof(PackedQKVParallelLinear), 11, "k_weight_scale");

    public static readonly ParameterInfo VWeightScale = new(typeof(PackedQKVParallelLinear), 12, "v_weight_scale");

    public static readonly ParameterInfo QOutput = new(typeof(PackedQKVParallelLinear), 13, "q_output");

    public static readonly ParameterInfo KOutput = new(typeof(PackedQKVParallelLinear), 14, "k_output");

    public static readonly ParameterInfo VOutput = new(typeof(PackedQKVParallelLinear), 15, "v_output");

    public long NumHeads { get; }

    public long NumKvHeads { get; }

    public override string DisplayProperty() => $"NumHeads: {NumHeads}, NumKvHeads: {NumKvHeads}";
}
