// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using Nncase.IR.Math;

namespace Nncase.IR.NN;

/// <summary>
/// Stateful gated delta network used by Qwen3.5 linear-attention layers.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class GatedDeltaNet : Op
{
    public static readonly ParameterInfo Input = new(typeof(GatedDeltaNet), 0, "input", ParameterKind.Input);

    public static readonly ParameterInfo State = new(
        typeof(GatedDeltaNet),
        1,
        "state",
        ParameterKind.Input,
        MemoryEffect.ChipReadWrite);

    public static readonly ParameterInfo QKVWeight = new(typeof(GatedDeltaNet), 2, "qkv_weight", ParameterKind.Input);

    public static readonly ParameterInfo QKVWeightScale = new(typeof(GatedDeltaNet), 3, "qkv_weight_scale", ParameterKind.Input);

    public static readonly ParameterInfo ZWeight = new(typeof(GatedDeltaNet), 4, "z_weight", ParameterKind.Input);

    public static readonly ParameterInfo ZWeightScale = new(typeof(GatedDeltaNet), 5, "z_weight_scale", ParameterKind.Input);

    public static readonly ParameterInfo BWeight = new(typeof(GatedDeltaNet), 6, "b_weight", ParameterKind.Input);

    public static readonly ParameterInfo AWeight = new(typeof(GatedDeltaNet), 7, "a_weight", ParameterKind.Input);

    public static readonly ParameterInfo ConvWeight = new(typeof(GatedDeltaNet), 8, "conv_weight", ParameterKind.Input);

    public static readonly ParameterInfo ALog = new(typeof(GatedDeltaNet), 9, "a_log", ParameterKind.Input);

    public static readonly ParameterInfo DtBias = new(typeof(GatedDeltaNet), 10, "dt_bias", ParameterKind.Input);

    public static readonly ParameterInfo NormWeight = new(typeof(GatedDeltaNet), 11, "norm_weight", ParameterKind.Input);

    public static readonly ParameterInfo OutputWeight = new(typeof(GatedDeltaNet), 12, "output_weight", ParameterKind.Input);

    public static readonly ParameterInfo OutputWeightScale = new(typeof(GatedDeltaNet), 13, "output_weight_scale", ParameterKind.Input);

    public static readonly ParameterInfo LayerId = new(
        typeof(GatedDeltaNet),
        14,
        "layer_id",
        TypePatternUtility.IsDimensionType(),
        ParameterKind.Attribute);

    public long NumKeyHeads { get; }

    public long NumValueHeads { get; }

    public long KeyHeadDim { get; }

    public long ValueHeadDim { get; }

    public long ConvKernelSize { get; }

    public float Epsilon { get; }

    public MatMulQuantizationMode QuantizationMode { get; }

    public long WeightBlockN { get; }

    public long WeightBlockK { get; }

    public override bool CanFoldConstCall => false;

    public override string DisplayProperty() =>
        $"NumKeyHeads: {NumKeyHeads}, NumValueHeads: {NumValueHeads}, " +
        $"KeyHeadDim: {KeyHeadDim}, ValueHeadDim: {ValueHeadDim}, " +
        $"ConvKernelSize: {ConvKernelSize}, Epsilon: {Epsilon}, " +
        $"QuantizationMode: {QuantizationMode}, WeightBlock: [{WeightBlockN}, {WeightBlockK}]";
}

/// <summary>
/// Applies the stateful convolution to an already projected QKV stream.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class GatedDeltaNetConvolution : Op
{
    public static readonly ParameterInfo QKV = new(typeof(GatedDeltaNetConvolution), 0, "qkv", ParameterKind.Input);

    public static readonly ParameterInfo State = new(
        typeof(GatedDeltaNetConvolution),
        1,
        "state",
        ParameterKind.Input,
        MemoryEffect.ChipReadWrite);

    public static readonly ParameterInfo ConvWeight = new(typeof(GatedDeltaNetConvolution), 2, "conv_weight", ParameterKind.Input);

    public static readonly ParameterInfo LayerId = new(
        typeof(GatedDeltaNetConvolution),
        3,
        "layer_id",
        TypePatternUtility.IsDimensionType(),
        ParameterKind.Attribute);

    public long ConvKernelSize { get; }

    public override bool CanFoldConstCall => false;

    public override string DisplayProperty() => $"ConvKernelSize: {ConvKernelSize}";
}

/// <summary>
/// Applies the recurrent state update and produces the gated value-head activation.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class GatedDeltaNetRecurrentCore : Op
{
    public static readonly ParameterInfo State = new(
        typeof(GatedDeltaNetRecurrentCore),
        0,
        "state",
        ParameterKind.Input,
        MemoryEffect.ChipReadWrite);

    public static readonly ParameterInfo QKV = new(typeof(GatedDeltaNetRecurrentCore), 1, "qkv", ParameterKind.Input);

    public static readonly ParameterInfo Z = new(typeof(GatedDeltaNetRecurrentCore), 2, "z", ParameterKind.Input);

    public static readonly ParameterInfo BProjection = new(typeof(GatedDeltaNetRecurrentCore), 3, "b_projection", ParameterKind.Input);

    public static readonly ParameterInfo AProjection = new(typeof(GatedDeltaNetRecurrentCore), 4, "a_projection", ParameterKind.Input);

    public static readonly ParameterInfo ALog = new(typeof(GatedDeltaNetRecurrentCore), 5, "a_log", ParameterKind.Input);

    public static readonly ParameterInfo DtBias = new(typeof(GatedDeltaNetRecurrentCore), 6, "dt_bias", ParameterKind.Input);

    public static readonly ParameterInfo NormWeight = new(typeof(GatedDeltaNetRecurrentCore), 7, "norm_weight", ParameterKind.Input);

    public static readonly ParameterInfo LayerId = new(
        typeof(GatedDeltaNetRecurrentCore),
        8,
        "layer_id",
        TypePatternUtility.IsDimensionType(),
        ParameterKind.Attribute);

    public long NumKeyHeads { get; }

    public long NumValueHeads { get; }

    public long KeyHeadDim { get; }

    public long ValueHeadDim { get; }

    public float Epsilon { get; }

    public override bool CanFoldConstCall => false;

    public override string DisplayProperty() =>
        $"NumKeyHeads: {NumKeyHeads}, NumValueHeads: {NumValueHeads}, " +
        $"KeyHeadDim: {KeyHeadDim}, ValueHeadDim: {ValueHeadDim}, Epsilon: {Epsilon}";
}
