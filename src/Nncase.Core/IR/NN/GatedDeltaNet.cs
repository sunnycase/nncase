// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

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

    public static readonly ParameterInfo ZWeight = new(typeof(GatedDeltaNet), 3, "z_weight", ParameterKind.Input);

    public static readonly ParameterInfo BWeight = new(typeof(GatedDeltaNet), 4, "b_weight", ParameterKind.Input);

    public static readonly ParameterInfo AWeight = new(typeof(GatedDeltaNet), 5, "a_weight", ParameterKind.Input);

    public static readonly ParameterInfo ConvWeight = new(typeof(GatedDeltaNet), 6, "conv_weight", ParameterKind.Input);

    public static readonly ParameterInfo ALog = new(typeof(GatedDeltaNet), 7, "a_log", ParameterKind.Input);

    public static readonly ParameterInfo DtBias = new(typeof(GatedDeltaNet), 8, "dt_bias", ParameterKind.Input);

    public static readonly ParameterInfo NormWeight = new(typeof(GatedDeltaNet), 9, "norm_weight", ParameterKind.Input);

    public static readonly ParameterInfo OutputWeight = new(typeof(GatedDeltaNet), 10, "output_weight", ParameterKind.Input);

    public static readonly ParameterInfo LayerId = new(
        typeof(GatedDeltaNet),
        11,
        "layer_id",
        TypePatternUtility.IsDimensionType(),
        ParameterKind.Attribute);

    public long NumKeyHeads { get; }

    public long NumValueHeads { get; }

    public long KeyHeadDim { get; }

    public long ValueHeadDim { get; }

    public long ConvKernelSize { get; }

    public float Epsilon { get; }

    public override bool CanFoldConstCall => false;

    public override string DisplayProperty() =>
        $"NumKeyHeads: {NumKeyHeads}, NumValueHeads: {NumValueHeads}, " +
        $"KeyHeadDim: {KeyHeadDim}, ValueHeadDim: {ValueHeadDim}, " +
        $"ConvKernelSize: {ConvKernelSize}, Epsilon: {Epsilon}";
}

/// <summary>
/// Computes the projected and convolved QKV stream for a gated delta network.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class GatedDeltaNetProjection : Op
{
    public static readonly ParameterInfo Input = new(typeof(GatedDeltaNetProjection), 0, "input", ParameterKind.Input);

    public static readonly ParameterInfo State = new(
        typeof(GatedDeltaNetProjection),
        1,
        "state",
        ParameterKind.Input,
        MemoryEffect.ChipReadWrite);

    public static readonly ParameterInfo QKVWeight = new(typeof(GatedDeltaNetProjection), 2, "qkv_weight", ParameterKind.Input);

    public static readonly ParameterInfo ConvWeight = new(typeof(GatedDeltaNetProjection), 3, "conv_weight", ParameterKind.Input);

    public static readonly ParameterInfo LayerId = new(
        typeof(GatedDeltaNetProjection),
        4,
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
    public static readonly ParameterInfo Input = new(typeof(GatedDeltaNetRecurrentCore), 0, "input", ParameterKind.Input);

    public static readonly ParameterInfo State = new(
        typeof(GatedDeltaNetRecurrentCore),
        1,
        "state",
        ParameterKind.Input,
        MemoryEffect.ChipReadWrite);

    public static readonly ParameterInfo QKV = new(typeof(GatedDeltaNetRecurrentCore), 2, "qkv", ParameterKind.Input);

    public static readonly ParameterInfo ZWeight = new(typeof(GatedDeltaNetRecurrentCore), 3, "z_weight", ParameterKind.Input);

    public static readonly ParameterInfo BWeight = new(typeof(GatedDeltaNetRecurrentCore), 4, "b_weight", ParameterKind.Input);

    public static readonly ParameterInfo AWeight = new(typeof(GatedDeltaNetRecurrentCore), 5, "a_weight", ParameterKind.Input);

    public static readonly ParameterInfo ALog = new(typeof(GatedDeltaNetRecurrentCore), 6, "a_log", ParameterKind.Input);

    public static readonly ParameterInfo DtBias = new(typeof(GatedDeltaNetRecurrentCore), 7, "dt_bias", ParameterKind.Input);

    public static readonly ParameterInfo NormWeight = new(typeof(GatedDeltaNetRecurrentCore), 8, "norm_weight", ParameterKind.Input);

    public static readonly ParameterInfo LayerId = new(
        typeof(GatedDeltaNetRecurrentCore),
        9,
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
