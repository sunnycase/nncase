// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
namespace Nncase.TIR.NTT;

/// <summary>
/// Channel-sharded stateful convolution stage of gated delta network.
/// </summary>
public sealed partial class GatedDeltaNetConvolution : NTTKernelOp
{
    public static readonly ParameterInfo QKV = new(typeof(GatedDeltaNetConvolution), 0, "qkv", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo State = new(typeof(GatedDeltaNetConvolution), 1, "state", memoryEffect: MemoryEffect.ChipReadWrite);
    public static readonly ParameterInfo ConvWeight = new(typeof(GatedDeltaNetConvolution), 2, "conv_weight", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo QKVOutput = new(typeof(GatedDeltaNetConvolution), 3, "qkv_output", memoryEffect: MemoryEffect.Write);
    public static readonly ParameterInfo LayerId = new(typeof(GatedDeltaNetConvolution), 4, "layer_id", TypePatternUtility.IsDimensionType(), memoryEffect: MemoryEffect.None);

    public long ConvKernelSize { get; }

    public override string DisplayProperty() => $"ConvKernelSize: {ConvKernelSize}";
}

/// <summary>
/// Head-sharded recurrent core of gated delta network.
/// </summary>
public sealed partial class GatedDeltaNetRecurrentCore : NTTKernelOp
{
    public static readonly ParameterInfo State = new(typeof(GatedDeltaNetRecurrentCore), 0, "state", memoryEffect: MemoryEffect.ChipReadWrite);
    public static readonly ParameterInfo QKV = new(typeof(GatedDeltaNetRecurrentCore), 1, "qkv", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo Z = new(typeof(GatedDeltaNetRecurrentCore), 2, "z", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo ProjectionInput = new(typeof(GatedDeltaNetRecurrentCore), 3, "projection_input", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo BWeight = new(typeof(GatedDeltaNetRecurrentCore), 4, "b_weight", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo AWeight = new(typeof(GatedDeltaNetRecurrentCore), 5, "a_weight", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo ALog = new(typeof(GatedDeltaNetRecurrentCore), 6, "a_log", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo DtBias = new(typeof(GatedDeltaNetRecurrentCore), 7, "dt_bias", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo NormWeight = new(typeof(GatedDeltaNetRecurrentCore), 8, "norm_weight", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo CoreScratch = new(typeof(GatedDeltaNetRecurrentCore), 9, "core_scratch", memoryEffect: MemoryEffect.ChipReadWrite);
    public static readonly ParameterInfo GatedOutput = new(typeof(GatedDeltaNetRecurrentCore), 10, "gated_output", memoryEffect: MemoryEffect.Write);
    public static readonly ParameterInfo LayerId = new(typeof(GatedDeltaNetRecurrentCore), 11, "layer_id", TypePatternUtility.IsDimensionType(), memoryEffect: MemoryEffect.None);

    public long NumKeyHeads { get; }

    public long NumValueHeads { get; }

    public long KeyHeadDim { get; }

    public long ValueHeadDim { get; }

    public float Epsilon { get; }

    public override string DisplayProperty() =>
        $"NumKeyHeads: {NumKeyHeads}, NumValueHeads: {NumValueHeads}, " +
        $"KeyHeadDim: {KeyHeadDim}, ValueHeadDim: {ValueHeadDim}, Epsilon: {Epsilon}";
}
