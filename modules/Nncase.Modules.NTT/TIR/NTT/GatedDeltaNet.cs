// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.TIR.NTT;

/// <summary>
/// Channel-sharded projection and convolution stage of gated delta network.
/// </summary>
public sealed partial class GatedDeltaNetProjection : NTTKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(GatedDeltaNetProjection), 0, "input", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo State = new(typeof(GatedDeltaNetProjection), 1, "state", memoryEffect: MemoryEffect.ChipReadWrite);
    public static readonly ParameterInfo QKVWeight = new(typeof(GatedDeltaNetProjection), 2, "qkv_weight", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo ConvWeight = new(typeof(GatedDeltaNetProjection), 3, "conv_weight", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo QKVOutput = new(typeof(GatedDeltaNetProjection), 4, "qkv_output", memoryEffect: MemoryEffect.Write);
    public static readonly ParameterInfo LayerId = new(typeof(GatedDeltaNetProjection), 5, "layer_id", TypePatternUtility.IsDimensionType(), memoryEffect: MemoryEffect.None);

    public long ConvKernelSize { get; }

    public override string DisplayProperty() => $"ConvKernelSize: {ConvKernelSize}";
}

/// <summary>
/// Head-sharded recurrent core of gated delta network.
/// </summary>
public sealed partial class GatedDeltaNetRecurrentCore : NTTKernelOp
{
    public static readonly ParameterInfo Input = new(typeof(GatedDeltaNetRecurrentCore), 0, "input", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo State = new(typeof(GatedDeltaNetRecurrentCore), 1, "state", memoryEffect: MemoryEffect.ChipReadWrite);
    public static readonly ParameterInfo QKV = new(typeof(GatedDeltaNetRecurrentCore), 2, "qkv", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo ZWeight = new(typeof(GatedDeltaNetRecurrentCore), 3, "z_weight", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo BWeight = new(typeof(GatedDeltaNetRecurrentCore), 4, "b_weight", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo AWeight = new(typeof(GatedDeltaNetRecurrentCore), 5, "a_weight", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo ALog = new(typeof(GatedDeltaNetRecurrentCore), 6, "a_log", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo DtBias = new(typeof(GatedDeltaNetRecurrentCore), 7, "dt_bias", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo NormWeight = new(typeof(GatedDeltaNetRecurrentCore), 8, "norm_weight", memoryEffect: MemoryEffect.Read);
    public static readonly ParameterInfo GatedOutput = new(typeof(GatedDeltaNetRecurrentCore), 9, "gated_output", memoryEffect: MemoryEffect.Write);
    public static readonly ParameterInfo LayerId = new(typeof(GatedDeltaNetRecurrentCore), 10, "layer_id", TypePatternUtility.IsDimensionType(), memoryEffect: MemoryEffect.None);

    public long NumKeyHeads { get; }

    public long NumValueHeads { get; }

    public long KeyHeadDim { get; }

    public long ValueHeadDim { get; }

    public float Epsilon { get; }

    public override string DisplayProperty() =>
        $"NumKeyHeads: {NumKeyHeads}, NumValueHeads: {NumValueHeads}, " +
        $"KeyHeadDim: {KeyHeadDim}, ValueHeadDim: {ValueHeadDim}, Epsilon: {Epsilon}";
}
