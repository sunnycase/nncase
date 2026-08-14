// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.TIR.NTT;

/// <summary>
/// Applies Q/K normalization and RoPE, writes the final Q tensor, and updates
/// the key/value cache in one block-local kernel.
/// </summary>
public sealed partial class QKVRoPEWithCache : NTTKernelOp
{
    public static readonly ParameterInfo Q = new(typeof(QKVRoPEWithCache), 0, "q", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo K = new(typeof(QKVRoPEWithCache), 1, "k", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo V = new(typeof(QKVRoPEWithCache), 2, "v", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo QScale = new(typeof(QKVRoPEWithCache), 3, "q_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo KScale = new(typeof(QKVRoPEWithCache), 4, "k_scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo QBias = new(typeof(QKVRoPEWithCache), 5, "q_bias", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo KBias = new(typeof(QKVRoPEWithCache), 6, "k_bias", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Cos = new(typeof(QKVRoPEWithCache), 7, "cos", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Sin = new(typeof(QKVRoPEWithCache), 8, "sin", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo KVCaches = new(typeof(QKVRoPEWithCache), 9, "kv_caches", memoryEffect: MemoryEffect.ChipWrite);

    public static readonly ParameterInfo LayerId = new(typeof(QKVRoPEWithCache), 10, "layer_id", IR.TypePatternUtility.IsDimensionType(), memoryEffect: MemoryEffect.None);

    public static readonly ParameterInfo QOutput = new(typeof(QKVRoPEWithCache), 11, "q_output", memoryEffect: MemoryEffect.Write);

    public int QAxis { get; }

    public float QEpsilon { get; }

    public bool QUseMean { get; }

    public int KAxis { get; }

    public float KEpsilon { get; }

    public bool KUseMean { get; }

    public IRArray<AttentionDimKind> QKVLayout { get; }

    public IRArray<AttentionDimKind> AttentionLayout { get; }

    public override string DisplayProperty() =>
        $"QAxis: {QAxis}, QEpsilon: {QEpsilon}, QUseMean: {QUseMean}, " +
        $"KAxis: {KAxis}, KEpsilon: {KEpsilon}, KUseMean: {KUseMean}, " +
        $"QKVLayout [{string.Join(',', QKVLayout)}], " +
        $"AttentionLayout [{string.Join(',', AttentionLayout)}]";
}
