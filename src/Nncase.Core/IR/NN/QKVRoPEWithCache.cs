// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.NN;

/// <summary>
/// Applies Q/K normalization and RoPE, returns the transformed query, and
/// updates the paged-attention key/value cache.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class QKVRoPEWithCache : Op
{
    /// <summary>
    /// Gets the tuple containing Q, K, and V tensors in that order.
    /// </summary>
    public static readonly ParameterInfo QKV = new(typeof(QKVRoPEWithCache), 0, "qkv", ParameterKind.Input);

    public static readonly ParameterInfo QStats = new(typeof(QKVRoPEWithCache), 1, "q_stats", ParameterKind.Input);

    public static readonly ParameterInfo KStats = new(typeof(QKVRoPEWithCache), 2, "k_stats", ParameterKind.Input);

    public static readonly ParameterInfo QScale = new(typeof(QKVRoPEWithCache), 3, "q_scale", ParameterKind.Input);

    public static readonly ParameterInfo KScale = new(typeof(QKVRoPEWithCache), 4, "k_scale", ParameterKind.Input);

    public static readonly ParameterInfo QBias = new(typeof(QKVRoPEWithCache), 5, "q_bias", ParameterKind.Input);

    public static readonly ParameterInfo KBias = new(typeof(QKVRoPEWithCache), 6, "k_bias", ParameterKind.Input);

    public static readonly ParameterInfo Cos = new(typeof(QKVRoPEWithCache), 7, "cos", ParameterKind.Input);

    public static readonly ParameterInfo Sin = new(typeof(QKVRoPEWithCache), 8, "sin", ParameterKind.Input);

    public static readonly ParameterInfo KVCaches = new(typeof(QKVRoPEWithCache), 9, "kv_caches", ParameterKind.Attribute);

    public static readonly ParameterInfo LayerId = new(typeof(QKVRoPEWithCache), 10, "layer_id", IsDimensionType(), ParameterKind.Attribute);

    public int QAxis { get; }

    public float QEpsilon { get; }

    public bool QUseMean { get; }

    public int KAxis { get; }

    public float KEpsilon { get; }

    public bool KUseMean { get; }

    /// <summary>
    /// Gets the semantic axis layout of the Q, K, and V inputs.
    /// </summary>
    public IRArray<AttentionDimKind> QKVLayout { get; }

    /// <summary>
    /// Gets the semantic axis layout of the returned query and cache slots.
    /// </summary>
    public IRArray<AttentionDimKind> AttentionLayout { get; }

    public override bool CanFoldConstCall => false;

    public override string DisplayProperty() =>
        $"QAxis: {QAxis}, QEpsilon: {QEpsilon}, QUseMean: {QUseMean}, " +
        $"KAxis: {KAxis}, KEpsilon: {KEpsilon}, KUseMean: {KUseMean}, " +
        $"QKVLayout [{string.Join(',', QKVLayout)}], " +
        $"AttentionLayout [{string.Join(',', AttentionLayout)}]";
}
