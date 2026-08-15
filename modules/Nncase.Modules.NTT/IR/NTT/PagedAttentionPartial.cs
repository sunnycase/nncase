// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.NN;
using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

/// <summary>
/// Computes per-partition online-softmax states for paged attention.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class PagedAttentionPartial : Op
{
    public static readonly ParameterInfo Q = new(typeof(PagedAttentionPartial), 0, "q", ParameterKind.Input);

    public static readonly ParameterInfo KVCaches = new(typeof(PagedAttentionPartial), 1, "kv_caches", ParameterKind.Attribute);

    public static readonly ParameterInfo Extra = new(typeof(PagedAttentionPartial), 2, "extra", ParameterKind.Input);

    public static readonly ParameterInfo Scale = new(typeof(PagedAttentionPartial), 3, "scale", ParameterKind.Attribute);

    public static readonly ParameterInfo LayerId = new(typeof(PagedAttentionPartial), 4, "layer_id", IR.TypePatternUtility.IsDimensionType(), ParameterKind.Attribute);

    public IRArray<AttentionDimKind> Layout { get; }

    public int HiddenSize { get; }

    public int SplitHierarchyAxis { get; }

    public int SplitCount { get; }

    public override string DisplayProperty()
        => $"Layout [{string.Join(',', Layout)}], HiddenSize: {HiddenSize}, " +
            $"SplitHierarchyAxis: {SplitHierarchyAxis}, SplitCount: {SplitCount}";
}
