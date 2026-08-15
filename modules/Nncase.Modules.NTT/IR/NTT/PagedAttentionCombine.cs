// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.NN;
using Nncase.PatternMatch;

namespace Nncase.IR.NTT;

/// <summary>
/// Merges paged-attention online-softmax states into the logical attention output.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class PagedAttentionCombine : Op
{
    public static readonly ParameterInfo MaxState = new(typeof(PagedAttentionCombine), 0, "max_state", ParameterKind.Input);

    public static readonly ParameterInfo SumState = new(typeof(PagedAttentionCombine), 1, "sum_state", ParameterKind.Input);

    public static readonly ParameterInfo AccState = new(typeof(PagedAttentionCombine), 2, "acc_state", ParameterKind.Input);

    public IRArray<AttentionDimKind> Layout { get; }

    public int HiddenSize { get; }

    public DataType OutputDataType { get; }

    public IRType OutputType { get; }

    public int SplitHierarchyAxis { get; }

    public int SplitCount { get; }

    public override string DisplayProperty()
        => $"Layout [{string.Join(',', Layout)}], HiddenSize: {HiddenSize}, " +
            $"OutputDataType: {OutputDataType}, OutputType: {OutputType}, " +
            $"SplitHierarchyAxis: {SplitHierarchyAxis}, " +
            $"SplitCount: {SplitCount}";
}
