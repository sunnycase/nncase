// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.TIR.NTT;

/// <summary>
/// Merges split paged-attention states and consumes the merged value directly
/// as the lhs of a packed matrix multiplication.
/// </summary>
public sealed partial class PagedAttentionMergePackedMatMul : NTTKernelOp
{
    public static readonly ParameterInfo MaxState = new(typeof(PagedAttentionMergePackedMatMul), 0, "max_state", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo SumState = new(typeof(PagedAttentionMergePackedMatMul), 1, "sum_state", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo AccState = new(typeof(PagedAttentionMergePackedMatMul), 2, "acc_state", memoryEffect: MemoryEffect.Read);

    /// <summary>
    /// Gets the metadata-only attention output view. The fused kernel does not
    /// store through this view; it carries the Seq/Head/Dim coordinate mapping.
    /// </summary>
    public static readonly ParameterInfo MergeOutputLayout = new(typeof(PagedAttentionMergePackedMatMul), 3, "merge_output_layout", memoryEffect: MemoryEffect.None);

    /// <summary>
    /// Gets the metadata-only packed-matmul lhs view. The fused kernel consumes
    /// the merged register value using this view's exact logical K layout.
    /// </summary>
    public static readonly ParameterInfo MergedLhsLayout = new(typeof(PagedAttentionMergePackedMatMul), 4, "merged_lhs_layout", memoryEffect: MemoryEffect.None);

    public static readonly ParameterInfo Rhs = new(typeof(PagedAttentionMergePackedMatMul), 5, "rhs", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Output = new(typeof(PagedAttentionMergePackedMatMul), 6, "output", memoryEffect: MemoryEffect.ReductionReadWrite);

    public static readonly ParameterInfo LoadC = new(typeof(PagedAttentionMergePackedMatMul), 7, "load_c", memoryEffect: MemoryEffect.None);

    public static readonly ParameterInfo Scale = new(typeof(PagedAttentionMergePackedMatMul), 8, "scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Addend = new(typeof(PagedAttentionMergePackedMatMul), 9, "addend", memoryEffect: MemoryEffect.Read);

    public IRArray<AttentionDimKind> Layout { get; }

    public int HiddenSize { get; }

    public int SplitHierarchyAxis { get; }

    public int SplitCount { get; }

    public IR.NTT.PackedMatMulRhsLayout RhsLayout { get; }

    public override string DisplayProperty()
        => $"RhsLayout: {RhsLayout}, SplitHierarchyAxis: {SplitHierarchyAxis}, SplitCount: {SplitCount}";
}
