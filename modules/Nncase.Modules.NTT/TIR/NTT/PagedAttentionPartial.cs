// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.TIR.NTT;

/// <summary>
/// Computes either one KV partition's FP32 online-softmax state or, for a
/// planner-selected short context, the complete attention output.
/// </summary>
public sealed partial class PagedAttentionPartial : NTTKernelOp
{
    public static readonly ParameterInfo Q = new(typeof(PagedAttentionPartial), 0, "q", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo KVCaches = new(typeof(PagedAttentionPartial), 1, "kvCaches", memoryEffect: MemoryEffect.ChipRead);

    public static readonly ParameterInfo Extra = new(typeof(PagedAttentionPartial), 2, "extra", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Scale = new(typeof(PagedAttentionPartial), 3, "scale", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo LayerId = new(typeof(PagedAttentionPartial), 4, "layerId", IR.TypePatternUtility.IsDimensionType(), memoryEffect: MemoryEffect.None);

    public static readonly ParameterInfo MaxState = new(typeof(PagedAttentionPartial), 5, "maxState", memoryEffect: MemoryEffect.Write);

    public static readonly ParameterInfo SumState = new(typeof(PagedAttentionPartial), 6, "sumState", memoryEffect: MemoryEffect.Write);

    public static readonly ParameterInfo AccState = new(typeof(PagedAttentionPartial), 7, "accState", memoryEffect: MemoryEffect.Write);

    public static readonly ParameterInfo Output = new(typeof(PagedAttentionPartial), 8, "output", memoryEffect: MemoryEffect.Write);

    public IRArray<AttentionDimKind> Layout { get; }

    public int HiddenSize { get; }

    public int SplitHierarchyAxis { get; }

    public int SplitCount { get; }

    /// <summary>
    /// Gets the largest single-sequence context length computed directly by
    /// every query owner instead of publishing and merging split state.
    /// </summary>
    public long DirectContextThreshold { get; }
}
