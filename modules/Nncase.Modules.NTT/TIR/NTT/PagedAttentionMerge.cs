// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.TIR.NTT;

/// <summary>
/// Merges FP32 online-softmax states from every KV partition.
/// </summary>
public sealed partial class PagedAttentionMerge : NTTKernelOp
{
    public static readonly ParameterInfo MaxState = new(typeof(PagedAttentionMerge), 0, "maxState", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo SumState = new(typeof(PagedAttentionMerge), 1, "sumState", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo AccState = new(typeof(PagedAttentionMerge), 2, "accState", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo OutputGate = new(typeof(PagedAttentionMerge), 3, "outputGate", memoryEffect: MemoryEffect.Read);

    public static readonly ParameterInfo Output = new(typeof(PagedAttentionMerge), 4, "output", memoryEffect: MemoryEffect.Write);

    public IRArray<AttentionDimKind> Layout { get; }

    public int HiddenSize { get; }

    public int SplitHierarchyAxis { get; }

    public int SplitCount { get; }
}
