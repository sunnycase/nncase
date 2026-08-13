// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.Utilities;

/// <summary>
/// Describes one legal tensor-axis/hierarchy-axis split considered by
/// AutoDistributed.
/// </summary>
public sealed record DistributedSplitCandidateContext(
    TensorType TensorType,
    int TensorAxis,
    Placement Placement,
    IRArray<int> HierarchyAxes,
    Dimension ContiguousGranularity,
    long MaximumExtent);

/// <summary>
/// Target-owned policy for constructing physical split-distribution
/// candidates. Operator evaluators remain independent of physical topology.
/// </summary>
public interface IDistributedSplitCandidateProvider
{
    IReadOnlyList<SBPSplit> GetCandidates(DistributedSplitCandidateContext context);
}

/// <summary>
/// Default split policy used by CPU and targets without a specialized physical
/// block distribution.
/// </summary>
public sealed class ContiguousDistributedSplitCandidateProvider : IDistributedSplitCandidateProvider
{
    public static readonly ContiguousDistributedSplitCandidateProvider Instance = new();

    private ContiguousDistributedSplitCandidateProvider()
    {
    }

    public IReadOnlyList<SBPSplit> GetCandidates(DistributedSplitCandidateContext context)
        => [SBP.SContiguous(context.HierarchyAxes, context.ContiguousGranularity)];
}
