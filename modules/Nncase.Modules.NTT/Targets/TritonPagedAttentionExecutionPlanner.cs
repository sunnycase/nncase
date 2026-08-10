// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.Utilities;

namespace Nncase.Targets;

/// <summary>
/// Chooses whether a persistent Triton block hierarchy should partition the KV
/// sequence of a paged-attention call.
/// </summary>
public sealed class TritonPagedAttentionExecutionPlanner
{
    private readonly TargetMachineModel _machine;
    private readonly TargetMemoryResourceSpec _rootMemory;

    public TritonPagedAttentionExecutionPlanner(TargetMachineModel machine)
    {
        _machine = machine ?? throw new ArgumentNullException(nameof(machine));
        _rootMemory = machine.GetMemoryResource(machine.GetMemorySpace(machine.RootMemorySpace));
    }

    public PagedAttentionExecutionPlan Plan(PagedAttentionExecutionPlanQuery query)
    {
        var placement = query.QueryType.Placement;
        var candidates = Enumerable.Range(0, placement.Rank)
            .Where(axis => placement.IsPhysicalBlockAxis(axis))
            .Where(axis => placement.Hierarchy[axis] > 1)
            .Where(axis => !query.UsesHierarchyAxis(axis))
            .OrderByDescending(axis => placement.Hierarchy[axis])
            .ThenBy(axis => axis)
            .ToArray();
        if (candidates.Length == 0)
        {
            return PagedAttentionExecutionPlan.Direct;
        }

        var directCycles = EstimateCycles(query, splitCount: 1);
        var best = candidates
            .SelectMany(axis => GetSplitCounts(placement.Hierarchy[axis])
                .Select(splitCount => new
                {
                    Axis = axis,
                    SplitCount = splitCount,
                }))
            .Select(candidate => new
            {
                candidate.Axis,
                candidate.SplitCount,
                Cycles = EstimateCycles(query, candidate.Axis, candidate.SplitCount),
            })
            .OrderBy(candidate => candidate.Cycles)
            .ThenBy(candidate => candidate.SplitCount)
            .ThenBy(candidate => candidate.Axis)
            .First();
        var plan = best.Cycles < directCycles
            ? new(
                PagedAttentionExecutionKind.SplitKV,
                best.Axis,
                best.SplitCount)
            : PagedAttentionExecutionPlan.Direct;
        plan.Validate(query);
        return plan;
    }

    private double EstimateCycles(PagedAttentionExecutionPlanQuery query, int splitCount)
    {
        if (splitCount != 1)
        {
            throw new ArgumentOutOfRangeException(nameof(splitCount));
        }

        var physicalBlocks = query.QueryType.Placement.Hierarchy.Aggregate(
            1.0,
            static (acc, extent) => acc * Math.Max(1, extent));
        var localQueryBytes = query.LocalQueryScalarElements * query.QueryElementSizeBytes;
        var loadBytes = (localQueryBytes + (query.KVScalarElements * query.KVElementSizeBytes)) * physicalBlocks;
        var storeBytes = localQueryBytes * physicalBlocks;
        return Math.Max(
            EstimateComputeCycles(query, query.ContextLength),
            EstimateMemoryCycles(loadBytes, storeBytes, physicalBlocks, physicalBlocks));
    }

    private double EstimateCycles(
        PagedAttentionExecutionPlanQuery query,
        int splitHierarchyAxis,
        int splitCount)
    {
        var placement = query.QueryType.Placement;
        var physicalBlocks = placement.Hierarchy.Aggregate(
            1.0,
            static (acc, extent) => acc * Math.Max(1, extent));
        var hierarchyExtent = placement.Hierarchy[splitHierarchyAxis];
        var splitGroups = physicalBlocks / hierarchyExtent;
        var activePartialBlocks = splitGroups * splitCount;
        var mergeOwnerBlocks = splitGroups;
        var localQueryBytes = query.LocalQueryScalarElements * query.QueryElementSizeBytes;
        var partialStateBytes = query.PartialStateScalarElements * sizeof(float);
        var localContextLength = MathUtility.CeilDiv(query.ContextLength, splitCount);

        var partialLoadBytes = (localQueryBytes * activePartialBlocks) +
            (query.KVScalarElements * query.KVElementSizeBytes * splitGroups);
        var partialStoreBytes = partialStateBytes * activePartialBlocks;
        var partialCycles = Math.Max(
            EstimateComputeCycles(query, localContextLength),
            EstimateMemoryCycles(
                partialLoadBytes,
                partialStoreBytes,
                activePartialBlocks,
                physicalBlocks));

        var mergeLoadBytes = partialStateBytes * splitCount * mergeOwnerBlocks;
        var mergeStoreBytes = localQueryBytes * mergeOwnerBlocks;
        var mergeComputeCycles = query.PartialStateScalarElements * splitCount /
            Math.Max(1.0, _machine.Compute.ElementwiseElementsPerCycle);
        var mergeCycles = Math.Max(
            mergeComputeCycles,
            EstimateMemoryCycles(
                mergeLoadBytes,
                mergeStoreBytes,
                mergeOwnerBlocks,
                physicalBlocks));
        return partialCycles + mergeCycles +
            (2.0 * _machine.Synchronization.GridCycles);
    }

    private double EstimateComputeCycles(
        PagedAttentionExecutionPlanQuery query,
        long contextLength)
    {
        var batch = query.QuerySequenceLength * (double)query.LocalKVHeads;
        var queryHeadsPerKVHead = MathUtility.CeilDiv(
            query.LocalQueryHeads,
            query.LocalKVHeads);
        var matrixCandidates = _machine.Compute.MatrixPrimitives
            .Where(primitive => primitive.Supports(query.QueryElementType, query.KVElementType))
            .Select(primitive =>
                EstimateMatrixCycles(
                    primitive,
                    queryHeadsPerKVHead,
                    contextLength,
                    query.HeadDimension,
                    batch) +
                EstimateMatrixCycles(
                    primitive,
                    queryHeadsPerKVHead,
                    query.HeadDimension,
                    contextLength,
                    batch))
            .ToArray();
        var dotCycles = matrixCandidates.Length > 0
            ? matrixCandidates.Min()
            : query.QuerySequenceLength * (double)query.LocalQueryHeads * contextLength *
                2.0 * query.HeadDimension / Math.Max(1.0, _machine.Compute.SimtFmaPerCycle);
        var softmaxCycles = query.QuerySequenceLength * (double)query.LocalQueryHeads *
            contextLength * 8.0 / Math.Max(1.0, _machine.Compute.ElementwiseElementsPerCycle);
        return dotCycles + softmaxCycles;
    }

    private double EstimateMatrixCycles(
        MatrixComputePrimitiveSpec primitive,
        long m,
        long n,
        long k,
        double batch)
    {
        var accumulatorChains = Math.Max(
            1.0,
            MathUtility.CeilDiv(m, primitive.M) *
            (double)MathUtility.CeilDiv(n, primitive.N) * batch);
        var dependentInstructions = MathUtility.CeilDiv(k, primitive.K);
        return MatrixComputeCostModel.EstimateCycles(
            primitive,
            accumulatorChains,
            dependentInstructions,
            _machine.Execution);
    }

    private double EstimateMemoryCycles(
        double loadBytes,
        double storeBytes,
        double activeBlocks,
        double physicalBlocks)
    {
        var bandwidthSaturatingBlocks = Math.Min(
            physicalBlocks,
            _machine.Execution.ComputeUnitCount);
        var activeFraction = Math.Min(
            1.0,
            activeBlocks / Math.Max(1.0, bandwidthSaturatingBlocks));
        var readBytesPerCycle = _rootMemory.ReadBytesPerCycle * activeFraction;
        var writeBytesPerCycle = _rootMemory.WriteBytesPerCycle * activeFraction;
        return (loadBytes / readBytesPerCycle) +
            (storeBytes / writeBytesPerCycle) +
            ((loadBytes + storeBytes) > 0 ? _rootMemory.LatencyCycles : 0);
    }

    private static IEnumerable<int> GetSplitCounts(int hierarchyExtent)
    {
        for (int splitCount = 2; splitCount <= hierarchyExtent; splitCount++)
        {
            yield return splitCount;
        }
    }
}
