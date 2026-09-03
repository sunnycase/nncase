// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Numerics;
using Nncase.IR;
using Nncase.Utilities;

namespace Nncase.Targets;

/// <summary>
/// Builds PyNTT split candidates. Inter-chip and inter-die stages remain
/// contiguous; physical block axes may use a block-cyclic stage so persistent
/// CTAs visit memory partitions in interleaved blocks.
/// </summary>
public sealed class PyNTTDistributedSplitCandidateProvider : IDistributedSplitCandidateProvider
{
    private readonly long _blockBytes;

    public PyNTTDistributedSplitCandidateProvider(long blockBytes)
    {
        if (blockBytes <= 0 || !BitOperations.IsPow2((ulong)blockBytes))
        {
            throw new ArgumentOutOfRangeException(
                nameof(blockBytes),
                blockBytes,
                "PyNTT block-cyclic byte granularity must be a positive power of two.");
        }

        _blockBytes = blockBytes;
    }

    public IReadOnlyList<SBPSplit> GetCandidates(DistributedSplitCandidateContext context)
    {
        ValidateContext(context);
        var contiguous = SBP.SContiguous(context.HierarchyAxes, context.ContiguousGranularity);
        var groups = GroupByPhysicalLevel(context);
        if (!groups.Any(group => group.Level == 'b'))
        {
            return [contiguous];
        }

        var candidates = new List<SBPSplit>();
        BuildCandidates(context, groups, 0, context.MaximumExtent, new SplitStage[groups.Count], candidates);
        return candidates;
    }

    private static void ValidateContext(DistributedSplitCandidateContext context)
    {
        if (context.HierarchyAxes.Count == 0)
        {
            throw new ArgumentException("A split candidate requires at least one hierarchy axis.", nameof(context));
        }

        if (context.HierarchyAxes.Any(axis => axis < 0 || axis >= context.Placement.Rank))
        {
            throw new ArgumentOutOfRangeException(nameof(context), "A split candidate hierarchy axis is outside its placement.");
        }

        if (context.MaximumExtent <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(context), "A split candidate maximum extent must be positive.");
        }
    }

    private static long HighestPowerOfTwoAtMost(long value)
    {
        if (value <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(value), value, "Power-of-two upper bound must be positive.");
        }

        return 1L << BitOperations.Log2((ulong)value);
    }

    private static List<PhysicalAxisGroup> GroupByPhysicalLevel(DistributedSplitCandidateContext context)
    {
        var levels = context.Placement.NormalizedHierarchyLevels;
        var groups = new List<PhysicalAxisGroup>();
        foreach (var hierarchyAxis in context.HierarchyAxes)
        {
            var level = levels[hierarchyAxis];
            if (groups.Count > 0 && groups[^1].Level == level)
            {
                groups[^1] = groups[^1] with
                {
                    HierarchyAxes = groups[^1].HierarchyAxes.Append(hierarchyAxis).ToArray(),
                };
            }
            else
            {
                groups.Add(new PhysicalAxisGroup(level, [hierarchyAxis]));
            }
        }

        return groups;
    }

    private static long Product(IEnumerable<int> factors)
    {
        long product = 1;
        foreach (var factor in factors)
        {
            product = checked(product * factor);
        }

        return product;
    }

    private void BuildCandidates(
        DistributedSplitCandidateContext context,
        IReadOnlyList<PhysicalAxisGroup> groups,
        int groupIndex,
        long parentMaximumExtent,
        SplitStage[] stages,
        List<SBPSplit> candidates)
    {
        if (groupIndex == groups.Count)
        {
            candidates.Add(SBP.S(stages.ToArray()));
            return;
        }

        var group = groups[groupIndex];
        var shardCount = Product(group.HierarchyAxes.Select(axis => context.Placement.Hierarchy[axis]));
        if (group.Level != 'b')
        {
            stages[groupIndex] = SplitStage.Contiguous(group.HierarchyAxes);
            BuildCandidates(
                context,
                groups,
                groupIndex + 1,
                MathUtility.CeilDiv(parentMaximumExtent, shardCount),
                stages,
                candidates);
            return;
        }

        foreach (var blockSize in GetBlockSizeCandidates(context, parentMaximumExtent, shardCount))
        {
            stages[groupIndex] = SplitStage.BlockCyclic(group.HierarchyAxes, blockSize);
            var localMaximumExtent = MathUtility.CeilDiv(parentMaximumExtent, shardCount * blockSize) * blockSize;
            BuildCandidates(context, groups, groupIndex + 1, localMaximumExtent, stages, candidates);
        }
    }

    private IReadOnlyList<long> GetBlockSizeCandidates(
        DistributedSplitCandidateContext context,
        long parentMaximumExtent,
        long shardCount)
    {
        var maximumUsefulBlockSize = Math.Max(1, parentMaximumExtent / shardCount);
        var preferredElements = Math.Max(1, _blockBytes / context.TensorType.DType.SizeInBytes);
        var preferredBlockSize = HighestPowerOfTwoAtMost(Math.Min(maximumUsefulBlockSize, preferredElements));
        if (parentMaximumExtent % checked(shardCount * preferredBlockSize) == 0)
        {
            return [preferredBlockSize];
        }

        // Retain the target's preferred transaction granule and add only the
        // largest finer granule that makes every shard carry the same number
        // of complete blocks. Other finer choices are dominated by this one.
        for (var blockSize = preferredBlockSize / 2; blockSize >= 1; blockSize /= 2)
        {
            if (parentMaximumExtent % checked(shardCount * blockSize) == 0)
            {
                return [preferredBlockSize, blockSize];
            }
        }

        return [preferredBlockSize];
    }

    private sealed record PhysicalAxisGroup(char Level, IRArray<int> HierarchyAxes);
}
