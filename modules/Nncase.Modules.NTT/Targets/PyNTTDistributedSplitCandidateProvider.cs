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

        long parentMaximumExtent = context.MaximumExtent;
        var stages = new SplitStage[groups.Count];
        for (var index = 0; index < groups.Count; index++)
        {
            var group = groups[index];
            var shardCount = Product(group.HierarchyAxes.Select(axis => context.Placement.Hierarchy[axis]));
            if (group.Level == 'b')
            {
                var maximumUsefulBlockSize = Math.Max(1, parentMaximumExtent / shardCount);
                var preferredElements = Math.Max(1, _blockBytes / context.TensorType.DType.SizeInBytes);
                var blockSize = HighestPowerOfTwoAtMost(Math.Min(maximumUsefulBlockSize, preferredElements));
                stages[index] = SplitStage.BlockCyclic(group.HierarchyAxes, blockSize);
                parentMaximumExtent = MathUtility.CeilDiv(parentMaximumExtent, shardCount * blockSize) * blockSize;
            }
            else
            {
                stages[index] = SplitStage.Contiguous(group.HierarchyAxes);
                parentMaximumExtent = MathUtility.CeilDiv(parentMaximumExtent, shardCount);
            }
        }

        return [SBP.S(stages)];
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

    private static long Product(IEnumerable<int> factors)
    {
        long product = 1;
        foreach (var factor in factors)
        {
            product = checked(product * factor);
        }

        return product;
    }

    private sealed record PhysicalAxisGroup(char Level, IRArray<int> HierarchyAxes);
}
