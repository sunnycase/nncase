// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Passes.Distributed;
using Nncase.Utilities;

namespace Nncase.Targets;

/// <summary>
/// Selects copy or alias realizations for PyNTT's persistent UMA execution model.
/// </summary>
public sealed class PyNTTDistributedReshardRealizationPolicy : IDistributedReshardRealizationPolicy
{
    private PyNTTDistributedReshardRealizationPolicy()
    {
    }

    /// <summary>
    /// Gets the shared policy instance.
    /// </summary>
    public static PyNTTDistributedReshardRealizationPolicy Instance { get; } = new();

    /// <inheritdoc/>
    public bool UsesShardedViewsForConstants(INTTTargetOptions targetOptions)
        => NTTDistributedReshardRealizationPolicy.Instance.UsesShardedViewsForConstants(targetOptions);

    /// <inheritdoc/>
    public DistributedReshardRealization Classify(DistributedReshardRealizationContext context)
    {
        if (!context.TargetOptions.UnifiedMemoryArch ||
            context.TargetOptions.MemoryAccessArch != MemoryAccessArchitecture.UMA)
        {
            return DistributedReshardRealization.Boxing;
        }

        if (context.SourceType is DistributedType { Partial: not null } partialSource &&
            context.TargetType is DistributedType { Partial: null } partialTarget &&
            context.SourceKind == DistributedReshardSourceKind.Internal &&
            context.UsageKind == DistributedReshardUsageKind.Internal &&
            DistributedReshardDecomposition
                .GetPartialReduceScatterIntermediates(partialSource, partialTarget)
                .Any(intermediate => CanMaterializeCanonicalChipView(intermediate, partialTarget)))
        {
            // PyNTT realizes an internal all-reduce as reduce-scatter into
            // canonical UMA storage followed by a zero-copy sharded view.
            return DistributedReshardRealization.Unsupported;
        }

        if (context.TargetType is not DistributedType targetType ||
            !DistributedUtility.TryValidateShardedView(context.SourceType, targetType, out _))
        {
            return DistributedReshardRealization.Boxing;
        }

        if (NTTDistributedReshardRealizationPolicy.CanAliasConstant(context))
        {
            return DistributedReshardRealization.ShardedView;
        }

        if (context.UsageKind is not (DistributedReshardUsageKind.Internal or DistributedReshardUsageKind.ProgramOutput) ||
            context.SourceType is not DistributedType sourceType)
        {
            return DistributedReshardRealization.Boxing;
        }

        if (CanAliasLocalShardSubviewWithinChip(sourceType, targetType))
        {
            // Narrowing a caller-owned or internal local region only changes the
            // descriptor seen by this block; it neither moves data nor widens
            // visibility to another block.
            return DistributedReshardRealization.ShardedView;
        }

        if (context.SourceKind != DistributedReshardSourceKind.Internal)
        {
            return DistributedReshardRealization.Boxing;
        }

        return CanMaterializeCanonicalChipView(sourceType, targetType)
            ? DistributedReshardRealization.ShardedView
            : DistributedReshardRealization.Boxing;
    }

    private static bool CanAliasLocalShardSubviewWithinChip(
        DistributedType sourceType,
        DistributedType targetType)
    {
        if (!DistributedUtility.IsLocalShardSubview(sourceType, targetType) ||
            !TryGetPlacementAxisOwners(sourceType, out var sourceOwners) ||
            !TryGetPlacementAxisOwners(targetType, out var targetOwners))
        {
            return false;
        }

        for (var axis = 0; axis < sourceType.Placement.Rank; axis++)
        {
            if (sourceType.Placement.Hierarchy[axis] > 1 &&
                !sourceType.Placement.IsPhysicalBlockAxis(axis) &&
                sourceOwners[axis] != targetOwners[axis])
            {
                return false;
            }
        }

        return true;
    }

    private static bool CanMaterializeCanonicalChipView(
        DistributedType sourceType,
        DistributedType targetType)
    {
        if (!TryGetPlacementAxisOwners(sourceType, out var sourceOwners) ||
            !TryGetPlacementAxisOwners(targetType, out var targetOwners))
        {
            return false;
        }

        for (var axis = 0; axis < sourceType.Placement.Rank; axis++)
        {
            if (sourceType.Placement.Hierarchy[axis] <= 1)
            {
                continue;
            }

            if (sourceType.Placement.IsPhysicalBlockAxis(axis))
            {
                // A split owner writes a disjoint region of the canonical backing.
                // No owner means this block axis is Broadcast: its replicas have
                // identical values and intentionally perform idempotent writes to
                // the same canonical region. A chip barrier makes those writes
                // visible before a wider view is consumed.
                continue;
            }

            if (sourceOwners[axis] != targetOwners[axis])
            {
                return false;
            }
        }

        return true;
    }

    private static bool TryGetPlacementAxisOwners(DistributedType type, out int[] owners)
    {
        owners = Enumerable.Repeat(-1, type.Placement.Rank).ToArray();
        for (var tensorAxis = 0; tensorAxis < type.AxisPolicies.Count; tensorAxis++)
        {
            if (type.AxisPolicies[tensorAxis] is not SBPSplit split)
            {
                continue;
            }

            foreach (var placementAxis in split.HierarchyAxes)
            {
                if ((uint)placementAxis >= (uint)owners.Length ||
                    owners[placementAxis] >= 0)
                {
                    return false;
                }

                owners[placementAxis] = tensorAxis;
            }
        }

        return true;
    }
}
