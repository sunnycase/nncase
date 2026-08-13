// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;

namespace Nncase.Passes.Distributed;

/// <summary>
/// Builds target-independent intermediate types for decomposed distributed
/// reshard operations.
/// </summary>
public static class DistributedReshardDecomposition
{
    /// <summary>
    /// Returns legal reduce-scatter intermediates which discharge
    /// <see cref="DistributedType.Partial"/> into an explicit split before
    /// reaching <paramref name="target"/>.
    /// </summary>
    public static IReadOnlyList<DistributedType> GetPartialReduceScatterIntermediates(
        DistributedType source,
        DistributedType target)
    {
        if (source.Partial is not { } partial ||
            target.Partial is not null ||
            source.TensorType != target.TensorType ||
            source.Placement != target.Placement ||
            !TryValidatePartialTransition(source, target, partial, out var remainingPartialAxes))
        {
            return Array.Empty<DistributedType>();
        }

        if (remainingPartialAxes.Length == 0)
        {
            return Array.Empty<DistributedType>();
        }

        var divisor = remainingPartialAxes.Aggregate(
            1,
            (product, placementAxis) => checked(product * source.Placement.Hierarchy[placementAxis]));
        var candidates = new List<DistributedType>();
        for (var tensorAxis = 0; tensorAxis < source.AxisPolicies.Count; tensorAxis++)
        {
            if (source.AxisPolicies[tensorAxis] is not SBPBroadCast ||
                target.AxisPolicies[tensorAxis] is not SBPBroadCast)
            {
                continue;
            }

            var policies = target.AxisPolicies.ToArray();
            policies[tensorAxis] = SBP.SContiguous(
                remainingPartialAxes,
                Dimension.CeilDiv(target.TensorType.Shape[tensorAxis], divisor));
            candidates.Add(new DistributedType(target.TensorType, policies, target.Placement));
        }

        return candidates;
    }

    private static bool TryValidatePartialTransition(
        DistributedType source,
        DistributedType target,
        SBPPartial partial,
        out int[] remainingPartialAxes)
    {
        remainingPartialAxes = Array.Empty<int>();
        var partialAxes = partial.Axes.ToHashSet();
        if (partialAxes.Count == 0 ||
            partialAxes.Any(axis => axis < 0 || axis >= source.Placement.Rank))
        {
            return false;
        }

        var targetSplitAxes = new HashSet<int>();
        for (var tensorAxis = 0; tensorAxis < source.AxisPolicies.Count; tensorAxis++)
        {
            switch (source.AxisPolicies[tensorAxis], target.AxisPolicies[tensorAxis])
            {
                case (SBPBroadCast, SBPBroadCast):
                    break;
                case (SBPBroadCast, SBPSplit targetSplit)
                    when targetSplit.HierarchyAxes.All(partialAxes.Contains):
                    if (!targetSplitAxes.AddRange(targetSplit.HierarchyAxes))
                    {
                        return false;
                    }

                    break;
                case (SBPSplit sourceSplit, SBPSplit targetSplit)
                    when sourceSplit == targetSplit:
                    if (!targetSplitAxes.AddRange(targetSplit.HierarchyAxes))
                    {
                        return false;
                    }

                    break;
                default:
                    return false;
            }
        }

        remainingPartialAxes = partial.Axes
            .Where(axis => !targetSplitAxes.Contains(axis))
            .ToArray();
        return true;
    }

    private static bool AddRange(this HashSet<int> destination, IEnumerable<int> values)
    {
        foreach (var value in values)
        {
            if (!destination.Add(value))
            {
                return false;
            }
        }

        return true;
    }
}
