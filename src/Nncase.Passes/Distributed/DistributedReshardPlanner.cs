// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;

namespace Nncase.Passes.Distributed;

internal sealed record DistributedReshardPlan(IReadOnlyList<IRType> StepTypes);

internal static class DistributedReshardPlanner
{
    public const int DefaultMaxHops = 3;

    public static IReadOnlyList<DistributedReshardPlan> Plan(
        IRType sourceType,
        IRType targetType,
        Func<IRType, IRType, bool> canBox,
        int maxHops = DefaultMaxHops)
    {
        if (maxHops < 1)
        {
            return Array.Empty<DistributedReshardPlan>();
        }

        var plans = new List<DistributedReshardPlan>();
        var seen = new HashSet<string>();
        AddPlanIfValid(sourceType, new[] { targetType }, canBox, maxHops, plans, seen);
        if (plans.Count > 0)
        {
            return plans;
        }

        if (maxHops == 1
            || sourceType is not DistributedType source
            || targetType is not DistributedType target
            || source.TensorType != target.TensorType
            || source.Placement != target.Placement)
        {
            return plans;
        }

        var sourceNoPartial = source.Partial is null
            ? source
            : new DistributedType(source.TensorType, source.AxisPolicies, source.Placement);
        var broadcast = CreateBroadcastType(sourceNoPartial);

        AddPlanIfValid(sourceType, new IRType[] { sourceNoPartial, targetType }, canBox, maxHops, plans, seen);
        AddPlanIfValid(sourceType, new IRType[] { broadcast, targetType }, canBox, maxHops, plans, seen);
        AddPlanIfValid(sourceType, new IRType[] { sourceNoPartial, broadcast, targetType }, canBox, maxHops, plans, seen);

        return plans;
    }

    private static DistributedType CreateBroadcastType(DistributedType type)
    {
        var policies = Enumerable.Repeat<SBP>(SBP.B, type.AxisPolicies.Count).ToArray();
        return new DistributedType(type.TensorType, policies, type.Placement);
    }

    private static void AddPlanIfValid(
        IRType sourceType,
        IReadOnlyList<IRType> candidateSteps,
        Func<IRType, IRType, bool> canBox,
        int maxHops,
        List<DistributedReshardPlan> plans,
        HashSet<string> seen)
    {
        var steps = NormalizeSteps(sourceType, candidateSteps);
        if (steps.Count == 0 || steps.Count > maxHops || !IsValidPath(sourceType, steps, canBox))
        {
            return;
        }

        var key = string.Join(" -> ", steps.Select(static type => type.ToString()));
        if (seen.Add(key))
        {
            plans.Add(new DistributedReshardPlan(steps));
        }
    }

    private static IReadOnlyList<IRType> NormalizeSteps(IRType sourceType, IReadOnlyList<IRType> candidateSteps)
    {
        var steps = new List<IRType>(candidateSteps.Count);
        var previous = sourceType;
        foreach (var step in candidateSteps)
        {
            if (SameType(previous, step))
            {
                continue;
            }

            steps.Add(step);
            previous = step;
        }

        return steps;
    }

    private static bool IsValidPath(IRType sourceType, IReadOnlyList<IRType> steps, Func<IRType, IRType, bool> canBox)
    {
        var previous = sourceType;
        foreach (var step in steps)
        {
            if (SameType(previous, step) || !canBox(previous, step))
            {
                return false;
            }

            previous = step;
        }

        return true;
    }

    private static bool SameType(IRType lhs, IRType rhs) => EqualityComparer<IRType>.Default.Equals(lhs, rhs);
}
