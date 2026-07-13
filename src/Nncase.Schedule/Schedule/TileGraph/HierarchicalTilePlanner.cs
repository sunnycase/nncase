// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Nncase.IR;
using QuikGraph;
using QuikGraph.Algorithms;

namespace Nncase.Schedule.TileGraph;

/// <summary>
/// Builds the derived schedule hierarchy for one immutable connection plan.
/// </summary>
internal static class TileScheduleBuilder
{
    public static bool TryBuild(
        TileRegion region,
        TileConnectionPlan plan,
        int levelCount,
        out TieredTileGraph scheduleGraph,
        out string failure)
    {
        scheduleGraph = region.BaseGraph.Clone(out _);
        failure = string.Empty;
        for (int level = levelCount - 1; level >= 0; level--)
        {
            foreach (var (use, connectionLevel) in plan.Connections)
            {
                if (connectionLevel < 0 || connectionLevel > level)
                {
                    continue;
                }

                try
                {
                    if (!TileScheduleComposer.TryApply(scheduleGraph, use, level, out var composeFailure))
                    {
                        failure = $"Connection {use}@L{connectionLevel} cannot be composed at hierarchy level {level}: {composeFailure}";
                        return false;
                    }
                }
                catch (QuikGraphException ex)
                {
                    failure = $"Connection {use}@L{connectionLevel} failed at hierarchy level {level} with {ex.GetType().Name}: {ex.Message}";
                    return false;
                }
            }
        }

        return TileExecutionPlanVerifier.TryVerifySchedule(
            region,
            plan,
            scheduleGraph,
            levelCount,
            out failure);
    }
}

/// <summary>
/// Verifies invariants of a materialized hierarchical tile schedule before it
/// reaches the constraint solver or lowering.
/// </summary>
internal static class TileExecutionPlanVerifier
{
    public static bool TryVerifySchedule(
        TileRegion region,
        TileConnectionPlan plan,
        TieredTileGraph scheduleGraph,
        int levelCount,
        out string failure)
    {
        foreach (var use in region.Uses)
        {
            var level = plan.GetLevel(use.Id);
            if (level < -1 || level >= levelCount)
            {
                failure = $"Connection {use.Id} selects invalid hierarchy level {level}.";
                return false;
            }

            if (level >= 0 && !use.IsLocallyConnectable)
            {
                failure = $"Connection {use.Id} crosses a required chip-visible materialization boundary.";
                return false;
            }
        }

        if (!scheduleGraph.Condense().IsDirectedAcyclicGraph())
        {
            failure = "The connection plan creates a non-convex fused region and a cycle in the schedule DAG.";
            return false;
        }

        foreach (var view in region.BaseGraph.Vertices.Where(vertex => vertex.IsPureBufferView))
        {
            var requiredVisibility = view.Attribute.HasFlag(TileGridAttribute.LiveOut)
                ? levelCount
                : region.Uses
                    .Where(use => use.Id.ProducerOpId == view.RegionOpId)
                    .Select(use => ToVisibilityLevel(plan.GetLevel(use.Id), levelCount))
                    .DefaultIfEmpty(0)
                    .Max();
            foreach (var sourceUse in region.Uses.Where(use => use.Id.ConsumerOpId == view.RegionOpId))
            {
                var sourceVisibility = ToVisibilityLevel(plan.GetLevel(sourceUse.Id), levelCount);
                if (sourceVisibility < requiredVisibility)
                {
                    failure = $"Buffer view Op{view.RegionOpId} requires backing storage visible at hierarchy level {requiredVisibility}, " +
                        $"but source connection {sourceUse.Id} is materialized at level {sourceVisibility}.";
                    return false;
                }
            }
        }

        failure = string.Empty;
        return true;
    }

    private static int ToVisibilityLevel(int connectionLevel, int levelCount)
        => connectionLevel < 0 ? levelCount : connectionLevel;
}

/// <summary>
/// Unified hierarchical AutoTiling planner. The maximal region is fixed; local
/// storage connections and tile schedules are optimized together through exact
/// component solves.
/// </summary>
public sealed class HierarchicalTilePlanner
{
    private readonly GraphTiler _graphTiler;

    public HierarchicalTilePlanner(GraphTiler graphTiler)
    {
        _graphTiler = graphTiler;
    }

    public TileExecutionPlan Plan(
        TileRegion region,
        string moduleKind,
        INTTTargetOptions targetOptions,
        DimVar[] dynamicDimVars)
    {
        var levelCount = targetOptions.TargetMachineModel.TilingMemorySpaces.Length;
        if (levelCount == 0)
        {
            throw new InvalidOperationException($"Target {targetOptions.TargetMachineModel.Id} exposes no AutoTiling storage levels.");
        }

        var rootPlan = TileConnectionPlan.CreateRoot(region);
        var maximalPlan = BuildMaximalLevelPlan(region, rootPlan, 0, levelCount);
        if (TrySolve(
            region,
            maximalPlan,
            levelCount,
            moduleKind,
            targetOptions,
            dynamicDimVars,
            out var result,
            out var failure))
        {
            DumpPlan(region, result.Connections, result, targetOptions);
            return result;
        }

        DumpRejectedSolverPlan(maximalPlan, failure);
        if (!TrySolve(
            region,
            rootPlan,
            levelCount,
            moduleKind,
            targetOptions,
            dynamicDimVars,
            out result,
            out var rootFailure))
        {
            throw new SolveFailedException(
                $"Hierarchical AutoTiling cannot solve even the fully cut root plan: {rootFailure}. Maximal-plan failure: {failure}");
        }

        var acceptedPlan = rootPlan;
        var promotableUses = region.Uses
            .Where(use => maximalPlan.GetLevel(use.Id) >= 0)
            .OrderByDescending(use => use.MaximumBytes)
            .ThenBy(use => use.Id.ProducerOpId)
            .ThenBy(use => use.Id.ConsumerOpId)
            .ThenBy(use => use.Id.ConsumerAccessIndex)
            .ToArray();

        void PromoteChunk(ReadOnlySpan<TileUse> uses)
        {
            if (uses.IsEmpty)
            {
                return;
            }

            var candidatePlan = acceptedPlan;
            foreach (var use in uses)
            {
                candidatePlan = candidatePlan.WithLevel(use.Id, 0);
            }

            if (candidatePlan.Equals(acceptedPlan))
            {
                return;
            }

            if (TrySolve(
                region,
                candidatePlan,
                levelCount,
                moduleKind,
                targetOptions,
                dynamicDimVars,
                out var candidateResult,
                out _))
            {
                acceptedPlan = candidatePlan;
                result = candidateResult;
                return;
            }

            if (uses.Length == 1)
            {
                return;
            }

            var middle = uses.Length / 2;
            PromoteChunk(uses[..middle]);
            PromoteChunk(uses[middle..]);
        }

        PromoteChunk(promotableUses);
        var changed = true;
        while (changed)
        {
            changed = false;
            var remainingUses = promotableUses
                .Where(use => acceptedPlan.GetLevel(use.Id) < 0)
                .ToArray();
            if (remainingUses.Length == 0)
            {
                break;
            }

            var candidatePlan = remainingUses.Aggregate(
                acceptedPlan,
                (plan, use) => plan.WithLevel(use.Id, 0));
            if (TrySolve(
                region,
                candidatePlan,
                levelCount,
                moduleKind,
                targetOptions,
                dynamicDimVars,
                out var candidateResult,
                out _))
            {
                acceptedPlan = candidatePlan;
                result = candidateResult;
                break;
            }

            foreach (var use in remainingUses)
            {
                candidatePlan = acceptedPlan.WithLevel(use.Id, 0);
                if (!TrySolve(
                    region,
                    candidatePlan,
                    levelCount,
                    moduleKind,
                    targetOptions,
                    dynamicDimVars,
                    out candidateResult,
                    out _))
                {
                    continue;
                }

                acceptedPlan = candidatePlan;
                result = candidateResult;
                changed = true;
            }
        }

        DumpPlan(region, result.Connections, result, targetOptions);
        return result;
    }

    private static TileConnectionPlan BuildMaximalLevelPlan(
        TileRegion region,
        TileConnectionPlan basePlan,
        int level,
        int levelCount)
    {
        var promotable = region.Uses
            .Where(use =>
            {
                var currentLevel = basePlan.GetLevel(use.Id);
                return use.IsLocallyConnectable && (currentLevel < 0 || currentLevel > level);
            })
            .OrderByDescending(use => use.MaximumBytes)
            .ThenBy(use => use.Id.ProducerOpId)
            .ThenBy(use => use.Id.ConsumerOpId)
            .ThenBy(use => use.Id.ConsumerAccessIndex)
            .ToArray();
        if (promotable.Length == 0)
        {
            return basePlan;
        }

        var completePlan = promotable.Aggregate(basePlan, (plan, use) => plan.Connect(use.Id, level));
        if (TileScheduleBuilder.TryBuild(region, completePlan, levelCount, out _, out var completeFailure))
        {
            return completePlan;
        }

        DumpRejectedFrontier(level, completePlan, completeFailure);

        var maximalPlan = basePlan;
        var changed = true;
        while (changed)
        {
            changed = false;
            foreach (var use in promotable.Where(use => maximalPlan.GetLevel(use.Id) != level))
            {
                var candidate = maximalPlan.Connect(use.Id, level);
                if (TileScheduleBuilder.TryBuild(region, candidate, levelCount, out _, out _))
                {
                    maximalPlan = candidate;
                    changed = true;
                }
            }
        }

        return maximalPlan;
    }

    private static void DumpRejectedFrontier(int level, TileConnectionPlan plan, string failure)
    {
        if (!Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            return;
        }

        using var stream = Diagnostics.DumpScope.Current.OpenFile($"tile_frontier_l{level}_rejected.yaml");
        using var writer = new StreamWriter(stream);
        writer.WriteLine($"level: {level}");
        writer.WriteLine($"failure: {failure}");
        writer.WriteLine($"plan: {plan}");
    }

    private static void DumpRejectedSolverPlan(TileConnectionPlan plan, string failure)
    {
        if (!Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            return;
        }

        using var stream = Diagnostics.DumpScope.Current.OpenFile("tile_maximal_plan_rejected.yaml");
        using var writer = new StreamWriter(stream);
        writer.WriteLine($"failure: {failure}");
        writer.WriteLine($"plan: {plan}");
    }

    private bool TrySolve(
        TileRegion region,
        TileConnectionPlan plan,
        int levelCount,
        string moduleKind,
        INTTTargetOptions targetOptions,
        DimVar[] dynamicDimVars,
        out TileExecutionPlan result,
        out string failure)
    {
        result = null!;
        if (!TileScheduleBuilder.TryBuild(region, plan, levelCount, out var graph, out failure))
        {
            return false;
        }

        try
        {
            graph.PruneDeadBufferViews();
            var solved = _graphTiler.SolveRootGraph(graph, moduleKind, targetOptions, dynamicDimVars);
            var uses = region.Uses.ToDictionary(use => use.Id);
            var placements = region.Uses.ToImmutableDictionary(
                use => use.Id,
                use => use.IsAliasView
                    ? TileUsePlacement.AliasView(plan.GetLevel(use.Id))
                    : TileUsePlacement.RootStorage);
            foreach (var (use, level) in solved.SelectedUseLevels)
            {
                if (!uses.TryGetValue(use, out var regionUse))
                {
                    throw new InvalidOperationException($"Tiling solver selected placement for unknown region use {use}.");
                }

                placements = placements.SetItem(
                    use,
                    regionUse.IsAliasView
                        ? TileUsePlacement.AliasView(level)
                        : TileUsePlacement.LocalStorage(level));
            }

            foreach (var (use, structuralLevel) in plan.Connections.Where(connection => connection.Value >= 0))
            {
                var regionUse = uses[use];
                if (!regionUse.IsAliasView && placements[use].Kind != TileUsePlacementKind.LocalStorage)
                {
                    throw new InvalidOperationException(
                        $"Maximal schedule connection {use}@L{structuralLevel} has no selected storage level.");
                }
            }

            result = new TileExecutionPlan(plan, placements, graph, solved.ArgumentMemo, solved.ObjectValue);
            failure = string.Empty;
            return true;
        }
        catch (Exception ex) when (ex is SolveFailedException or QuikGraphException)
        {
            failure = ex.Message;
            return false;
        }
    }

    private static void DumpPlan(
        TileRegion region,
        TileConnectionPlan plan,
        TileExecutionPlan result,
        INTTTargetOptions targetOptions)
    {
        if (!Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            return;
        }

        using var stream = Diagnostics.DumpScope.Current.OpenFile("tile_execution_plan.yaml");
        using var writer = new StreamWriter(stream);
        writer.WriteLine($"target: {targetOptions.TargetMachineModel.Id}");
        writer.WriteLine($"objective_cycles: {result.ObjectiveValue}");
        writer.WriteLine("connections:");
        foreach (var (use, structuralLevel) in plan.Connections)
        {
            var regionUse = region.Uses.Single(candidate => candidate.Id == use);
            var placement = result.Placements[use];
            var storage = placement.Kind switch
            {
                TileUsePlacementKind.RootStorage => targetOptions.TargetMachineModel
                    .GetMemorySpace(targetOptions.TargetMachineModel.RootMemorySpace).Id.ToString(),
                TileUsePlacementKind.LocalStorage => targetOptions.TargetMachineModel
                    .TilingMemorySpaces[placement.Level].Id.ToString(),
                TileUsePlacementKind.AliasView => "alias",
                _ => throw new InvalidOperationException($"Unsupported tile use placement {placement}."),
            };
            writer.WriteLine($"  - use: {use}");
            writer.WriteLine($"    locally_connectable: {(regionUse.IsLocallyConnectable ? "true" : "false")}");
            writer.WriteLine($"    alias_view: {(regionUse.IsAliasView ? "true" : "false")}");
            writer.WriteLine($"    maximum_bytes: {regionUse.MaximumBytes}");
            writer.WriteLine($"    structural_level: {structuralLevel}");
            writer.WriteLine($"    placement_kind: {placement.Kind}");
            writer.WriteLine($"    placement_level: {placement.Level}");
            writer.WriteLine($"    storage: {storage}");
        }
    }
}
