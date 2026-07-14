// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using QuikGraph;
using QuikGraph.Algorithms;

namespace Nncase.Schedule.TileGraph;

/// <summary>
/// Builds the derived hierarchy for one immutable structural schedule.
/// </summary>
internal static class TileScheduleBuilder
{
    public static bool TryBuild(
        TileRegion region,
        TileStructuralSchedule schedule,
        int levelCount,
        out TieredTileGraph scheduleGraph,
        out string failure)
    {
        scheduleGraph = region.GetBaseGraph().Clone(out _);
        failure = string.Empty;
        foreach (var (scope, loopOrder) in schedule.LoopOrders)
        {
            var matches = new List<TieredTileGraph>();
            scheduleGraph.Walk(node =>
            {
                if (node is TieredTileGraph graph && graph.OpId == scope.AnchorOpId && graph.Level == scope.Level)
                {
                    matches.Add(graph);
                }
            });
            if (matches.Count != 1)
            {
                failure = $"Structural scope {scope} resolves to {matches.Count} scopes before composition.";
                return false;
            }

            matches[0].SetLoopOrder(loopOrder);
        }

        for (int level = levelCount - 1; level >= 0; level--)
        {
            foreach (var (use, fusionLevel) in schedule.FusionLevels)
            {
                if (fusionLevel < 0 || fusionLevel > level)
                {
                    continue;
                }

                try
                {
                    if (!TileScheduleComposer.TryApply(scheduleGraph, use, level, out var composeFailure))
                    {
                        failure = $"Fusion {use}@L{fusionLevel} cannot be composed at hierarchy level {level}: {composeFailure}";
                        return false;
                    }
                }
                catch (QuikGraphException ex)
                {
                    failure = $"Fusion {use}@L{fusionLevel} failed at hierarchy level {level} with {ex.GetType().Name}: {ex.Message}";
                    return false;
                }
            }
        }

        return TileExecutionPlanVerifier.TryVerifySchedule(
            region,
            schedule,
            scheduleGraph,
            levelCount,
            out failure);
    }
}

/// <summary>
/// Verifies a structural schedule before it reaches the exact solver.
/// </summary>
internal static class TileExecutionPlanVerifier
{
    public static bool TryVerifySchedule(
        TileRegion region,
        TileStructuralSchedule schedule,
        TieredTileGraph scheduleGraph,
        int levelCount,
        out string failure)
    {
        foreach (var use in region.Uses)
        {
            var level = schedule.GetFusionLevel(use.Id);
            if (level < -1 || level >= levelCount)
            {
                failure = $"Connection {use.Id} selects invalid hierarchy level {level}.";
                return false;
            }

            if (level >= 0 && !use.CanFuseAtLevel(level, levelCount))
            {
                failure = $"Connection {use.Id} requires {use.RequiredMemoryScope} visibility and cannot be fused at hierarchy level {level}.";
                return false;
            }

            if (use.RequiredMemoryScope == MemoryAccessScope.Chip && level != levelCount - 1)
            {
                failure = $"Chip-visible connection {use.Id} must form a sequential phase at the outermost hierarchy level {levelCount - 1}, got {level}.";
                return false;
            }
        }

        if (!scheduleGraph.Condense().IsDirectedAcyclicGraph())
        {
            failure = "The structural schedule creates a non-convex fused region and a cycle in the schedule DAG.";
            return false;
        }

        var sequentialScopes = new List<TieredTileGraph>();
        scheduleGraph.Walk(node =>
        {
            if (node is TieredTileGraph { ScopeKind: TileScopeKind.Sequential } sequence)
            {
                sequentialScopes.Add(sequence);
            }
        });
        foreach (var sequence in sequentialScopes)
        {
            if (sequence.Level != levelCount - 1 ||
                sequence.DomainRelation.Map.Domains.Length != 0 ||
                sequence.DomainRelation.Map.Results.Length != 0 ||
                sequence.DomainBoundExprs.Length != 0 ||
                sequence.LoopOrder.Length != 0)
            {
                failure = $"Sequential scope {sequence} must be a zero-dimensional outermost scope at L{levelCount - 1}.";
                return false;
            }

            var phases = sequence.Clusters.OfType<TieredTileGraph>().ToArray();
            if (phases.Length < 2 || phases.Any(phase =>
                    phase.Level != sequence.Level ||
                    phase.ScopeKind != TileScopeKind.Iteration))
            {
                failure = $"Sequential scope {sequence} must contain at least two independent iteration scopes at L{sequence.Level}.";
                return false;
            }

            var phaseEdges = scheduleGraph.Edges.Where(edge =>
            {
                var sourcePhase = phases.SingleOrDefault(phase => phase.ContainsVertex(edge.Source));
                var targetPhase = phases.SingleOrDefault(phase => phase.ContainsVertex(edge.Target));
                return sourcePhase is not null &&
                    targetPhase is not null &&
                    !ReferenceEquals(sourcePhase, targetPhase);
            }).ToArray();
            if (phaseEdges.Length == 0 || phaseEdges.Any(edge =>
                    GraphExtensions.GetRequiredFusionScope(edge.Source, edge.Target, edge.Tag) != MemoryAccessScope.Chip))
            {
                failure = $"Sequential scope {sequence} may contain only chip-visible dependencies between child phases.";
                return false;
            }
        }

        foreach (var view in region.GetBaseGraph().Vertices.Where(vertex => vertex.IsPureBufferView))
        {
            var requiredVisibility = view.Attribute.HasFlag(TileGridAttribute.LiveOut)
                ? levelCount
                : region.Uses
                    .Where(use => use.Id.ProducerOpId == view.RegionOpId)
                    .Select(use => GetStorageVisibility(use, schedule, levelCount))
                    .DefaultIfEmpty(0)
                    .Max();
            foreach (var sourceUse in region.Uses.Where(use => use.Id.ConsumerOpId == view.RegionOpId))
            {
                var sourceVisibility = GetStorageVisibility(sourceUse, schedule, levelCount);
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

    private static int GetStorageVisibility(
        TileUse use,
        TileStructuralSchedule schedule,
        int levelCount)
    {
        // Chip-visible data is caller/root materialized even though its
        // producer and consumer execute as adjacent phases in the outermost
        // block scope. Structural fusion and storage visibility are separate
        // decisions.
        if (use.RequiredMemoryScope == MemoryAccessScope.Chip)
        {
            return levelCount;
        }

        var connectionLevel = schedule.GetFusionLevel(use.Id);
        return connectionLevel < 0 ? levelCount : connectionLevel;
    }
}
