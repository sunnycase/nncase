// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Nncase.IR;
using QuikGraph;

namespace Nncase.Schedule.TileGraph;

/// <summary>
/// Unified hierarchical AutoTiling planner. Structural search owns fusion and
/// lexical order; the exact solver owns tile sizes and materialization.
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

        var search = new TileStructuralSearch(
            region,
            levelCount,
            schedule => TrySolve(region, schedule, levelCount, moduleKind, targetOptions, dynamicDimVars));
        var result = search.Run();
        DumpPlan(region, result, targetOptions);
        return result;
    }

    private static void DumpPlan(
        TileRegion region,
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
        writer.WriteLine("fusion:");
        foreach (var (use, fusionLevel) in result.Structure.FusionLevels)
        {
            var regionUse = region.Uses.Single(candidate => candidate.Id == use);
            writer.WriteLine($"  - use: {use}");
            writer.WriteLine($"    level: {fusionLevel}");
            writer.WriteLine($"    required_memory_scope: {regionUse.RequiredMemoryScope}");
            writer.WriteLine($"    alias_view: {(regionUse.IsAliasView ? "true" : "false")}");
            writer.WriteLine($"    maximum_bytes: {regionUse.MaximumBytes}");
        }

        writer.WriteLine("loop_orders:");
        foreach (var (scope, order) in result.Structure.LoopOrders)
        {
            writer.WriteLine($"  {scope}: [{string.Join(", ", order.Select(axis => $"d{axis}"))}]");
        }

        writer.WriteLine("materializations:");
        foreach (var materialization in result.Materializations)
        {
            writer.WriteLine($"  - value: {materialization.Value}");
            writer.WriteLine($"    creation_scope: {materialization.CreationScope}");
            writer.WriteLine($"    loop_entry: {materialization.LoopEntry}");
            writer.WriteLine($"    storage: {materialization switch
            {
                TileStorageMaterialization storage => storage.StorageSpace.ToString(),
                TileAliasMaterialization => "alias",
                TileRootMaterialization root => $"root:{root.RequiredMemoryScope}",
                _ => throw new InvalidOperationException($"Unknown tile materialization {materialization.GetType().Name}."),
            }}");
            writer.WriteLine($"    uses: [{string.Join(", ", materialization.Uses)}]");
        }
    }

    private (TileExecutionPlan? Plan, string Failure) TrySolve(
        TileRegion region,
        TileStructuralSchedule schedule,
        int levelCount,
        string moduleKind,
        INTTTargetOptions targetOptions,
        DimVar[] dynamicDimVars)
    {
        if (!TileScheduleBuilder.TryBuild(region, schedule, levelCount, out var graph, out var failure))
        {
            return (null, failure);
        }

        try
        {
            graph.PruneDeadBufferViews();
            var (argumentMemo, objectValue, materializations) = _graphTiler.SolveRootGraph(graph, moduleKind, targetOptions, dynamicDimVars);
            var knownUses = region.Uses.Select(use => use.Id).ToHashSet();
            foreach (var materialization in materializations)
            {
                foreach (var use in materialization.Uses)
                {
                    if (!knownUses.Contains(use))
                    {
                        throw new InvalidOperationException(
                            $"Tiling solver selected materialization {materialization.Value} for unknown region use {use}.");
                    }
                }
            }

            var materializedUses = materializations.SelectMany(materialization => materialization.Uses).ToHashSet();
            foreach (var (use, fusionLevel) in schedule.FusionLevels.Where(item => item.Value >= 0))
            {
                if (!materializedUses.Contains(use))
                {
                    throw new InvalidOperationException(
                        $"Structural fusion {use}@L{fusionLevel} has no exact-solver materialization or alias placement.");
                }
            }

            return (new TileExecutionPlan(
                schedule,
                materializations.ToImmutableArray(),
                graph,
                argumentMemo,
                objectValue), string.Empty);
        }
        catch (Exception ex) when (ex is SolveFailedException or QuikGraphException)
        {
            return (null, ex.Message);
        }
    }
}
