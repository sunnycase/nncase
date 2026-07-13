// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Google.OrTools.ConstraintSolver;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.Graphs;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Shapes;
using Nncase.Schedule.TileGraph;
using Nncase.TIR;
using QuikGraph;
using QuikGraph.Algorithms;
using QuikGraph.Graphviz;

namespace Nncase.Schedule;

public sealed class GraphTiler
{
    public Dictionary<TileNode, TiledFunc> SolveMemo { get; } = new Dictionary<TileNode, TiledFunc>(new ITreeNodeComparer());

    /// <summary>
    /// a simple cost model.
    /// </summary>
    public static TreeSolveResult SolvePrimGraph(TileNode primTree, Dictionary<TieredTileGraph, BufferGraph> bufferGraphMemo, INTTTargetOptions targetOptions, string moduleKind)
    {
        var machine = targetOptions.TargetMachineModel;
        var tilingMemorySpaces = machine.TilingMemorySpaces;
        var rootMemorySpace = machine.GetMemorySpace(machine.RootMemorySpace);
        var memCapacities = tilingMemorySpaces.Select(machine.GetMaximumUsableAllocationBytes).ToArray();
        var levelCount = tilingMemorySpaces.Length;
        TreeSolverInitializer.Init(primTree, bufferGraphMemo, levelCount, targetOptions, out var solver, out var opNodeMemo, out var tileNodeMemo, out var tileableNodeMemo);
        var primBufferGraph = bufferGraphMemo[primTree.Wrapped];
        var (externalInputs, externalOutputs) = primBufferGraph.GetInputsOutputs(primBufferGraph.Parent as BufferGraph);

        static int GetStoragePosition(BufferIdentity bid, TileNodeBufferInfo<IntExpr> bufferInfo)
            => bid.Access.BindingMode == GridBindingMode.Root ? 0 : bufferInfo.GetLastRelatedPos();

        bool RequiresLocalAllocation(BufferIdentity bid, int level)
        {
            if (bid.IsOutput && bid.Node.TryGetAliasSourceAccess(bid.Index, out _))
            {
                return false;
            }

            if (bid.Node.LocalAccessEffects[bid.Index].Scope == MemoryAccessScope.Chip)
            {
                return false;
            }

            var isExternal = externalInputs.Contains(bid) || externalOutputs.Contains(bid);
            return !isExternal ||
                (bid.Access.BindingMode != GridBindingMode.Root && machine.RequiresExplicitTransfer(level));
        }

        // 0. External views exist at every hierarchy level. Internal def-use
        // connections choose exactly one (memory level, creation position)
        // across the maximal fused hierarchy.
        var eachLevelStoreBufferConstrains = new Dictionary<int, Constraint[]>();
        var internalPlacementCandidates = new Dictionary<BufferIdentity, List<(TileNode Node, TileNodeBufferInfo<IntExpr> Info)>>();
        for (int level = 0; level < levelCount; level++)
        {
            var cons = new List<Constraint>();
            foreach (var (tileNode, nodeInfo) in tileNodeMemo.Where(kv => kv.Key.Level == level))
            {
                foreach (var (bid, bufferInfo) in nodeInfo.BufferInfoMap)
                {
                    if (nodeInfo.DefUseMap.ContainsKey(bid))
                    {
                        if (!internalPlacementCandidates.TryGetValue(bid, out var candidates))
                        {
                            candidates = new();
                            internalPlacementCandidates.Add(bid, candidates);
                        }

                        candidates.Add((tileNode, bufferInfo));
                        for (int ci = 0; ci < bufferInfo.Places.Length; ci++)
                        {
                            for (int sl = 0; sl < bufferInfo.Places[ci].Length; sl++)
                            {
                                if (sl == level)
                                {
                                    continue;
                                }

                                var c = solver.MakeEquality(bufferInfo.Places[ci][sl], 0);
                                c.SetName($"n_internal_store[{tileNode}_{bid}_cl{ci}_sl{sl}]");
                                solver.Add(c);
                                cons.Add(c);
                            }
                        }

                        continue;
                    }

                    var pos = GetStoragePosition(bid, bufferInfo);
                    for (int ci = 0; ci < bufferInfo.Places.Length; ci++)
                    {
                        for (int sl = 0; sl < bufferInfo.Places[ci].Length; sl++)
                        {
                            if (ci == pos && sl == level)
                            {
                                var c = solver.MakeEquality(bufferInfo.Places[ci][sl], 1);
                                c.SetName($"store[{tileNode}_{bid}_cl{ci}_sl{sl}]");
                                solver.Add(c);
                                cons.Add(c);
                            }
                            else
                            {
                                var c = solver.MakeEquality(bufferInfo.Places[ci][sl], 0);
                                c.SetName($"n_store[{tileNode}_{bid}_cl{ci}_sl{sl}]");
                                solver.Add(c);
                                cons.Add(c);
                            }
                        }
                    }
                }
            }

            eachLevelStoreBufferConstrains.Add(level, cons.ToArray());
        }

        foreach (var (bid, candidates) in internalPlacementCandidates)
        {
            var placementVars = candidates
                .SelectMany(candidate => candidate.Info.Places.Select(place => place[candidate.Node.Level]))
                .ToArray();
            if (placementVars.Length == 0)
            {
                throw new InvalidOperationException($"Internal tile buffer {bid} has no hierarchy placement candidates.");
            }

            var constraint = solver.MakeEquality(solver.MakeSum(placementVars), 1);
            constraint.SetName($"one_internal_store[{bid}]");
            solver.Add(constraint);
            var ownerLevel = candidates.Max(candidate => candidate.Node.Level);
            eachLevelStoreBufferConstrains[ownerLevel] = eachLevelStoreBufferConstrains[ownerLevel]
                .Append(constraint)
                .ToArray();
        }

        // 1. tile var constraints
        var tileVarConstraints = new Dictionary<OpNode, Constraint[]>();
        foreach (var opNode in opNodeMemo.Keys)
        {
            var domainInfo = tileableNodeMemo[opNode];
            var constraints = new Constraint[domainInfo.TileVars.Length];
            for (int i = 0; i < domainInfo.TileVars.Length; i++)
            {
                constraints[i] = solver.MakeEquality(domainInfo.ForwardExtents[i], opNode.DomainBounds[i]);
                constraints[i].SetName($"bound[op{opNode.OpId}, d{i}]");
                solver.Add(constraints[i]);
            }

            tileVarConstraints.Add(opNode, constraints);
        }

        // Reduction state is semantically live across L0 reduction tiles, but
        // its physical representation belongs to the backend. Constrain the
        // logical state here without placing it in any TIR memory hierarchy.
        var reductionStateBytes = new Dictionary<OpNode, IntExpr>();
        var reductionStateConstraints = new Dictionary<OpNode, Constraint>();
        foreach (var (opNode, opNodeInfo) in opNodeMemo)
        {
            if (opNode.GetTileWorkload() is not IReductionStateTileWorkload statefulWorkload)
            {
                continue;
            }

            var stateBytes = statefulWorkload.GetReductionStateBytes(
                opNodeInfo.Shapes,
                solver,
                opNode.GetTileWorkloadContext(),
                machine);
            var constraint = solver.MakeLessOrEqual(
                stateBytes,
                machine.Execution.BackendPrivateAccumulatorCapacityBytes);
            constraint.SetName($"backend_private_accumulator_le[op{opNode.OpId}]");
            solver.Add(constraint);
            reductionStateBytes.Add(opNode, stateBytes);
            reductionStateConstraints.Add(opNode, constraint);
        }

        // 5. add the memory schedule constraints, each level has own memory plan schedule.
        // 5.1. sum(place[cl,b,ci,sl]*size[cl,b,ci], sl), sl = [0,toplevel)
        var levelBufferSizes = new Dictionary<int, Dictionary<NodeWithBuffer, IntExpr>>();
        var levelBufferShapes = new Dictionary<int, Dictionary<NodeWithBuffer, IntExpr[]>>();
        var levelBufferLifeness = new Dictionary<int, Dictionary<NodeWithBuffer, Tuple<int, int>>>();
        var levelBufferPlacementInfos = new Dictionary<int, Dictionary<NodeWithBuffer, TileNodeBufferInfo<IntExpr>>>();
        var levelBufferLifenessConstraints = new Dictionary<int, Constraint[]>();
        for (int sl = 0; sl < levelCount; sl++)
        {
            var nodeBufferSizes = levelBufferSizes[sl] = new();
            var nodeBufferLiveness = levelBufferLifeness[sl] = new();
            var nodeBufferShapes = levelBufferShapes[sl] = new();
            var nodeBufferPlacementInfos = levelBufferPlacementInfos[sl] = new();
            var occupancyByTime = new SortedDictionary<int, List<IntExpr>>();

            foreach (var (tileNode, nodeInfo) in tileNodeMemo.Where(kv => kv.Key.Level == sl)) // only consider create and store at same level.
            {
                foreach (var (bid, bufferInfo) in nodeInfo.BufferInfoMap)
                {
                    var nodeBuffer = new NodeWithBuffer(tileNode, bid);
                    var requiresLocalAllocation = RequiresLocalAllocation(bid, sl);
                    var sizeTerms = new List<IntExpr>(bufferInfo.Places.Length);
                    var shapeTerms = Enumerable.Range(0, bufferInfo.Shapes[0].Length)
                        .Select(_ => new List<IntExpr>(bufferInfo.Places.Length))
                        .ToArray();
                    var lifetimeStart = int.MaxValue;
                    var lifetimeEnd = int.MinValue;
                    for (int ci = 0; ci < bufferInfo.Places.Length; ci++)
                    {
                        var place = bufferInfo.Places[ci][sl];
                        var placedSize = solver.MakeProd(place, bufferInfo.Sizes[ci]);
                        sizeTerms.Add(placedSize);
                        for (int axis = 0; axis < shapeTerms.Length; axis++)
                        {
                            shapeTerms[axis].Add(solver.MakeProd(place, bufferInfo.Shapes[ci][axis]));
                        }

                        lifetimeStart = Math.Min(lifetimeStart, bufferInfo.Liveness[ci].Item1);
                        lifetimeEnd = Math.Max(lifetimeEnd, bufferInfo.Liveness[ci].Item2);
                        if (requiresLocalAllocation)
                        {
                            for (int time = bufferInfo.Liveness[ci].Item1; time <= bufferInfo.Liveness[ci].Item2; time++)
                            {
                                if (!occupancyByTime.TryGetValue(time, out var occupancy))
                                {
                                    occupancy = new();
                                    occupancyByTime.Add(time, occupancy);
                                }

                                occupancy.Add(placedSize);
                            }
                        }

                        if (!IsObjectBuffer(nodeBuffer.Id) &&
                            !nodeBuffer.Id.Node.TryGetAliasSourceAccess(nodeBuffer.Id.Index, out _))
                        {
                            var hasPositiveSize = solver.MakeIsGreaterCstVar(bufferInfo.Sizes[ci], 0);
                            solver.Add(solver.MakeLessOrEqual(place, hasPositiveSize));
                        }
                    }

                    nodeBufferSizes[nodeBuffer] = requiresLocalAllocation
                        ? solver.MakeSum(sizeTerms.ToArray())
                        : solver.MakeIntConst(0);
                    nodeBufferShapes[nodeBuffer] = shapeTerms
                        .Select(terms => solver.MakeSum(terms.ToArray()))
                        .ToArray();
                    nodeBufferLiveness[nodeBuffer] = new(lifetimeStart, lifetimeEnd);
                    nodeBufferPlacementInfos[nodeBuffer] = bufferInfo;
                }
            }

            // Add constraints according to liveness.
            if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
            {
                DumpGantt(nodeBufferSizes, nodeBufferLiveness, primTree, sl);
            }

            var constraints = new List<Constraint>();
            foreach (var (time, occupancy) in occupancyByTime)
            {
                var totalSize = occupancy.Count == 0
                    ? solver.MakeIntConst(0)
                    : solver.MakeSum(occupancy.ToArray());
                var constraint = solver.MakeLessOrEqual(totalSize, memCapacities[sl]);
                constraint.SetName($"capacity_le[sl{sl}, t{time}]");
                solver.Add(constraint);
                constraints.Add(constraint);
            }

            levelBufferLifenessConstraints.Add(sl, constraints.ToArray());
        }

        // when buffer is read, the data read from last level memory.
        // when buffer is write, the data write to current level memory.
        var memoryLevelCount = levelCount + 1;
        var levelDataReads = Enumerable.Range(0, memoryLevelCount).Select(i => (IntExpr)solver.MakeIntConst(0)).ToArray();
        var levelDataWrites = Enumerable.Range(0, memoryLevelCount).Select(i => (IntExpr)solver.MakeIntConst(0)).ToArray();
        var levelTransferReads = Enumerable.Range(0, levelCount).Select(i => (IntExpr)solver.MakeIntConst(0)).ToArray();
        var levelTransferWrites = Enumerable.Range(0, levelCount).Select(i => (IntExpr)solver.MakeIntConst(0)).ToArray();
        IntExpr synchronizationCycles = solver.MakeIntConst(0);
        foreach (var (tileNode, nodeInfo) in tileNodeMemo)
        {
            var nodeWrites = Enumerable.Range(0, memoryLevelCount).Select(_ => new List<IntExpr>()).ToArray();
            var nodeReads = Enumerable.Range(0, memoryLevelCount).Select(_ => new List<IntExpr>()).ToArray();
            var nodeTransferReads = Enumerable.Range(0, levelCount).Select(_ => new List<IntExpr>()).ToArray();
            var nodeTransferWrites = Enumerable.Range(0, levelCount).Select(_ => new List<IntExpr>()).ToArray();
            foreach (var (bid, bufferInfo) in nodeInfo.BufferInfoMap)
            {
                var reused = nodeInfo.DefUseMap.ContainsKey(bid);
                var localEffect = bid.Node.LocalAccessEffects[bid.Index];
                for (int sl = 0; sl <= tileNode.Level; sl++)
                {
                    var requiresLocalAllocation = RequiresLocalAllocation(bid, sl);
                    for (int ci = 0; ci < bufferInfo.Places.Length; ci++)
                    {
                        var volume = bufferInfo.Places[ci][sl] * bufferInfo.Trips[ci] * bufferInfo.Sizes[ci];
                        if (localEffect.Mode.HasFlag(MemoryAccessMode.Read))
                        {
                            if (requiresLocalAllocation)
                            {
                                nodeWrites[sl].Add(volume);
                                nodeReads[sl].Add(volume);
                            }

                            if (!reused)
                            {
                                nodeReads[sl + 1].Add(volume);
                                if (requiresLocalAllocation)
                                {
                                    nodeTransferReads[sl].Add(volume);
                                }
                            }
                        }

                        if (localEffect.Mode.HasFlag(MemoryAccessMode.Write))
                        {
                            if (requiresLocalAllocation)
                            {
                                nodeWrites[sl].Add(volume);
                                if (reused)
                                {
                                    nodeReads[sl].Add(volume);
                                }
                            }

                            if (!reused)
                            {
                                nodeWrites[sl + 1].Add(volume);
                                if (requiresLocalAllocation)
                                {
                                    nodeTransferWrites[sl].Add(volume);
                                }
                            }
                        }

                        if (reused && tilingMemorySpaces[sl].RequiresExplicitSynchronization)
                        {
                            synchronizationCycles += bufferInfo.Places[ci][sl]
                                * bufferInfo.Trips[ci]
                                * machine.Synchronization.BlockCycles;
                        }
                    }
                }
            }

            for (int l = 0; l < memoryLevelCount; l++)
            {
                if (nodeWrites[l].Any())
                {
                    levelDataWrites[l] = levelDataWrites[l] + solver.MakeSum(nodeWrites[l]);
                }

                if (nodeReads[l].Any())
                {
                    levelDataReads[l] = levelDataReads[l] + solver.MakeSum(nodeReads[l]);
                }
            }

            for (int l = 0; l < levelCount; l++)
            {
                if (nodeTransferReads[l].Any())
                {
                    levelTransferReads[l] += solver.MakeSum(nodeTransferReads[l]);
                }

                if (nodeTransferWrites[l].Any())
                {
                    levelTransferWrites[l] += solver.MakeSum(nodeTransferWrites[l]);
                }
            }
        }

        var activeBlockCount = GetActiveBlockCount(targetOptions);
        var storageSpaces = tilingMemorySpaces.Append(rootMemorySpace).ToArray();
        var memoryResourceGroups = storageSpaces
            .Select((space, level) => (space, level))
            .GroupBy(item => item.space.ResourceId)
            .OrderBy(group => group.Min(item => item.level))
            .ToArray();
        var memoryCycles = new IntExpr[memoryResourceGroups.Length];
        for (int i = 0; i < memoryResourceGroups.Length; i++)
        {
            var group = memoryResourceGroups[i];
            var resource = machine.GetMemoryResource(group.Key);
            var reads = solver.MakeSum(group.Select(item => levelDataReads[item.level]).ToArray());
            var writes = solver.MakeSum(group.Select(item => levelDataWrites[item.level]).ToArray());
            var contentionFactor = group.Any(item => item.space.Scope == MemorySharingScope.Chip)
                ? activeBlockCount
                : 1;
            var latency = solver.MakeIsGreaterCstVar(reads + writes, 0) * resource.LatencyCycles;
            memoryCycles[i] = reads.ScaleAndCeilDiv(contentionFactor, resource.ReadBytesPerCycle)
                + writes.ScaleAndCeilDiv(contentionFactor, resource.WriteBytesPerCycle)
                + latency;
        }

        var transferCycles = new IntExpr[levelCount];
        for (int i = 0; i < levelCount; i++)
        {
            var localMemorySpace = tilingMemorySpaces[i];
            var parentMemorySpace = i + 1 < levelCount
                ? tilingMemorySpaces[i + 1]
                : rootMemorySpace;
            var readTransfer = machine.GetTransfer(parentMemorySpace.Id, localMemorySpace.Id);
            var writeTransfer = machine.GetTransfer(localMemorySpace.Id, parentMemorySpace.Id);
            var reads = levelTransferReads[i];
            var writes = levelTransferWrites[i];
            if (parentMemorySpace.ResourceId == localMemorySpace.ResourceId)
            {
                transferCycles[i] = solver.MakeIntConst(0);
                continue;
            }

            if (parentMemorySpace.Scope == MemorySharingScope.Chip || localMemorySpace.Scope == MemorySharingScope.Chip)
            {
                reads *= activeBlockCount;
                writes *= activeBlockCount;
            }

            var readEvent = solver.MakeIsGreaterCstVar(reads, 0);
            var writeEvent = solver.MakeIsGreaterCstVar(writes, 0);
            transferCycles[i] = reads.CeilDiv(readTransfer.BytesPerCycle)
                + writes.CeilDiv(writeTransfer.BytesPerCycle)
                + (readEvent * readTransfer.LatencyCycles)
                + (writeEvent * writeTransfer.LatencyCycles);
            if (readTransfer.RequiresSynchronization)
            {
                synchronizationCycles += readEvent * machine.Synchronization.BlockCycles;
            }

            if (writeTransfer.RequiresSynchronization)
            {
                synchronizationCycles += writeEvent * machine.Synchronization.BlockCycles;
            }
        }

        IntExpr computeCycles = solver.MakeIntConst(0);
        foreach (var (opNode, opNodeInfo) in opNodeMemo)
        {
            var parent = (TileNode?)opNode.Parent
                ?? throw new InvalidOperationException($"AutoTiling Op{opNode.OpId} has no parent tile node.");
            var workload = opNode.GetTileWorkload();
            var context = opNode.GetTileWorkloadContext();
            computeCycles += EstimateTotalBlockComputeCycles(
                machine,
                workload,
                opNodeInfo.Shapes,
                tileNodeMemo[parent].TripCounts[^1],
                solver,
                context,
                opNode.OpId);
        }

        var overlappedCycles = memoryCycles.Concat(transferCycles).Aggregate(computeCycles, solver.MakeMax);
        var totalCycles = overlappedCycles + synchronizationCycles;

        var totalCyclesVar = totalCycles.Var();
        totalCyclesVar.SetRange(0, long.MaxValue / memoryResourceGroups.Min(group => machine.GetMemoryResource(group.Key).ReadBytesPerCycle)); /* avoid crash. */
        var objectiveMonitor = solver.MakeMinimize(totalCyclesVar, 1);
        var collector = solver.MakeNBestValueSolutionCollector(5, false);
        collector.AddObjective(totalCyclesVar);
        collector.Add(totalCyclesVar);
        collector.Add(levelDataReads.Select(i => i.Var()).ToArray());
        collector.Add(levelDataWrites.Select(i => i.Var()).ToArray());
        collector.Add(levelTransferReads.Select(i => i.Var()).ToArray());
        collector.Add(levelTransferWrites.Select(i => i.Var()).ToArray());
        collector.Add(computeCycles.Var());
        collector.Add(synchronizationCycles.Var());
        collector.Add(memoryCycles.Select(i => i.Var()).ToArray());
        collector.Add(transferCycles.Select(i => i.Var()).ToArray());
        collector.Add(reductionStateBytes.Values.Select(value => value.Var()).ToArray());

        var searchAbleVars = new List<IntVar>();
        foreach (var (node, diminfo) in tileableNodeMemo)
        {
            searchAbleVars.AddRange(diminfo.TileVars.Select(i => i.Var()).Reverse());
            collector.Add(diminfo.TileVars.Select(i => i.Var()).ToArray());
            collector.Add(diminfo.ForwardExtents.Select(x => x.Var()).ToArray());
        }

        foreach (var (node, info) in opNodeMemo)
        {
            collector.Add(info.Shapes.SelectMany(i => i).Select(i => i.Var()).ToArray());
            collector.Add(info.Sizes.Select(i => i.Var()).ToArray());
        }

        foreach (var (node, info) in tileNodeMemo)
        {
            collector.Add(info.TripCounts.Select(i => i.Var()).ToArray());
            collector.Add(info.BackWardExtents.Select(i => i.Select(j => j.Var())).SelectMany(i => i).ToArray());
            foreach (var (bid, bufferInfo) in info.BufferInfoMap)
            {
                var placeVars = bufferInfo.Places.SelectMany(i => i).ToArray();
                searchAbleVars.AddRange(placeVars.Select(i => i.Var()));
                collector.Add(placeVars.Select(i => i.Var()).ToArray());
                collector.Add(bufferInfo.Shapes.SelectMany(i => i).Select(i => i.Var()).ToArray());
                collector.Add(bufferInfo.Sizes.Where(v => v is not null).Select(i => i.Var()).ToArray());
                collector.Add(bufferInfo.Trips.Where(v => v is not null).Select(i => i.Var()).ToArray());
            }
        }

        foreach (var (level, nodeBufferSizes) in levelBufferSizes)
        {
            foreach (var (nodeBuffer, bufferSize) in nodeBufferSizes)
            {
                collector.Add(bufferSize.Var());

                foreach (var shape in levelBufferShapes[level][nodeBuffer])
                {
                    collector.Add(shape.Var());
                }
            }
        }

        foreach (var (_, v) in levelBufferLifenessConstraints)
        {
            foreach (var item in v)
            {
                collector.Add(item.Var());
            }
        }

        // var defaultPhaseParameters = new DefaultPhaseParameters();
        // var decisionBuilder = solver.MakeDefaultPhase(searchAbleVars.ToArray(), defaultPhaseParameters);
        DecisionBuilder decisionBuilder;
        {
            var phaseTileVars = new List<IntVar>();
            var phaseOtherVars = new List<IntVar>();
            foreach (var (node, info) in opNodeMemo)
            {
                foreach (var tilevar in tileableNodeMemo[node].TileVars.Reverse())
                {
                    if (searchAbleVars.Contains(tilevar.Var()))
                    {
                        phaseTileVars.Add(tilevar.Var());
                    }
                }
            }

            var phasePlacementVars = internalPlacementCandidates
                .SelectMany(pair => pair.Value
                    .OrderBy(candidate => candidate.Node.Level)
                    .SelectMany(candidate => candidate.Info.Places.Select(place => place[candidate.Node.Level].Var())))
                .Distinct()
                .ToArray();
            phaseOtherVars.AddRange(searchAbleVars.Except(phaseTileVars).Except(phasePlacementVars));
            var phaseTiles = solver.MakePhase(phaseTileVars.ToArray(), Solver.CHOOSE_FIRST_UNBOUND, Solver.ASSIGN_MAX_VALUE);
            var phasePlacements = solver.MakePhase(phasePlacementVars, Solver.CHOOSE_FIRST_UNBOUND, Solver.ASSIGN_MAX_VALUE);
            var phaseOthers = solver.MakePhase(phaseOtherVars.ToArray(), Solver.CHOOSE_FIRST_UNBOUND, Solver.ASSIGN_MIN_VALUE);
            decisionBuilder = solver.Compose(phaseTiles, phasePlacements, phaseOthers);
        }

        var solve_max_time = 30;
        if (System.Environment.GetEnvironmentVariable("NNCASE_TILING_MAX_TIME") is string s_solve_max_time)
        {
            try
            {
                solve_max_time = int.Parse(s_solve_max_time);
            }
            catch (System.Exception)
            {
            }
        }

        var solve_max_solutions = 15;
        if (System.Environment.GetEnvironmentVariable("NNCASE_TILING_MAX_SOLUTIONS") is string s_solve_max_solutions)
        {
            try
            {
                solve_max_solutions = int.Parse(s_solve_max_solutions);
            }
            catch (System.Exception)
            {
            }
        }

        var monitors = new List<SearchMonitor>() { collector, objectiveMonitor, solver.MakeTimeLimit(solve_max_time * 1000) };
        if (solve_max_solutions > 0)
        {
            monitors.Add(solver.MakeSolutionsLimit(solve_max_solutions));
        }

        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            monitors.Add(solver.MakeSearchLog(10000, totalCyclesVar));
        }

        var status = solver.Solve(decisionBuilder, monitors.ToArray());
        if (!status)
        {
            DumpAssgin(primTree, new TreeSolverPrinter(null, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, targetOptions), tileVarConstraints, reductionStateBytes, reductionStateConstraints, eachLevelStoreBufferConstrains, levelBufferLifenessConstraints, levelBufferSizes, levelDataReads, levelDataWrites, levelTransferReads, levelTransferWrites, memoryCycles, transferCycles, computeCycles, synchronizationCycles, totalCyclesVar);
            throw new SolveFailedException(
                $"tiling solve failed after {solver.WallTime()} ms, {solver.Branches()} branches, and {solver.Failures()} failures.");
        }

        var sol = collector.Solution(collector.SolutionCount() - 1);

        var levelBufferInfos = new Dictionary<int, Dictionary<NodeWithBuffer, NodeWithBufferInfo>>();
        foreach (var (level, nodeBufferSizes) in levelBufferSizes)
        {
            var nodeBufferInfos = new Dictionary<NodeWithBuffer, NodeWithBufferInfo>();
            foreach (var (nodeBuffer, sizeVar) in nodeBufferSizes)
            {
                var placementInfo = levelBufferPlacementInfos[level][nodeBuffer];
                var selectedPositions = Enumerable.Range(0, placementInfo.Places.Length)
                    .Where(ci => sol.Value(placementInfo.Places[ci][level].Var()) == 1)
                    .ToArray();
                if (selectedPositions.Length > 1)
                {
                    throw new InvalidOperationException(
                        $"Tile buffer {nodeBuffer} selected multiple creation positions at memory level {level}: {string.Join(",", selectedPositions)}.");
                }

                var liveness = selectedPositions.Length == 1
                    ? placementInfo.Liveness[selectedPositions[0]]
                    : new Tuple<int, int>(0, 0);
                var shapes = levelBufferShapes[level][nodeBuffer].Select(s => sol.Value(s.Var())).ToArray();
                var strides = TensorUtilities.GetDefaultStrides(shapes);
                nodeBufferInfos[nodeBuffer] = new NodeWithBufferInfo(sol.Value(sizeVar.Var()), liveness, shapes, strides);
            }

            levelBufferInfos[level] = nodeBufferInfos;
        }

        var opNodeMemoAssgin = opNodeMemo.ToDictionary(kv => kv.Key, kv => new OpNodeInfo<long>(kv.Value.Maps, sol.Value(kv.Value.Shapes), sol.Value(kv.Value.Sizes)));
        var tileNodeMemoAssgin = tileNodeMemo.ToDictionary(kv => kv.Key, kv => new TileNodeInfo<long>(sol.Value(kv.Value.TripCounts), sol.Value(kv.Value.BackWardExtents), kv.Value.DefUseMap, kv.Value.BufferInfoMap.ToDictionary(p => p.Key, p => new TileNodeBufferInfo<long>(p.Value.Liveness, p.Value.Map, sol.Value(p.Value.Places), sol.Value(p.Value.Shapes), sol.Value(p.Value.Sizes), sol.Value(p.Value.Trips), p.Value.Mask))));
        var tileableNodeMemoAssgin = tileableNodeMemo.ToDictionary(kv => kv.Key, kv => new DomainInfo<long>(sol.Value(kv.Value.TileVars), sol.Value(kv.Value.ForwardExtents), kv.Value.DimsMap));
        var selectedProducerLevels = new Dictionary<(int ProducerOpId, int ProducerOutputIndex), int>();
        foreach (var (source, candidates) in internalPlacementCandidates)
        {
            var selectedLevels = candidates
                .Where(candidate => candidate.Info.Places.Any(place => sol.Value(place[candidate.Node.Level].Var()) == 1))
                .Select(candidate => candidate.Node.Level)
                .Distinct()
                .ToArray();
            if (selectedLevels.Length != 1)
            {
                throw new InvalidOperationException(
                    $"Internal producer result {source} must select exactly one storage level, got [{string.Join(",", selectedLevels)}].");
            }

            var producer = (
                ProducerOpId: source.Node.RegionOpId,
                ProducerOutputIndex: source.OutputIndex);
            if (selectedProducerLevels.TryGetValue(producer, out var existingLevel) && existingLevel != selectedLevels[0])
            {
                throw new InvalidOperationException(
                    $"Producer Op{producer.ProducerOpId}.out{producer.ProducerOutputIndex} selected conflicting memory levels {existingLevel} and {selectedLevels[0]}.");
            }

            selectedProducerLevels[producer] = selectedLevels[0];
        }

        var selectedUseLevels = new Dictionary<TileUseId, int>();
        foreach (var (tileNode, nodeInfo) in tileNodeMemoAssgin)
        {
            foreach (var (source, sink) in nodeInfo.DefUseMap)
            {
                var producer = (
                    ProducerOpId: source.Node.RegionOpId,
                    ProducerOutputIndex: source.OutputIndex);
                if (!selectedProducerLevels.TryGetValue(producer, out var selectedLevel))
                {
                    continue;
                }

                var use = new TileUseId(
                    source.Node.RegionOpId,
                    source.OutputIndex,
                    sink.Node.RegionOpId,
                    sink.Index);
                if (selectedUseLevels.TryGetValue(use, out var existingLevel) && existingLevel != selectedLevel)
                {
                    throw new InvalidOperationException(
                        $"Tile use {use} selected conflicting memory levels {existingLevel} and {selectedLevel}.");
                }

                selectedUseLevels[use] = selectedLevel;
            }
        }

        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            DumpAssgin(primTree, new TreeSolverPrinter(sol, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, targetOptions), tileVarConstraints, reductionStateBytes, reductionStateConstraints, eachLevelStoreBufferConstrains, levelBufferLifenessConstraints, levelBufferSizes, levelDataReads, levelDataWrites, levelTransferReads, levelTransferWrites, memoryCycles, transferCycles, computeCycles, synchronizationCycles, totalCyclesVar);

            DumpAssgin(primTree, new TreeSolverPythonPrinter(sol, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, targetOptions), tileVarConstraints, reductionStateBytes, reductionStateConstraints, eachLevelStoreBufferConstrains, levelBufferLifenessConstraints, levelBufferSizes, levelDataReads, levelDataWrites, levelTransferReads, levelTransferWrites, memoryCycles, transferCycles, computeCycles, synchronizationCycles, totalCyclesVar);
        }

        return new TreeSolveResult(primBufferGraph, sol.ObjectiveValue(), levelBufferInfos, opNodeMemoAssgin, tileNodeMemoAssgin, tileableNodeMemoAssgin, selectedUseLevels, targetOptions, moduleKind);
    }

    public static void DumpAssgin(ITreeNode tree, TreeSolverPythonPrinter printer, Dictionary<OpNode, Constraint[]> tileVarConstraints, Dictionary<OpNode, IntExpr> reductionStateBytes, Dictionary<OpNode, Constraint> reductionStateConstraints, Dictionary<int, Constraint[]> lowestStoreBufferNumsConstrains, Dictionary<int, Constraint[]> levelBufferLifenessConstraints, Dictionary<int, Dictionary<NodeWithBuffer, IntExpr>> levelBufferSizes, IntExpr[] levelDataReads, IntExpr[] levelDataWrites, IntExpr[] levelTransferReads, IntExpr[] levelTransferWrites, IntExpr[] memoryCycles, IntExpr[] transferCycles, IntExpr computeCycles, IntExpr synchronizationCycles, IntVar totalCycles)
    {
        using (var stream = Diagnostics.DumpScope.Current.OpenFile($"modeling.py"))
        {
            using var baseWriter = new StreamWriter(stream);
            using var writer = new System.CodeDom.Compiler.IndentedTextWriter(baseWriter, "  ");
            tree.Accept(printer, (null, writer));
        }
    }

    public static void DumpAssgin(ITreeNode tree, TreeSolverPrinter printer, Dictionary<OpNode, Constraint[]> tileVarConstraints, Dictionary<OpNode, IntExpr> reductionStateBytes, Dictionary<OpNode, Constraint> reductionStateConstraints, Dictionary<int, Constraint[]> eachLevelStoreBufferNumsConstrains, Dictionary<int, Constraint[]> levelBufferLifenessConstraints, Dictionary<int, Dictionary<NodeWithBuffer, IntExpr>> levelBufferSizes, IntExpr[] levelDataReads, IntExpr[] levelDataWrites, IntExpr[] levelTransferReads, IntExpr[] levelTransferWrites, IntExpr[] memoryCycles, IntExpr[] transferCycles, IntExpr computeCycles, IntExpr synchronizationCycles, IntVar totalCycles)
    {
        using (var stream = Diagnostics.DumpScope.Current.OpenFile($"modeling.yaml"))
        {
            using var baseWriter = new StreamWriter(stream);
            using var writer = new System.CodeDom.Compiler.IndentedTextWriter(baseWriter, "  ");
            WriteTargetMachine(writer, printer.TargetOptions.TargetMachineModel);
            tree.Accept(printer, writer);
            writer.WriteLine("tileVarConstraints:");
            writer.Indent++;
            foreach (var (opnode, consts) in tileVarConstraints)
            {
                TreeSolverPrinter.WriteIntExprVector(writer, opnode.ToString(), consts, printer.Solution);
            }

            writer.Indent--;

            writer.WriteLine("BackendPrivateAccumulatorUsage:");
            writer.Indent++;
            foreach (var (opNode, bytes) in reductionStateBytes)
            {
                TreeSolverPrinter.WriteIntExpr(writer, opNode.ToString(), bytes, printer.Solution);
            }

            writer.Indent--;

            writer.WriteLine("BackendPrivateAccumulatorConstraints:");
            writer.Indent++;
            foreach (var (opNode, constraint) in reductionStateConstraints)
            {
                TreeSolverPrinter.WriteIntExprVector(writer, opNode.ToString(), new[] { constraint }, printer.Solution);
            }

            writer.Indent--;

            writer.WriteLine("EachLevelStoreBufferNumsConstrains:");
            writer.Indent++;
            foreach (var (node, cons) in eachLevelStoreBufferNumsConstrains)
            {
                TreeSolverPrinter.WriteIntExprVector(writer, node.ToString(), cons, printer.Solution);
            }

            writer.Indent--;

            writer.WriteLine("EachLevelBufferLifenessConstraints:");
            writer.Indent++;
            foreach (var (node, cons) in levelBufferLifenessConstraints)
            {
                TreeSolverPrinter.WriteIntExprVector(writer, node.ToString(), cons, printer.Solution);
            }

            writer.Indent--;

            writer.WriteLine("LevelMemoryUsage:");
            {
                writer.Indent++;
                foreach (var (sl, nodeMemoryUsage) in levelBufferSizes)
                {
                    writer.WriteLine($"Level_{sl}:");
                    writer.Indent++;
                    foreach (var (node, usage) in nodeMemoryUsage)
                    {
                        TreeSolverPrinter.WriteIntExpr(writer, $"- {node}", usage, printer.Solution);
                    }

                    writer.Indent--;
                }

                writer.Indent--;
            }

            TreeSolverPrinter.WriteIntExprVector(writer, "LevelDataReads", levelDataReads, printer.Solution);
            TreeSolverPrinter.WriteIntExprVector(writer, "LevelDataWrites", levelDataWrites, printer.Solution);
            TreeSolverPrinter.WriteIntExprVector(writer, "LevelTransferReads", levelTransferReads, printer.Solution);
            TreeSolverPrinter.WriteIntExprVector(writer, "LevelTransferWrites", levelTransferWrites, printer.Solution);
            TreeSolverPrinter.WriteIntExprVector(writer, "MemoryCycles", memoryCycles, printer.Solution);
            TreeSolverPrinter.WriteIntExprVector(writer, "TransferCycles", transferCycles, printer.Solution);
            TreeSolverPrinter.WriteIntExpr(writer, "ComputeCycles", computeCycles, printer.Solution);
            TreeSolverPrinter.WriteIntExpr(writer, "SynchronizationCycles", synchronizationCycles, printer.Solution);
            TreeSolverPrinter.WriteIntExpr(writer, "TotalCycles", totalCycles, printer.Solution);
        }
    }

    public (Dictionary<BufferIdentity, Expr> ArgumentMemo, long ObjectValue, Dictionary<TileUseId, int> SelectedUseLevels) SolveRootGraph(TieredTileGraph rootGraph, string moduleKind, INTTTargetOptions targetOptions, DimVar[] dynamicDimVars)
    {
        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            rootGraph.Dump($"root_tile_graph");
        }

        // bufferize root graph.
        var bufferGraphMemo = rootGraph.Bufferize();
        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            bufferGraphMemo[rootGraph].Dump($"root_buffer_graph");
        }

        // condense the root graph.
        var condensedGraph = rootGraph.Condense();
        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            using (var file = Diagnostics.DumpScope.Current.OpenFile($"root_condensed_graph.dot"))
            {
                using var writer = new StreamWriter(file);
                writer.Write(condensedGraph.ToGraphviz(init =>
                {
                    init.FormatVertex += (_, arg) =>
                    {
                        if (arg.Vertex is TieredTileGraph t)
                        {
                            arg.VertexFormat.Label = t.ToString();
                        }
                    };
                }));
            }
        }

        // convert root graph as tree.
        var rootTree = TileNode.FromTileGraph(rootGraph, out var treeGraphMemo);

        var argumentMemo = bufferGraphMemo[rootGraph].GetInputsOutputs(null).Inputs.ToDictionary(k => k, k => k.Node.Grid.GetArgument(k.Index));
        var selectedUseLevels = new Dictionary<TileUseId, int>();
        long objectValue = 0;
        foreach (var (primGraph, i) in condensedGraph.TopologicalSort().Select((s, i) => (s, i)))
        {
            if (IsAliasOnlyComponent(primGraph))
            {
                var (aliasInputBids, aliasOutputBids) = bufferGraphMemo[primGraph].GetInputsOutputs(bufferGraphMemo[rootGraph]);
                var orderedAliasInputs = OrderBufferIdentities(aliasInputBids);
                var orderedAliasOutputs = OrderBufferIdentities(aliasOutputBids);
                var residualOutputs = CreateResidualAliasOutputs(orderedAliasInputs, orderedAliasOutputs, argumentMemo);
                BindComponentOutputs(
                    orderedAliasOutputs,
                    residualOutputs,
                    bufferGraphMemo[rootGraph],
                    argumentMemo);
                continue;
            }

            var funcName = CompileSessionScope.GetCurrentThrowIfNull().GetRequiredService<INamingProvider>().GetName("device_func");
            using var subSubScope = new Diagnostics.DumpScope(funcName, Diagnostics.DumpFlags.Tiling);
            var primTree = treeGraphMemo[primGraph];
            HashSet<BufferIdentity> inputBids;
            HashSet<BufferIdentity> outputBids;

            if (!SolveMemo.TryGetValue(primTree, out var tiled))
            {
                var result = SolvePrimGraph(primTree, bufferGraphMemo, targetOptions, moduleKind);
                (inputBids, outputBids) = (result.Inputs, result.Outputs);
                var inputBidsOrdered = OrderBufferIdentities(inputBids);
                var outputBidsOrdered = OrderBufferIdentities(outputBids);
                var maxAlign = result.ScheduleBuffers();
                var bodyBuilder = T.Sequential();
                var initOffsets = Enumerable.Repeat(new DimConst(0), primTree.DomainBoundExprs.Length).ToArray();
                var initBounds = primTree.DomainBoundExprs.ToArray();
                result.Visit(primTree, new(bodyBuilder, initOffsets, initBounds));
                var parameters = inputBidsOrdered.Select(k => result.InputOutputVars[k]).Concat(
                    dynamicDimVars.Select(v => (IVar)v.With())).Concat(
                    result.OutputParameters).ToArray();
                var primFunc = new PrimFunction(
                    funcName,
                    moduleKind,
                    bodyBuilder.Build(),
                    new Return(outputBidsOrdered.Select(bid => result.OutputValues[bid]).ToArray()),
                    parameters)
                {
                    Role = FunctionRole.ScheduledRegion,
                };
                {
                    // note noneed to rewrite shapeof, because we don't use shapeof new.
                    // var gridBufferToVarMap = inputBids.Concat(outputBids).Select(bid => bid.Node.Grid.GetArgument(bid.Index)).Zip(parameters.Where(p => p is not DimVar)).ToDictionary(p => p.First, p => (Expr)p.Second, (IEqualityComparer<Expr>)ReferenceEqualityComparer.Instance);
                    // var mutator = new AtShapeOfRewriter(gridBufferToVarMap);
                    // mutator.Visit(primFunc, default);
                }

                primFunc.SchedResult.IsScheduled = true; // avoid buffersize pass schedule it again.
                primFunc.SchedResult.DataAlign = (ulong)maxAlign;
                var typeHints = inputBidsOrdered.Select(bid => bid.Node.Grid.GetArgument(bid.Index).CheckedType)
                    .Concat(dynamicDimVars.Select(v => new DimensionType(DimensionKind.Dynamic)))
                    .Concat(result.OutputParameters.Select(parameter => parameter.CheckedType))
                    .ToArray();
                tiled = new(
                    new PrimFunctionWrapper(
                        primFunc,
                        inputBidsOrdered.Length + dynamicDimVars.Length,
                        typeHints),
                    result.ObjectiveValue,
                    CanonicalizeSelectedUseLevels(primTree, result.SelectedUseLevels));
                SolveMemo.Add(primTree, tiled);
            }
            else
            {
                (inputBids, outputBids) = bufferGraphMemo[primGraph].GetInputsOutputs(bufferGraphMemo[rootGraph]);
            }

            var orderedInputBids = OrderBufferIdentities(inputBids);
            var orderedOutputBids = OrderBufferIdentities(outputBids);
            objectValue += tiled.ObjectValue;
            foreach (var (use, level) in MaterializeSelectedUseLevels(primTree, tiled.SelectedUseLevels))
            {
                if (selectedUseLevels.TryGetValue(use, out var existingLevel) && existingLevel != level)
                {
                    throw new InvalidOperationException(
                        $"Tile use {use} selected conflicting levels {existingLevel} and {level} across solved components.");
                }

                selectedUseLevels[use] = level;
            }

            var finalCall = new Call(tiled.Func, orderedInputBids.Select(bid => argumentMemo[bid]).Concat(dynamicDimVars.OfType<BaseExpr>()).ToArray());
            var componentOutputs = orderedOutputBids.Length == 1
                ? new Expr[] { finalCall }
                : orderedOutputBids.Select((_, outputIndex) => IR.F.Tensors.GetItem(finalCall, outputIndex)).ToArray();
            BindComponentOutputs(
                orderedOutputBids,
                componentOutputs,
                bufferGraphMemo[rootGraph],
                argumentMemo);
        }

        return (argumentMemo, objectValue, selectedUseLevels);
    }

    public BaseExpr Tile(BaseExpr preExpr, string moduleKind, INTTTargetOptions targetOptions, DimVar[] dynamicDimVars)
    {
        var levelCount = targetOptions.TargetMachineModel.TilingMemorySpaces.Length;
        var rootGraph = TieredTileGraphBuilder.Build(preExpr, levelCount, out var exprMemo);
        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            rootGraph.Dump($"tile_graph");
        }

        var region = TileRegion.Create(rootGraph);
        var plan = new HierarchicalTilePlanner(this).Plan(region, moduleKind, targetOptions, dynamicDimVars);

        var replaces = new Dictionary<BaseExpr, BaseExpr>(ReferenceEqualityComparer.Instance);
        var multiOutputReplaces = new Dictionary<Grid, Dictionary<int, BaseExpr>>(ReferenceEqualityComparer.Instance);

        foreach (var (bid, value) in plan.ArgumentMemo)
        {
            // use bid to find the old expr.
            if (bid.IsOutput)
            {
                var grid = bid.Node.Grid;
                var outputIndex = bid.OutputIndex;
                if (grid.Accesses.ToArray().Count(access => access.IsWrite) == 1)
                {
                    replaces.TryAdd(grid, value);
                }
                else
                {
                    if (!multiOutputReplaces.TryGetValue(grid, out var outputMap))
                    {
                        outputMap = new Dictionary<int, BaseExpr>();
                        multiOutputReplaces.Add(grid, outputMap);
                    }

                    outputMap.TryAdd(outputIndex, value);
                }
            }
            else
            {
                var oldExpr = bid.Node.Grid.GetArgument(bid.Index);
                replaces.TryAdd(oldExpr, value);
            }
        }

        foreach (var (grid, outputMap) in multiOutputReplaces)
        {
            var outputCount = grid.Accesses.ToArray().Count(access => access.IsWrite);
            if (outputMap.Count != outputCount)
            {
                continue;
            }

            var fields = Enumerable.Range(0, outputCount)
                .Select(i => outputMap.TryGetValue(i, out var output)
                    ? output
                    : throw new InvalidOperationException($"Missing tiled output {i} for Op{exprMemo[grid].OpId}."))
                .ToArray();
            replaces.TryAdd(grid, new IR.Tuple(fields));
        }

        var cloner = new TiledOutputReplacingExprCloner(replaces, multiOutputReplaces);
        return cloner.Clone(preExpr, default);
    }

    private static bool IsAliasOnlyComponent(TieredTileGraph graph)
    {
        var vertices = graph.Vertices.ToArray();
        return vertices.Length != 0 && vertices.All(vertex => vertex.IsPureBufferView);
    }

    private static Expr[] CreateResidualAliasOutputs(
        IReadOnlyList<BufferIdentity> inputBids,
        IReadOnlyList<BufferIdentity> outputBids,
        IReadOnlyDictionary<BufferIdentity, Expr> argumentMemo)
    {
        var inputReplacements = new Dictionary<BaseExpr, BaseExpr>(ReferenceEqualityComparer.Instance);
        foreach (var inputBid in inputBids)
        {
            if (!argumentMemo.TryGetValue(inputBid, out var selectedInput))
            {
                throw new InvalidOperationException($"Residual buffer alias input {inputBid} has no selected producer value.");
            }

            var sourceValue = inputBid.Node.Grid.GetArgument(inputBid.Index);
            if (inputReplacements.TryGetValue(sourceValue, out var existing) && !ReferenceEquals(existing, selectedInput))
            {
                throw new InvalidOperationException(
                    $"Residual buffer alias source {sourceValue} resolves to multiple selected values.");
            }

            inputReplacements[sourceValue] = selectedInput;
        }

        var originalOutputs = outputBids.Select(GetBufferOutputExpression).ToArray();
        BaseExpr originalResult = originalOutputs.Length == 1
            ? originalOutputs[0]
            : new IR.Tuple(originalOutputs);
        var residualCloner = new ReplacingExprCloner(inputReplacements)
        {
            CloneUnmutated = false,
        };
        var residualResult = residualCloner.Clone(originalResult, Unit.Default);
        return residualResult is IR.Tuple tuple
            ? tuple.Fields.AsValueEnumerable().Select(field => (Expr)field).ToArray()
            : [(Expr)residualResult];
    }

    private static Expr GetBufferOutputExpression(BufferIdentity outputBid)
    {
        if (!outputBid.IsOutput)
        {
            throw new ArgumentException($"Expected an output buffer identity, got {outputBid}.", nameof(outputBid));
        }

        var grid = outputBid.Node.Grid;
        return grid.Accesses.ToArray().Count(access => access.IsWrite) == 1
            ? grid
            : IR.F.Tensors.GetItem(grid, outputBid.OutputIndex);
    }

    private static void BindComponentOutputs(
        IReadOnlyList<BufferIdentity> outputBids,
        IReadOnlyList<Expr> outputValues,
        BufferGraph rootBufferGraph,
        Dictionary<BufferIdentity, Expr> argumentMemo)
    {
        if (outputBids.Count != outputValues.Count)
        {
            throw new ArgumentException(
                $"Component output count mismatch: buffers={outputBids.Count}, values={outputValues.Count}.");
        }

        for (var outputIndex = 0; outputIndex < outputBids.Count; outputIndex++)
        {
            var outputBid = outputBids[outputIndex];
            if (argumentMemo.ContainsKey(outputBid))
            {
                continue;
            }

            var outputValue = outputValues[outputIndex];
            argumentMemo.Add(outputBid, outputValue);
            foreach (var sinkBid in rootBufferGraph.OutEdges(outputBid)
                .Where(edge => edge.Tag is BufferEdgeKind.Inter)
                .Select(edge => edge.Target))
            {
                argumentMemo.TryAdd(sinkBid, outputValue);
            }
        }
    }

    private static BufferIdentity[] OrderBufferIdentities(IEnumerable<BufferIdentity> identities)
        => identities
            .OrderBy(identity => identity.Node.OpId)
            .ThenBy(identity => identity.Index)
            .ToArray();

    private static void WriteTargetMachine(System.CodeDom.Compiler.IndentedTextWriter writer, TargetMachineModel machine)
    {
        writer.WriteLine("TargetMachine:");
        writer.Indent++;
        writer.WriteLine($"Id: {machine.Id}");
        writer.WriteLine($"Execution: {machine.Execution.Kind}");
        writer.WriteLine($"ComputeUnits: {machine.Execution.ComputeUnitCount}");
        writer.WriteLine($"WorkersPerBlock: {machine.Execution.WorkersPerBlock}");
        writer.WriteLine($"WorkerWidth: {machine.Execution.WorkerWidth}");
        writer.WriteLine($"BackendPrivateAccumulatorCapacityBytes: {machine.Execution.BackendPrivateAccumulatorCapacityBytes}");
        writer.WriteLine($"BackendPrivateMatrixAccumulatorMinM: {machine.Execution.BackendPrivateMatrixAccumulatorMinM}");
        writer.WriteLine($"BackendPrivateMatrixAccumulatorMinN: {machine.Execution.BackendPrivateMatrixAccumulatorMinN}");
        writer.WriteLine($"BackendPrivateGemvAccumulatorMinN: {machine.Execution.BackendPrivateGemvAccumulatorMinN}");
        writer.WriteLine("MemorySpaces:");
        writer.Indent++;
        foreach (var memorySpace in machine.MemorySpaces.Values.OrderBy(space => space.Id.Value, StringComparer.Ordinal))
        {
            var resource = machine.GetMemoryResource(memorySpace);
            writer.WriteLine($"- {memorySpace.Id}: resource={resource.Id}, kind={resource.Kind}, scope={memorySpace.Scope}, allocation_limit={memorySpace.MaxAllocationBytesPerScope}, resource_capacity={resource.CapacityBytes}, read_bpc={resource.ReadBytesPerCycle}, write_bpc={resource.WriteBytesPerCycle}, latency={resource.LatencyCycles}, tiling_level={memorySpace.TilingLevel}");
        }

        writer.Indent--;
        writer.WriteLine("MatrixPrimitives:");
        writer.Indent++;
        foreach (var primitive in machine.Compute.MatrixPrimitives)
        {
            writer.WriteLine($"- {primitive.Name}: m={primitive.M}, n={primitive.N}, k={primitive.K}, instructions_per_cycle={primitive.InstructionsPerCyclePerBlock}, supported={primitive.IsSupported}");
        }

        writer.Indent--;
        writer.Indent--;
    }

    private static long GetActiveBlockCount(INTTTargetOptions targetOptions)
    {
        var hierarchy = targetOptions.Hierarchies.FirstOrDefault()
            ?? throw new InvalidOperationException("AutoTiling requires at least one target hierarchy.");
        var placement = new Placement(hierarchy, targetOptions.HierarchyNames, targetOptions.HierarchyLevels);
        var activeBlockCount = Math.Max(1, placement.GetPhysicalLevelSize('b'));
        if (activeBlockCount > targetOptions.TargetMachineModel.Execution.ComputeUnitCount)
        {
            throw new InvalidOperationException(
                $"Configured block hierarchy requires {activeBlockCount} blocks, but target machine {targetOptions.TargetMachineModel.Id} exposes only {targetOptions.TargetMachineModel.Execution.ComputeUnitCount} compute units.");
        }

        return activeBlockCount;
    }

    private static IntExpr EstimateTotalBlockComputeCycles(
        TargetMachineModel machine,
        TileWorkload workload,
        IntExpr[][] bufferShapes,
        IntExpr loopTripCount,
        Solver solver,
        TileWorkloadContext context,
        int opId)
    {
        var fullBufferShapes = context.BufferShapes
            .Select(shape => shape.Select(dimension => (IntExpr)solver.MakeIntConst(dimension)).ToArray())
            .ToArray();
        if (workload is BufferAliasTileWorkload)
        {
            return solver.MakeIntConst(0);
        }

        if (workload is ElementwiseTileWorkload elementwise)
        {
            var elementwiseWork = elementwise.GetWork(fullBufferShapes, solver, context);
            return DivideByRate(elementwiseWork, machine.Compute.ElementwiseElementsPerCycle);
        }

        if (workload is ReductionTileWorkload reduction)
        {
            var reductionWork = reduction.GetWork(fullBufferShapes, solver, context);
            return DivideByRate(reductionWork, machine.Compute.ElementwiseElementsPerCycle);
        }

        if (workload is not MatrixTileWorkload matrix)
        {
            throw new NotSupportedException($"Unsupported AutoTiling workload {workload.GetType().Name} for {context.Op.GetType().Name}.");
        }

        var fullShape = matrix.GetShape(fullBufferShapes, solver, context);
        var fullWork = fullShape.GetWork();
        var simtCycles = DivideByRate(fullWork, machine.Compute.SimtFmaPerCycle);
        if (!context.BufferDataTypes.Take(2).Any(IsVectorDataType))
        {
            return simtCycles;
        }

        var shape = matrix.GetShape(bufferShapes, solver, context);
        var operandDataTypes = context.BufferDataTypes.Take(2).ToArray();
        if (operandDataTypes.Length != 2)
        {
            throw new InvalidOperationException($"Matrix tile workload {context.Op.GetType().Name} must expose at least two operand buffers.");
        }

        var fullWorkUpperBound = fullWork.Var().Max();
        if (fullWorkUpperBound <= 0 || fullWork.Var().Min() != fullWorkUpperBound)
        {
            throw new InvalidOperationException($"AutoTiling full compute work for Op{opId} must be a positive compile-time constant.");
        }

        var candidates = machine.Compute.MatrixPrimitives
            .Where(primitive => primitive.Supports(operandDataTypes[0], operandDataTypes[1]))
            .Select((primitive, primitiveIndex) =>
            {
                var localInstructionCount = shape.M.CeilDiv(primitive.M)
                    * shape.N.CeilDiv(primitive.N)
                    * shape.K.CeilDiv(primitive.K)
                    * shape.Multiplicity;
                var totalInstructionCount = solver.MakeIntVar(
                    1,
                    fullWorkUpperBound,
                    $"op{opId}_matrix_primitive_{primitiveIndex}_instructions");
                solver.Add(solver.MakeEquality(totalInstructionCount, localInstructionCount * loopTripCount));
                return DivideByRate(totalInstructionCount, primitive.InstructionsPerCyclePerBlock);
            })
            .ToArray();
        return candidates.Aggregate(simtCycles, solver.MakeMin);
    }

    private static IntExpr DivideByRate(IntExpr work, double unitsPerCycle)
    {
        const long scale = 1024;
        if (!double.IsFinite(unitsPerCycle) || unitsPerCycle <= 0)
        {
            throw new InvalidOperationException($"Target throughput must be finite and positive, got {unitsPerCycle}.");
        }

        var scaledRate = checked((long)Math.Round(unitsPerCycle * scale));
        if (scaledRate <= 0)
        {
            throw new InvalidOperationException($"Target throughput {unitsPerCycle} is below the supported fixed-point precision.");
        }

        return (work * scale).CeilDiv(scaledRate);
    }

    private static bool IsVectorDataType(DataType dataType)
        => dataType is VectorType;

    private static void DumpGantt(Dictionary<NodeWithBuffer, IntExpr> nodeBufferSizes, Dictionary<NodeWithBuffer, Tuple<int, int>> nodeBufferLiveness, TileNode primTree, int storeLevel)
    {
        string GetStartStr(string name, int start) => $"[{name}] starts D+{start}";
        string GetDurationStr(string name, int duration) => $"[{name}] requires {duration} days";
        using (var fs = Diagnostics.DumpScope.Current.OpenFile($"Op{primTree.OpId}_{primTree.Level}_store_{storeLevel}_gantt.md"))
        {
            using var writer = new StreamWriter(fs);
            writer.WriteLine("```plantuml");
            writer.WriteLine("@startgantt");
            writer.WriteLine("printscale daily zoom 10");

            foreach (var ((node, bid), liveness) in nodeBufferLiveness)
            {
                var name = $"cl{node.Level} op{bid.Node.OpId} {bid.Index}";
                writer.WriteLine(GetDurationStr(name, liveness.Item2 - liveness.Item1));
                writer.WriteLine(GetStartStr(name, liveness.Item1));
            }

            writer.WriteLine("@endgantt");
            writer.WriteLine("```");
        }
    }

    private static bool IsObjectBuffer(BufferIdentity bid) => bid.Access.Buffer.CheckedDataType is ReferenceType;

    private static IReadOnlyDictionary<CanonicalTileUseId, int> CanonicalizeSelectedUseLevels(
        TileNode tree,
        IReadOnlyDictionary<TileUseId, int> selectedUseLevels)
    {
        var ordinals = GetRegionOpIds(tree)
            .Select((regionOpId, ordinal) => (regionOpId, ordinal))
            .ToDictionary(item => item.regionOpId, item => item.ordinal);
        var canonical = new Dictionary<CanonicalTileUseId, int>();
        foreach (var (use, level) in selectedUseLevels)
        {
            if (!ordinals.TryGetValue(use.ProducerOpId, out var producerOrdinal) ||
                !ordinals.TryGetValue(use.ConsumerOpId, out var consumerOrdinal))
            {
                throw new InvalidOperationException(
                    $"Selected tile use {use} does not belong to the solved component.");
            }

            var canonicalUse = new CanonicalTileUseId(
                producerOrdinal,
                use.ProducerOutputIndex,
                consumerOrdinal,
                use.ConsumerAccessIndex);
            if (canonical.TryGetValue(canonicalUse, out var existingLevel) && existingLevel != level)
            {
                throw new InvalidOperationException(
                    $"Canonical tile use {canonicalUse} selected conflicting levels {existingLevel} and {level}.");
            }

            canonical[canonicalUse] = level;
        }

        return canonical;
    }

    private static IReadOnlyDictionary<TileUseId, int> MaterializeSelectedUseLevels(
        TileNode tree,
        IReadOnlyDictionary<CanonicalTileUseId, int> selectedUseLevels)
    {
        var regionOpIds = GetRegionOpIds(tree);
        var materialized = new Dictionary<TileUseId, int>();
        foreach (var (use, level) in selectedUseLevels)
        {
            if ((uint)use.ProducerOrdinal >= (uint)regionOpIds.Count ||
                (uint)use.ConsumerOrdinal >= (uint)regionOpIds.Count)
            {
                throw new InvalidOperationException(
                    $"Canonical tile use {use} is incompatible with a component containing {regionOpIds.Count} operations.");
            }

            var materializedUse = new TileUseId(
                regionOpIds[use.ProducerOrdinal],
                use.ProducerOutputIndex,
                regionOpIds[use.ConsumerOrdinal],
                use.ConsumerAccessIndex);
            if (materialized.TryGetValue(materializedUse, out var existingLevel) && existingLevel != level)
            {
                throw new InvalidOperationException(
                    $"Tile use {materializedUse} selected conflicting levels {existingLevel} and {level} after component remapping.");
            }

            materialized[materializedUse] = level;
        }

        return materialized;
    }

    private static IReadOnlyList<int> GetRegionOpIds(TileNode tree)
    {
        var ids = new List<int>();
        var seen = new HashSet<int>();

        void Visit(ITreeNode node)
        {
            if (node is OpNode opNode)
            {
                if (seen.Add(opNode.Wrapped.RegionOpId))
                {
                    ids.Add(opNode.Wrapped.RegionOpId);
                }

                return;
            }

            foreach (var child in ((TileNode)node).Children)
            {
                Visit(child);
            }
        }

        Visit(tree);
        return ids;
    }

    public readonly record struct CanonicalTileUseId(
        int ProducerOrdinal,
        int ProducerOutputIndex,
        int ConsumerOrdinal,
        int ConsumerAccessIndex);

    public sealed record TiledFunc(
        PrimFunctionWrapper Func,
        long ObjectValue,
        IReadOnlyDictionary<CanonicalTileUseId, int> SelectedUseLevels)
    {
    }

    private sealed class TiledOutputReplacingExprCloner : ExprCloner<Unit>
    {
        private readonly IReadOnlyDictionary<BaseExpr, BaseExpr> _replaces;
        private readonly IReadOnlyDictionary<Grid, Dictionary<int, BaseExpr>> _multiOutputReplaces;

        public TiledOutputReplacingExprCloner(
            IReadOnlyDictionary<BaseExpr, BaseExpr> replaces,
            IReadOnlyDictionary<Grid, Dictionary<int, BaseExpr>> multiOutputReplaces)
        {
            _replaces = replaces;
            _multiOutputReplaces = multiOutputReplaces;
            CloneUnmutated = false;
        }

        protected override BaseExpr DispatchVisit(BaseExpr expr, Unit context)
        {
            if (expr is Call { Target: IR.Tensors.GetItem } getItem &&
                getItem[IR.Tensors.GetItem.Input] is Grid grid &&
                getItem[IR.Tensors.GetItem.Index] is DimConst index &&
                _multiOutputReplaces.TryGetValue(grid, out var outputs) &&
                outputs.TryGetValue(checked((int)index.Value), out var output))
            {
                return output;
            }

            if (_replaces.TryGetValue(expr, out var replacement))
            {
                return replacement;
            }

            return base.DispatchVisit(expr, context);
        }
    }

    private sealed class AtShapeOfRewriter : ExprRewriter
    {
        private readonly Dictionary<Expr, Expr> _exprMap;

        public AtShapeOfRewriter(Dictionary<Expr, Expr> exprMap)
            : base(false)
        {
            _exprMap = exprMap;
        }

        protected override BaseExpr VisitDimAt(DimAt at, Unit context)
        {
            if (at.Shape is ShapeOf { Value: Expr expr } && _exprMap.TryGetValue(expr, out var newExpr))
            {
                return new DimAt(new ShapeOf(newExpr), at.Index)
                {
                    Metadata = at.Metadata,
                };
            }

            return base.VisitDimAt(at, context);
        }
    }
}

internal sealed class SolveFailedException : Exception
{
    public SolveFailedException(string message)
        : base(message)
    {
    }
}
