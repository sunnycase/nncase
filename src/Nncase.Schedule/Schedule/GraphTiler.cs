// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
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
    public static TreeSolveResult SolvePrimGraph(TileNode primTree, Dictionary<TieredTileGraph, BufferGraph> bufferGraphMemo, INTTTargetOptions targetOptions, string moduleKind, string owningScheduledFunctionId)
    {
        var machine = targetOptions.TargetMachineModel;
        var tilingMemorySpaces = machine.TilingMemorySpaces;
        var rootMemorySpace = machine.GetMemorySpace(machine.RootMemorySpace);
        var memCapacities = tilingMemorySpaces.Select(machine.GetMaximumUsableAllocationBytes).ToArray();
        var levelCount = tilingMemorySpaces.Length;
        var activeBlockCount = GetActiveBlockCount(targetOptions);
        TreeSolverInitializer.Init(primTree, bufferGraphMemo, levelCount, targetOptions, out var solver, out var opNodeMemo, out var tileNodeMemo, out var tileableNodeMemo, out var coverageConstraints, out var opLifetimes);
        var primBufferGraph = bufferGraphMemo[primTree.Wrapped];
        var (externalInputs, externalOutputs) = primBufferGraph.GetInputsOutputs(primBufferGraph.Parent as BufferGraph);
        var rootMaterializationEdges = primBufferGraph.GetOwnedRootMaterializationEdges().ToArray();
        var rootMaterializationEndpoints = rootMaterializationEdges
            .SelectMany(edge => new[] { edge.Source, edge.Target })
            .ToHashSet();
        var rootEndpoints = externalInputs
            .Concat(externalOutputs)
            .Concat(rootMaterializationEndpoints)
            .ToHashSet();

        bool TryResolveRootEndpoint(BufferIdentity buffer, out BufferIdentity rootEndpoint)
        {
            if (rootEndpoints.Contains(buffer))
            {
                rootEndpoint = buffer;
                return true;
            }

            if (TileBufferPlacementUtility.TryGetAliasReadBuffer(buffer, out var aliasRead) &&
                rootEndpoints.Contains(aliasRead))
            {
                rootEndpoint = aliasRead;
                return true;
            }

            rootEndpoint = null!;
            return false;
        }

        static int GetStoragePosition(BufferIdentity bid, TileNodeBufferInfo<IntExpr> bufferInfo)
            => bid.Access.BindingMode == GridBindingMode.Root ? 0 : bufferInfo.GetLastRelatedPos();

        bool RequiresLocalAllocation(BufferIdentity bid, int level)
        {
            if (TileBufferAliasAnalysis.IsPureAliasEndpoint(bid) ||
                (bid.IsOutput && bid.Node.TryGetAliasSourceAccess(bid.Index, out _)))
            {
                return false;
            }

            if (bid.Node.LocalAccessEffects[bid.Index].Scope == MemoryAccessScope.Chip)
            {
                return false;
            }

            var isExternal = externalInputs.Contains(bid) ||
                externalOutputs.Contains(bid) ||
                rootMaterializationEndpoints.Contains(bid);
            return !isExternal ||
                (bid.Access.BindingMode != GridBindingMode.Root && machine.RequiresExplicitTransfer(level));
        }

        bool RequiresUseLocalMaterialization(TileNode tileNode, BufferIdentity bid, int level)
        {
            if (bid.IsOutput ||
                tileNode.Wrapped.IsPureBufferViewScope() ||
                bid.Access.BindingMode != GridBindingMode.Subview ||
                !machine.RequiresExplicitTransfer(level))
            {
                return false;
            }

            var effect = bid.Node.LocalAccessEffects[bid.Index];
            return effect.Scope != MemoryAccessScope.Chip &&
                effect.Mode.HasFlag(MemoryAccessMode.Read);
        }

        // Internal values may be created in any lexical scope that contains the
        // producer and dominates every use. The selected scope chooses one loop
        // entry and any storage space at or below its creation level. Other
        // occurrences are logical views of that single materialization.
        var eachLevelStoreBufferConstrains = new Dictionary<int, Constraint[]>();
        var internalPlacementCandidates = new Dictionary<BufferIdentity, List<(TileNode Node, TileNodeBufferInfo<IntExpr> Info)>>();
        var internalSourcesByEndpoint = new Dictionary<BufferIdentity, HashSet<BufferIdentity>>();
        var internalUsesBySource = new Dictionary<BufferIdentity, HashSet<TileUseId>>();
        var internalOwnerNodes = new Dictionary<BufferIdentity, HashSet<TileNode>>();
        foreach (var tileNode in tileNodeMemo.Keys)
        {
            foreach (var edge in bufferGraphMemo[tileNode.Wrapped].GetOwnedInterEdges())
            {
                if (!internalOwnerNodes.TryGetValue(edge.Source, out var owners))
                {
                    owners = new();
                    internalOwnerNodes.Add(edge.Source, owners);
                }

                owners.Add(tileNode);
                foreach (var endpoint in new[] { edge.Source, edge.Target })
                {
                    if (!internalSourcesByEndpoint.TryGetValue(endpoint, out var sources))
                    {
                        sources = new();
                        internalSourcesByEndpoint.Add(endpoint, sources);
                    }

                    sources.Add(edge.Source);
                }

                if (!internalUsesBySource.TryGetValue(edge.Source, out var uses))
                {
                    uses = new();
                    internalUsesBySource.Add(edge.Source, uses);
                }

                uses.Add(new TileUseId(
                    edge.Source.Node.RegionOpId,
                    edge.Source.OutputIndex,
                    edge.Target.Node.RegionOpId,
                    edge.Target.Index));
            }
        }

        // Build block-microkernel choices before fixing buffer placements:
        // candidate-specific direct-access contracts participate in the same
        // global decision as tile extents and physical materialization.
        var reductionStateBytes = new Dictionary<OpNode, IntExpr>();
        var reductionStateConstraints = new Dictionary<OpNode, Constraint>();
        var baseComputeCyclesByOp = new Dictionary<OpNode, IntExpr>();
        var microKernelDecisions = new Dictionary<OpNode, MicroKernelSolverDecision>();
        foreach (var (opNode, opNodeInfo) in opNodeMemo)
        {
            var workload = opNode.GetTileWorkload();
            var context = opNode.GetTileWorkloadContext();
            var fullBufferShapes = context.BufferShapes
                .Select(shape => shape.Select(extent => (IntExpr)solver.MakeIntConst(extent)).ToArray())
                .ToArray();
            var baseComputeCycles = EstimateTotalBlockComputeCycles(
                machine,
                workload,
                opNodeInfo.Shapes,
                solver,
                context);
            baseComputeCyclesByOp.Add(opNode, baseComputeCycles);
            if (workload is not IReductionStateTileWorkload statefulWorkload)
            {
                continue;
            }

            var stateDescriptors = statefulWorkload.GetReductionStates(
                opNodeInfo.Shapes,
                solver,
                context);
            if (stateDescriptors.Count == 0)
            {
                throw new InvalidOperationException(
                    $"Reduction workload {context.Op.GetType().Name} returned no logical state descriptors.");
            }

            var stateBytes = solver.MakeSum(stateDescriptors.Select(state => state.GetLogicalBytes()).ToArray());
            reductionStateBytes.Add(opNode, stateBytes);
            var unprunedCandidates = targetOptions.BlockMicroKernelModel.GetCandidates(
                new(
                    context.Op,
                    workload,
                    context,
                    opNodeInfo.Shapes,
                    fullBufferShapes,
                    baseComputeCycles,
                    machine,
                    solver)
                {
                    ChipActiveBlockCount = activeBlockCount,
                });
            ValidateMicroKernelCandidates(
                opNode,
                unprunedCandidates,
                machine);
            var candidates = PruneMicroKernelCandidates(unprunedCandidates);

            var selectionVars = candidates
                .Select((candidate, index) => solver.MakeBoolVar(
                    $"microkernel[op{opNode.OpId},{index},{candidate.Variant}]"))
                .ToArray();
            var exactlyOne = solver.MakeEquality(solver.MakeSum(selectionVars), 1);
            exactlyOne.SetName($"microkernel_exactly_one[op{opNode.OpId}]");
            solver.Add(exactlyOne);
            reductionStateConstraints.Add(opNode, exactlyOne);

            for (var candidateIndex = 0; candidateIndex < candidates.Count; candidateIndex++)
            {
                var candidate = candidates[candidateIndex];
                var selected = selectionVars[candidateIndex];
                var legal = solver.MakeLessOrEqual(selected, candidate.IsLegal);
                legal.SetName($"microkernel_legal[op{opNode.OpId},{candidateIndex}]");
                solver.Add(legal);
                foreach (var usage in candidate.Resources)
                {
                    var resource = machine.GetPrivateResource(usage.Resource);
                    var allocatedUnits = GetAllocatedPrivateResourceUnits(usage.Units, resource, solver);
                    var capacity = solver.MakeLessOrEqual(selected * allocatedUnits, resource.CapacityUnits);
                    capacity.SetName($"microkernel_resource_le[op{opNode.OpId},{candidateIndex},{resource.Id}]");
                    solver.Add(capacity);
                }
            }

            var selectedRegionCycles = solver.MakeSum(candidates
                .Select((candidate, index) => selectionVars[index] * candidate.ExecutionCost.RegionCycles)
                .ToArray());
            microKernelDecisions.Add(
                opNode,
                new(candidates, selectionVars, selectedRegionCycles));
        }

        var microKernelDecisionsByGrid = microKernelDecisions.ToDictionary(
            pair => pair.Key.Wrapped,
            pair => pair.Value);

        static IReadOnlyList<OpNode> EnumerateOperationNodes(ITreeNode node)
        {
            var result = new List<OpNode>();
            Visit(node);
            return result;

            void Visit(ITreeNode current)
            {
                if (current is OpNode operationNode)
                {
                    result.Add(operationNode);
                    return;
                }

                foreach (var child in ((TileNode)current).Children)
                {
                    Visit(child);
                }
            }
        }

        for (int level = 0; level < levelCount; level++)
        {
            var cons = new List<Constraint>();
            foreach (var (tileNode, nodeInfo) in tileNodeMemo.Where(kv => kv.Key.Level == level))
            {
                foreach (var (bid, bufferInfo) in nodeInfo.BufferInfoMap)
                {
                    if (tileNode.Wrapped.IsPureBufferViewScope())
                    {
                        AddNoPlacementConstraints(tileNode, bid, bufferInfo, cons);
                        continue;
                    }

                    if (TileBufferAliasAnalysis.IsPureAliasSource(bid) && machine.RequiresExplicitTransfer(level))
                    {
                        AddNoPlacementConstraints(tileNode, bid, bufferInfo, cons);
                        continue;
                    }

                    if (internalSourcesByEndpoint.TryGetValue(bid, out var internalSources))
                    {
                        var ownedSources = internalSources
                            .Where(source => source.Equals(bid))
                            .ToArray();
                        if (ownedSources.Length > 1)
                        {
                            throw new InvalidOperationException(
                                $"Tile buffer {bid} is the creation endpoint of multiple internal values at {tileNode}.");
                        }

                        if (ownedSources.Length == 1)
                        {
                            var source = ownedSources[0];
                            if (!internalPlacementCandidates.TryGetValue(source, out var candidates))
                            {
                                candidates = new();
                                internalPlacementCandidates.Add(source, candidates);
                            }

                            candidates.Add((tileNode, bufferInfo));
                            continue;
                        }

                        if (!RequiresUseLocalMaterialization(tileNode, bid, level))
                        {
                            AddNoPlacementConstraints(tileNode, bid, bufferInfo, cons);
                            continue;
                        }
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

        void AddNoPlacementConstraints(
            TileNode tileNode,
            BufferIdentity bid,
            TileNodeBufferInfo<IntExpr> bufferInfo,
            ICollection<Constraint> constraints)
        {
            for (int ci = 0; ci < bufferInfo.Places.Length; ci++)
            {
                for (int sl = 0; sl < bufferInfo.Places[ci].Length; sl++)
                {
                    var constraint = solver.MakeEquality(bufferInfo.Places[ci][sl], 0);
                    constraint.SetName($"n_propagated_store[{tileNode}_{bid}_ci{ci}_sl{sl}]");
                    solver.Add(constraint);
                    constraints.Add(constraint);
                }
            }
        }

        foreach (var (bid, candidates) in internalPlacementCandidates)
        {
            var ownerNodes = internalOwnerNodes[bid];
            var invalidCandidates = candidates
                .Where(candidate => ownerNodes.Any(owner => !Dominates(candidate.Node, owner)))
                .ToArray();
            foreach (var candidate in invalidCandidates)
            {
                var constraints = new List<Constraint>();
                AddNoPlacementConstraints(candidate.Node, bid, candidate.Info, constraints);
                eachLevelStoreBufferConstrains[candidate.Node.Level] = eachLevelStoreBufferConstrains[candidate.Node.Level]
                    .Concat(constraints)
                    .ToArray();
                candidates.Remove(candidate);
            }

            var placementVars = candidates
                .SelectMany(candidate => candidate.Info.Places.SelectMany(place => place))
                .ToArray();
            if (placementVars.Length == 0)
            {
                throw new SolveFailedException(
                    $"Internal tile buffer {bid} has no creation scope that dominates use owners " +
                    $"[{string.Join(", ", ownerNodes)}].");
            }

            var constraint = solver.MakeEquality(solver.MakeSum(placementVars), 1);
            constraint.SetName($"one_internal_store[{bid}]");
            solver.Add(constraint);
            var ownerLevel = candidates.Max(candidate => candidate.Node.Level);
            eachLevelStoreBufferConstrains[ownerLevel] = eachLevelStoreBufferConstrains[ownerLevel]
                .Append(constraint)
                .ToArray();
        }

        static bool Dominates(TileNode ancestor, TileNode descendant)
        {
            for (ITreeNode? current = descendant; current is not null; current = current.Parent)
            {
                if (ReferenceEquals(current, ancestor))
                {
                    return true;
                }
            }

            return false;
        }

        foreach (var (tileNode, nodeInfo) in tileNodeMemo)
        {
            var aliasConstraints = new List<Constraint>();
            foreach (var (aliasBid, aliasInfo) in nodeInfo.BufferInfoMap)
            {
                if (!aliasBid.IsOutput ||
                    !aliasBid.Node.TryGetAliasSourceAccess(aliasBid.Index, out var sourceAccessIndex))
                {
                    continue;
                }

                var sourceBid = new BufferIdentity(aliasBid.Node, sourceAccessIndex, BufferEndpoint.Input);
                for (int aliasEntry = 0; aliasEntry < aliasInfo.Places.Length; aliasEntry++)
                {
                    var aliasAtEntry = solver.MakeSum(aliasInfo.Places[aliasEntry]);
                    var visibleSourcePlacements = GetVisibleSourcePlacements(tileNode, sourceBid, aliasEntry);
                    if (visibleSourcePlacements.Length == 0)
                    {
                        throw new SolveFailedException(
                            $"Alias buffer {aliasBid} at {tileNode} has no current or ancestor backing view for {sourceBid}.");
                    }

                    var sourceIsVisible = solver.MakeSum(visibleSourcePlacements);
                    var constraint = solver.MakeLessOrEqual(aliasAtEntry, sourceIsVisible);
                    constraint.SetName($"alias_source_dominates[{tileNode}_{aliasBid}_ci{aliasEntry}]");
                    solver.Add(constraint);
                    aliasConstraints.Add(constraint);
                }
            }

            if (aliasConstraints.Count != 0)
            {
                eachLevelStoreBufferConstrains[tileNode.Level] = eachLevelStoreBufferConstrains[tileNode.Level]
                    .Concat(aliasConstraints)
                    .ToArray();
            }
        }

        IntExpr[] GetVisibleSourcePlacements(TileNode node, BufferIdentity sourceBid, int loopEntry)
        {
            var placements = new List<IntExpr>();
            var currentNode = node;
            var currentBid = sourceBid;
            var isCurrentNode = true;
            while (true)
            {
                var currentInfo = tileNodeMemo[currentNode];
                var storageBid = currentInfo.DefUseMap.TryGetByValue(currentBid, out var producerBid)
                    ? producerBid
                    : currentBid;
                if (currentInfo.BufferInfoMap.TryGetValue(storageBid, out var storageInfo))
                {
                    var visibleEntries = isCurrentNode
                        ? storageInfo.Places.Take(loopEntry + 1)
                        : storageInfo.Places;
                    placements.AddRange(visibleEntries.SelectMany(place => place));
                }

                if (currentNode.Parent is not TileNode parentNode ||
                    parentNode.OpId == -1 ||
                    !tileNodeMemo[parentNode].TryGetByChildBuffer(storageBid, out var parentBid))
                {
                    break;
                }

                currentNode = parentNode;
                currentBid = parentBid;
                isCurrentNode = false;
            }

            return placements.Distinct().ToArray();
        }

        var transferSourceDecisions = new Dictionary<TileBufferPlacement, TileTransferSourceDecision>();
        foreach (var (tileNode, nodeInfo) in tileNodeMemo.Where(pair => pair.Key.ScopeKind == TileScopeKind.Iteration))
        {
            foreach (var (bid, bufferInfo) in nodeInfo.BufferInfoMap)
            {
                var effect = bid.Node.LocalAccessEffects[bid.Index];
                if (bid.IsOutput ||
                    effect.Scope == MemoryAccessScope.Chip ||
                    !MemoryEffectUtility.GetPhysicalBufferAccessMode(effect).HasFlag(MemoryAccessMode.Read))
                {
                    continue;
                }

                for (var loopEntry = 0; loopEntry < bufferInfo.Places.Length; loopEntry++)
                {
                    for (var storageLevel = 0; storageLevel < bufferInfo.Places[loopEntry].Length; storageLevel++)
                    {
                        if (!RequiresLocalAllocation(bid, storageLevel) ||
                            !machine.RequiresExplicitTransfer(storageLevel))
                        {
                            continue;
                        }

                        var destination = new TileBufferPlacement(tileNode, bid, loopEntry, storageLevel);
                        var sourceDecision = CreateTransferSourceDecision(destination);
                        transferSourceDecisions.Add(destination, sourceDecision);
                        var sourceDominates = solver.MakeLessOrEqual(
                            bufferInfo.Places[loopEntry][storageLevel],
                            sourceDecision.SourceIsAvailable);
                        sourceDominates.SetName($"transfer_source_dominates[{tileNode}_{bid}_ci{loopEntry}_sl{storageLevel}]");
                        solver.Add(sourceDominates);
                        eachLevelStoreBufferConstrains[tileNode.Level] = eachLevelStoreBufferConstrains[tileNode.Level]
                            .Append(sourceDominates)
                            .ToArray();
                    }
                }
            }
        }

        TileTransferSourceDecision CreateTransferSourceDecision(TileBufferPlacement destination)
        {
            var sourceMemorySpace = machine.GetTilingParentMemorySpace(destination.StorageLevel).Id;
            var destinationMemorySpace = tilingMemorySpaces[destination.StorageLevel].Id;
            var visible = TileBufferPlacementUtility.EnumerateVisiblePlacementsBefore(destination, tileNodeMemo);
            var matchingSources = new List<TileTransferSourceChoice>();
            IntExpr noNearerView = solver.MakeIntConst(1);
            foreach (var sourcePlacement in visible.Placements)
            {
                var sourceSelected = tileNodeMemo[sourcePlacement.Node]
                    .BufferInfoMap[sourcePlacement.Buffer]
                    .Places[sourcePlacement.LoopEntry][sourcePlacement.StorageLevel];
                var nearest = (noNearerView * sourceSelected).Var();
                nearest.SetName($"nearest_transfer_source[{destination},{sourcePlacement}]");
                nearest.SetRange(0, 1);
                if (tilingMemorySpaces[sourcePlacement.StorageLevel].Id == sourceMemorySpace)
                {
                    matchingSources.Add(new(
                        new SelectedTileBufferPlacementSource(sourcePlacement),
                        nearest));
                }

                noNearerView = (noNearerView * (1 - sourceSelected)).Var();
                noNearerView.SetRange(0, 1);
            }

            if (machine.RootMemorySpace == sourceMemorySpace &&
                TryResolveRootEndpoint(visible.RootEndpoint, out var rootEndpoint))
            {
                matchingSources.Add(new(
                    new SelectedTileBufferRootSource(rootEndpoint),
                    noNearerView));
            }

            var sourceIsAvailable = matchingSources.Count == 0
                ? solver.MakeIntConst(0)
                : solver.MakeSum(matchingSources.Select(source => source.Selected).ToArray()).Var();
            sourceIsAvailable.SetName($"transfer_source_available[{destination},{sourceMemorySpace}]");
            sourceIsAvailable.SetRange(0, 1);
            return new(
                destination,
                sourceMemorySpace,
                destinationMemorySpace,
                matchingSources,
                sourceIsAvailable);
        }

        IntExpr GetMicroKernelTrafficOverride(
            TileGrid grid,
            int accessIndex,
            MemoryAccessMode mode)
        {
            if (!microKernelDecisionsByGrid.TryGetValue(grid, out var decision))
            {
                return solver.MakeIntConst(0);
            }

            var terms = decision.Candidates
                .Select((candidate, candidateIndex) => candidate.MemoryAccesses.Any(access =>
                    access.BufferAccessIndex == accessIndex && access.Mode == mode)
                    ? (IntExpr)decision.SelectionVars[candidateIndex]
                    : solver.MakeIntConst(0))
                .ToArray();
            var result = solver.MakeSum(terms).Var();
            result.SetRange(0, 1);
            return result;
        }

        const int asynchronousStageCount = 2;
        var loopPipelineDecisions = new Dictionary<LoopPipelineKey, LoopPipelineSolverDecision>();
        if (targetOptions.LoopPipelineBackend.SupportsStageCount(asynchronousStageCount))
        {
            foreach (var (tileNode, nodeInfo) in tileNodeMemo.Where(pair => pair.Key.ScopeKind == TileScopeKind.Iteration))
            {
                var regionOpIds = EnumerateOperationNodes(tileNode)
                    .Select(opNode => opNode.Wrapped.RegionOpId)
                    .Distinct()
                    .Order()
                    .ToImmutableArray();
                for (var loopEntry = 1; loopEntry <= tileNode.LoopOrder.Length; loopEntry++)
                {
                    var channels = new List<LoopPipelineChannelDecision>();
                    foreach (var (bid, bufferInfo) in nodeInfo.BufferInfoMap)
                    {
                        var effect = bid.Node.LocalAccessEffects[bid.Index];
                        if (bid.IsOutput ||
                            effect.Scope == MemoryAccessScope.Chip ||
                            !MemoryEffectUtility.GetPhysicalBufferAccessMode(effect).HasFlag(MemoryAccessMode.Read))
                        {
                            continue;
                        }

                        for (var storageLevel = 0; storageLevel <= tileNode.Level; storageLevel++)
                        {
                            if (!RequiresLocalAllocation(bid, storageLevel) ||
                                !machine.RequiresExplicitTransfer(storageLevel))
                            {
                                continue;
                            }

                            var destinationPlacement = new TileBufferPlacement(
                                tileNode,
                                bid,
                                loopEntry,
                                storageLevel);
                            if (!transferSourceDecisions.TryGetValue(destinationPlacement, out var transferSource))
                            {
                                throw new InvalidOperationException(
                                    $"Pipeline destination {destinationPlacement} has no explicit transfer-source decision.");
                            }

                            var destination = machine.GetMemorySpace(transferSource.DestinationMemorySpace);
                            var source = machine.GetMemorySpace(transferSource.SourceMemorySpace);
                            var channelId = $"{TileSemanticNaming.GetBufferEndpointName(bid)}.entry{loopEntry}.{destination.Id}";
                            var backendLegality = targetOptions.LoopPipelineBackend.GetChannelLegality(
                                new(
                                    source,
                                    destination,
                                    bid.Node.BufferDataTypes[bid.Index],
                                    bufferInfo.Shapes[loopEntry].ToImmutableArray(),
                                    bid.Node.BufferShapes[bid.Index]
                                        .Select(extent => (IntExpr)solver.MakeIntConst(extent))
                                        .ToImmutableArray(),
                                    machine,
                                    solver),
                                asynchronousStageCount);
                            var legality = (backendLegality * transferSource.SourceIsAvailable).Var();
                            legality.SetName($"loop_pipeline_channel_legal[{channelId}]");
                            legality.SetRange(0, 1);
                            channels.Add(new(
                                channelId,
                                new(tileNode, bid),
                                bufferInfo,
                                loopEntry,
                                storageLevel,
                                transferSource,
                                bufferInfo.Places[loopEntry][storageLevel].Var(),
                                legality));
                        }
                    }

                    if (channels.Count == 0)
                    {
                        continue;
                    }

                    var serial = solver.MakeBoolVar($"loop_stage_1[{tileNode},entry{loopEntry}]");
                    var asynchronous = solver.MakeBoolVar($"loop_stage_2[{tileNode},entry{loopEntry}]");
                    var exactlyOne = solver.MakeEquality(serial + asynchronous, 1);
                    exactlyOne.SetName($"loop_stage_exactly_one[{tileNode},entry{loopEntry}]");
                    solver.Add(exactlyOne);
                    var hasChannel = solver.MakeLessOrEqual(
                        asynchronous,
                        solver.MakeSum(channels.Select(channel => (IntExpr)channel.Placement).ToArray()));
                    hasChannel.SetName($"loop_stage_2_has_channel[{tileNode},entry{loopEntry}]");
                    solver.Add(hasChannel);
                    foreach (var channel in channels)
                    {
                        var representable = solver.MakeLessOrEqual(
                            asynchronous + channel.Placement,
                            channel.Legality + 1);
                        representable.SetName($"loop_stage_2_channel_legal[{channel.ChannelId}]");
                        solver.Add(representable);
                    }

                    var stageCount = (serial + (asynchronousStageCount * asynchronous)).Var();
                    stageCount.SetName($"loop_stage_count[{tileNode},entry{loopEntry}]");
                    stageCount.SetRange(1, asynchronousStageCount);
                    var key = new LoopPipelineKey(tileNode, loopEntry);
                    loopPipelineDecisions.Add(
                        key,
                        new(
                            key,
                            tileNode.LoopOrder[loopEntry - 1],
                            serial,
                            asynchronous,
                            stageCount,
                            channels,
                            regionOpIds));
                }
            }

            // Two schedules may overlap only when their consumer regions are
            // disjoint. This constraint is derived from lexical region
            // membership, not from a backend candidate's claimed ownership.
            var decisions = loopPipelineDecisions.Values.ToArray();
            for (var left = 0; left < decisions.Length; left++)
            {
                for (var right = left + 1; right < decisions.Length; right++)
                {
                    if (!decisions[left].RegionOpIds.Intersect(decisions[right].RegionOpIds).Any())
                    {
                        continue;
                    }

                    var exclusive = solver.MakeLessOrEqual(
                        decisions[left].AsynchronousSelected + decisions[right].AsynchronousSelected,
                        1);
                    exclusive.SetName($"non_overlapping_loop_pipelines[{decisions[left].Key},{decisions[right].Key}]");
                    solver.Add(exclusive);
                }
            }
        }

        StagedAllocationContext? GetStagedAllocationContext(
            NodeWithBuffer buffer,
            int loopEntry,
            int storageLevel)
        {
            if (!loopPipelineDecisions.TryGetValue(new(buffer.Node, loopEntry), out var decision))
            {
                return null;
            }

            var channels = decision.Channels
                .Where(channel => channel.Buffer == buffer && channel.StorageLevel == storageLevel)
                .ToArray();
            return channels.Length switch
            {
                0 => null,
                1 => new StagedAllocationContext(channels[0].ChannelId, decision.StageCount),
                _ => throw new InvalidOperationException(
                    $"Loop schedule {decision.Key} contains duplicate staging channels for {buffer}/L{storageLevel}."),
            };
        }

        foreach (var privateResource in machine.PrivateResources.Values)
        {
            var times = opLifetimes
                .Where(pair => microKernelDecisions.ContainsKey(pair.Key))
                .SelectMany(pair => Enumerable.Range(pair.Value.FirstPhase, pair.Value.PhaseCount))
                .Distinct()
                .Order()
                .ToArray();
            foreach (var time in times)
            {
                var terms = new List<IntExpr>();
                foreach (var (opNode, decision) in microKernelDecisions)
                {
                    var lifetime = opLifetimes[opNode];
                    if (time < lifetime.FirstPhase || time > lifetime.LastPhase)
                    {
                        continue;
                    }

                    for (var candidateIndex = 0; candidateIndex < decision.Candidates.Count; candidateIndex++)
                    {
                        var usage = decision.Candidates[candidateIndex].Resources
                            .Where(item => item.Resource == privateResource.Id)
                            .Select(item => item.Units)
                            .ToArray();
                        if (usage.Length == 0)
                        {
                            continue;
                        }

                        var allocatedUnits = GetAllocatedPrivateResourceUnits(
                            solver.MakeSum(usage),
                            privateResource,
                            solver);
                        terms.Add(decision.SelectionVars[candidateIndex] * allocatedUnits);
                    }
                }

                if (terms.Count == 0)
                {
                    continue;
                }

                var constraint = solver.MakeLessOrEqual(
                    solver.MakeSum(terms.ToArray()),
                    privateResource.CapacityUnits);
                constraint.SetName($"private_resource_capacity_le[{privateResource.Id},t{time}]");
                solver.Add(constraint);
            }
        }

        // 5. add the memory schedule constraints, each level has own memory plan schedule.
        // 5.1. sum(place[cl,b,ci,sl]*size[cl,b,ci], sl), sl = [0,toplevel)
        var levelBufferSizes = new Dictionary<int, Dictionary<NodeWithBuffer, IntExpr>>();
        var levelBufferShapes = new Dictionary<int, Dictionary<NodeWithBuffer, IntExpr[]>>();
        var levelBufferLifetimes = new Dictionary<int, Dictionary<NodeWithBuffer, TileLifetime>>();
        var levelBufferPlacementInfos = new Dictionary<int, Dictionary<NodeWithBuffer, TileNodeBufferInfo<IntExpr>>>();
        var levelBufferLifetimeConstraints = new Dictionary<int, Constraint[]>();
        var levelOccupancyByTime = new Dictionary<int, SortedDictionary<int, List<IntExpr>>>();
        var managedArenaAllocationBytes = new Dictionary<int, IntExpr>();
        var storageEncodingDecisions = new Dictionary<StorageEncodingPlacementKey, StorageEncodingSolverDecision>();
        IntExpr storageEncodingCycles = solver.MakeIntConst(0);
        for (int sl = 0; sl < levelCount; sl++)
        {
            var nodeBufferSizes = levelBufferSizes[sl] = new();
            var nodeBufferLifetimes = levelBufferLifetimes[sl] = new();
            var nodeBufferShapes = levelBufferShapes[sl] = new();
            var nodeBufferPlacementInfos = levelBufferPlacementInfos[sl] = new();
            var occupancyByTime = new SortedDictionary<int, List<IntExpr>>();
            levelOccupancyByTime.Add(sl, occupancyByTime);

            foreach (var (tileNode, nodeInfo) in tileNodeMemo.Where(kv => kv.Key.Level >= sl))
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
                        IntExpr placedSize;
                        if (requiresLocalAllocation && !IsObjectBuffer(nodeBuffer.Id))
                        {
                            var stagingContext = GetStagedAllocationContext(nodeBuffer, ci, sl);
                            var encodingContext = new TargetStorageEncodingModelContext(
                                tilingMemorySpaces[sl],
                                nodeBuffer.Id.Node.BufferDataTypes[nodeBuffer.Id.Index],
                                bufferInfo.Shapes[ci].ToImmutableArray(),
                                bufferInfo.Sizes[ci],
                                machine,
                                solver)
                            {
                                StagedAllocation = stagingContext,
                            };
                            var encodingCandidates = targetOptions.StorageEncodingModel.GetCandidates(encodingContext);
                            ValidateStorageEncodingCandidates(nodeBuffer, ci, sl, encodingCandidates, machine, stagingContext);
                            var hasIndependentSelectionVars = encodingCandidates.Count > 1;
                            var encodingVars = hasIndependentSelectionVars
                                ? encodingCandidates
                                    .Select((candidate, index) => solver.MakeBoolVar(
                                        $"storage_encoding[{tileNode}_{bid}_ci{ci}_sl{sl},{index},{candidate.Id}]"))
                                    .ToArray()
                                : [place.Var()];
                            if (hasIndependentSelectionVars)
                            {
                                var selectWhenPlaced = solver.MakeEquality(solver.MakeSum(encodingVars), place);
                                selectWhenPlaced.SetName($"storage_encoding_exactly_one_if_placed[{tileNode}_{bid}_ci{ci}_sl{sl}]");
                                solver.Add(selectWhenPlaced);
                            }

                            for (int encodingIndex = 0; encodingIndex < encodingCandidates.Count; encodingIndex++)
                            {
                                var candidate = encodingCandidates[encodingIndex];
                                var legal = solver.MakeLessOrEqual(encodingVars[encodingIndex], candidate.IsLegal);
                                legal.SetName($"storage_encoding_legal[{tileNode}_{bid}_ci{ci}_sl{sl},{candidate.Id}]");
                                solver.Add(legal);
                            }

                            var encodingDecision = new StorageEncodingSolverDecision(
                                encodingCandidates,
                                encodingVars,
                                hasIndependentSelectionVars,
                                stagingContext);
                            var encodingKey = new StorageEncodingPlacementKey(nodeBuffer, ci, sl);
                            if (!storageEncodingDecisions.TryAdd(encodingKey, encodingDecision))
                            {
                                throw new InvalidOperationException($"Duplicate storage encoding decision for {encodingKey}.");
                            }

                            var usesMultipleStages = stagingContext is null
                                ? (IntExpr)solver.MakeIntConst(0)
                                : solver.MakeIsGreaterCstVar(stagingContext.StageCount, 1);
                            placedSize = solver.MakeSum(encodingCandidates
                                .Select((candidate, index) => encodingVars[index] * (stagingContext is null
                                    ? candidate.PhysicalBytes
                                    : (usesMultipleStages * stagingContext.StageCount * candidate.StageStrideBytes!)
                                        + ((1 - usesMultipleStages) * candidate.PhysicalBytes)))
                                .ToArray());
                            storageEncodingCycles += solver.MakeSum(encodingCandidates
                                .Select((candidate, index) => encodingVars[index] * candidate.EstimatedCycles)
                                .ToArray());
                        }
                        else
                        {
                            placedSize = solver.MakeProd(place, bufferInfo.Sizes[ci]);
                        }

                        sizeTerms.Add(placedSize);
                        for (int axis = 0; axis < shapeTerms.Length; axis++)
                        {
                            shapeTerms[axis].Add(solver.MakeProd(place, bufferInfo.Shapes[ci][axis]));
                        }

                        lifetimeStart = Math.Min(lifetimeStart, bufferInfo.Lifetimes[ci].FirstPhase);
                        lifetimeEnd = Math.Max(lifetimeEnd, bufferInfo.Lifetimes[ci].LastPhase);
                        if (requiresLocalAllocation)
                        {
                            for (int time = bufferInfo.Lifetimes[ci].FirstPhase; time <= bufferInfo.Lifetimes[ci].LastPhase; time++)
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
                    nodeBufferLifetimes[nodeBuffer] = new(lifetimeStart, lifetimeEnd);
                    nodeBufferPlacementInfos[nodeBuffer] = bufferInfo;
                }
            }

            // Add constraints according to liveness.
            if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
            {
                DumpGantt(nodeBufferSizes, nodeBufferLifetimes, primTree, sl);
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

            levelBufferLifetimeConstraints.Add(sl, constraints.ToArray());
        }

        // A block microkernel constrains the physical encoding of the operand
        // materialization passed directly to that op, not an upstream staging
        // buffer with the same logical value. This joins microkernel and
        // placement choices without introducing target details into
        // Grid/ParameterInfo.
        foreach (var (opNode, decision) in microKernelDecisions)
        {
            for (int candidateIndex = 0; candidateIndex < decision.Candidates.Count; candidateIndex++)
            {
                var selected = decision.SelectionVars[candidateIndex];
                var candidate = decision.Candidates[candidateIndex];
                foreach (var requirement in candidate.BufferEncodingRequirements)
                {
                    var operandDecisions = GetAccessStorageEncodingDecisions(
                        opNode,
                        requirement.BufferAccessIndex,
                        requirement.MemorySpace);
                    var placementVars = operandDecisions
                        .SelectMany(encodingDecision => encodingDecision.SelectionVars)
                        .ToArray();
                    var hasOperandMaterialization = placementVars.Length == 0
                        ? (IntExpr)solver.MakeIntConst(0)
                        : solver.MakeSum(placementVars);
                    var materialized = solver.MakeLessOrEqual(selected, hasOperandMaterialization);
                    materialized.SetName(
                        $"microkernel_storage_encoding[op{opNode.OpId},{candidateIndex},access{requirement.BufferAccessIndex},{requirement.MemorySpace}]");
                    solver.Add(materialized);

                    for (int placementIndex = 0; placementIndex < operandDecisions.Count; placementIndex++)
                    {
                        var encodingDecision = operandDecisions[placementIndex];
                        var placed = solver.MakeSum(encodingDecision.SelectionVars);
                        var acceptedVars = encodingDecision.Candidates
                            .Select((candidate, encodingIndex) => (candidate, variable: encodingDecision.SelectionVars[encodingIndex]))
                            .Where(item => requirement.AcceptedEncodings.Contains(item.candidate.Id))
                            .Select(item => item.variable)
                            .ToArray();
                        var accepted = acceptedVars.Length == 0
                            ? (IntExpr)solver.MakeIntConst(0)
                            : solver.MakeSum(acceptedVars);
                        var compatibleWhenPlaced = solver.MakeLessOrEqual(selected + placed, accepted + 1);
                        compatibleWhenPlaced.SetName(
                            $"microkernel_storage_encoding_at_placement[op{opNode.OpId},{candidateIndex},access{requirement.BufferAccessIndex},{requirement.MemorySpace},p{placementIndex}]");
                        solver.Add(compatibleWhenPlaced);
                    }
                }
            }
        }

        IReadOnlyList<StorageEncodingSolverDecision> GetAccessStorageEncodingDecisions(
            OpNode opNode,
            int accessIndex,
            TargetMemorySpaceId memorySpace)
        {
            var storageLevel = tilingMemorySpaces
                .Select((space, index) => (space, index))
                .Where(item => item.space.Id == memorySpace)
                .Select(item => item.index)
                .DefaultIfEmpty(-1)
                .Single();
            if (storageLevel < 0)
            {
                throw new InvalidOperationException(
                    $"Block microkernel Op{opNode.OpId} requires non-tiling memory space {memorySpace}.");
            }

            if (opNode.Parent is not TileNode ownerNode)
            {
                throw new InvalidOperationException($"Block microkernel Op{opNode.OpId} has no owning tile scope.");
            }

            var access = opNode.Wrapped.Grid.Accesses[accessIndex];
            var endpoint = access.IsRead ? BufferEndpoint.Input : BufferEndpoint.Output;
            var operandBid = new BufferIdentity(opNode.Wrapped, accessIndex, endpoint);
            var ownerInfo = tileNodeMemo[ownerNode];
            if (!ownerInfo.BufferInfoMap.TryGetValue(operandBid, out var bufferInfo))
            {
                return Array.Empty<StorageEncodingSolverDecision>();
            }

            var result = new List<StorageEncodingSolverDecision>();
            var nodeBuffer = new NodeWithBuffer(ownerNode, operandBid);
            for (int loopEntry = 0; loopEntry < bufferInfo.Places.Length; loopEntry++)
            {
                var key = new StorageEncodingPlacementKey(nodeBuffer, loopEntry, storageLevel);
                if (storageEncodingDecisions.TryGetValue(key, out var encodingDecision))
                {
                    result.Add(encodingDecision);
                }
            }

            return result;
        }

        // TIR buffers are emitted as one static arena per memory level. Model
        // the target allocation policy here, rather than treating phase-local
        // live bytes as the physical allocation seen by the backend.
        for (int level = 0; level < levelCount; level++)
        {
            var memorySpace = tilingMemorySpaces[level];
            if (!machine.PrivateResources.Values.Any(resource => resource.BackingMemoryResource == memorySpace.ResourceId))
            {
                continue;
            }

            var liveByteTotals = levelOccupancyByTime[level].Values
                .Select(occupancy => occupancy.Count == 0
                    ? (IntExpr)solver.MakeIntConst(0)
                    : solver.MakeSum(occupancy.ToArray()))
                .ToArray();
            var requiredBytes = liveByteTotals.Length == 0
                ? solver.MakeIntConst(0)
                : liveByteTotals.Aggregate((IntExpr)solver.MakeIntConst(0), solver.MakeMax);
            var allocationBytes = GetManagedArenaAllocationBytes(machine, memorySpace, requiredBytes, solver);
            managedArenaAllocationBytes.Add(level, allocationBytes);
            foreach (var (time, liveBytes) in levelOccupancyByTime[level].Keys.Zip(liveByteTotals))
            {
                var constraint = solver.MakeLessOrEqual(liveBytes, allocationBytes);
                constraint.SetName($"managed_arena_capacity_le[{memorySpace.Id},t{time}]");
                solver.Add(constraint);
            }
        }

        // Backend-private storage and compiler-managed TIR arenas contend for
        // the same physical resource. The TIR arena is a static allocation;
        // backend-private storage remains phase-local and may be reused.
        foreach (var memoryResource in machine.MemoryResources.Values)
        {
            var backedPrivateResources = machine.PrivateResources.Values
                .Where(resource => resource.BackingMemoryResource == memoryResource.Id)
                .ToArray();
            if (backedPrivateResources.Length == 0)
            {
                continue;
            }

            var times = levelOccupancyByTime
                .Where(pair => tilingMemorySpaces[pair.Key].ResourceId == memoryResource.Id)
                .SelectMany(pair => pair.Value.Keys)
                .Concat(opLifetimes
                    .Where(pair => microKernelDecisions.ContainsKey(pair.Key))
                    .SelectMany(pair => Enumerable.Range(pair.Value.FirstPhase, pair.Value.PhaseCount)))
                .Distinct()
                .Order()
                .ToArray();
            foreach (var time in times)
            {
                var terms = new List<IntExpr>();
                foreach (var (level, allocationBytes) in managedArenaAllocationBytes)
                {
                    if (tilingMemorySpaces[level].ResourceId == memoryResource.Id)
                    {
                        terms.Add(allocationBytes);
                    }
                }

                foreach (var (opNode, decision) in microKernelDecisions)
                {
                    var lifetime = opLifetimes[opNode];
                    if (time < lifetime.FirstPhase || time > lifetime.LastPhase)
                    {
                        continue;
                    }

                    for (var candidateIndex = 0; candidateIndex < decision.Candidates.Count; candidateIndex++)
                    {
                        var selected = decision.SelectionVars[candidateIndex];
                        foreach (var usage in decision.Candidates[candidateIndex].Resources)
                        {
                            if (backedPrivateResources.Any(resource => resource.Id == usage.Resource))
                            {
                                var resource = machine.GetPrivateResource(usage.Resource);
                                terms.Add(selected * GetAllocatedPrivateResourceUnits(usage.Units, resource, solver));
                            }
                        }
                    }
                }

                if (terms.Count == 0)
                {
                    continue;
                }

                var constraint = solver.MakeLessOrEqual(solver.MakeSum(terms.ToArray()), memoryResource.CapacityBytes);
                constraint.SetName($"physical_resource_capacity_le[{memoryResource.Id},t{time}]");
                solver.Add(constraint);
            }
        }

        // Materialization traffic moves a TIR buffer between memory levels.
        // Execution traffic is what an operation consumes after target-local
        // reuse. A target microkernel may replace the generic execution term
        // for a semantic access without changing materialization traffic.
        var memoryLevelCount = levelCount + 1;
        var levelMaterializationReads = Enumerable.Range(0, memoryLevelCount).Select(_ => (IntExpr)solver.MakeIntConst(0)).ToArray();
        var levelMaterializationWrites = Enumerable.Range(0, memoryLevelCount).Select(_ => (IntExpr)solver.MakeIntConst(0)).ToArray();
        var levelExecutionReads = Enumerable.Range(0, memoryLevelCount).Select(_ => (IntExpr)solver.MakeIntConst(0)).ToArray();
        var levelExecutionWrites = Enumerable.Range(0, memoryLevelCount).Select(_ => (IntExpr)solver.MakeIntConst(0)).ToArray();
        var levelTransferReads = Enumerable.Range(0, levelCount).Select(i => (IntExpr)solver.MakeIntConst(0)).ToArray();
        var levelTransferWrites = Enumerable.Range(0, levelCount).Select(i => (IntExpr)solver.MakeIntConst(0)).ToArray();
        var levelTransferReadSynchronizationVolumes = Enumerable.Range(0, levelCount).Select(_ => (IntExpr)solver.MakeIntConst(0)).ToArray();
        var levelTransferWriteSynchronizationVolumes = Enumerable.Range(0, levelCount).Select(_ => (IntExpr)solver.MakeIntConst(0)).ToArray();
        var microKernelMemoryReads = machine.MemoryResources.Keys.ToDictionary(
            resource => resource,
            _ => (IntExpr)solver.MakeIntConst(0));
        var microKernelMemoryWrites = machine.MemoryResources.Keys.ToDictionary(
            resource => resource,
            _ => (IntExpr)solver.MakeIntConst(0));
        foreach (var decision in microKernelDecisions.Values)
        {
            for (var candidateIndex = 0; candidateIndex < decision.Candidates.Count; candidateIndex++)
            {
                var selected = decision.SelectionVars[candidateIndex];
                foreach (var access in decision.Candidates[candidateIndex].MemoryAccesses)
                {
                    var traffic = selected * access.Bytes;
                    if (access.Mode == MemoryAccessMode.Read)
                    {
                        microKernelMemoryReads[access.Resource] += traffic;
                    }
                    else
                    {
                        microKernelMemoryWrites[access.Resource] += traffic;
                    }
                }
            }
        }

        IntExpr synchronizationCycles = solver.MakeIntConst(0);
        foreach (var (tileNode, nodeInfo) in tileNodeMemo)
        {
            var nodeMaterializationWrites = Enumerable.Range(0, memoryLevelCount).Select(_ => new List<IntExpr>()).ToArray();
            var nodeMaterializationReads = Enumerable.Range(0, memoryLevelCount).Select(_ => new List<IntExpr>()).ToArray();
            var nodeExecutionWrites = Enumerable.Range(0, memoryLevelCount).Select(_ => new List<IntExpr>()).ToArray();
            var nodeExecutionReads = Enumerable.Range(0, memoryLevelCount).Select(_ => new List<IntExpr>()).ToArray();
            var nodeTransferReads = Enumerable.Range(0, levelCount).Select(_ => new List<IntExpr>()).ToArray();
            var nodeTransferWrites = Enumerable.Range(0, levelCount).Select(_ => new List<IntExpr>()).ToArray();
            var nodeTransferReadSynchronizationVolumes = Enumerable.Range(0, levelCount).Select(_ => new List<IntExpr>()).ToArray();
            var nodeTransferWriteSynchronizationVolumes = Enumerable.Range(0, levelCount).Select(_ => new List<IntExpr>()).ToArray();
            foreach (var (bid, bufferInfo) in nodeInfo.BufferInfoMap)
            {
                var reused = nodeInfo.DefUseMap.ContainsKey(bid);
                var localEffect = bid.Node.LocalAccessEffects[bid.Index];
                var isReductionAccumulator = localEffect.Kind == MemoryEffectKind.ReductionAccumulator;
                if (isReductionAccumulator && (!bid.IsOutput || !localEffect.Mode.HasFlag(MemoryAccessMode.Write)))
                {
                    throw new InvalidOperationException(
                        $"Reduction accumulator {bid} must be a writable output, got {localEffect}.");
                }

                var physicalAccessMode = MemoryEffectUtility.GetPhysicalBufferAccessMode(localEffect);

                for (int sl = 0; sl <= tileNode.Level; sl++)
                {
                    var requiresLocalAllocation = RequiresLocalAllocation(bid, sl);
                    for (int ci = 0; ci < bufferInfo.Places.Length; ci++)
                    {
                        var volume = bufferInfo.Places[ci][sl] * bufferInfo.TransferBytes[ci];
                        var executionLevel = requiresLocalAllocation ? sl : sl + 1;

                        // A ReductionAccumulator read is feedback through backend-private
                        // state, not a physical buffer access for every reduction tile.
                        // Only its final write is committed to the selected memory space.
                        if (physicalAccessMode.HasFlag(MemoryAccessMode.Read))
                        {
                            var overrideTraffic = GetMicroKernelTrafficOverride(
                                bid.Node,
                                bid.Index,
                                MemoryAccessMode.Read);
                            nodeExecutionReads[executionLevel].Add((1 - overrideTraffic) * volume);
                            if (requiresLocalAllocation && !reused)
                            {
                                nodeMaterializationWrites[sl].Add(volume);
                                nodeMaterializationReads[sl + 1].Add(volume);
                                nodeTransferReads[sl].Add(volume);
                                nodeTransferReadSynchronizationVolumes[sl].Add(volume);
                            }
                        }

                        if (physicalAccessMode.HasFlag(MemoryAccessMode.Write))
                        {
                            var overrideTraffic = GetMicroKernelTrafficOverride(
                                bid.Node,
                                bid.Index,
                                MemoryAccessMode.Write);
                            nodeExecutionWrites[executionLevel].Add((1 - overrideTraffic) * volume);
                            if (reused)
                            {
                                if (!nodeInfo.DefUseMap.TryGetByKey(bid, out var consumers) || consumers.Count == 0)
                                {
                                    throw new InvalidOperationException(
                                        $"Reused tile buffer {bid} has no consumer endpoint.");
                                }

                                foreach (var consumer in consumers)
                                {
                                    var consumerOverride = GetMicroKernelTrafficOverride(
                                        consumer.Node,
                                        consumer.Index,
                                        MemoryAccessMode.Read);
                                    nodeExecutionReads[executionLevel].Add((1 - consumerOverride) * volume);
                                }
                            }
                            else if (requiresLocalAllocation)
                            {
                                nodeMaterializationReads[sl].Add(volume);
                                nodeMaterializationWrites[sl + 1].Add(volume);
                                nodeTransferWrites[sl].Add(volume);
                                nodeTransferWriteSynchronizationVolumes[sl].Add(volume);
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
                if (nodeMaterializationWrites[l].Any())
                {
                    levelMaterializationWrites[l] += solver.MakeSum(nodeMaterializationWrites[l]);
                }

                if (nodeMaterializationReads[l].Any())
                {
                    levelMaterializationReads[l] += solver.MakeSum(nodeMaterializationReads[l]);
                }

                if (nodeExecutionWrites[l].Any())
                {
                    levelExecutionWrites[l] += solver.MakeSum(nodeExecutionWrites[l]);
                }

                if (nodeExecutionReads[l].Any())
                {
                    levelExecutionReads[l] += solver.MakeSum(nodeExecutionReads[l]);
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

                if (nodeTransferReadSynchronizationVolumes[l].Any())
                {
                    levelTransferReadSynchronizationVolumes[l] += solver.MakeSum(nodeTransferReadSynchronizationVolumes[l]);
                }

                if (nodeTransferWriteSynchronizationVolumes[l].Any())
                {
                    levelTransferWriteSynchronizationVolumes[l] += solver.MakeSum(nodeTransferWriteSynchronizationVolumes[l]);
                }
            }
        }

        var levelDataReads = levelMaterializationReads
            .Zip(levelExecutionReads, (materialization, execution) => materialization + execution)
            .ToArray();
        var levelDataWrites = levelMaterializationWrites
            .Zip(levelExecutionWrites, (materialization, execution) => materialization + execution)
            .ToArray();

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
            var reads = solver.MakeSum(group.Select(item => levelDataReads[item.level]).ToArray())
                + microKernelMemoryReads[group.Key];
            var writes = solver.MakeSum(group.Select(item => levelDataWrites[item.level]).ToArray())
                + microKernelMemoryWrites[group.Key];
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
            var readSynchronizationVolume = levelTransferReadSynchronizationVolumes[i];
            var writeSynchronizationVolume = levelTransferWriteSynchronizationVolumes[i];
            if (parentMemorySpace.ResourceId == localMemorySpace.ResourceId)
            {
                transferCycles[i] = solver.MakeIntConst(0);
                continue;
            }

            if (parentMemorySpace.Scope == MemorySharingScope.Chip || localMemorySpace.Scope == MemorySharingScope.Chip)
            {
                reads *= activeBlockCount;
                writes *= activeBlockCount;
                readSynchronizationVolume *= activeBlockCount;
                writeSynchronizationVolume *= activeBlockCount;
            }

            var readEvent = solver.MakeIsGreaterCstVar(reads, 0);
            var writeEvent = solver.MakeIsGreaterCstVar(writes, 0);
            var readSynchronizationEvent = solver.MakeIsGreaterCstVar(readSynchronizationVolume, 0);
            var writeSynchronizationEvent = solver.MakeIsGreaterCstVar(writeSynchronizationVolume, 0);
            transferCycles[i] = reads.CeilDiv(readTransfer.BytesPerCycle)
                + writes.CeilDiv(writeTransfer.BytesPerCycle)
                + (readEvent * readTransfer.LatencyCycles)
                + (writeEvent * writeTransfer.LatencyCycles);
            if (readTransfer.RequiresSynchronization)
            {
                synchronizationCycles += readSynchronizationEvent * machine.Synchronization.BlockCycles;
            }

            if (writeTransfer.RequiresSynchronization)
            {
                synchronizationCycles += writeSynchronizationEvent * machine.Synchronization.BlockCycles;
            }
        }

        IntExpr computeCycles = solver.MakeIntConst(0);
        foreach (var (opNode, _) in opNodeMemo)
        {
            computeCycles += microKernelDecisions.TryGetValue(opNode, out var decision)
                ? decision.SelectedRegionCycles
                : baseComputeCyclesByOp[opNode];
        }

        // Stage one is a serial loop schedule. Only a selected stage-two loop
        // is allowed to replace P + C by the template's fill/steady/drain
        // estimate. The decision is built from the same loop entry, transfer
        // placements, and operation region that will construct PipelineFor.
        IntExpr pipelineOverlapSavings = solver.MakeIntConst(0);
        foreach (var pipelineDecision in loopPipelineDecisions.Values)
        {
            var totalIterationCount = pipelineDecision.Channels[0].BufferInfo.Trips[pipelineDecision.Key.LoopEntry];
            var invocationCount = pipelineDecision.Channels[0].BufferInfo.Trips[pipelineDecision.Key.LoopEntry - 1];
            var iterationCount = solver.MakeDiv(totalIterationCount, invocationCount).Var();
            iterationCount.SetName($"loop_iteration_count[{pipelineDecision.Key}]");
            iterationCount.SetRange(1, totalIterationCount.Var().Max());

            var hasTwoIterations = solver.MakeIsGreaterCstVar(iterationCount, 1);
            var enoughIterations = solver.MakeLessOrEqual(
                pipelineDecision.AsynchronousSelected,
                hasTwoIterations);
            enoughIterations.SetName($"loop_stage_2_requires_two_iterations[{pipelineDecision.Key}]");
            solver.Add(enoughIterations);

            var producerRegionCycles = solver.MakeSum(pipelineDecision.Channels
                .Select(channel =>
                {
                    var source = machine.GetMemorySpace(channel.SourceMemorySpace);
                    var destination = machine.GetMemorySpace(channel.DestinationMemorySpace);
                    var transfer = machine.GetTransfer(source.Id, destination.Id);
                    var contentionFactor = source.Scope == MemorySharingScope.Chip ||
                        destination.Scope == MemorySharingScope.Chip
                        ? activeBlockCount
                        : 1;
                    var bytes = channel.Placement * channel.BufferInfo.TransferBytes[channel.LoopEntry];
                    var events = channel.Placement * channel.BufferInfo.Trips[channel.LoopEntry];
                    return bytes.ScaleAndCeilDiv(contentionFactor, transfer.BytesPerCycle)
                        + (events * transfer.LatencyCycles);
                })
                .ToArray());
            var producerCycles = producerRegionCycles.CeilDiv(totalIterationCount).Var();
            producerCycles.SetName($"loop_producer_cycles[{pipelineDecision.Key}]");
            producerCycles.SetRange(0, producerRegionCycles.Var().Max());

            var consumerRegionCycles = solver.MakeSum(EnumerateOperationNodes(pipelineDecision.Key.Node)
                .Select(opNode => microKernelDecisions.TryGetValue(opNode, out var microKernelDecision)
                    ? microKernelDecision.SelectedRegionCycles
                    : baseComputeCyclesByOp[opNode])
                .ToArray());
            var consumerCycles = consumerRegionCycles.CeilDiv(totalIterationCount).Var();
            consumerCycles.SetName($"loop_consumer_cycles[{pipelineDecision.Key}]");
            consumerCycles.SetRange(0, consumerRegionCycles.Var().Max());

            var template = targetOptions.LoopPipelineBackend.GetTemplate(asynchronousStageCount, machine);
            var controlCosts = pipelineDecision.Channels
                .Select(channel => template.Synchronization.GetControlCost(
                    machine,
                    machine.GetTransfer(channel.SourceMemorySpace, channel.DestinationMemorySpace)))
                .ToArray();
            var estimate = LoopPipelineScheduleEstimate.Create(
                solver,
                iterationCount,
                invocationCount,
                producerCycles,
                consumerCycles,
                solver.MakeIntConst(controlCosts.Max(cost => cost.ProducerCommitCycles)),
                solver.MakeIntConst(controlCosts.Max(cost => cost.ConsumerWaitAcquireCycles)),
                solver.MakeIntConst(controlCosts.Max(cost => cost.ConsumerReleaseCycles)));
            pipelineDecision.Estimate = estimate;
            var savings = solver.MakeMax(
                solver.MakeIntConst(0),
                estimate.SerialRegionCycles - estimate.PipelinedRegionCycles);
            pipelineOverlapSavings += pipelineDecision.AsynchronousSelected * savings;
        }

        var serialCycles = computeCycles
            + solver.MakeSum(memoryCycles)
            + solver.MakeSum(transferCycles)
            + synchronizationCycles
            + storageEncodingCycles;
        var totalCycles = solver.MakeMax(solver.MakeIntConst(0), serialCycles - pipelineOverlapSavings);

        // Predicted latency is the primary objective. Target-owned candidate
        // priority is a strict secondary objective used only when two complete
        // schedules have exactly the same predicted latency. Scalarization is
        // exact because the multiplier is greater than the largest possible
        // aggregate priority in this solve.
        var maximumMicroKernelSelectionPriority = microKernelDecisions.Values.Aggregate(
            0L,
            (sum, decision) => checked(sum + decision.Candidates.Max(candidate => (long)candidate.SelectionPriority)));
        var maximumStorageEncodingSelectionPriority = storageEncodingDecisions.Values.Aggregate(
            0L,
            (sum, decision) => checked(sum + decision.Candidates.Max(candidate => (long)candidate.SelectionPriority)));
        var maximumPipelineSelectionPriority = loopPipelineDecisions.Count;
        var maximumSelectionPriority = checked(
            maximumMicroKernelSelectionPriority +
            maximumStorageEncodingSelectionPriority +
            maximumPipelineSelectionPriority);
        var microKernelSelectionPriorityTerms = microKernelDecisions.Values
            .SelectMany(decision => decision.Candidates.Select(
                (candidate, index) => (IntExpr)(decision.SelectionVars[index] * candidate.SelectionPriority)))
            .ToArray();
        var storageEncodingSelectionPriorityTerms = storageEncodingDecisions.Values
            .SelectMany(decision => decision.Candidates.Select(
                (candidate, index) => (IntExpr)(decision.SelectionVars[index] * candidate.SelectionPriority)))
            .ToArray();
        var pipelineSelectionPriorityTerms = loopPipelineDecisions.Values
            .Select(decision => (IntExpr)decision.AsynchronousSelected)
            .ToArray();
        var selectionPriorityTerms = microKernelSelectionPriorityTerms
            .Concat(storageEncodingSelectionPriorityTerms)
            .Concat(pipelineSelectionPriorityTerms)
            .ToArray();
        var selectionPriority = (selectionPriorityTerms.Length == 0
            ? solver.MakeIntConst(0)
            : solver.MakeSum(selectionPriorityTerms)).Var();
        selectionPriority.SetRange(0, maximumSelectionPriority);
        var objectiveScale = checked(maximumSelectionPriority + 1);

        var totalCyclesVar = totalCycles.Var();
        var arithmeticCycleUpperBound = (long.MaxValue - maximumSelectionPriority) / objectiveScale;
        var memoryCycleUpperBound = long.MaxValue /
            memoryResourceGroups.Min(group => machine.GetMemoryResource(group.Key).ReadBytesPerCycle);
        totalCyclesVar.SetRange(0, Math.Min(arithmeticCycleUpperBound, memoryCycleUpperBound));
        var optimizationObjective = ((totalCyclesVar * objectiveScale) + selectionPriority).Var();
        var objectiveMonitor = solver.MakeMinimize(optimizationObjective, 1);
        var collector = solver.MakeNBestValueSolutionCollector(5, false);
        collector.AddObjective(optimizationObjective);
        collector.Add(optimizationObjective);
        collector.Add(totalCyclesVar);
        collector.Add(selectionPriority);
        collector.Add(levelDataReads.Select(i => i.Var()).ToArray());
        collector.Add(levelDataWrites.Select(i => i.Var()).ToArray());
        collector.Add(levelMaterializationReads.Select(i => i.Var()).ToArray());
        collector.Add(levelMaterializationWrites.Select(i => i.Var()).ToArray());
        collector.Add(levelExecutionReads.Select(i => i.Var()).ToArray());
        collector.Add(levelExecutionWrites.Select(i => i.Var()).ToArray());
        collector.Add(microKernelMemoryReads.Values.Select(i => i.Var()).ToArray());
        collector.Add(microKernelMemoryWrites.Values.Select(i => i.Var()).ToArray());
        collector.Add(levelTransferReads.Select(i => i.Var()).ToArray());
        collector.Add(levelTransferWrites.Select(i => i.Var()).ToArray());
        collector.Add(computeCycles.Var());
        collector.Add(pipelineOverlapSavings.Var());
        collector.Add(synchronizationCycles.Var());
        collector.Add(storageEncodingCycles.Var());
        collector.Add(memoryCycles.Select(i => i.Var()).ToArray());
        collector.Add(transferCycles.Select(i => i.Var()).ToArray());
        collector.Add(reductionStateBytes.Values.Select(value => value.Var()).ToArray());
        collector.Add(managedArenaAllocationBytes.Values.Select(value => value.Var()).ToArray());

        var searchAbleVars = new List<IntVar>();
        foreach (var decision in storageEncodingDecisions.Values)
        {
            if (decision.HasIndependentSelectionVars)
            {
                searchAbleVars.AddRange(decision.SelectionVars);
            }

            collector.Add(decision.SelectionVars);
            collector.Add(decision.Candidates.Select(candidate => candidate.PhysicalBytes.Var()).ToArray());
            collector.Add(decision.Candidates
                .Where(candidate => candidate.StageStrideBytes is not null)
                .Select(candidate => candidate.StageStrideBytes!.Var())
                .ToArray());
            if (decision.StagedAllocation is { } stagedAllocation)
            {
                collector.Add(stagedAllocation.StageCount.Var());
            }

            collector.Add(decision.Candidates.Select(candidate => candidate.EstimatedCycles.Var()).ToArray());
            collector.Add(decision.Candidates
                .SelectMany(candidate => candidate.Parameters)
                .Select(parameter => parameter.Value.Var())
                .ToArray());
        }

        foreach (var decision in microKernelDecisions.Values)
        {
            searchAbleVars.AddRange(decision.SelectionVars);
            collector.Add(decision.SelectionVars);
            collector.Add(decision.Candidates.Select(candidate => candidate.ExecutionCost.RegionCycles.Var()).ToArray());
            collector.Add(decision.Candidates
                .SelectMany(candidate => candidate.Resources)
                .Select(resource => resource.Units.Var())
                .ToArray());
            collector.Add(decision.Candidates
                .SelectMany(candidate => candidate.MemoryAccesses)
                .Select(access => access.Bytes.Var())
                .ToArray());
            collector.Add(decision.Candidates
                .SelectMany(candidate => candidate.Parameters)
                .Select(parameter => parameter.Value.Var())
                .ToArray());
        }

        foreach (var decision in transferSourceDecisions.Values)
        {
            collector.Add(decision.SourceIsAvailable.Var());
            collector.Add(decision.Sources.Select(source => source.Selected.Var()).ToArray());
        }

        foreach (var decision in loopPipelineDecisions.Values)
        {
            searchAbleVars.Add(decision.SerialSelected);
            searchAbleVars.Add(decision.AsynchronousSelected);
            collector.Add(decision.SerialSelected);
            collector.Add(decision.AsynchronousSelected);
            collector.Add(decision.StageCount);
            collector.Add(decision.Channels.Select(channel => channel.Legality.Var()).ToArray());
            if (decision.Estimate is not { } estimate)
            {
                throw new InvalidOperationException($"Loop pipeline {decision.Key} has no schedule estimate.");
            }

            collector.Add(new[]
            {
                estimate.IterationCount.Var(),
                estimate.InvocationCount.Var(),
                estimate.ProducerCycles.Var(),
                estimate.ConsumerCycles.Var(),
                estimate.InitiationIntervalCycles.Var(),
                estimate.SerialRegionCycles.Var(),
                estimate.PipelinedRegionCycles.Var(),
            });
        }

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
                collector.Add(bufferInfo.TransferBytes.Where(v => v is not null).Select(i => i.Var()).ToArray());
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

        foreach (var (_, v) in levelBufferLifetimeConstraints)
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
                    .SelectMany(candidate => candidate.Info.Places.SelectMany(place => place.Select(value => value.Var()))))
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
            monitors.Add(solver.MakeSearchLog(10000, optimizationObjective));
        }

        var status = solver.Solve(decisionBuilder, monitors.ToArray());
        if (!status)
        {
            DumpAssgin(primTree, new TreeSolverPrinter(null, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, targetOptions), coverageConstraints, reductionStateBytes, reductionStateConstraints, new Dictionary<int, BlockMicroKernelSelection>(), eachLevelStoreBufferConstrains, levelBufferLifetimeConstraints, levelBufferSizes, levelDataReads, levelDataWrites, levelTransferReads, levelTransferWrites, memoryCycles, transferCycles, computeCycles, synchronizationCycles, totalCyclesVar);
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
                    ? placementInfo.Lifetimes[selectedPositions[0]]
                    : new TileLifetime(0, 0);
                var shapes = levelBufferShapes[level][nodeBuffer].Select(s => sol.Value(s.Var())).ToArray();
                var strides = TensorUtilities.GetDefaultStrides(shapes);
                TargetStorageEncodingSelection? storageEncoding = null;
                StagedBufferLayout? stagedLayout = null;
                var alignment = checked((int)Math.Max(1, nodeBuffer.Id.Node.GetBufferElemSize(nodeBuffer.Id.Index)));
                if (selectedPositions.Length == 1 && sol.Value(sizeVar.Var()) > 0)
                {
                    var encodingKey = new StorageEncodingPlacementKey(nodeBuffer, selectedPositions[0], level);
                    if (!storageEncodingDecisions.TryGetValue(encodingKey, out var encodingDecision))
                    {
                        throw new InvalidOperationException(
                            $"Allocated tile buffer {nodeBuffer} at L{level}/entry{selectedPositions[0]} has no storage encoding decision.");
                    }

                    var selectedEncodingIndices = encodingDecision.SelectionVars
                        .Select((variable, index) => (index, selected: sol.Value(variable)))
                        .Where(item => item.selected == 1)
                        .Select(item => item.index)
                        .ToArray();
                    if (selectedEncodingIndices.Length != 1)
                    {
                        throw new InvalidOperationException(
                            $"Allocated tile buffer {nodeBuffer} at L{level}/entry{selectedPositions[0]} must select exactly one " +
                            $"storage encoding, got {selectedEncodingIndices.Length}.");
                    }

                    var candidate = encodingDecision.Candidates[selectedEncodingIndices[0]];
                    var stagePhysicalBytes = sol.Value(candidate.PhysicalBytes.Var());
                    var selectedSize = sol.Value(sizeVar.Var());
                    var expectedSize = stagePhysicalBytes;
                    long? stageStrideBytes = null;
                    int? stageCount = null;
                    if (encodingDecision.StagedAllocation is { } stagedAllocation)
                    {
                        stageCount = checked((int)sol.Value(stagedAllocation.StageCount.Var()));
                        stageStrideBytes = sol.Value(candidate.StageStrideBytes!.Var());
                        expectedSize = stageCount.Value > 1
                            ? checked(stageCount.Value * stageStrideBytes.Value)
                            : stagePhysicalBytes;
                    }

                    if (expectedSize != selectedSize)
                    {
                        throw new InvalidOperationException(
                            $"Storage encoding {candidate.Id} selected {expectedSize} bytes for {nodeBuffer}, " +
                            $"but the scheduled buffer size is {selectedSize} bytes.");
                    }

                    storageEncoding = new TargetStorageEncodingSelection(
                        candidate.Id,
                        stagePhysicalBytes,
                        candidate.AlignmentBytes,
                        candidate.Parameters.Select(parameter => new KeyValuePair<string, long>(
                            parameter.Name,
                            sol.Value(parameter.Value.Var()))));
                    if (stageCount is { } concreteStageCount && concreteStageCount > 1 && stageStrideBytes is { } concreteStageStride)
                    {
                        stagedLayout = storageEncoding.CreateStagedBufferLayout(concreteStageCount, concreteStageStride);
                    }

                    alignment = candidate.AlignmentBytes;
                }

                nodeBufferInfos[nodeBuffer] = new NodeWithBufferInfo(
                    sol.Value(sizeVar.Var()),
                    liveness,
                    shapes,
                    strides,
                    alignment,
                    storageEncoding,
                    stagedLayout);
            }

            levelBufferInfos[level] = nodeBufferInfos;
        }

        var opNodeMemoAssgin = opNodeMemo.ToDictionary(kv => kv.Key, kv => new OpNodeInfo<long>(kv.Value.Maps, sol.Value(kv.Value.Shapes), sol.Value(kv.Value.Sizes)));
        var tileNodeMemoAssgin = tileNodeMemo.ToDictionary(kv => kv.Key, kv => new TileNodeInfo<long>(sol.Value(kv.Value.TripCounts), sol.Value(kv.Value.BackWardExtents), kv.Value.DefUseMap, kv.Value.BufferInfoMap.ToDictionary(p => p.Key, p => new TileNodeBufferInfo<long>(p.Value.Lifetimes, p.Value.Map, sol.Value(p.Value.Places), sol.Value(p.Value.Shapes), sol.Value(p.Value.Sizes), sol.Value(p.Value.Trips), sol.Value(p.Value.TransferBytes), p.Value.Mask))));
        var tileableNodeMemoAssgin = tileableNodeMemo.ToDictionary(kv => kv.Key, kv => new DomainInfo<long>(sol.Value(kv.Value.TileVars), sol.Value(kv.Value.ForwardExtents), kv.Value.DimsMap));
        var selectedMicroKernels = new Dictionary<int, BlockMicroKernelSelection>();
        foreach (var (opNode, decision) in microKernelDecisions)
        {
            var selectedIndices = decision.SelectionVars
                .Select((variable, index) => (index, selected: sol.Value(variable)))
                .Where(item => item.selected == 1)
                .Select(item => item.index)
                .ToArray();
            if (selectedIndices.Length != 1)
            {
                throw new InvalidOperationException(
                    $"Op{opNode.OpId} must select exactly one block microkernel, got {selectedIndices.Length}.");
            }

            var candidate = decision.Candidates[selectedIndices[0]];
            var parameters = candidate.Parameters.ToImmutableDictionary(
                parameter => parameter.Name,
                parameter => sol.Value(parameter.Value.Var()),
                StringComparer.Ordinal);
            var resources = candidate.Resources
                .GroupBy(usage => usage.Resource)
                .ToImmutableDictionary(
                    group => group.Key.Value,
                    group =>
                    {
                        var resource = machine.GetPrivateResource(group.Key);
                        var units = group.Sum(usage => sol.Value(usage.Units.Var()));
                        return AlignUp(units, resource.AllocationGranularityUnits);
                    },
                    StringComparer.Ordinal);
            var selection = new BlockMicroKernelSelection(
                candidate.Family,
                candidate.Variant,
                sol.Value(candidate.ExecutionCost.RegionCycles.Var()),
                resources,
                parameters);
            if (!selectedMicroKernels.TryAdd(opNode.Wrapped.RegionOpId, selection))
            {
                throw new InvalidOperationException(
                    $"Scheduled region contains multiple block microkernel decisions for source Op{opNode.Wrapped.RegionOpId}.");
            }
        }

        var selectedTransferSources = new Dictionary<TileBufferPlacement, SelectedTileBufferSource>();
        foreach (var decision in transferSourceDecisions.Values)
        {
            var destinationSelected = tileNodeMemo[decision.Destination.Node]
                .BufferInfoMap[decision.Destination.Buffer]
                .Places[decision.Destination.LoopEntry][decision.Destination.StorageLevel];
            if (sol.Value(destinationSelected.Var()) == 0)
            {
                continue;
            }

            var sources = decision.Sources
                .Where(source => sol.Value(source.Selected.Var()) == 1)
                .ToArray();
            if (sources.Length != 1)
            {
                throw new InvalidOperationException(
                    $"Selected transfer destination {decision.Destination} requires exactly one " +
                    $"{decision.SourceMemorySpace} source, got {sources.Length}.");
            }

            selectedTransferSources.Add(decision.Destination, sources[0].Source);
        }

        var selectedLoopPipelines = new Dictionary<TileNode, SelectedLoopPipeline>();
        foreach (var decision in loopPipelineDecisions.Values)
        {
            if (sol.Value(decision.AsynchronousSelected) == 0)
            {
                continue;
            }

            var channels = decision.Channels
                .Where(channel => sol.Value(channel.Placement) == 1)
                .Select(channel =>
                {
                    var destination = new TileBufferPlacement(
                        decision.Key.Node,
                        channel.Buffer.Id,
                        channel.LoopEntry,
                        channel.StorageLevel);
                    if (!selectedTransferSources.TryGetValue(destination, out var source))
                    {
                        throw new InvalidOperationException(
                            $"Selected pipeline channel {channel.ChannelId} has no solved transfer source.");
                    }

                    return new SelectedLoopPipelineChannel(
                        channel.ChannelId,
                        channel.Buffer,
                        channel.StorageLevel,
                        source,
                        channel.SourceMemorySpace,
                        channel.DestinationMemorySpace);
                })
                .ToImmutableArray();
            if (channels.IsDefaultOrEmpty)
            {
                throw new InvalidOperationException(
                    $"Selected loop pipeline {decision.Key} has no staged transfer channel.");
            }

            var template = targetOptions.LoopPipelineBackend.GetTemplate(asynchronousStageCount, machine);
            var plan = new PipelineRegionPlan(
                $"{template.Id}.axis{decision.DomainAxis}.entry{decision.Key.LoopEntry}",
                template.Id,
                template.Synchronization,
                asynchronousStageCount,
                prefetchDistance: 1,
                PipelineTailPolicy.Serial,
                channels.Select(channel => new PipelineStageChannelPlan(
                    channel.ChannelId,
                    channel.SourceMemorySpace,
                    channel.DestinationMemorySpace)));
            var estimate = decision.Estimate
                ?? throw new InvalidOperationException($"Selected loop pipeline {decision.Key} has no cost estimate.");
            var selection = new SelectedLoopPipeline(
                decision.Key.LoopEntry,
                decision.DomainAxis,
                plan,
                new SelectedLoopPipelineScheduleEstimate(
                    sol.Value(estimate.IterationCount.Var()),
                    sol.Value(estimate.InvocationCount.Var()),
                    sol.Value(estimate.ProducerCycles.Var()),
                    sol.Value(estimate.ConsumerCycles.Var()),
                    sol.Value(estimate.InitiationIntervalCycles.Var()),
                    sol.Value(estimate.SerialRegionCycles.Var()),
                    sol.Value(estimate.PipelinedRegionCycles.Var())),
                channels);
            if (!selectedLoopPipelines.TryAdd(decision.Key.Node, selection))
            {
                throw new InvalidOperationException(
                    $"Tile scope {decision.Key.Node} selected more than one asynchronous loop schedule.");
            }
        }

        var selectedMaterializations = new List<TileMaterialization>();
        foreach (var (source, candidates) in internalPlacementCandidates)
        {
            var selections = candidates
                .SelectMany(candidate => candidate.Info.Places.SelectMany(
                    (places, loopEntry) => places.Select(
                        (place, storageLevel) => (candidate.Node, LoopEntry: loopEntry, StorageLevel: storageLevel, Selected: sol.Value(place.Var())))))
                .Where(selection => selection.Selected == 1)
                .ToArray();
            if (selections.Length != 1)
            {
                throw new InvalidOperationException(
                    $"Internal producer result {source} must select exactly one materialization, got {selections.Length}.");
            }

            var (node, loopEntry, storageLevel, _) = selections[0];
            var uses = internalUsesBySource[source]
                .OrderBy(use => use.ProducerOpId)
                .ThenBy(use => use.ProducerOutputIndex)
                .ThenBy(use => use.ConsumerOpId)
                .ThenBy(use => use.ConsumerAccessIndex)
                .ToImmutableArray();
            var value = new TileValueId(source.Node.RegionOpId, source.OutputIndex);
            var creationScope = new TileScopeId(GetStableScopeAnchor(node), node.Level);
            selectedMaterializations.Add(source.Node.TryGetAliasSourceAccess(source.Index, out _)
                ? new TileAliasMaterialization(value, creationScope, loopEntry, uses)
                : new TileStorageMaterialization(
                    value,
                    creationScope,
                    loopEntry,
                    uses,
                    tilingMemorySpaces[storageLevel].Id));
        }

        foreach (var group in rootMaterializationEdges
                     .GroupBy(edge => edge.Source)
                     .OrderBy(group => group.Key.Node.RegionOpId)
                     .ThenBy(group => group.Key.OutputIndex))
        {
            var source = group.Key;
            var uses = group
                .Select(edge => new TileUseId(
                    edge.Source.Node.RegionOpId,
                    edge.Source.OutputIndex,
                    edge.Target.Node.RegionOpId,
                    edge.Target.Index))
                .OrderBy(use => use.ProducerOpId)
                .ThenBy(use => use.ProducerOutputIndex)
                .ThenBy(use => use.ConsumerOpId)
                .ThenBy(use => use.ConsumerAccessIndex)
                .ToImmutableArray();
            selectedMaterializations.Add(new TileRootMaterialization(
                new TileValueId(source.Node.RegionOpId, source.OutputIndex),
                new TileScopeId(GetStableScopeAnchor(primTree), primTree.Level),
                0,
                uses,
                MemoryAccessScope.Chip));
        }

        if (Diagnostics.DumpScope.Current.IsEnabled(Diagnostics.DumpFlags.Tiling))
        {
            DumpAssgin(primTree, new TreeSolverPrinter(sol, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, targetOptions), coverageConstraints, reductionStateBytes, reductionStateConstraints, selectedMicroKernels, eachLevelStoreBufferConstrains, levelBufferLifetimeConstraints, levelBufferSizes, levelDataReads, levelDataWrites, levelTransferReads, levelTransferWrites, memoryCycles, transferCycles, computeCycles, synchronizationCycles, totalCyclesVar);

            DumpAssgin(primTree, new TreeSolverPythonPrinter(sol, solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, targetOptions), coverageConstraints, reductionStateBytes, reductionStateConstraints, selectedMicroKernels, eachLevelStoreBufferConstrains, levelBufferLifetimeConstraints, levelBufferSizes, levelDataReads, levelDataWrites, levelTransferReads, levelTransferWrites, memoryCycles, transferCycles, computeCycles, synchronizationCycles, totalCyclesVar);
        }

        return new TreeSolveResult(primBufferGraph, sol.Value(totalCyclesVar), levelBufferInfos, opNodeMemoAssgin, tileNodeMemoAssgin, tileableNodeMemoAssgin, selectedMaterializations, selectedMicroKernels, selectedLoopPipelines, selectedTransferSources, targetOptions, moduleKind, owningScheduledFunctionId);

        static int GetStableScopeAnchor(TileNode node)
        {
            var regionOpIds = new List<int>();
            node.Walk(child =>
            {
                if (child is OpNode opNode)
                {
                    regionOpIds.Add(opNode.Wrapped.RegionOpId);
                }
            });
            if (regionOpIds.Count == 0)
            {
                throw new InvalidOperationException($"Tile scope {node} contains no operations.");
            }

            return regionOpIds.Contains(node.OpId) ? node.OpId : regionOpIds.Min();
        }
    }

    public static void DumpAssgin(ITreeNode tree, TreeSolverPythonPrinter printer, Dictionary<OpNode, Constraint[]> coverageConstraints, Dictionary<OpNode, IntExpr> reductionStateBytes, Dictionary<OpNode, Constraint> reductionStateConstraints, IReadOnlyDictionary<int, BlockMicroKernelSelection> selectedMicroKernels, Dictionary<int, Constraint[]> lowestStoreBufferNumsConstrains, Dictionary<int, Constraint[]> levelBufferLifetimeConstraints, Dictionary<int, Dictionary<NodeWithBuffer, IntExpr>> levelBufferSizes, IntExpr[] levelDataReads, IntExpr[] levelDataWrites, IntExpr[] levelTransferReads, IntExpr[] levelTransferWrites, IntExpr[] memoryCycles, IntExpr[] transferCycles, IntExpr computeCycles, IntExpr synchronizationCycles, IntVar totalCycles)
    {
        using (var stream = Diagnostics.DumpScope.Current.OpenFile($"modeling.py"))
        {
            using var baseWriter = new StreamWriter(stream);
            using var writer = new System.CodeDom.Compiler.IndentedTextWriter(baseWriter, "  ");
            tree.Accept(printer, (null, writer));
        }
    }

    public static void DumpAssgin(ITreeNode tree, TreeSolverPrinter printer, Dictionary<OpNode, Constraint[]> coverageConstraints, Dictionary<OpNode, IntExpr> reductionStateBytes, Dictionary<OpNode, Constraint> reductionStateConstraints, IReadOnlyDictionary<int, BlockMicroKernelSelection> selectedMicroKernels, Dictionary<int, Constraint[]> eachLevelStoreBufferNumsConstrains, Dictionary<int, Constraint[]> levelBufferLifetimeConstraints, Dictionary<int, Dictionary<NodeWithBuffer, IntExpr>> levelBufferSizes, IntExpr[] levelDataReads, IntExpr[] levelDataWrites, IntExpr[] levelTransferReads, IntExpr[] levelTransferWrites, IntExpr[] memoryCycles, IntExpr[] transferCycles, IntExpr computeCycles, IntExpr synchronizationCycles, IntVar totalCycles)
    {
        using (var stream = Diagnostics.DumpScope.Current.OpenFile($"modeling.yaml"))
        {
            using var baseWriter = new StreamWriter(stream);
            using var writer = new System.CodeDom.Compiler.IndentedTextWriter(baseWriter, "  ");
            WriteTargetMachine(writer, printer.TargetOptions.TargetMachineModel);
            tree.Accept(printer, writer);
            writer.WriteLine("coverageConstraints:");
            writer.Indent++;
            foreach (var (opnode, consts) in coverageConstraints)
            {
                TreeSolverPrinter.WriteIntExprVector(writer, opnode.ToString(), consts, printer.Solution);
            }

            writer.Indent--;

            writer.WriteLine("LogicalReductionStateBytes:");
            writer.Indent++;
            foreach (var (opNode, bytes) in reductionStateBytes)
            {
                TreeSolverPrinter.WriteIntExpr(writer, opNode.ToString(), bytes, printer.Solution);
            }

            writer.Indent--;

            writer.WriteLine("BlockMicroKernelSelectionConstraints:");
            writer.Indent++;
            foreach (var (opNode, constraint) in reductionStateConstraints)
            {
                TreeSolverPrinter.WriteIntExprVector(writer, opNode.ToString(), new[] { constraint }, printer.Solution);
            }

            writer.Indent--;

            writer.WriteLine("SelectedBlockMicroKernels:");
            writer.Indent++;
            foreach (var (regionOpId, selection) in selectedMicroKernels.OrderBy(pair => pair.Key))
            {
                writer.WriteLine($"Op{regionOpId}: {selection.Family}/{selection.Variant}");
                writer.Indent++;
                writer.WriteLine($"RegionCycles: {selection.RegionCycles}");
                writer.WriteLine($"Resources: {string.Join(", ", selection.Resources.OrderBy(pair => pair.Key).Select(pair => $"{pair.Key}={pair.Value}"))}");
                writer.WriteLine($"Parameters: {string.Join(", ", selection.Parameters.OrderBy(pair => pair.Key).Select(pair => $"{pair.Key}={pair.Value}"))}");
                writer.Indent--;
            }

            writer.Indent--;

            writer.WriteLine("EachLevelStoreBufferNumsConstrains:");
            writer.Indent++;
            foreach (var (node, cons) in eachLevelStoreBufferNumsConstrains)
            {
                TreeSolverPrinter.WriteIntExprVector(writer, node.ToString(), cons, printer.Solution);
            }

            writer.Indent--;

            writer.WriteLine("EachLevelBufferLifetimeConstraints:");
            writer.Indent++;
            foreach (var (node, cons) in levelBufferLifetimeConstraints)
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

    public (Dictionary<BufferIdentity, Expr> ArgumentMemo, long ObjectValue, IReadOnlyList<TileMaterialization> Materializations) SolveRootGraph(TieredTileGraph rootGraph, string moduleKind, INTTTargetOptions targetOptions, DimVar[] dynamicDimVars)
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
        var materializations = new List<TileMaterialization>();
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

            var primTree = treeGraphMemo[primGraph];
            var fusionName = TileSemanticNaming.DescribeFusion(primTree);
            var funcName = CompileSessionScope.GetCurrentThrowIfNull()
                .GetRequiredService<INamingProvider>()
                .GetName($"device_{fusionName.Symbol}");
            using var subSubScope = new Diagnostics.DumpScope(funcName, Diagnostics.DumpFlags.Tiling);
            HashSet<BufferIdentity> inputBids;
            HashSet<BufferIdentity> outputBids;

            if (!SolveMemo.TryGetValue(primTree, out var tiled))
            {
                var result = SolvePrimGraph(primTree, bufferGraphMemo, targetOptions, moduleKind, funcName);
                (inputBids, outputBids) = (result.Inputs, result.Outputs);
                var inputBidsOrdered = OrderBufferIdentities(inputBids);
                var outputBidsOrdered = OrderBufferIdentities(outputBids);
                var bufferSchedule = result.ScheduleBuffers();
                var bodyBuilder = T.Sequential();
                var initOffsets = Enumerable.Repeat(new DimConst(0), primTree.DomainBoundExprs.Length).ToArray();
                var initBounds = primTree.DomainBoundExprs.ToArray();
                result.Visit(primTree, new(bodyBuilder, initOffsets, initBounds));
                var body = bodyBuilder.Build().With(traceScopeName: fusionName.TraceName);
                TileLoweringVerifier.Verify(body, funcName);
                var parameters = inputBidsOrdered.Select(k => result.InputOutputVars[k]).Concat(
                    result.RootParameterBindings.Select(binding => (IVar)binding.Parameter)).Concat(
                    dynamicDimVars.Select(v => (IVar)v)).Concat(
                    result.OutputParameters).ToArray();
                var primFunc = new PrimFunction(
                    funcName,
                    moduleKind,
                    body,
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
                PublishScheduledDataPools(primFunc.SchedResult, bufferSchedule);
                var typeHints = inputBidsOrdered.Select(bid => bid.Node.Grid.GetArgument(bid.Index).CheckedType)
                    .Concat(result.RootParameterBindings.Select(binding => binding.Argument.CheckedType))
                    .Concat(dynamicDimVars.Select(v => new DimensionType(DimensionKind.Dynamic)))
                    .Concat(result.OutputParameters.Select(parameter => parameter.CheckedType))
                    .ToArray();
                tiled = new(
                    new PrimFunctionWrapper(
                        primFunc,
                        inputBidsOrdered.Length + result.RootParameterBindings.Count + dynamicDimVars.Length,
                        typeHints),
                    result.ObjectiveValue,
                    CanonicalizeMaterializations(primTree, result.Materializations),
                    CanonicalizeRootParameters(primTree, result.RootParameterBindings));
                SolveMemo.Add(primTree, tiled);
            }
            else
            {
                (inputBids, outputBids) = bufferGraphMemo[primGraph].GetInputsOutputs(bufferGraphMemo[rootGraph]);
            }

            var orderedInputBids = OrderBufferIdentities(inputBids);
            var orderedOutputBids = OrderBufferIdentities(outputBids);
            objectValue += tiled.ObjectValue;
            materializations.AddRange(MaterializeMaterializations(primTree, tiled.Materializations));

            var finalCall = new Call(
                tiled.Func,
                orderedInputBids.Select(bid => argumentMemo[bid])
                    .Concat(MaterializeRootArguments(primTree, tiled.RootParameters))
                    .Concat(dynamicDimVars.OfType<BaseExpr>())
                    .ToArray());
            var componentOutputs = orderedOutputBids.Length == 1
                ? new Expr[] { finalCall }
                : orderedOutputBids.Select((_, outputIndex) => IR.F.Tensors.GetItem(finalCall, outputIndex)).ToArray();
            BindComponentOutputs(
                orderedOutputBids,
                componentOutputs,
                bufferGraphMemo[rootGraph],
                argumentMemo);
        }

        return (argumentMemo, objectValue, materializations);
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

    private static void PublishScheduledDataPools(
        SchedFunctionResult scheduleResult,
        TileBufferScheduleResult bufferSchedule)
    {
        foreach (var pool in bufferSchedule.Pools.Where(pool => pool.AllocationBytes != 0))
        {
            if (pool.Binding.Hierarchy != 0 &&
                pool.Binding.Location is MemoryLocation.Data or MemoryLocation.ChipLocalData or MemoryLocation.BlockLocalData)
            {
                throw new InvalidOperationException(
                    $"Externally allocated data pool {pool.MemorySpace} uses unsupported TIR hierarchy {pool.Binding.Hierarchy}.");
            }

            var bytes = checked((ulong)pool.AllocationBytes);
            switch (pool.Binding.Location)
            {
                case MemoryLocation.Data:
                    scheduleResult.DataUsage = Math.Max(scheduleResult.DataUsage, bytes);
                    break;
                case MemoryLocation.ChipLocalData:
                    scheduleResult.ChipLocalDataPoolSize = Math.Max(scheduleResult.ChipLocalDataPoolSize, bytes);
                    break;
                case MemoryLocation.BlockLocalData:
                    scheduleResult.BlockLocalDataPoolSize = Math.Max(scheduleResult.BlockLocalDataPoolSize, bytes);
                    break;
                default:
                    continue;
            }

            scheduleResult.DataAlign = Math.Max(scheduleResult.DataAlign, checked((ulong)pool.Alignment));
        }
    }

    private static bool IsAliasOnlyComponent(TieredTileGraph graph)
        => graph.IsPureBufferViewScope();

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
                .Where(edge => edge.Tag is BufferEdgeKind.Inter or BufferEdgeKind.RootMaterialization)
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

    private static IReadOnlyList<BlockMicroKernelCandidate> PruneMicroKernelCandidates(
        IReadOnlyList<BlockMicroKernelCandidate> candidates)
    {
        var result = new List<BlockMicroKernelCandidate>(candidates.Count);
        for (var candidateIndex = 0; candidateIndex < candidates.Count; candidateIndex++)
        {
            var candidate = candidates[candidateIndex];
            var dominated = candidates.Where((_, index) => index != candidateIndex).Any(other => Dominates(other, candidate));
            if (!dominated)
            {
                result.Add(candidate);
            }
        }

        return result;

        static bool Dominates(BlockMicroKernelCandidate lhs, BlockMicroKernelCandidate rhs)
        {
            if (lhs.IsLegal.Var().Min() != 1 || lhs.IsLegal.Var().Max() != 1 ||
                rhs.IsLegal.Var().Min() != 1 || rhs.IsLegal.Var().Max() != 1 ||
                !HaveEquivalentBufferEncodingRequirements(lhs, rhs) ||
                !TryGetConstant(lhs.ExecutionCost.RegionCycles, out var lhsCycles) ||
                !TryGetConstant(rhs.ExecutionCost.RegionCycles, out var rhsCycles) ||
                lhsCycles > rhsCycles ||
                lhs.SelectionPriority > rhs.SelectionPriority)
            {
                return false;
            }

            var lhsResources = lhs.Resources.ToDictionary(usage => usage.Resource);
            var rhsResources = rhs.Resources.ToDictionary(usage => usage.Resource);
            var strictlyBetter = lhsCycles < rhsCycles ||
                lhs.SelectionPriority < rhs.SelectionPriority;
            foreach (var resource in lhsResources.Keys.Union(rhsResources.Keys))
            {
                if (!lhsResources.TryGetValue(resource, out var lhsUsage) ||
                    !rhsResources.TryGetValue(resource, out var rhsUsage) ||
                    !TryGetConstant(lhsUsage.Units, out var lhsUnits) ||
                    !TryGetConstant(rhsUsage.Units, out var rhsUnits) ||
                    lhsUnits > rhsUnits)
                {
                    return false;
                }

                strictlyBetter |= lhsUnits < rhsUnits;
            }

            var lhsMemory = lhs.MemoryAccesses.ToDictionary(
                access => (access.BufferAccessIndex, access.Resource, access.Mode));
            var rhsMemory = rhs.MemoryAccesses.ToDictionary(
                access => (access.BufferAccessIndex, access.Resource, access.Mode));
            if (!lhsMemory.Keys.ToHashSet().SetEquals(rhsMemory.Keys))
            {
                return false;
            }

            foreach (var key in lhsMemory.Keys)
            {
                if (!TryGetConstant(lhsMemory[key].Bytes, out var lhsBytes) ||
                    !TryGetConstant(rhsMemory[key].Bytes, out var rhsBytes) ||
                    lhsBytes > rhsBytes)
                {
                    return false;
                }

                strictlyBetter |= lhsBytes < rhsBytes;
            }

            return strictlyBetter;
        }

        static bool HaveEquivalentBufferEncodingRequirements(
            BlockMicroKernelCandidate lhs,
            BlockMicroKernelCandidate rhs)
        {
            if (lhs.BufferEncodingRequirements.Length != rhs.BufferEncodingRequirements.Length)
            {
                return false;
            }

            return lhs.BufferEncodingRequirements.All(lhsRequirement =>
                rhs.BufferEncodingRequirements.Any(rhsRequirement =>
                    lhsRequirement.BufferAccessIndex == rhsRequirement.BufferAccessIndex
                    && lhsRequirement.MemorySpace == rhsRequirement.MemorySpace
                    && lhsRequirement.AcceptedEncodings.ToHashSet().SetEquals(rhsRequirement.AcceptedEncodings)));
        }

        static bool TryGetConstant(IntExpr expression, out long value)
        {
            var variable = expression.Var();
            value = variable.Min();
            return value == variable.Max();
        }
    }

    private static void ValidateMicroKernelCandidates(
        OpNode opNode,
        IReadOnlyList<BlockMicroKernelCandidate> candidates,
        TargetMachineModel machine)
    {
        if (candidates.Count == 0)
        {
            throw new InvalidOperationException(
                $"Target {machine.Id} exposes no block microkernel candidate for Op{opNode.OpId} ({opNode.Op.GetType().Name}).");
        }

        var duplicate = candidates
            .GroupBy(candidate => (candidate.Family, candidate.Variant))
            .FirstOrDefault(group => group.Count() > 1);
        if (duplicate is not null)
        {
            throw new InvalidOperationException(
                $"Target {machine.Id} exposes duplicate block microkernel " +
                $"{duplicate.Key.Family}/{duplicate.Key.Variant} for Op{opNode.OpId}.");
        }

        foreach (var candidate in candidates)
        {
            if (string.IsNullOrWhiteSpace(candidate.Family) || string.IsNullOrWhiteSpace(candidate.Variant))
            {
                throw new InvalidOperationException($"Op{opNode.OpId} has a block microkernel candidate with an empty identity.");
            }

            if (candidate.SelectionPriority < 0)
            {
                throw new InvalidOperationException(
                    $"Block microkernel {candidate.Family}/{candidate.Variant} has negative selection priority " +
                    $"{candidate.SelectionPriority}.");
            }

            var legality = candidate.IsLegal.Var();
            if (legality.Min() < 0 || legality.Max() > 1)
            {
                throw new InvalidOperationException(
                    $"Block microkernel {candidate.Family}/{candidate.Variant} legality must be boolean, got [{legality.Min()}, {legality.Max()}].");
            }

            var estimatedCycles = candidate.ExecutionCost.RegionCycles.Var();
            if (estimatedCycles.Min() < 0)
            {
                throw new InvalidOperationException(
                    $"Block microkernel {candidate.Family}/{candidate.Variant} has invalid estimated cycles " +
                    $"[{estimatedCycles.Min()}, {estimatedCycles.Max()}]: {candidate.ExecutionCost.RegionCycles}.");
            }

            var duplicateResource = candidate.Resources.GroupBy(usage => usage.Resource).FirstOrDefault(group => group.Count() > 1);
            if (duplicateResource is not null)
            {
                throw new InvalidOperationException(
                    $"Block microkernel {candidate.Family}/{candidate.Variant} repeats resource {duplicateResource.Key}.");
            }

            foreach (var usage in candidate.Resources)
            {
                _ = machine.GetPrivateResource(usage.Resource);
                if (usage.Units.Var().Min() < 0)
                {
                    throw new InvalidOperationException(
                        $"Block microkernel {candidate.Family}/{candidate.Variant} has negative use of {usage.Resource}.");
                }
            }

            var duplicateMemoryAccess = candidate.MemoryAccesses
                .GroupBy(access => (access.BufferAccessIndex, access.Resource, access.Mode))
                .FirstOrDefault(group => group.Count() > 1);
            if (duplicateMemoryAccess is not null)
            {
                throw new InvalidOperationException(
                    $"Block microkernel {candidate.Family}/{candidate.Variant} repeats memory access {duplicateMemoryAccess.Key}.");
            }

            foreach (var access in candidate.MemoryAccesses)
            {
                _ = machine.GetMemoryResource(access.Resource);
                if (access.Mode is not (MemoryAccessMode.Read or MemoryAccessMode.Write))
                {
                    throw new InvalidOperationException(
                        $"Block microkernel {candidate.Family}/{candidate.Variant} memory traffic must be an exact read or write, got {access.Mode}.");
                }

                if (access.Bytes.Var().Min() < 0)
                {
                    throw new InvalidOperationException(
                        $"Block microkernel {candidate.Family}/{candidate.Variant} has negative {access.Mode} traffic for {access.Resource}.");
                }

                if (access.BufferAccessIndex is { } accessIndex)
                {
                    if ((uint)accessIndex >= (uint)opNode.LocalAccessEffects.Length)
                    {
                        throw new InvalidOperationException(
                            $"Block microkernel {candidate.Family}/{candidate.Variant} references missing buffer access {accessIndex} on Op{opNode.OpId}.");
                    }

                    var physicalMode = MemoryEffectUtility.GetPhysicalBufferAccessMode(opNode.LocalAccessEffects[accessIndex]);
                    if (!physicalMode.HasFlag(access.Mode))
                    {
                        throw new InvalidOperationException(
                            $"Block microkernel {candidate.Family}/{candidate.Variant} declares {access.Mode} traffic for Op{opNode.OpId} access {accessIndex}, whose physical effect is {physicalMode}.");
                    }
                }
            }

            var duplicateParameter = candidate.Parameters.GroupBy(parameter => parameter.Name, StringComparer.Ordinal).FirstOrDefault(group => group.Count() > 1);
            if (duplicateParameter is not null || candidate.Parameters.Any(parameter => string.IsNullOrWhiteSpace(parameter.Name)))
            {
                throw new InvalidOperationException(
                    $"Block microkernel {candidate.Family}/{candidate.Variant} has invalid or duplicate parameters.");
            }

            var duplicateRequirement = candidate.BufferEncodingRequirements
                .GroupBy(requirement => (requirement.BufferAccessIndex, requirement.MemorySpace))
                .FirstOrDefault(group => group.Count() > 1);
            if (duplicateRequirement is not null)
            {
                throw new InvalidOperationException(
                    $"Block microkernel {candidate.Family}/{candidate.Variant} repeats storage requirement {duplicateRequirement.Key}.");
            }

            foreach (var requirement in candidate.BufferEncodingRequirements)
            {
                if ((uint)requirement.BufferAccessIndex >= (uint)opNode.LocalAccessEffects.Length)
                {
                    throw new InvalidOperationException(
                        $"Block microkernel {candidate.Family}/{candidate.Variant} storage requirement references missing " +
                        $"buffer access {requirement.BufferAccessIndex} on Op{opNode.OpId}.");
                }

                if (!MemoryEffectUtility.GetPhysicalBufferAccessMode(opNode.LocalAccessEffects[requirement.BufferAccessIndex])
                    .HasFlag(MemoryAccessMode.Read))
                {
                    throw new InvalidOperationException(
                        $"Block microkernel {candidate.Family}/{candidate.Variant} storage requirement references non-readable " +
                        $"buffer access {requirement.BufferAccessIndex} on Op{opNode.OpId}.");
                }

                _ = machine.GetMemorySpace(requirement.MemorySpace);
                if (requirement.AcceptedEncodings.IsDefaultOrEmpty ||
                    requirement.AcceptedEncodings.Any(encoding => string.IsNullOrWhiteSpace(encoding.Value)) ||
                    requirement.AcceptedEncodings.Distinct().Count() != requirement.AcceptedEncodings.Length)
                {
                    throw new InvalidOperationException(
                        $"Block microkernel {candidate.Family}/{candidate.Variant} has invalid accepted storage encodings " +
                        $"for access {requirement.BufferAccessIndex}.");
                }
            }
        }
    }

    private static void ValidateStorageEncodingCandidates(
        NodeWithBuffer buffer,
        int loopEntry,
        int storageLevel,
        IReadOnlyList<TargetStorageEncodingCandidate> candidates,
        TargetMachineModel machine,
        StagedAllocationContext? staging)
    {
        if (candidates.Count == 0)
        {
            throw new InvalidOperationException(
                $"Target {machine.Id} exposes no storage encoding for {buffer} at entry {loopEntry}/L{storageLevel}.");
        }

        var duplicate = candidates.GroupBy(candidate => candidate.Id).FirstOrDefault(group => group.Count() > 1);
        if (duplicate is not null)
        {
            throw new InvalidOperationException(
                $"Target {machine.Id} exposes duplicate storage encoding {duplicate.Key} for {buffer}.");
        }

        foreach (var candidate in candidates)
        {
            if (string.IsNullOrWhiteSpace(candidate.Id.Value))
            {
                throw new InvalidOperationException($"Target {machine.Id} exposes an empty storage encoding identity for {buffer}.");
            }

            var legality = candidate.IsLegal.Var();
            if (legality.Min() < 0 || legality.Max() > 1)
            {
                throw new InvalidOperationException(
                    $"Storage encoding {candidate.Id} legality must be boolean, got [{legality.Min()}, {legality.Max()}].");
            }

            if (candidate.PhysicalBytes.Var().Min() < 0 || candidate.EstimatedCycles.Var().Min() < 0)
            {
                throw new InvalidOperationException(
                    $"Storage encoding {candidate.Id} for {buffer} has negative physical size or cost.");
            }

            if (candidate.SelectionPriority < 0)
            {
                throw new InvalidOperationException(
                    $"Storage encoding {candidate.Id} for {buffer} has negative selection priority {candidate.SelectionPriority}.");
            }

            if (staging is null && candidate.StageStrideBytes is not null)
            {
                throw new InvalidOperationException(
                    $"Ordinary storage encoding {candidate.Id} for {buffer} unexpectedly declares a stage stride.");
            }

            if (staging is not null)
            {
                if (candidate.StageStrideBytes is not { } stageStride)
                {
                    throw new InvalidOperationException(
                        $"Staged storage encoding {candidate.Id} for {buffer} must declare a stage stride.");
                }

                if (stageStride.Var().Min() < candidate.PhysicalBytes.Var().Min())
                {
                    throw new InvalidOperationException(
                        $"Staged storage encoding {candidate.Id} for {buffer} has stage stride " +
                        $"[{stageStride.Var().Min()},{stageStride.Var().Max()}] smaller than its encoded stage " +
                        $"[{candidate.PhysicalBytes.Var().Min()},{candidate.PhysicalBytes.Var().Max()}].");
                }
            }

            if (candidate.AlignmentBytes <= 0 ||
                !System.Numerics.BitOperations.IsPow2((uint)candidate.AlignmentBytes))
            {
                throw new InvalidOperationException(
                    $"Storage encoding {candidate.Id} for {buffer} has invalid alignment {candidate.AlignmentBytes}.");
            }

            var duplicateParameter = candidate.Parameters
                .GroupBy(parameter => parameter.Name, StringComparer.Ordinal)
                .FirstOrDefault(group => group.Count() > 1);
            if (duplicateParameter is not null || candidate.Parameters.Any(parameter => string.IsNullOrWhiteSpace(parameter.Name)))
            {
                throw new InvalidOperationException(
                    $"Storage encoding {candidate.Id} for {buffer} has invalid or duplicate parameters.");
            }
        }
    }

    private static void WriteTargetMachine(System.CodeDom.Compiler.IndentedTextWriter writer, TargetMachineModel machine)
    {
        writer.WriteLine("TargetMachine:");
        writer.Indent++;
        writer.WriteLine($"Id: {machine.Id}");
        writer.WriteLine($"Execution: {machine.Execution.Kind}");
        writer.WriteLine($"ComputeUnits: {machine.Execution.ComputeUnitCount}");
        writer.WriteLine($"WorkersPerBlock: {machine.Execution.WorkersPerBlock}");
        writer.WriteLine($"WorkerWidth: {machine.Execution.WorkerWidth}");
        writer.WriteLine("PrivateResources:");
        writer.Indent++;
        foreach (var resource in machine.PrivateResources.Values.OrderBy(resource => resource.Id.Value, StringComparer.Ordinal))
        {
            writer.WriteLine($"- {resource.Id}: unit={resource.Unit}, capacity={resource.CapacityUnits}, granularity={resource.AllocationGranularityUnits}, backing={resource.BackingMemoryResource?.ToString() ?? "none"}");
        }

        writer.Indent--;
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
            writer.WriteLine(
                $"- {primitive.Name}: m={primitive.M}, n={primitive.N}, k={primitive.K}, " +
                $"dependency_latency_cycles={primitive.DependencyLatencyCycles}, " +
                $"worker_reciprocal_throughput_cycles={primitive.ReciprocalThroughputCyclesPerWorker}, " +
                $"max_instructions_per_cycle={primitive.MaxInstructionsPerCyclePerBlock}, " +
                $"cooperative_workers={primitive.CooperativeWorkers}, supported={primitive.IsSupported}");
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
        Solver solver,
        TileWorkloadContext context)
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

        var candidates = machine.Compute.MatrixPrimitives
            .Where(primitive => primitive.Supports(operandDataTypes[0], operandDataTypes[1]))
            .Select(primitive =>
            {
                // Keep output accumulator chains separate from their dependent
                // reduction length so the target timing model can enforce both
                // instruction latency and throughput bounds.
                var accumulatorChains = GetTotalPrimitiveTileCount(fullShape.M, shape.M, primitive.M, solver)
                    * GetTotalPrimitiveTileCount(fullShape.N, shape.N, primitive.N, solver)
                    * fullShape.Multiplicity;
                var dependentInstructionsPerChain = GetTotalPrimitiveTileCount(
                    fullShape.K,
                    shape.K,
                    primitive.K,
                    solver);
                return MatrixComputeCostModel.EstimateCycles(
                    primitive,
                    accumulatorChains,
                    dependentInstructionsPerChain,
                    machine.Execution,
                    solver);
            })
            .ToArray();
        return candidates.Aggregate(simtCycles, solver.MakeMin);
    }

    private static IntExpr GetTotalPrimitiveTileCount(
        IntExpr logicalExtent,
        IntExpr nominalTileExtent,
        long primitiveExtent,
        Solver solver)
    {
        var logicalExtentVar = logicalExtent.Var();
        if (logicalExtentVar.Min() < 0 || logicalExtentVar.Min() != logicalExtentVar.Max())
        {
            throw new InvalidOperationException(
                $"Primitive tile-count modeling requires a constant non-negative logical extent, got [{logicalExtentVar.Min()}, {logicalExtentVar.Max()}].");
        }

        var logicalBound = logicalExtentVar.Max();
        var fullTileCount = solver.MakeDiv(logicalExtent, nominalTileExtent).Var();
        fullTileCount.SetRange(0, logicalBound);
        var tailExtent = (logicalExtent - (fullTileCount * nominalTileExtent)).Var();
        tailExtent.SetRange(0, logicalBound);
        var hasTail = solver.MakeIsGreaterCstVar(tailExtent, 0);
        return (fullTileCount * nominalTileExtent.CeilDiv(primitiveExtent))
            + (hasTail * tailExtent.CeilDiv(primitiveExtent));
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

    private static void DumpGantt(Dictionary<NodeWithBuffer, IntExpr> nodeBufferSizes, Dictionary<NodeWithBuffer, TileLifetime> nodeBufferLifetimes, TileNode primTree, int storeLevel)
    {
        string GetStartStr(string name, int start) => $"[{name}] starts D+{start}";
        string GetDurationStr(string name, int duration) => $"[{name}] requires {duration} days";
        using (var fs = Diagnostics.DumpScope.Current.OpenFile($"Op{primTree.OpId}_{primTree.Level}_store_{storeLevel}_gantt.md"))
        {
            using var writer = new StreamWriter(fs);
            writer.WriteLine("```plantuml");
            writer.WriteLine("@startgantt");
            writer.WriteLine("printscale daily zoom 10");

            foreach (var ((node, bid), lifetime) in nodeBufferLifetimes)
            {
                var name = $"cl{node.Level} op{bid.Node.OpId} {bid.Index}";
                writer.WriteLine(GetDurationStr(name, lifetime.PhaseCount));
                writer.WriteLine(GetStartStr(name, lifetime.FirstPhase));
            }

            writer.WriteLine("@endgantt");
            writer.WriteLine("```");
        }
    }

    private static bool IsObjectBuffer(BufferIdentity bid) => bid.Access.Buffer.CheckedDataType is ReferenceType;

    private static IntExpr GetManagedArenaAllocationBytes(
        TargetMachineModel machine,
        TargetMemorySpaceSpec memorySpace,
        IntExpr requiredBytes,
        Solver solver)
    {
        var maximumBytes = machine.GetMaximumUsableAllocationBytes(memorySpace);
        return memorySpace.AllocationSizePolicy switch
        {
            TargetMemoryAllocationSizePolicy.GranularityAligned =>
                requiredBytes.CeilDiv(machine.GetMemoryResource(memorySpace).AllocationGranularityBytes)
                * machine.GetMemoryResource(memorySpace).AllocationGranularityBytes,
            TargetMemoryAllocationSizePolicy.PowerOfTwo => GetPowerOfTwoAllocationBytes(
                machine,
                memorySpace,
                requiredBytes,
                maximumBytes,
                solver),
            _ => throw new ArgumentOutOfRangeException(
                    nameof(memorySpace),
                    memorySpace.AllocationSizePolicy,
                    "Unknown target memory allocation size policy."),
        };
    }

    private static IntExpr GetPowerOfTwoAllocationBytes(
        TargetMachineModel machine,
        TargetMemorySpaceSpec memorySpace,
        IntExpr requiredBytes,
        long maximumBytes,
        Solver solver)
    {
        var minimumBytes = machine.GetAllocationSizeBytes(memorySpace, 1);
        if (minimumBytes > maximumBytes)
        {
            throw new InvalidOperationException(
                $"Memory space {memorySpace.Id} has no positive allocation permitted by its " +
                $"{memorySpace.AllocationSizePolicy} policy and {maximumBytes}-byte limit.");
        }

        IntExpr allocationBytes = solver.MakeIntConst(0);
        long previousBytes = 0;
        for (var bytes = minimumBytes; bytes <= maximumBytes;)
        {
            var exceedsPrevious = solver.MakeIsGreaterCstVar(requiredBytes, previousBytes);
            var fitsCurrent = 1 - solver.MakeIsGreaterCstVar(requiredBytes, bytes);
            allocationBytes += exceedsPrevious * fitsCurrent * bytes;
            previousBytes = bytes;
            if (bytes > maximumBytes / 2)
            {
                break;
            }

            bytes = checked(bytes * 2);
        }

        return allocationBytes;
    }

    private static IntExpr GetAllocatedPrivateResourceUnits(
        IntExpr requestedUnits,
        TargetPrivateResourceSpec resource,
        Solver solver)
    {
        var granularity = resource.AllocationGranularityUnits;
        return solver.MakeDiv(requestedUnits + (granularity - 1), granularity) * granularity;
    }

    private static long AlignUp(long value, long alignment)
        => checked(((value + alignment - 1) / alignment) * alignment);

    private static IReadOnlyList<CanonicalTileMaterialization> CanonicalizeMaterializations(
        TileNode tree,
        IReadOnlyList<TileMaterialization> materializations)
    {
        var ordinals = GetRegionOpIds(tree)
            .Select((regionOpId, ordinal) => (regionOpId, ordinal))
            .ToDictionary(item => item.regionOpId, item => item.ordinal);
        return materializations.Select<TileMaterialization, CanonicalTileMaterialization>(materialization =>
        {
            if (!ordinals.TryGetValue(materialization.Value.ProducerOpId, out var producerOrdinal) ||
                !ordinals.TryGetValue(materialization.CreationScope.AnchorOpId, out var creationAnchorOrdinal))
            {
                throw new InvalidOperationException(
                    $"Selected tile materialization {materialization.Value} at {materialization.CreationScope} does not belong to the solved component.");
            }

            var value = new CanonicalTileValueId(producerOrdinal, materialization.Value.ProducerOutputIndex);
            var uses = materialization.Uses.Select(use => CanonicalizeUse(use, ordinals)).ToImmutableArray();
            return materialization switch
            {
                TileStorageMaterialization storage => new CanonicalTileStorageMaterialization(
                    value,
                    creationAnchorOrdinal,
                    materialization.CreationScope.Level,
                    materialization.LoopEntry,
                    uses,
                    storage.StorageSpace),
                TileAliasMaterialization => new CanonicalTileAliasMaterialization(
                    value,
                    creationAnchorOrdinal,
                    materialization.CreationScope.Level,
                    materialization.LoopEntry,
                    uses),
                TileRootMaterialization root => new CanonicalTileRootMaterialization(
                    value,
                    creationAnchorOrdinal,
                    materialization.CreationScope.Level,
                    materialization.LoopEntry,
                    uses,
                    root.RequiredMemoryScope),
                _ => throw new InvalidOperationException(
                    $"Unknown tile materialization {materialization.GetType().Name}."),
            };
        }).ToArray();

        static CanonicalTileUseId CanonicalizeUse(TileUseId use, IReadOnlyDictionary<int, int> ordinals)
        {
            if (!ordinals.TryGetValue(use.ProducerOpId, out var producerOrdinal) ||
                !ordinals.TryGetValue(use.ConsumerOpId, out var consumerOrdinal))
            {
                throw new InvalidOperationException($"Selected tile use {use} does not belong to the solved component.");
            }

            return new CanonicalTileUseId(
                producerOrdinal,
                use.ProducerOutputIndex,
                consumerOrdinal,
                use.ConsumerAccessIndex);
        }
    }

    private static IReadOnlyList<CanonicalTileValueId> CanonicalizeRootParameters(
        TileNode tree,
        IReadOnlyList<TileRootParameterBinding> bindings)
    {
        var ordinals = GetRegionOpIds(tree)
            .Select((regionOpId, ordinal) => (regionOpId, ordinal))
            .ToDictionary(item => item.regionOpId, item => item.ordinal);
        return bindings.Select(binding =>
        {
            if (!ordinals.TryGetValue(binding.Source.Node.RegionOpId, out var producerOrdinal))
            {
                throw new InvalidOperationException(
                    $"Root parameter source {binding.Source} does not belong to the solved component.");
            }

            return new CanonicalTileValueId(producerOrdinal, binding.Source.OutputIndex);
        }).ToArray();
    }

    private static IReadOnlyList<Expr> MaterializeRootArguments(
        TileNode tree,
        IReadOnlyList<CanonicalTileValueId> parameters)
    {
        var regionOpIds = GetRegionOpIds(tree);
        var opNodes = new List<OpNode>();
        tree.Walk(node =>
        {
            if (node is OpNode opNode)
            {
                opNodes.Add(opNode);
            }
        });

        return parameters.Select(parameter =>
        {
            if ((uint)parameter.ProducerOrdinal >= (uint)regionOpIds.Count)
            {
                throw new InvalidOperationException(
                    $"Root parameter {parameter} is incompatible with a component containing {regionOpIds.Count} operations.");
            }

            var regionOpId = regionOpIds[parameter.ProducerOrdinal];
            var producer = opNodes.Single(node => node.Wrapped.RegionOpId == regionOpId);
            var accessIndex = producer.Wrapped.GetWriteAccessIndex(parameter.ProducerOutputIndex);
            return producer.Wrapped.Grid.Accesses[accessIndex].Buffer;
        }).ToArray();
    }

    private static IReadOnlyList<TileMaterialization> MaterializeMaterializations(
        TileNode tree,
        IReadOnlyList<CanonicalTileMaterialization> materializations)
    {
        var regionOpIds = GetRegionOpIds(tree);
        return materializations.Select<CanonicalTileMaterialization, TileMaterialization>(materialization =>
        {
            ValidateOrdinal(materialization.Value.ProducerOrdinal, regionOpIds.Count, materialization.Value.ToString());
            ValidateOrdinal(materialization.CreationAnchorOrdinal, regionOpIds.Count, materialization.ToString());
            var value = new TileValueId(
                regionOpIds[materialization.Value.ProducerOrdinal],
                materialization.Value.ProducerOutputIndex);
            var creationScope = new TileScopeId(
                regionOpIds[materialization.CreationAnchorOrdinal],
                materialization.CreationLevel);
            var uses = materialization.Uses.Select(use => MaterializeUse(use, regionOpIds)).ToImmutableArray();
            return materialization switch
            {
                CanonicalTileStorageMaterialization storage => new TileStorageMaterialization(
                    value,
                    creationScope,
                    materialization.LoopEntry,
                    uses,
                    storage.StorageSpace),
                CanonicalTileAliasMaterialization => new TileAliasMaterialization(
                    value,
                    creationScope,
                    materialization.LoopEntry,
                    uses),
                CanonicalTileRootMaterialization root => new TileRootMaterialization(
                    value,
                    creationScope,
                    materialization.LoopEntry,
                    uses,
                    root.RequiredMemoryScope),
                _ => throw new InvalidOperationException(
                    $"Unknown canonical tile materialization {materialization.GetType().Name}."),
            };
        }).ToArray();

        static TileUseId MaterializeUse(CanonicalTileUseId use, IReadOnlyList<int> regionOpIds)
        {
            ValidateOrdinal(use.ProducerOrdinal, regionOpIds.Count, use.ToString());
            ValidateOrdinal(use.ConsumerOrdinal, regionOpIds.Count, use.ToString());
            return new TileUseId(
                regionOpIds[use.ProducerOrdinal],
                use.ProducerOutputIndex,
                regionOpIds[use.ConsumerOrdinal],
                use.ConsumerAccessIndex);
        }

        static void ValidateOrdinal(int ordinal, int count, string owner)
        {
            if ((uint)ordinal >= (uint)count)
            {
                throw new InvalidOperationException(
                    $"Canonical tile identity {owner} is incompatible with a component containing {count} operations.");
            }
        }
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

    public readonly record struct CanonicalTileValueId(int ProducerOrdinal, int ProducerOutputIndex);

    public abstract record CanonicalTileMaterialization(
        CanonicalTileValueId Value,
        int CreationAnchorOrdinal,
        int CreationLevel,
        int LoopEntry,
        ImmutableArray<CanonicalTileUseId> Uses);

    public sealed record CanonicalTileStorageMaterialization(
        CanonicalTileValueId Value,
        int CreationAnchorOrdinal,
        int CreationLevel,
        int LoopEntry,
        ImmutableArray<CanonicalTileUseId> Uses,
        TargetMemorySpaceId StorageSpace)
        : CanonicalTileMaterialization(Value, CreationAnchorOrdinal, CreationLevel, LoopEntry, Uses);

    public sealed record CanonicalTileAliasMaterialization(
        CanonicalTileValueId Value,
        int CreationAnchorOrdinal,
        int CreationLevel,
        int LoopEntry,
        ImmutableArray<CanonicalTileUseId> Uses)
        : CanonicalTileMaterialization(Value, CreationAnchorOrdinal, CreationLevel, LoopEntry, Uses);

    public sealed record CanonicalTileRootMaterialization(
        CanonicalTileValueId Value,
        int CreationAnchorOrdinal,
        int CreationLevel,
        int LoopEntry,
        ImmutableArray<CanonicalTileUseId> Uses,
        MemoryAccessScope RequiredMemoryScope)
        : CanonicalTileMaterialization(Value, CreationAnchorOrdinal, CreationLevel, LoopEntry, Uses);

    public sealed record TiledFunc(
        PrimFunctionWrapper Func,
        long ObjectValue,
        IReadOnlyList<CanonicalTileMaterialization> Materializations,
        IReadOnlyList<CanonicalTileValueId> RootParameters)
    {
    }

    private sealed record LoopPipelineKey(TileNode Node, int LoopEntry);

    private sealed record MicroKernelSolverDecision(
        IReadOnlyList<BlockMicroKernelCandidate> Candidates,
        IntVar[] SelectionVars,
        IntExpr SelectedRegionCycles);

    private sealed record TileTransferSourceChoice(
        SelectedTileBufferSource Source,
        IntExpr Selected);

    private sealed record TileTransferSourceDecision(
        TileBufferPlacement Destination,
        TargetMemorySpaceId SourceMemorySpace,
        TargetMemorySpaceId DestinationMemorySpace,
        IReadOnlyList<TileTransferSourceChoice> Sources,
        IntExpr SourceIsAvailable);

    private sealed record LoopPipelineChannelDecision(
        string ChannelId,
        NodeWithBuffer Buffer,
        TileNodeBufferInfo<IntExpr> BufferInfo,
        int LoopEntry,
        int StorageLevel,
        TileTransferSourceDecision TransferSource,
        IntVar Placement,
        IntExpr Legality)
    {
        public TargetMemorySpaceId SourceMemorySpace => TransferSource.SourceMemorySpace;

        public TargetMemorySpaceId DestinationMemorySpace => TransferSource.DestinationMemorySpace;
    }

    private sealed class LoopPipelineSolverDecision
    {
        public LoopPipelineSolverDecision(
            LoopPipelineKey key,
            int domainAxis,
            IntVar serialSelected,
            IntVar asynchronousSelected,
            IntVar stageCount,
            IReadOnlyList<LoopPipelineChannelDecision> channels,
            ImmutableArray<int> regionOpIds)
        {
            Key = key;
            DomainAxis = domainAxis;
            SerialSelected = serialSelected;
            AsynchronousSelected = asynchronousSelected;
            StageCount = stageCount;
            Channels = channels;
            RegionOpIds = regionOpIds;
        }

        public LoopPipelineKey Key { get; }

        public int DomainAxis { get; }

        public IntVar SerialSelected { get; }

        public IntVar AsynchronousSelected { get; }

        public IntVar StageCount { get; }

        public IReadOnlyList<LoopPipelineChannelDecision> Channels { get; }

        public ImmutableArray<int> RegionOpIds { get; }

        public LoopPipelineScheduleEstimate? Estimate { get; set; }
    }

    private sealed record StorageEncodingPlacementKey(
        NodeWithBuffer Buffer,
        int LoopEntry,
        int StorageLevel);

    private sealed record StorageEncodingSolverDecision(
        IReadOnlyList<TargetStorageEncodingCandidate> Candidates,
        IntVar[] SelectionVars,
        bool HasIndependentSelectionVars,
        StagedAllocationContext? StagedAllocation);

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
