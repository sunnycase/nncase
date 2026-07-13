// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Diagnostics.CodeAnalysis;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Shapes;
using QuikGraph;
using QuikGraph.Algorithms;
using QuikGraph.Algorithms.ShortestPath;
using QuikGraph.Graphviz;

namespace Nncase.Schedule.TileGraph;

/// <summary>
/// Materializes an immutable connection plan as a derived schedule graph.
/// The graph is mutable only while this builder is constructing one plan.
/// </summary>
internal sealed class TileScheduleComposer
{
    private TileScheduleComposer(TileGrid opConsumer, TileGrid opProducer, int level, int consumerAccessIndex)
    {
        ConsumerOp = opConsumer;
        ProducerOp = opProducer;
        TargetLevel = level;
        ConsumerAccessIndex = consumerAccessIndex;
        RootGraph = null!;
    }

    public TileGrid ConsumerOp { get; }

    public TileGrid ProducerOp { get; }

    public int TargetLevel { get; }

    public int ConsumerAccessIndex { get; }

    public TieredTileGraph RootGraph { get; set; }

    private string Failure { get; set; } = string.Empty;

    public static bool TryApply(TieredTileGraph graph, TileUseId use, int level, out string failure)
    {
        var edge = graph.Edges.SingleOrDefault(candidate =>
            candidate.Source.RegionOpId == use.ProducerOpId &&
            candidate.Target.RegionOpId == use.ConsumerOpId &&
            candidate.Tag == use.ConsumerAccessIndex);
        if (edge is null)
        {
            throw new InvalidOperationException($"Tile region no longer contains use {use}.");
        }

        var producer = edge.Source;
        var consumer = edge.Target;

        var actualOutputIndex = GraphExtensions.GetProducerOutputIndex(
            consumer.Grid.Accesses[use.ConsumerAccessIndex].Value,
            producer);
        if (actualOutputIndex != use.ProducerOutputIndex)
        {
            throw new InvalidOperationException(
                $"Tile use {use} resolves to producer output {actualOutputIndex} while constructing its schedule.");
        }

        var composer = new TileScheduleComposer(consumer, producer, level, use.ConsumerAccessIndex);
        var success = composer.Visit(graph);
        failure = success
            ? string.Empty
            : string.IsNullOrEmpty(composer.Failure)
                ? "No legal composition path reaches the requested hierarchy level."
                : composer.Failure;
        return success;
    }

    public bool Visit(TieredTileGraph graph)
    {
        RootGraph = graph;
        if (ProducerOp.IsPureBufferView && ConsumerAccessIndex < 0)
        {
            throw new InvalidOperationException($"Buffer-view fusion requires an explicit consumer access index for {ProducerOp} -> {ConsumerOp}.");
        }

        if (ProducerOp.IsPureBufferView &&
            (ProducerOp.Attribute.HasFlag(TileGridAttribute.LiveOut) || RootGraph.OutDegree(ProducerOp) > 1))
        {
            return TryMergeSharedBufferViewUse();
        }

        return VisitRecursion(graph);
    }

    private bool TryMergeSharedBufferViewUse()
    {
        var useEdge = RootGraph.Edges.FirstOrDefault(edge =>
            ReferenceEquals(edge.Source, ProducerOp) &&
            ReferenceEquals(edge.Target, ConsumerOp) &&
            edge.Tag == ConsumerAccessIndex);
        if (useEdge is null || !TryFindStandaloneTopLevelCluster(ProducerOp, out var producerTopLevel))
        {
            Failure = $"Shared buffer view {ProducerOp} has no standalone hierarchy that can be cloned for this use.";
            return false;
        }

        var scheduleOpId = RootGraph.Vertices.Select(vertex => vertex.OpId).DefaultIfEmpty(-1).Max() + 1;
        var clonedView = new TileGrid(
            ProducerOp.Grid,
            ProducerOp.Op,
            scheduleOpId,
            ProducerOp.RegionOpId,
            ProducerOp.DomainBounds,
            new DomainRelation(scheduleOpId, scheduleOpId, AffineMap.Identity(ProducerOp.DomainBounds.Length)),
            ProducerOp.DomainBoundExprs.ToArray(),
            ProducerOp.DomainDynamic,
            ProducerOp.BufferShapes.Select(shape => shape.AsEnumerable()),
            TileGridAttribute.None);
        CloneStandaloneHierarchy(producerTopLevel, clonedView, scheduleOpId);

        foreach (var inputEdge in RootGraph.Edges.Where(edge => ReferenceEquals(edge.Target, ProducerOp)).ToArray())
        {
            RootGraph.AddEdge(new(inputEdge.Source, clonedView, inputEdge.Tag));
        }

        RootGraph.RemoveEdge(useEdge);
        RootGraph.AddEdge(new(clonedView, ConsumerOp, ConsumerAccessIndex));
        return new TileScheduleComposer(ConsumerOp, clonedView, TargetLevel, ConsumerAccessIndex).Visit(RootGraph);
    }

    private bool TryFindStandaloneTopLevelCluster(
        TileGrid view,
        [MaybeNullWhen(false)] out TieredTileGraph topLevel)
    {
        topLevel = RootGraph.Clusters
            .OfType<TieredTileGraph>()
            .SingleOrDefault(cluster => cluster.ContainsVertex(view));
        if (topLevel is null || topLevel.Vertices.Count() != 1)
        {
            return false;
        }

        var current = topLevel;
        while (current.ClustersCount != 0)
        {
            var children = current.Clusters.OfType<TieredTileGraph>().ToArray();
            if (children.Length != 1 || children[0].Vertices.Count() != 1)
            {
                return false;
            }

            current = children[0];
        }

        return current.ContainsVertex(view);
    }

    private void CloneStandaloneHierarchy(TieredTileGraph source, TileGrid clonedView, int scheduleOpId)
    {
        TieredTileGraph CloneLevel(TieredTileGraph sourceLevel, TieredTileGraph destinationParent)
        {
            var relation = new DomainRelation(scheduleOpId, scheduleOpId, sourceLevel.DomainRelation.Map);
            var destination = destinationParent.CreateCluster<TieredTileGraph>(
                sourceLevel.Level,
                scheduleOpId,
                relation,
                sourceLevel.DomainBoundExprs.ToArray(),
                sourceLevel.DomainDynamic.ToArray());
            var child = sourceLevel.Clusters.OfType<TieredTileGraph>().SingleOrDefault();
            if (child is null)
            {
                destination.AddVertex(clonedView);
            }
            else
            {
                CloneLevel(child, destination);
            }

            return destination;
        }

        CloneLevel(source, RootGraph);
    }

    private bool TryMerge(TieredTileGraph graph)
    {
        if (!GatherSubGraphs(graph, out var producerGraph, out var consumerGraph))
        {
            Failure = $"Could not locate distinct producer and consumer clusters below {graph}.";
            return false;
        }

        if (!CheckSubGraphsDenpendence(producerGraph, consumerGraph))
        {
            Failure = $"Fusing {producerGraph} into {consumerGraph} violates dependence convexity or a chip-visible memory effect.";
            return false;
        }

        System.Diagnostics.Trace.Assert(ReferenceEquals(producerGraph.Parent, consumerGraph.Parent));
        System.Diagnostics.Trace.Assert(producerGraph.Level.Equals(consumerGraph.Level));

        var commonAncestor = producerGraph.Parent!;

        // 1. find the dataflow graph
        // 1.1 find the directly connected opnode with producer op.
        var algo = new FloydWarshallAllShortestPathAlgorithm<TileGrid, EquatableTaggedEdge<TileGrid, int>>(RootGraph, (_) => 1.0f);
        algo.Compute();
        if (!algo.TryGetPath(ProducerOp, ConsumerOp, out var dependencePath))
        {
            Failure = $"No data-dependence path exists from {ProducerOp} to {ConsumerOp}.";
            return false;
        }

        var relayOp = dependencePath.First().Target;

        // 1.2. build the direct domain relation from the consumer use to the
        // producer result.
        var consumerAccessIndex = dependencePath.First().Tag;
        var readAccess = relayOp.GetAccessMap(consumerAccessIndex);
        var producerOutputIndex = GraphExtensions.GetProducerOutputIndex(relayOp.Grid.Accesses[consumerAccessIndex].Value, ProducerOp);
        var producerWriteAccess = ProducerOp.GetWriteAccess(producerOutputIndex);
        var relation = readAccess * AffineUtility.Inverse(producerWriteAccess, ProducerOp.DomainBounds.Select(Convert.ToInt64).ToArray());
        if (!relation.IsRectangularProjection(true))
        {
            Failure = $"Connection relation from {ProducerOp} to {relayOp} does not preserve rectangular tile regions: {relation}.";
            return false;
        }

        var consumerToRelay = GetDescendantRelation(consumerGraph, relayOp);
        var consumerToProducerOp = consumerToRelay.ApplyRange(
            new DomainRelation(relayOp.OpId, ProducerOp.OpId, relation));
        var producerGraphToProducerOp = GetDescendantRelation(producerGraph, ProducerOp);
        if (!producerGraphToProducerOp.Map.IsRectangularProjection(true))
        {
            Failure = $"Producer cluster relation to {ProducerOp} does not preserve rectangular tile regions: {producerGraphToProducerOp.Map}.";
            return false;
        }

        var producerGraphBounds = CompilerServices.GetMaxShape(
            new RankedShape(producerGraph.DomainBoundExprs.ToArray()));
        var producerOpToGraph = new DomainRelation(
            ProducerOp.OpId,
            producerGraph.OpId,
            AffineUtility.Inverse(producerGraphToProducerOp.Map, producerGraphBounds));
        var domainRel = consumerToProducerOp.ApplyRange(producerOpToGraph);

        // 4. merge producerGraph's subgrph into the consumerGraph.
        commonAncestor.RemoveCluster(producerGraph);
        if (producerGraph.ClustersCount == 0)
        {
            foreach (var vertex in producerGraph.Vertices)
            {
                vertex.DomainRelation = domainRel.ApplyRange(vertex.DomainRelation);
            }
        }
        else
        {
            foreach (var producerChild in producerGraph.Clusters.OfType<TieredTileGraph>())
            {
                producerChild.DomainRelation = domainRel.ApplyRange(producerChild.DomainRelation);
                consumerGraph.AddCluster(producerChild);
            }
        }

        consumerGraph.AddVertexRange(producerGraph.Vertices);
        return true;
    }

    private static DomainRelation GetDescendantRelation(TieredTileGraph ancestor, TileGrid descendant)
    {
        if (!ancestor.ContainsVertex(descendant))
        {
            throw new InvalidOperationException($"{descendant} is not contained by {ancestor}.");
        }

        DomainRelation? relation = null;
        TieredTileGraph current = ancestor;
        while (true)
        {
            var childGraph = current.Clusters
                .OfType<TieredTileGraph>()
                .SingleOrDefault(cluster => cluster.ContainsVertex(descendant));
            ITileable child = childGraph is null ? descendant : childGraph;
            relation = relation is null
                ? child.DomainRelation
                : relation.ApplyRange(child.DomainRelation);
            if (childGraph is null)
            {
                return relation;
            }

            current = childGraph;
        }
    }

    private bool CheckSubGraphsDenpendence(TieredTileGraph producer, TieredTileGraph consumer)
    {
        // 1. ensure there is no dependence cycle between producer and consumer.
        var subGraphGraph = new AdjacencyGraph<TieredTileGraph, Edge<TieredTileGraph>>();
        foreach (var edge in RootGraph.Edges)
        {
            var crossesSubGraphs = (producer.ContainsVertex(edge.Source) && consumer.ContainsVertex(edge.Target)) ||
                (consumer.ContainsVertex(edge.Source) && producer.ContainsVertex(edge.Target));
            if (crossesSubGraphs && !GraphExtensions.IsFusionLegal(edge.Source, edge.Target, edge.Tag))
            {
                return false;
            }

            if (producer.ContainsVertex(edge.Source) && consumer.ContainsVertex(edge.Target))
            {
                subGraphGraph.AddVerticesAndEdge(new(producer, consumer));
            }
            else if (producer.ContainsVertex(edge.Target) && consumer.ContainsVertex(edge.Source))
            {
                subGraphGraph.AddVerticesAndEdge(new(consumer, producer));
            }
        }

#if false
        var graphviz = subGraphGraph.ToGraphviz(init => { init.FormatVertex += (_, arg) => arg.VertexFormat.Label = $"{arg.Vertex.OpId}@{arg.Vertex.Level}"; });
#endif

        bool hasCycles = false;
        bool hasDependence = false;
        var dfs = new QuikGraph.Algorithms.Search.EdgeDepthFirstSearchAlgorithm<TieredTileGraph, Edge<TieredTileGraph>>(subGraphGraph);
        dfs.BackEdge += (edge) =>
        {
            hasCycles = true;
        };

        dfs.TreeEdge += (edge) =>
        {
            if (ReferenceEquals(edge.Source, producer) && ReferenceEquals(edge.Target, consumer))
            {
                hasDependence = true;
            }
        };

        dfs.Compute();

        return hasDependence && !hasCycles;
    }

    private bool GatherSubGraphs(TieredTileGraph graph, [MaybeNullWhen(false)] out TieredTileGraph producer, [MaybeNullWhen(false)] out TieredTileGraph consumer)
    {
        producer = graph.Clusters
            .OfType<TieredTileGraph>()
            .SingleOrDefault(cluster => cluster.ContainsVertex(ProducerOp));
        consumer = graph.Clusters
            .OfType<TieredTileGraph>()
            .SingleOrDefault(cluster => cluster.ContainsVertex(ConsumerOp));
        return producer is not null &&
            consumer is not null &&
            !ReferenceEquals(producer, consumer);
    }

    private bool VisitRecursion(TieredTileGraph graph)
    {
        if (graph.Level == TargetLevel &&
            graph.ContainsVertex(ProducerOp) &&
            graph.ContainsVertex(ConsumerOp))
        {
            return true;
        }

        if (graph.Level > 0 && graph.Level <= TargetLevel)
        {
            return false;
        }

        if (TryMerge(graph))
        {
            return true;
        }

        foreach (var subGraph in graph.Clusters.OfType<TieredTileGraph>())
        {
            if (VisitRecursion(subGraph))
            {
                return true;
            }
        }

        return false;
    }
}
