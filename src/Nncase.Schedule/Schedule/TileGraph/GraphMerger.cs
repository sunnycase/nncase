// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Diagnostics.CodeAnalysis;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using QuikGraph;
using QuikGraph.Algorithms;
using QuikGraph.Algorithms.ShortestPath;
using QuikGraph.Graphviz;

namespace Nncase.Schedule.TileGraph;

public sealed record MergePoint(TileGrid Consumer, TileGrid Producer, int Level, int ConsumerAccessIndex)
{
    public override string ToString() => $"merge({Consumer}.in{ConsumerAccessIndex},{Producer},{Level})";
}

public sealed class GraphMerger
{
    public GraphMerger(TileGrid opConsumer, TileGrid opProducer, int level, int consumerAccessIndex)
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
            return false;
        }

        var scheduleOpId = RootGraph.Vertices.Select(vertex => vertex.OpId).DefaultIfEmpty(-1).Max() + 1;
        var clonedView = new TileGrid(
            ProducerOp.Grid,
            ProducerOp.Op,
            scheduleOpId,
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
        return new GraphMerger(ConsumerOp, clonedView, TargetLevel, ConsumerAccessIndex).Visit(RootGraph);
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
            return false;
        }

        if (!CheckSubGraphsDenpendence(producerGraph, consumerGraph))
        {
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
            return false;
        }

        var relayOp = dependencePath.First().Target;

        // 1.2. build the dataflow graph from consumer graph -> sub graph -> relay node.
        ITileable consumerParent = consumerGraph;
        var relationChain = new List<ITileable>();
        while (consumerParent is TieredTileGraph tileGraph)
        {
            ITileable consumerChild = tileGraph.Clusters.OfType<TieredTileGraph>().Where(sg => sg.ContainsVertex(relayOp)).Cast<ITileable>().FirstOrDefault(relayOp);
            relationChain.Add(consumerChild);
            consumerParent = consumerChild;
        }

        // 1.3 build the domain relation betwwen relay op -> producer op.
        var consumerAccessIndex = dependencePath.First().Tag;
        var readAccess = relayOp.GetAccessMap(consumerAccessIndex);
        var producerOutputIndex = GraphExtensions.GetProducerOutputIndex(relayOp.Grid.Accesses[consumerAccessIndex].Value, ProducerOp);
        var producerWriteAccess = ProducerOp.GetWriteAccess(producerOutputIndex);
        var relation = readAccess * AffineUtility.Inverse(producerWriteAccess, ProducerOp.DomainBounds.Select(Convert.ToInt64).ToArray());
        if (!relation.IsProjectedPermutation(true))
        {
            return false;
        }

        var domainRel = new DomainRelation(relayOp.OpId, ProducerOp.OpId, relation);

        // 1.4 apply domain relation until consumerGraph
        foreach (var mappable in relationChain.Reverse<ITileable>())
        {
            domainRel = mappable.DomainRelation.ApplyRange(domainRel);
        }

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
        producer = null!;
        consumer = null!;
        foreach (var s1 in graph.Clusters.OfType<TieredTileGraph>().Where(s => s.OpId == ProducerOp.OpId))
        {
            foreach (var s2 in graph.Clusters.OfType<TieredTileGraph>().Where(s => s.OpId == ConsumerOp.OpId))
            {
                if (s1.ContainsVertex(ProducerOp) && !s1.ContainsVertex(ConsumerOp) &&
                    !s2.ContainsVertex(ProducerOp) && s2.ContainsVertex(ConsumerOp))
                {
                    producer = s1;
                    consumer = s2;
                    return true;
                }
            }
        }

        return false;
    }

    private bool VisitRecursion(TieredTileGraph graph)
    {
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
