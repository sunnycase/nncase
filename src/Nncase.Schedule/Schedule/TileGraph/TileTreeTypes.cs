// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Nncase.Graphs;
using Nncase.IR;
using Nncase.IR.Affine;
using QuikGraph;
using QuikGraph.Algorithms;
using QuikGraph.Collections;

namespace Nncase.Schedule.TileGraph;

public interface ITreeNode : ITileable
{
    ITreeNode? Parent { get; }

    TileDomainAxisSemantics DomainAxisSemantics { get; }

    TReturn Accept<TArg1, TReturn>(ITreeNodeVisitor<TArg1, TReturn> visitor, TArg1 arg1);
}

public interface ITreeNodeVisitor<in TArg1, out TReturn>
{
    TReturn Visit(TileNode value, TArg1 arg1);

    TReturn Visit(OpNode value, TArg1 arg1);
}

public sealed class OpNode : ITreeNode
{
    private readonly TileGrid _wrapped;

    public OpNode(ITreeNode? parent, TileGrid wrapped)
    {
        Parent = parent;
        _wrapped = wrapped;
    }

    public TileGrid Wrapped => _wrapped;

    public int Level => _wrapped.Level;

    public int OpId => _wrapped.OpId;

    public DomainRelation DomainRelation { get => _wrapped.DomainRelation; set => throw new NotSupportedException(); }

    public ITreeNode? Parent { get; }

    public Grid Grid => _wrapped.Grid;

    public Op Op => _wrapped.Op;

    public ImmutableArray<bool> DomainDynamic => _wrapped.DomainDynamic;

    public ImmutableArray<long> DomainBounds => _wrapped.DomainBounds;

    public ImmutableArray<Dimension> DomainBoundExprs => _wrapped.DomainBoundExprs;

    public ImmutableArray<ImmutableArray<long>> BufferShapes => _wrapped.BufferShapes;

    public ImmutableArray<DataType> BufferDataTypes => _wrapped.BufferDataTypes;

    public ImmutableArray<AffineMap> AccessMaps => _wrapped.AccessMaps;

    public ImmutableArray<GridTileAxisPolicy> TileAxisPolicies => _wrapped.TileAxisPolicies;

    public TileDomainAxisSemantics DomainAxisSemantics => _wrapped.DomainAxisSemantics;

    public ImmutableArray<MemoryEffect> LocalAccessEffects => _wrapped.LocalAccessEffects;

    public ImmutableArray<GridBufferAlias> BufferAliases => _wrapped.BufferAliases;

    public bool IsPureBufferView => _wrapped.IsPureBufferView;

    public ImmutableArray<int> ReadAccessIndices => _wrapped.ReadAccessIndices;

    public ImmutableArray<int> WriteAccessIndices => _wrapped.WriteAccessIndices;

    public bool TryGetAliasSourceAccess(int resultAccessIndex, out int sourceAccessIndex)
        => _wrapped.TryGetAliasSourceAccess(resultAccessIndex, out sourceAccessIndex);

    public AffineMap GetAccessMap(int accessIndex) => _wrapped.GetAccessMap(accessIndex);

    public AffineMap GetWriteAccess(int outputIndex) => _wrapped.GetWriteAccess(outputIndex);

    public int GetWriteAccessIndex(int outputIndex) => _wrapped.GetWriteAccessIndex(outputIndex);

    public long GetBufferElemSize(int i) => _wrapped.GetBufferElemSize(i);

    public TileWorkloadContext GetTileWorkloadContext() => new(Op, BufferShapes, BufferDataTypes);

    public TileWorkload GetTileWorkload() => CompilerServices.GetTileWorkload(Op, GetTileWorkloadContext());

    public TReturn Accept<TArg1, TReturn>(ITreeNodeVisitor<TArg1, TReturn> visitor, TArg1 arg1) => visitor.Visit(this, arg1);

    public override string ToString() => _wrapped.ToString();
}

public sealed class TileNode : ITreeNode
{
    private readonly TieredTileGraph _wrapped;
    private readonly ITreeNode[] _children;

    private TileNode(ITreeNode? parent, TieredTileGraph wrapped, int childCount)
    {
        Parent = parent;
        _wrapped = wrapped;
        _children = new ITreeNode[childCount];
    }

    public ITreeNode? Parent { get; private set; }

    public TieredTileGraph Wrapped => _wrapped;

    public ReadOnlySpan<ITreeNode> Children => _children;

    public int Level => _wrapped.Level;

    public int OpId => _wrapped.OpId;

    public TileScopeKind ScopeKind => _wrapped.ScopeKind;

    public DomainRelation DomainRelation { get => _wrapped.DomainRelation; set => throw new NotSupportedException(); }

    public ImmutableArray<bool> DomainDynamic => _wrapped.DomainDynamic;

    public ImmutableArray<Dimension> DomainBoundExprs => _wrapped.DomainBoundExprs;

    public ImmutableArray<int> LoopOrder => _wrapped.LoopOrder;

    public TileDomainAxisSemantics DomainAxisSemantics { get; private set; } = null!;

    public static TileNode FromTileGraph(TieredTileGraph rootGraph, out Dictionary<TieredTileGraph, TileNode> memo)
    {
        memo = new();
        return ConvertToTree(null, rootGraph, rootGraph, memo);
    }

    TReturn ITreeNode.Accept<TArg1, TReturn>(ITreeNodeVisitor<TArg1, TReturn> visitor, TArg1 arg1) => visitor.Visit(this, arg1);

    public override string ToString()
    {
        return _wrapped.ToString();
    }

    private static TileNode ConvertToTree(ITreeNode? parent, TieredTileGraph tileGraph, TieredTileGraph rootGraph, Dictionary<TieredTileGraph, TileNode> memo)
    {
        if (!memo.TryGetValue(tileGraph, out var tnode))
        {
            if (tileGraph.ScopeKind == TileScopeKind.Sequential && tileGraph.ClustersCount < 2)
            {
                throw new InvalidOperationException(
                    $"Sequential tile scope {tileGraph} must contain at least two independent child phases.");
            }

            if (tileGraph.ClustersCount == 0)
            {
                // sort
                var tempGraph = new AdjacencyGraph<TileGrid, Edge<TileGrid>>(allowParallelEdges: false);
                var childVertices = tileGraph.Vertices.ToArray();
                tempGraph.AddVertexRange(childVertices);
                foreach (var edge in rootGraph.Edges)
                {
                    var producers = childVertices.Where(c => c.Equals(edge.Source)).ToArray();
                    var consumers = childVertices.Where(c => c.Equals(edge.Target)).ToArray();
                    foreach (var producer in producers)
                    {
                        foreach (var consumer in consumers)
                        {
                            if (!ReferenceEquals(producer, consumer))
                            {
                                tempGraph.AddEdge(new(producer, consumer));
                            }
                        }
                    }
                }

                tnode = new TileNode(parent, tileGraph, tileGraph.VertexCount);
                int count = 0;
                foreach (var item in tempGraph.TopologicalSort())
                {
                    tnode._children[count++] = new OpNode(tnode, item);
                }

                tnode.CompleteConstruction();
            }
            else
            {
                // sort child clusters
                var tempGraph = new AdjacencyGraph<TieredTileGraph, Edge<TieredTileGraph>>(allowParallelEdges: false);
                var childClusters = tileGraph.Clusters.OfType<TieredTileGraph>().ToArray();
                tempGraph.AddVertexRange(childClusters);
                foreach (var edge in rootGraph.Edges)
                {
                    var producers = childClusters.Where(c => c.ContainsVertex(edge.Source)).ToArray();
                    var consumers = childClusters.Where(c => c.ContainsVertex(edge.Target)).ToArray();
                    foreach (var producer in producers)
                    {
                        foreach (var consumer in consumers)
                        {
                            if (!ReferenceEquals(producer, consumer))
                            {
                                tempGraph.AddEdge(new(producer, consumer));
                            }
                        }
                    }
                }

                tnode = new TileNode(parent, tileGraph, tileGraph.ClustersCount);
                int count = 0;
                foreach (var item in tempGraph.TopologicalSort())
                {
                    tnode._children[count++] = ConvertToTree(tnode, item, rootGraph, memo);
                }

                tnode.CompleteConstruction();
            }

            memo.Add(tileGraph, tnode);
        }

        return tnode;
    }

    private void CompleteConstruction()
    {
        if (DomainAxisSemantics is not null)
        {
            throw new InvalidOperationException($"Tile-tree node {this} was completed more than once.");
        }

        if (_children.Any(child => child is null))
        {
            throw new InvalidOperationException($"Tile-tree node {this} has uninitialized children.");
        }

        var rank = DomainRelation.Map.Results.Length;
        if (DomainBoundExprs.Length != rank)
        {
            throw new InvalidOperationException(
                $"Tile-tree node {this} has rank-{rank} domain relation but {DomainBoundExprs.Length} domain bounds.");
        }

        DomainAxisSemantics = OpId == -1 || ScopeKind == TileScopeKind.Sequential
            ? TileDomainAxisSemantics.Empty(rank)
            : TileDomainAxisSemantics.Compose(
                ToString(),
                rank,
                _children.Select(child => (child.DomainRelation, child.DomainAxisSemantics)));
    }
}
