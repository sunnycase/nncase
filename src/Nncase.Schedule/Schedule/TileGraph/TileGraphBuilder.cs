// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Utilities;
using QuikGraph;
using Isl = IntegerSetLibrary;

namespace Nncase.Schedule.TileGraph;

public sealed class TieredTileGraphBuilder : ExprVisitor<Unit, Unit>
{
    private readonly Dictionary<Grid, TileGrid> _memo;
    private readonly Dictionary<Grid, TieredTileGraph> _exprMemo;
    private int _opId;

    private TieredTileGraphBuilder(int levelCount, HashSet<Grid> outputGrids)
    {
        RootGraph = new(new AdjacencyGraph<TileGrid, EquatableTaggedEdge<TileGrid, int>>());
        _memo = new();
        _exprMemo = new();
        LevelCount = levelCount;
        OutputGrids = outputGrids;
    }

    public TieredTileGraph RootGraph { get; }

    public int LevelCount { get; }

    /// <summary>
    /// Gets the output grids, for mark the grid is required by outside.
    /// </summary>
    public HashSet<Grid> OutputGrids { get; }

    public static TieredTileGraph Build(BaseExpr expr, int levelCount, out Dictionary<Grid, TieredTileGraph> exprMemo)
    {
        HashSet<Grid> outputGrids = new();
        CollectOutputGrids(expr, outputGrids);

        var builder = new TieredTileGraphBuilder(levelCount, outputGrids);
        builder.Visit(expr);
        exprMemo = builder._exprMemo;
        return builder.RootGraph;
    }

    protected override Unit DefaultVisitLeaf(BaseExpr expr) => default;

    protected override Unit VisitLeafGrid(Grid current)
    {
        if (_memo.TryGetValue(current, out var node))
        {
            return default;
        }

        var accesses = current.Accesses.ToArray();
        var buffers = accesses.Select(access => access.Buffer).ToArray();
        var bufferShapeValues = buffers.Select(b => TilingUtilities.GetBufferShape(b, true).ToValueArray()).ToArray();
        var affineDomainRank = accesses.First(access => access.IsAffine).AffineMap.Domains.Length;
        var constraintAccesses = accesses
            .Where(access => access.IsAffine && access.DomainMode == GridDomainMode.Constraint)
            .ToArray();
        var physicalRuntimeShapes = constraintAccesses
            .Select(access => AffineDomainInference.GetBufferRuntimeShape(access.Buffer))
            .ToArray();
        var physicalAccessMaps = constraintAccesses
            .Select(access => access.AffineMap)
            .ToArray();
        var domainBoundExprs = AffineDomainInference.IntersectDomainBounds(
            current.DomainBounds,
            physicalRuntimeShapes,
            physicalAccessMaps);
        var domainBoundValues = CompilerServices.GetMaxShape(new RankedShape(domainBoundExprs));
        var domainDynamic = domainBoundExprs.Select(bound => bound is not DimConst).ToArray();

        var copId = _opId++;
        var domainDims = affineDomainRank;
        var dimNames = Enumerable.Range(0, domainDims).Select(i => $"Op{copId}_d{i}").ToArray();
        if (current.Body[0] is not Call { Target: Op op })
        {
            throw new InvalidOperationException("body is not call");
        }

        var attr = TileGridAttribute.None;
        if (OutputGrids.Contains(current))
        {
            attr |= TileGridAttribute.LiveOut;
        }

        var opNode = new TileGrid(current, op, copId, copId, domainBoundValues, new DomainRelation(copId, copId, AffineMap.Identity(domainDims)), domainBoundExprs, domainDynamic, bufferShapeValues, attr);

        var tileNodeRoot = RootGraph.CreateCluster<TieredTileGraph>(LevelCount - 1, copId, new DomainRelation(copId, copId, AffineMap.Identity(domainDims)), domainBoundExprs, domainDynamic);
        var tileNodeTail = tileNodeRoot;
        for (int l = LevelCount - 2; l >= 0; l--)
        {
            tileNodeTail = tileNodeTail.CreateCluster<TieredTileGraph>(l, copId, new DomainRelation(copId, copId, AffineMap.Identity(domainDims)), domainBoundExprs, domainDynamic);
        }

        tileNodeTail.AddVertex(opNode);

        for (int i = 0; i < current.Accesses.Length; i++)
        {
            var access = current.Accesses[i];
            if (access.IsRead && GraphExtensions.TryGetProducerGrid(access.Value, out var producer, out _))
            {
                var producerNode = _memo[producer];
                RootGraph.AddEdge(new(producerNode, opNode, i));
            }
        }

        _memo.Add(current, opNode);
        _exprMemo.Add(current, tileNodeRoot);

        return default;
    }

    private static void CollectOutputGrids(BaseExpr expr, HashSet<Grid> outputGrids)
    {
        switch (expr)
        {
            case Grid grid:
                outputGrids.Add(grid);
                break;
            case IR.Tuple tuple:
                foreach (var field in tuple.Fields)
                {
                    CollectOutputGrids(field, outputGrids);
                }

                break;
            case Expr tensorExpr when GraphExtensions.TryGetProducerGrid(tensorExpr, out var producer, out _):
                outputGrids.Add(producer);
                break;
            default:
                foreach (var operand in expr.Operands)
                {
                    CollectOutputGrids(operand, outputGrids);
                }

                break;
        }
    }
}
