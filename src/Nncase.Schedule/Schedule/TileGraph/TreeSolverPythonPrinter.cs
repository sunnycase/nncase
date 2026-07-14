// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.CodeDom.Compiler;
using System.Reactive;
using Google.OrTools.ConstraintSolver;

namespace Nncase.Schedule.TileGraph;

public sealed class TreeSolverPythonPrinter : TreeSolverBase<IntExpr>, ITreeNodeVisitor<(ITreeNode? Parent, IndentedTextWriter Writer), Unit>
{
    private readonly Dictionary<TileNode, List<long>> _bounds = new();

    public TreeSolverPythonPrinter(Assignment solution, Solver solver, Dictionary<OpNode, OpNodeInfo<IntExpr>> opNodeMemo, Dictionary<TileNode, TileNodeInfo<IntExpr>> tileNodeMemo, Dictionary<ITileable, DomainInfo<IntExpr>> tileableNodeMemo, INTTTargetOptions targetOptions)
        : base(solver, opNodeMemo, tileNodeMemo, tileableNodeMemo, targetOptions)
    {
        Solution = solution;
    }

    public Assignment Solution { get; }

    public Unit Visit(TileNode value, (ITreeNode? Parent, IndentedTextWriter Writer) context)
    {
        var (parent, writer) = context;
        writer.WriteLine($"# {value}");
        writer.WriteLine($"# ScopeKind: {value.ScopeKind}");
        writer.WriteLine($"# Domain Relation: {value.DomainRelation}");
        var domainInfo = TileableNodeMemo[value];
        var bounds = Enumerable.Repeat(0L, domainInfo.TileVars.Length).ToList();
        for (int position = 0; position < value.LoopOrder.Length; position++)
        {
            var axis = value.LoopOrder[position];
            WriteBuffer(value, position, writer);
            var trip = Solution.Value(TileNodeMemo[value].TripCounts[position + 1].Var());

            // 2. write loop.
            long parentBounds = 0;
            if (parent is null)
            {
                value.Walk(child =>
                {
                    if (child is OpNode opNode && opNode.OpId == value.OpId)
                    {
                        parentBounds = opNode.DomainBounds[axis];
                    }
                });
            }
            else if (parent is TileNode parentTile)
            {
                if (axis < _bounds[parentTile].Count)
                {
                    parentBounds = _bounds[parentTile][axis];
                }
                else
                {
                    value.Walk(child =>
                    {
                        if (child is OpNode opNode && opNode.OpId == value.OpId)
                        {
                            parentBounds = opNode.DomainBounds[axis];
                        }
                    });
                }
            }

            var tile = Solution.Value(domainInfo.TileVars[axis].Var());
            writer.WriteLine($"for {domainInfo.TileVars[axis].ToSimplifyString()} in range(0, {parentBounds}, {parentBounds / tile}):  # trip: {trip}");
            bounds[axis] = parentBounds / tile;
            writer.Indent++;
        }

        WriteBuffer(value, domainInfo.TileVars.Length, writer);

        _bounds.Add(value, bounds);

        foreach (var item in value.Children)
        {
            item.Accept(this, (value, writer));
        }

        for (int i = 0; i < domainInfo.TileVars.Length; i++)
        {
            writer.Indent--;
        }

        return default;
    }

    public void WriteBuffer(TileNode value, int i, IndentedTextWriter writer)
    {
        foreach (var (bid, bufferInfo) in TileNodeMemo[value].BufferInfoMap)
        {
            if (!bufferInfo.Places.Any())
            {
                continue;
            }

            var place = bufferInfo.Places[i];
            for (int sl = place.Length - 1; sl >= 0; sl--)
            {
                if (Solution.Value(place[sl].Var()) == 1)
                {
                    var shape = bufferInfo.Shapes[i].Select(s => Solution.Value(s.Var())).ToArray();
                    var size = Solution.Value(bufferInfo.Sizes[i].Var());
                    writer.WriteLine($"{bid}[{string.Join(", ", shape)}] @ L{sl + 1}  # size: {size}");
                }
            }
        }
    }

    public Unit Visit(OpNode value, (ITreeNode? Parent, IndentedTextWriter Writer) context)
    {
        var (parent, writer) = context;
        var opinfo = OpNodeMemo[value];
        var shapes = string.Join(", ", opinfo.Shapes.Select((sp, i) =>
        {
            var endpoint = value.Grid.Accesses[i].IsRead ? BufferEndpoint.Input : BufferEndpoint.Output;
            return $"{new BufferIdentity(value.Wrapped, i, endpoint)}[" + string.Join(',', sp.Select(s => Solution.Value(s.Var()))) + "]";
        }));
        var size = string.Join(", ", opinfo.Sizes.Select(s => Solution.Value(s.Var())));
        writer.WriteLine($"{value.Op.GetType()}({value.Op.DisplayProperty()}, {shapes})  # size: {size}");
        return default;
    }
}
