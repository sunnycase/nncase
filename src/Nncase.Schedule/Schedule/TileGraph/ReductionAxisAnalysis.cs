// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.Affine;

namespace Nncase.Schedule.TileGraph;

/// <summary>
/// Projects leaf Grid reduction-axis roles onto a hierarchy node's domain.
/// </summary>
internal static class ReductionAxisAnalysis
{
    public static bool[] GetReductionAxes(ITreeNode node)
    {
        var reductionAxes = new bool[node.DomainBoundExprs.Length];
        Visit(node, AffineMap.Identity(reductionAxes.Length), reductionAxes);
        return reductionAxes;
    }

    private static void Visit(ITreeNode node, AffineMap relation, bool[] reductionAxes)
    {
        switch (node)
        {
            case OpNode opNode:
                if (relation.Results.Length != opNode.TileAxisPolicies.Length)
                {
                    throw new InvalidOperationException(
                        $"Reduction-axis relation for Op{opNode.OpId} has {relation.Results.Length} results, " +
                        $"but the Grid has {opNode.TileAxisPolicies.Length} axes.");
                }

                for (var axis = 0; axis < opNode.TileAxisPolicies.Length; axis++)
                {
                    if (opNode.TileAxisPolicies[axis].AxisKind == GridAxisKind.Reduction)
                    {
                        var range = relation.Results[axis];
                        MarkDomainDependencies(range.Offset, reductionAxes);
                        MarkDomainDependencies(range.Extent, reductionAxes);
                    }
                }

                break;
            case TileNode tileNode:
                foreach (var child in tileNode.Children)
                {
                    Visit(child, relation * child.DomainRelation.Map, reductionAxes);
                }

                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(node), node, "Unsupported tile-tree node.");
        }
    }

    private static void MarkDomainDependencies(AffineExpr expression, bool[] reductionAxes)
    {
        switch (expression)
        {
            case AffineDim dim:
                MarkDomainAxis(dim.Position, reductionAxes);
                break;
            case AffineExtent extent:
                MarkDomainAxis(extent.Position, reductionAxes);
                break;
            case AffineAddBinary add:
                MarkDomainDependencies(add.Lhs, reductionAxes);
                MarkDomainDependencies(add.Rhs, reductionAxes);
                break;
            case AffineMulBinary multiply:
                MarkDomainDependencies(multiply.Lhs, reductionAxes);
                MarkDomainDependencies(multiply.Rhs, reductionAxes);
                break;
            case AffineDivBinary divide:
                MarkDomainDependencies(divide.Lhs, reductionAxes);
                MarkDomainDependencies(divide.Rhs, reductionAxes);
                break;
            case AffineConstant or AffineSymbol:
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(expression), expression, "Unsupported affine reduction-axis expression.");
        }
    }

    private static void MarkDomainAxis(int position, bool[] reductionAxes)
    {
        if ((uint)position >= (uint)reductionAxes.Length)
        {
            throw new InvalidOperationException(
                $"Reduction-axis affine expression references domain axis {position} in a rank-{reductionAxes.Length} domain.");
        }

        reductionAxes[position] = true;
    }
}
