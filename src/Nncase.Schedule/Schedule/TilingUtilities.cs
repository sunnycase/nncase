// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Google.OrTools.ConstraintSolver;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Utilities;
using Isl = IntegerSetLibrary;

namespace Nncase.Schedule;

public static class TilingUtilities
{
    public static Shape GetBufferShape(Expr buffer, bool maxShape)
    {
        return buffer.CheckedType switch
        {
            TensorType t => maxShape ? CompilerServices.GetMaxShape(t.Shape) : t.Shape,
            DistributedType dt => Utilities.DistributedUtility.GetDividedTensorType(dt, maxShape ? DistributedUtility.DivideFlags.MaxShape | DistributedUtility.DivideFlags.FloorDiv : DistributedUtility.DivideFlags.FloorDiv).Shape,
            _ => throw new NotSupportedException(),
        };
    }

    public static Shape GetBufferRuntimeShape(Expr buffer)
    {
        int GetDivisor(SBP sbp, Placement placement)
        {
            var divisor = 1;
            if (sbp is SBPSplit split)
            {
                divisor = split.Axes.Select(t => placement.Hierarchy[t]).Aggregate(1, (a, b) => a * b);
            }

            return divisor;
        }

        return buffer.CheckedType switch
        {
            TensorType t => t.Shape,
            DistributedType dt => new RankedShape(dt.TensorType.Shape.Select((d, i) =>
             d switch
             {
                 DimConst c => c / GetDivisor(dt.AxisPolicies[i], dt.Placement),
                 Dimension dim => dt.AxisPolicies[i] switch
                 {
                     SBPSplit s => new AsDim(IR.F.Tensors.LocalShardDim(dim, s, dt.Placement)),
                     SBPBroadCast => dim,
                     _ => throw new NotSupportedException(),
                 },
                 _ => throw new NotSupportedException(),
             }).ToArray()),
            _ => throw new NotSupportedException(),
        };
    }

    public static (Isl.set DomainSet, bool[] DomainDynamic, long[] DomainBoundValues, Dimension[] DomainBoundExprs) InferDomainBounds(Shape[] bufferRuntimeShapes, Isl.set[] shapeDomains, AffineMap[] affineMaps, HashSet<DimVar> dimVars)
    {
        var accessMaps = affineMaps.Select(AffineUtility.AsMap).ToArray();
        var reversedAccessMaps = accessMaps.Zip(shapeDomains).Select(pair =>
        {
            var reverse = pair.First.reverse();
            if (pair.Second.n_dim() == 0)
            {
                return reverse;
            }

            return reverse.intersect_domain(pair.Second);
        }).ToArray();
        Isl.map domainMap = null!;
        var shapeExprMap = new Dictionary<string, Dimension>();
        int z = 0;
        for (int i = 0; i < shapeDomains.Length; i++)
        {
            var reversedAccess = reversedAccessMaps[i];
            domainMap = domainMap is null ? reversedAccess : domainMap.flat_domain_product(reversedAccess!);
            for (int j = 0; j < shapeDomains[i].n_dim(); j++)
            {
                domainMap = domainMap.set_dim_name(Isl.dim_type.in_, (uint)z++, $"d{i}_{j}");
                shapeExprMap.Add($"d{i}_{j}", bufferRuntimeShapes[i][j]);
            }
        }

        var domainSet = domainMap.range();
        var domainBoundMpas = domainSet.max_multi_pw_aff();
        var domainDynamic = new bool[domainSet.n_dim()];
        var domainBoundValues = new long[domainSet.n_dim()];
        var domainBoundExprs = new Dimension[domainSet.n_dim()];

        for (int i = 0; i < domainSet.n_dim(); i++)
        {
            var boundMpa = domainBoundMpas.at(i);
            domainDynamic[i] = !boundMpa.is_cst();
            if (domainDynamic[i])
            {
                var dimExpr = ISLUtility.ToDimension(domainMap.max_multi_pw_aff().at(i), shapeExprMap, shapeExprMap.Keys.ToArray());
                dimExpr.Metadata = new()
                {
                    Range = new(boundMpa.min_val().num_si() + 1, boundMpa.max_val().num_si() + 1),
                };
                domainBoundExprs[i] = dimExpr;
                domainBoundValues[i] = boundMpa.max_val().num_si() + 1;
            }
            else
            {
                domainBoundExprs[i] = domainBoundValues[i] = boundMpa.max_val().num_si() + 1;
            }
        }

        if (affineMaps.Length == 0 || domainBoundValues.Length == affineMaps[0].Domains.Length)
        {
            return (domainSet, domainDynamic, domainBoundValues, domainBoundExprs);
        }

        if (TryInferProjectedPermutationDomainBounds(bufferRuntimeShapes, affineMaps, out var directDomainSet, out var directDomainDynamic, out var directDomainBoundValues, out var directDomainBoundExprs))
        {
            return (directDomainSet, directDomainDynamic, directDomainBoundValues, directDomainBoundExprs);
        }

        throw new InvalidOperationException($"Unable to infer complete affine domain bounds. Expected {affineMaps[0].Domains.Length} domains, got {domainBoundValues.Length}.");
    }

    private static bool TryInferProjectedPermutationDomainBounds(
        Shape[] bufferRuntimeShapes,
        AffineMap[] accessMaps,
        out Isl.set domainSet,
        out bool[] domainDynamic,
        out long[] domainBoundValues,
        out Dimension[] domainBoundExprs)
    {
        domainSet = null!;
        domainDynamic = Array.Empty<bool>();
        domainBoundValues = Array.Empty<long>();
        domainBoundExprs = Array.Empty<Dimension>();

        if (accessMaps.Length == 0)
        {
            return false;
        }

        var domainRank = accessMaps[0].Domains.Length;
        if (accessMaps.Any(map => map.Domains.Length != domainRank))
        {
            throw new InvalidOperationException("All affine access maps in a grid must use the same domain rank.");
        }

        var found = new bool[domainRank];
        var exprs = new Dimension[domainRank];
        var values = new long[domainRank];
        for (int bufferIndex = 0; bufferIndex < accessMaps.Length; bufferIndex++)
        {
            var accessMap = accessMaps[bufferIndex];
            var shape = bufferRuntimeShapes[bufferIndex];
            if (accessMap.Results.Length > shape.Rank)
            {
                return false;
            }

            for (int resultIndex = 0; resultIndex < accessMap.Results.Length; resultIndex++)
            {
                var range = accessMap.Results[resultIndex];
                if (range.Offset is not AffineDim dim || range.Extent is not AffineExtent extent || dim.Position != extent.Position)
                {
                    continue;
                }

                var candidate = shape[resultIndex];
                if (!CompilerServices.TryGetMaxShape(new RankedShape(new[] { candidate }), out var maxShape))
                {
                    return false;
                }

                var position = dim.Position;
                if (position < 0 || position >= domainRank)
                {
                    throw new InvalidOperationException($"Affine domain position {position} is outside domain rank {domainRank}.");
                }

                if (found[position] && values[position] != maxShape[0])
                {
                    return false;
                }

                found[position] = true;
                exprs[position] = candidate;
                values[position] = maxShape[0];
            }
        }

        if (found.Any(x => !x))
        {
            return false;
        }

        domainBoundExprs = exprs;
        domainBoundValues = values;
        domainDynamic = exprs.Select(expr => expr is not DimConst).ToArray();
        domainSet = ISLUtility.ToDomain(new RankedShape(domainBoundExprs), out _);
        return true;
    }

    /// <summary>
    /// some times we need to get the min/max value of an affine map.
    /// </summary>
    public static (Isl.multi_pw_aff MinMpa, Isl.multi_pw_aff MaxMpa) ToMinMaxMpa(AffineMap map)
    {
        var domains = string.Join(", ", Enumerable.Range(0, map.Domains.Length).Select(i => $"d{i}"));
        if (map.Symbols.Length > 0)
        {
            throw new NotSupportedException("Isl map does not support symbols yet.");
        }

        var minResults = StringUtility.Join(", ", map.Results.ToArray().Select(expr => expr.Offset switch
        {
            AffineConstant c => c.Value.ToString(),
            _ => expr.Offset.GetDisplayString(map.Symbols),
        }));

        var maxResults = StringUtility.Join(", ", map.Results.ToArray().Select(expr => expr.Offset switch
        {
            AffineConstant c => expr.Extent.GetDisplayString(map.Symbols),
            _ => expr.Offset.GetDisplayString(map.Symbols),
        }));

        return (new Isl.multi_pw_aff(Isl.ctx.Current, $"{{ [{domains}] -> [{minResults}] }}"),
                new Isl.multi_pw_aff(Isl.ctx.Current, $"{{ [{domains}] -> [{maxResults}] }}"));
    }
}
