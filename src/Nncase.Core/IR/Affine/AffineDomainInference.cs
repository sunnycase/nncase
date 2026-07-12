// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Utilities;
using Isl = IntegerSetLibrary;

namespace Nncase.IR.Affine;

/// <summary>
/// Infers a bounded grid iteration domain from its logical constraint accesses.
/// </summary>
public static class AffineDomainInference
{
    public static Shape GetBufferRuntimeShape(Expr buffer)
    {
        static int GetDivisor(SBP sbp, Placement placement)
        {
            var divisor = 1;
            if (sbp is SBPSplit split)
            {
                divisor = split.Axes.Select(axis => placement.Hierarchy[axis]).Aggregate(1, (lhs, rhs) => lhs * rhs);
            }

            return divisor;
        }

        static Dimension GetLocalShardDimension(Dimension globalDimension, SBPSplit split, Placement placement)
        {
            var divisor = GetDivisor(split, placement);
            if (globalDimension is DimConst fixedDimension &&
                split.Granularity is null &&
                fixedDimension.Value % divisor == 0)
            {
                return fixedDimension / divisor;
            }

            if (globalDimension is DimConst fixedGranularDimension &&
                split.Granularity is DimConst fixedGranularity &&
                fixedGranularDimension.Value == fixedGranularity.Value * divisor)
            {
                return fixedGranularity;
            }

            var globalMax = CompilerServices.GetMaxShape(new RankedShape([globalDimension]))[0];
            var localMax = split.Granularity is { } granularity
                ? System.Math.Min(globalMax, CompilerServices.GetMaxShape(new RankedShape([granularity]))[0])
                : MathUtility.CeilDiv(globalMax, divisor);
            var runtimeDimension = new AsDim(F.Tensors.LocalShardDim(globalDimension, split, placement))
            {
                Metadata = new()
                {
                    Range = new(0, localMax),
                },
            };
            return runtimeDimension;
        }

        return buffer.CheckedType switch
        {
            TensorType tensorType => tensorType.Shape,
            DistributedType distributedType => new RankedShape(distributedType.TensorType.Shape.Select((dimension, axis) =>
                distributedType.AxisPolicies[axis] switch
                {
                    SBPSplit split => GetLocalShardDimension(dimension, split, distributedType.Placement),
                    SBPBroadCast => dimension,
                    _ => throw new NotSupportedException($"Unsupported distributed axis policy {distributedType.AxisPolicies[axis]} for affine domain inference."),
                }).ToArray()),
            _ => throw new NotSupportedException($"Affine domain inference requires a tensor buffer, got {buffer.CheckedType}."),
        };
    }

    public static Dimension[] InferDomainBounds(Shape[] bufferRuntimeShapes, AffineMap[] affineMaps)
    {
        if (bufferRuntimeShapes.Length != affineMaps.Length)
        {
            throw new ArgumentException(
                $"Affine domain inference expects one shape per access, got {bufferRuntimeShapes.Length} shapes and {affineMaps.Length} maps.");
        }

        if (TryInferProjectedPermutationDomainBounds(
            bufferRuntimeShapes,
            affineMaps,
            requireAllResultsProjected: false,
            out var directDomainBoundExprs))
        {
            return directDomainBoundExprs;
        }

        using var context = Isl.ctx.Create();
        var (shapeDomains, shapeParameters) = CreateSymbolicDomains(bufferRuntimeShapes);
        return InferDomainBoundsCore(
            bufferRuntimeShapes,
            shapeDomains,
            affineMaps,
            shapeParameters).DomainBoundExprs;
    }

    /// <summary>
    /// Intersects an existing logical grid domain with the domains reachable through
    /// the supplied physical buffer accesses.
    /// </summary>
    public static Dimension[] IntersectDomainBounds(
        ReadOnlySpan<Dimension> logicalDomainBounds,
        Shape[] bufferRuntimeShapes,
        AffineMap[] affineMaps)
    {
        if (bufferRuntimeShapes.Length != affineMaps.Length)
        {
            throw new ArgumentException(
                $"Affine domain intersection expects one shape per access, got {bufferRuntimeShapes.Length} shapes and {affineMaps.Length} maps.");
        }

        if (logicalDomainBounds.IsEmpty)
        {
            if (affineMaps.Any(map => map.Domains.Length != 0))
            {
                throw new InvalidOperationException("Rank-0 logical domains require rank-0 physical affine accesses.");
            }

            return Array.Empty<Dimension>();
        }

        var logicalDomainRank = logicalDomainBounds.Length;
        if (affineMaps.Any(map => map.Domains.Length != logicalDomainRank))
        {
            throw new InvalidOperationException(
                $"Physical affine accesses must use the rank-{logicalDomainRank} logical grid domain.");
        }

        if (affineMaps.Length == 0)
        {
            return logicalDomainBounds.ToArray();
        }

        var constrainedShapes = new[] { (Shape)new RankedShape(logicalDomainBounds.ToArray()) }
            .Concat(bufferRuntimeShapes)
            .ToArray();
        var constrainedMaps = new[] { AffineMap.Identity(logicalDomainRank) }
            .Concat(affineMaps)
            .ToArray();

        // The logical identity access is an explicit domain constraint. Only use the
        // projection fast path when every physical access is separable; otherwise the
        // generic ISL path must see all constraints together.
        if (TryInferProjectedPermutationDomainBounds(
            constrainedShapes,
            constrainedMaps,
            requireAllResultsProjected: true,
            out var directDomainBoundExprs))
        {
            return directDomainBoundExprs;
        }

        using var context = Isl.ctx.Create();
        var (shapeDomains, shapeParameters) = CreateSymbolicDomains(constrainedShapes);
        return InferDomainBoundsCore(
            constrainedShapes,
            shapeDomains,
            constrainedMaps,
            shapeParameters).DomainBoundExprs;
    }

    private static (Isl.set[] Domains, Dictionary<string, Dimension> Parameters) CreateSymbolicDomains(Shape[] shapes)
    {
        var parameterNames = new Dictionary<Dimension, string>();
        foreach (var shape in shapes)
        {
            for (var axis = 0; axis < shape.Rank; axis++)
            {
                var dimension = shape[axis];
                if (dimension is not DimConst && !parameterNames.ContainsKey(dimension))
                {
                    parameterNames.Add(dimension, $"p{parameterNames.Count}");
                }
            }
        }

        var domains = shapes
            .Select(shape => ISLUtility.ToSymbolicDomain(shape, parameterNames, out _))
            .ToArray();
        var parameters = parameterNames.ToDictionary(pair => pair.Value, pair => pair.Key);
        return (domains, parameters);
    }

    private static (Isl.set DomainSet, bool[] DomainDynamic, long[] DomainBoundValues, Dimension[] DomainBoundExprs) InferDomainBoundsCore(
        Shape[] bufferRuntimeShapes,
        Isl.set[] shapeDomains,
        AffineMap[] affineMaps,
        IReadOnlyDictionary<string, Dimension> shapeParameters)
    {
        var accessMaps = affineMaps.Select(AffineUtility.AsMap).ToArray();
        var reversedAccessMaps = accessMaps.Zip(shapeDomains).Select(pair =>
        {
            var reverse = pair.First.reverse();
            return pair.Second.n_dim() == 0 ? reverse : reverse.intersect_domain(pair.Second);
        }).ToArray();
        Isl.map domainMap = null!;
        var shapeExprMap = new Dictionary<string, Dimension>(shapeParameters);
        var flattenedDimension = 0;
        for (var bufferIndex = 0; bufferIndex < shapeDomains.Length; bufferIndex++)
        {
            var reversedAccess = reversedAccessMaps[bufferIndex];
            domainMap = domainMap is null ? reversedAccess : domainMap.flat_domain_product(reversedAccess);
            for (var axis = 0; axis < shapeDomains[bufferIndex].n_dim(); axis++)
            {
                domainMap = domainMap.set_dim_name(Isl.dim_type.in_, (uint)flattenedDimension++, $"d{bufferIndex}_{axis}");
                shapeExprMap.Add($"d{bufferIndex}_{axis}", bufferRuntimeShapes[bufferIndex][axis]);
            }
        }

        var domainSet = domainMap.range();
        var domainBoundMpas = domainSet.max_multi_pw_aff();
        var domainDynamic = new bool[domainSet.n_dim()];
        var domainBoundValues = new long[domainSet.n_dim()];
        var domainBoundExprs = new Dimension[domainSet.n_dim()];
        for (var axis = 0; axis < domainSet.n_dim(); axis++)
        {
            var boundMpa = domainBoundMpas.at(axis);
            domainDynamic[axis] = !boundMpa.is_cst();
            if (domainDynamic[axis])
            {
                var dimension = (ISLUtility.ToDimension(boundMpa, shapeExprMap) + 1).Simplify();
                dimension.Metadata = new()
                {
                    Range = new(boundMpa.min_val().num_si() + 1, boundMpa.max_val().num_si() + 1),
                };
                domainBoundExprs[axis] = dimension;
                domainBoundValues[axis] = boundMpa.max_val().num_si() + 1;
            }
            else
            {
                domainBoundExprs[axis] = domainBoundValues[axis] = boundMpa.max_val().num_si() + 1;
            }
        }

        if (affineMaps.Length == 0 || domainBoundValues.Length == affineMaps[0].Domains.Length)
        {
            return (domainSet, domainDynamic, domainBoundValues, domainBoundExprs);
        }

        throw new InvalidOperationException(
            $"Unable to infer complete affine domain bounds. Expected {affineMaps[0].Domains.Length} domains, got {domainBoundValues.Length}.");
    }

    private static bool TryInferProjectedPermutationDomainBounds(
        Shape[] bufferRuntimeShapes,
        AffineMap[] accessMaps,
        bool requireAllResultsProjected,
        out Dimension[] domainBoundExprs)
    {
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
        var expressions = new Dimension[domainRank];
        for (var bufferIndex = 0; bufferIndex < accessMaps.Length; bufferIndex++)
        {
            var accessMap = accessMaps[bufferIndex];
            var shape = bufferRuntimeShapes[bufferIndex];
            if (accessMap.Results.Length > shape.Rank)
            {
                return false;
            }

            for (var resultIndex = 0; resultIndex < accessMap.Results.Length; resultIndex++)
            {
                var range = accessMap.Results[resultIndex];
                if (!TryGetScaledProjection(range, out var position, out var scale))
                {
                    if (requireAllResultsProjected && !IsConstantRange(range))
                    {
                        return false;
                    }

                    continue;
                }

                var candidate = scale == 1
                    ? shape[resultIndex]
                    : Dimension.CeilDiv(shape[resultIndex], scale).Simplify();
                if (position < 0 || position >= domainRank)
                {
                    throw new InvalidOperationException($"Affine domain position {position} is outside domain rank {domainRank}.");
                }

                if (found[position])
                {
                    if (!expressions[position].Simplify().Equals(candidate.Simplify()))
                    {
                        expressions[position] = Dimension.Min(expressions[position], candidate).Simplify();
                    }

                    continue;
                }

                found[position] = true;
                expressions[position] = candidate;
            }
        }

        if (found.Any(value => !value))
        {
            return false;
        }

        domainBoundExprs = expressions;
        return true;
    }

    private static bool IsConstantRange(AffineRange range)
        => range.Offset is AffineConstant && range.Extent is AffineConstant;

    private static bool TryGetScaledProjection(AffineRange range, out int position, out long scale)
    {
        position = -1;
        scale = 0;
        if (!TryGetProjection(range.Offset, out var offsetPosition, out var offsetScale) ||
            !TryGetProjection(range.Extent, out var extentPosition, out var extentScale, requireExtent: true) ||
            offsetPosition != extentPosition ||
            (extentScale != 1 && extentScale != offsetScale))
        {
            return false;
        }

        position = offsetPosition;
        scale = offsetScale;
        return true;

        static bool TryGetProjection(
            AffineExpr expression,
            out int projectionPosition,
            out long projectionScale,
            bool requireExtent = false)
        {
            projectionPosition = -1;
            projectionScale = 0;
            (int Position, long Scale)? projection = expression switch
            {
                AffineDim dim when !requireExtent => (dim.Position, 1),
                AffineExtent extent when requireExtent => (extent.Position, 1),
                AffineMulBinary { Lhs: AffineDim dim, Rhs: AffineConstant constant } when !requireExtent => (dim.Position, constant.Value),
                AffineMulBinary { Lhs: AffineConstant constant, Rhs: AffineDim dim } when !requireExtent => (dim.Position, constant.Value),
                AffineMulBinary { Lhs: AffineExtent extent, Rhs: AffineConstant constant } when requireExtent => (extent.Position, constant.Value),
                AffineMulBinary { Lhs: AffineConstant constant, Rhs: AffineExtent extent } when requireExtent => (extent.Position, constant.Value),
                _ => null,
            };
            if (projection is not { Scale: > 0 } value)
            {
                return false;
            }

            projectionPosition = value.Position;
            projectionScale = value.Scale;
            return true;
        }
    }
}
