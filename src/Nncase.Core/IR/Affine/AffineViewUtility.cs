// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Utilities;

namespace Nncase.IR.Affine;

/// <summary>
/// Builds and validates target-independent affine tensor views.
/// </summary>
public static class AffineViewUtility
{
    /// <summary>
    /// Tries to build a storage-preserving affine transform between two tensor types.
    /// </summary>
    public static bool TryCreate(IRType sourceType, IRType resultType, out AffineViewTransform transform)
    {
        transform = null!;
        if (!HaveCompatibleDistributedPlacement(sourceType, resultType))
        {
            return false;
        }

        if (!TryGetTensorType(sourceType, out var sourceTensor) ||
            !TryGetTensorType(resultType, out var resultTensor) ||
            sourceTensor.Shape is not RankedShape sourceShape ||
            resultTensor.Shape is not RankedShape resultShape)
        {
            return false;
        }

        var storageUnitBytes = GreatestCommonDivisor(sourceTensor.DType.SizeInBytes, resultTensor.DType.SizeInBytes);
        var sourceLanes = sourceTensor.DType.SizeInBytes / storageUnitBytes;
        var resultLanes = resultTensor.DType.SizeInBytes / storageUnitBytes;
        if (sourceLanes == resultLanes && HaveSameShape(sourceShape, resultShape))
        {
            transform = AffineViewTransform.Identity(sourceShape);
            return HaveCompatibleDistributedStorage(sourceType, resultType, transform);
        }

        if (TryGetSingletonDimensionMaps(sourceShape, resultShape, sourceLanes, resultLanes, out transform))
        {
            return HaveCompatibleDistributedStorage(sourceType, resultType, transform);
        }

        if (sourceShape.Rank == 0 || resultShape.Rank == 0)
        {
            if (sourceShape.Rank == resultShape.Rank && sourceTensor.DType.SizeInBytes == resultTensor.DType.SizeInBytes)
            {
                transform = AffineViewTransform.Identity(sourceShape);
                return HaveCompatibleDistributedStorage(sourceType, resultType, transform);
            }

            return false;
        }

        var prefixRank = GetCommonPrefixRank(sourceShape, resultShape);
        var created = TryGetFlatToFlatMaps(sourceShape, resultShape, prefixRank, sourceLanes, resultLanes, out transform) ||
            TryGetFlatInputToOutputMajorMaps(sourceShape, resultShape, prefixRank, sourceLanes, resultLanes, out transform) ||
            TryGetInputMajorToFlatOutputMaps(sourceShape, resultShape, prefixRank, sourceLanes, resultLanes, out transform) ||
            (!HasSuffixSplit(sourceType, prefixRank) &&
                !HasSuffixSplit(resultType, prefixRank) &&
                TryGetPrefixFullTileMaps(sourceShape, resultShape, prefixRank, sourceLanes, resultLanes, out transform));
        return created && HaveCompatibleDistributedStorage(sourceType, resultType, transform);
    }

    /// <summary>
    /// Verifies the structural and storage invariants of an affine view.
    /// </summary>
    public static string? Verify(IRType sourceType, IRType resultType, AffineViewTransform transform)
    {
        if (!TryGetTensorType(sourceType, out var sourceTensor) || !TryGetTensorType(resultType, out var resultTensor))
        {
            return $"AffineView requires tensor or distributed tensor types, got {sourceType} and {resultType}.";
        }

        if (sourceType is DistributedType { Partial: not null } || resultType is DistributedType { Partial: not null })
        {
            return "AffineView does not support partial distributed tensors.";
        }

        if (!HaveCompatibleDistributedPlacement(sourceType, resultType))
        {
            return $"AffineView distributed placement is incompatible: source={sourceType}, result={resultType}.";
        }

        if (sourceTensor.Shape is not RankedShape sourceShape || resultTensor.Shape is not RankedShape resultShape)
        {
            return "AffineView requires ranked source and result shapes.";
        }

        if (transform.SourceMap.Results.Length != sourceShape.Rank || transform.ResultMap.Results.Length != resultShape.Rank)
        {
            return $"AffineView map rank mismatch: source map={transform.SourceMap.Results.Length}, source rank={sourceShape.Rank}, result map={transform.ResultMap.Results.Length}, result rank={resultShape.Rank}.";
        }

        if (!CoversShape(transform.SourceMap, transform.DomainBounds, sourceShape) ||
            !CoversShape(transform.ResultMap, transform.DomainBounds, resultShape))
        {
            return $"AffineView maps must cover the complete source/result shapes: transform={transform}.";
        }

        var sourceBytes = (sourceShape.Prod() * sourceTensor.DType.SizeInBytes).Simplify();
        var resultBytes = (resultShape.Prod() * resultTensor.DType.SizeInBytes).Simplify();
        if (!sourceBytes.Equals(resultBytes))
        {
            return $"AffineView storage size mismatch: source={sourceBytes} bytes, result={resultBytes} bytes.";
        }

        if (!HaveCompatibleDistributedStorage(sourceType, resultType, transform))
        {
            return $"AffineView distributed shard regions are incompatible: source={sourceType}, result={resultType}.";
        }

        return null;
    }

    private static bool TryGetFlatToFlatMaps(RankedShape sourceShape, RankedShape resultShape, int prefixRank, int sourceLane, int resultLane, out AffineViewTransform transform)
    {
        transform = null!;
        if (sourceShape.Rank != prefixRank + 1 ||
            resultShape.Rank != prefixRank + 1 ||
            !HasFixedSuffix(sourceShape, prefixRank) ||
            !HasFixedSuffix(resultShape, prefixRank) ||
            GetScalarSuffixElementCount(sourceShape, prefixRank, sourceLane) != GetScalarSuffixElementCount(resultShape, prefixRank, resultLane))
        {
            return false;
        }

        var domains = F.Affine.Domains(prefixRank + 1);
        if (resultLane % sourceLane == 0)
        {
            var sourceScale = resultLane / sourceLane;
            transform = CreateTransform(
                domains,
                BuildFlatSuffixRanges(domains, sourceShape, prefixRank, sourceScale, suffixDomainIsIdentity: false),
                BuildFlatSuffixRanges(domains, resultShape, prefixRank, 1, suffixDomainIsIdentity: true),
                resultShape.Dimensions[..(prefixRank + 1)]);
            return true;
        }

        if (sourceLane % resultLane == 0)
        {
            var resultScale = sourceLane / resultLane;
            transform = CreateTransform(
                domains,
                BuildFlatSuffixRanges(domains, sourceShape, prefixRank, 1, suffixDomainIsIdentity: true),
                BuildFlatSuffixRanges(domains, resultShape, prefixRank, resultScale, suffixDomainIsIdentity: false),
                sourceShape.Dimensions[..(prefixRank + 1)]);
            return true;
        }

        return false;
    }

    private static bool TryGetSingletonDimensionMaps(RankedShape sourceShape, RankedShape resultShape, int sourceLane, int resultLane, out AffineViewTransform transform)
    {
        transform = null!;
        if (sourceLane != resultLane)
        {
            return false;
        }

        var sourceDomainBounds = sourceShape.Dimensions.ToArray().Where(dimension => !IsUnitDimension(dimension)).ToArray();
        var resultDomainBounds = resultShape.Dimensions.ToArray().Where(dimension => !IsUnitDimension(dimension)).ToArray();
        if (sourceDomainBounds.Length != resultDomainBounds.Length ||
            !sourceDomainBounds.Zip(resultDomainBounds).All(pair => IsSameDimension(pair.First, pair.Second)))
        {
            return false;
        }

        var domains = F.Affine.Domains(sourceDomainBounds.Length);
        transform = CreateTransform(
            domains,
            BuildSingletonProjectedRanges(domains, sourceShape),
            BuildSingletonProjectedRanges(domains, resultShape),
            sourceDomainBounds);
        return true;
    }

    private static bool TryGetFlatInputToOutputMajorMaps(RankedShape sourceShape, RankedShape resultShape, int prefixRank, int sourceLane, int resultLane, out AffineViewTransform transform)
    {
        transform = null!;
        if (sourceShape.Rank != prefixRank + 1 ||
            resultShape.Rank <= prefixRank + 1 ||
            !HasFixedSuffix(sourceShape, prefixRank) ||
            !HasFixedSuffix(resultShape, prefixRank) ||
            GetScalarSuffixElementCount(sourceShape, prefixRank, sourceLane) != GetScalarSuffixElementCount(resultShape, prefixRank, resultLane))
        {
            return false;
        }

        var scalarElementsPerResultMajor = ProductFixedSuffix(resultShape, prefixRank + 1) * resultLane;
        if (scalarElementsPerResultMajor % sourceLane != 0)
        {
            return false;
        }

        var domains = F.Affine.Domains(prefixRank + 1);
        transform = CreateTransform(
            domains,
            BuildFlatSuffixRanges(domains, sourceShape, prefixRank, scalarElementsPerResultMajor / sourceLane, suffixDomainIsIdentity: false),
            BuildMajorSuffixRanges(domains, resultShape, prefixRank),
            resultShape.Dimensions[..(prefixRank + 1)]);
        return true;
    }

    private static bool TryGetInputMajorToFlatOutputMaps(RankedShape sourceShape, RankedShape resultShape, int prefixRank, int sourceLane, int resultLane, out AffineViewTransform transform)
    {
        transform = null!;
        if (sourceShape.Rank <= prefixRank + 1 ||
            resultShape.Rank != prefixRank + 1 ||
            !HasFixedSuffix(sourceShape, prefixRank) ||
            !HasFixedSuffix(resultShape, prefixRank) ||
            GetScalarSuffixElementCount(sourceShape, prefixRank, sourceLane) != GetScalarSuffixElementCount(resultShape, prefixRank, resultLane))
        {
            return false;
        }

        var scalarElementsPerSourceMajor = ProductFixedSuffix(sourceShape, prefixRank + 1) * sourceLane;
        if (scalarElementsPerSourceMajor % resultLane != 0)
        {
            return false;
        }

        var domains = F.Affine.Domains(prefixRank + 1);
        transform = CreateTransform(
            domains,
            BuildMajorSuffixRanges(domains, sourceShape, prefixRank),
            BuildFlatSuffixRanges(domains, resultShape, prefixRank, scalarElementsPerSourceMajor / resultLane, suffixDomainIsIdentity: false),
            sourceShape.Dimensions[..(prefixRank + 1)]);
        return true;
    }

    private static bool TryGetPrefixFullTileMaps(RankedShape sourceShape, RankedShape resultShape, int prefixRank, int sourceLane, int resultLane, out AffineViewTransform transform)
    {
        transform = null!;
        if (!HasFixedSuffix(sourceShape, prefixRank) ||
            !HasFixedSuffix(resultShape, prefixRank) ||
            GetScalarSuffixElementCount(sourceShape, prefixRank, sourceLane) != GetScalarSuffixElementCount(resultShape, prefixRank, resultLane))
        {
            return false;
        }

        var domains = F.Affine.Domains(prefixRank);
        transform = CreateTransform(
            domains,
            BuildPrefixFullTileRanges(domains, sourceShape, prefixRank),
            BuildPrefixFullTileRanges(domains, resultShape, prefixRank),
            sourceShape.Dimensions[..prefixRank]);
        return true;
    }

    private static AffineViewTransform CreateTransform(AffineDomain[] domains, AffineRange[] sourceRanges, AffineRange[] resultRanges, ReadOnlySpan<Dimension> domainBounds)
        => new(
            new AffineMap(domains, default, sourceRanges),
            new AffineMap(domains, default, resultRanges),
            new IRArray<Dimension>(domainBounds));

    private static AffineRange[] BuildPrefixFullTileRanges(AffineDomain[] domains, RankedShape shape, int prefixRank)
    {
        var ranges = new AffineRange[shape.Rank];
        for (var axis = 0; axis < shape.Rank; axis++)
        {
            ranges[axis] = axis < prefixRank
                ? new AffineRange(domains[axis].Offset, domains[axis].Extent)
                : new AffineRange(0, GetFixedDimension(shape[axis]));
        }

        return ranges;
    }

    private static AffineRange[] BuildSingletonProjectedRanges(AffineDomain[] domains, RankedShape shape)
    {
        var ranges = new AffineRange[shape.Rank];
        var domainIndex = 0;
        for (var axis = 0; axis < shape.Rank; axis++)
        {
            ranges[axis] = IsUnitDimension(shape[axis])
                ? new AffineRange(0, 1)
                : new AffineRange(domains[domainIndex].Offset, domains[domainIndex++].Extent);
        }

        return ranges;
    }

    private static AffineRange[] BuildFlatSuffixRanges(AffineDomain[] domains, RankedShape shape, int prefixRank, long scale, bool suffixDomainIsIdentity)
    {
        var ranges = new AffineRange[shape.Rank];
        for (var axis = 0; axis < prefixRank; axis++)
        {
            ranges[axis] = new AffineRange(domains[axis].Offset, domains[axis].Extent);
        }

        var suffixDomain = domains[prefixRank];
        ranges[prefixRank] = suffixDomainIsIdentity
            ? new AffineRange(suffixDomain.Offset, suffixDomain.Extent)
            : ScaledRange(suffixDomain, scale);
        return ranges;
    }

    private static AffineRange[] BuildMajorSuffixRanges(AffineDomain[] domains, RankedShape shape, int prefixRank)
    {
        var ranges = new AffineRange[shape.Rank];
        for (var axis = 0; axis < shape.Rank; axis++)
        {
            ranges[axis] = axis switch
            {
                _ when axis < prefixRank => new AffineRange(domains[axis].Offset, domains[axis].Extent),
                _ when axis == prefixRank => new AffineRange(domains[prefixRank].Offset, domains[prefixRank].Extent),
                _ => new AffineRange(0, GetFixedDimension(shape[axis])),
            };
        }

        return ranges;
    }

    private static AffineRange ScaledRange(AffineDomain domain, long scale)
        => scale == 1
            ? new AffineRange(domain.Offset, domain.Extent)
            : new AffineRange(domain.Offset * scale, domain.Extent * scale);

    private static bool HasFixedSuffix(RankedShape shape, int start)
    {
        for (var axis = start; axis < shape.Rank; axis++)
        {
            if (shape[axis] is not DimConst)
            {
                return false;
            }
        }

        return true;
    }

    private static int GetCommonPrefixRank(RankedShape sourceShape, RankedShape resultShape)
    {
        var prefixRank = 0;
        var minRank = System.Math.Min(sourceShape.Rank, resultShape.Rank);
        while (prefixRank < minRank && IsSameDimension(sourceShape[prefixRank], resultShape[prefixRank]))
        {
            prefixRank++;
        }

        return prefixRank;
    }

    private static bool HasSuffixSplit(IRType type, int prefixRank)
        => type is DistributedType distributedType && distributedType.AxisPolicies.Any(policy =>
            policy is SBPSplit split && split.Axes.Any(axis => axis >= prefixRank));

    private static bool HaveCompatibleDistributedPlacement(IRType sourceType, IRType resultType)
        => (sourceType, resultType) switch
        {
            (DistributedType { Partial: not null }, _) => false,
            (_, DistributedType { Partial: not null }) => false,
            (DistributedType source, DistributedType result) =>
                HasValidDistributedLayout(source) &&
                HasValidDistributedLayout(result) &&
                source.Placement == result.Placement,
            (DistributedType source, _) => HasValidDistributedLayout(source),
            (_, DistributedType result) => HasValidDistributedLayout(result),
            _ => true,
        };

    private static bool HasValidDistributedLayout(DistributedType type)
        => type.TensorType.Shape is RankedShape shape &&
            type.AxisPolicies.Count == shape.Rank &&
            type.AxisPolicies.All(policy => policy is not SBPSplit split ||
                split.Axes.All(axis => axis >= 0 && axis < type.Placement.Rank));

    private static bool HaveCompatibleDistributedStorage(IRType sourceType, IRType resultType, AffineViewTransform transform)
    {
        if (sourceType is not DistributedType source || resultType is not DistributedType result)
        {
            return true;
        }

        if (!HaveCompatibleDistributedPlacement(source, result))
        {
            return false;
        }

        if (source.TensorType == result.TensorType &&
            DistributedUtility.AreSamePolicies(source.AxisPolicies, result.AxisPolicies))
        {
            return true;
        }

        var domainBounds = CompilerServices.GetMaxShape(new RankedShape(transform.DomainBounds.ToArray()));
        var sourceInverse = AffineUtility.Inverse(transform.SourceMap, domainBounds);
        var resultInverse = AffineUtility.Inverse(transform.ResultMap, domainBounds);
        var hierarchy = source.Placement.Hierarchy.ToArray();
        var shardCount = checked((int)TensorUtilities.GetProduct(hierarchy));
        for (var linearIndex = 0; linearIndex < shardCount; linearIndex++)
        {
            var shardIndex = DistributedUtility.GetUnraveledIndex(linearIndex, hierarchy);
            var (sourceOffset, sourceShape) = DistributedUtility.GetLocalOffsetAndShape(source, shardIndex);
            var (resultOffset, resultShape) = DistributedUtility.GetLocalOffsetAndShape(result, shardIndex);
            var sourceDomain = sourceInverse.Apply(sourceOffset, sourceShape);
            var resultDomain = resultInverse.Apply(resultOffset, resultShape);
            if (!AreSameRanges(sourceDomain, resultDomain))
            {
                return false;
            }
        }

        return true;
    }

    private static bool CoversShape(AffineMap map, IRArray<Dimension> domainBounds, RankedShape shape)
    {
        if (map.Domains.Length != domainBounds.Count || map.Results.Length != shape.Rank)
        {
            return false;
        }

        var domainOffsets = Enumerable.Repeat<Dimension>(Dimension.Zero, domainBounds.Count).ToArray();
        var ranges = map.Apply(domainOffsets, domainBounds.ToArray());
        for (var axis = 0; axis < ranges.Length; axis++)
        {
            if (!ranges[axis].Start.Simplify().Equals(Dimension.Zero) ||
                !ranges[axis].Stop.Simplify().Equals(shape[axis].Simplify()))
            {
                return false;
            }
        }

        return true;
    }

    private static bool AreSameRanges(ReadOnlySpan<TIR.Range> lhs, ReadOnlySpan<TIR.Range> rhs)
    {
        if (lhs.Length != rhs.Length)
        {
            return false;
        }

        for (var i = 0; i < lhs.Length; i++)
        {
            if (!lhs[i].Start.Simplify().Equals(rhs[i].Start.Simplify()) ||
                !lhs[i].Stop.Simplify().Equals(rhs[i].Stop.Simplify()))
            {
                return false;
            }
        }

        return true;
    }

    private static bool TryGetTensorType(IRType type, out TensorType tensorType)
    {
        tensorType = type switch
        {
            TensorType tensor => tensor,
            DistributedType distributed => distributed.TensorType,
            _ => null!,
        };
        return tensorType is not null;
    }

    private static int GreatestCommonDivisor(int lhs, int rhs)
    {
        while (rhs != 0)
        {
            (lhs, rhs) = (rhs, lhs % rhs);
        }

        return lhs;
    }

    private static long GetScalarSuffixElementCount(RankedShape shape, int start, int lanes)
        => ProductFixedSuffix(shape, start) * lanes;

    private static long ProductFixedSuffix(RankedShape shape, int start)
    {
        var product = 1L;
        for (var axis = start; axis < shape.Rank; axis++)
        {
            product *= GetFixedDimension(shape[axis]);
        }

        return product;
    }

    private static long GetFixedDimension(Dimension dimension)
        => dimension is DimConst dimConst
            ? dimConst.Value
            : throw new ArgumentException($"Expected fixed dimension, got {dimension}.");

    private static bool IsSameDimension(Dimension lhs, Dimension rhs)
        => lhs.Equals(rhs) || (lhs is DimConst l && rhs is DimConst r && l.Value == r.Value);

    private static bool HaveSameShape(RankedShape lhs, RankedShape rhs)
        => lhs.Rank == rhs.Rank && lhs.Dimensions.ToArray().Zip(rhs.Dimensions.ToArray()).All(pair => IsSameDimension(pair.First, pair.Second));

    private static bool IsUnitDimension(Dimension dimension)
        => dimension is DimConst { Value: 1 };
}
