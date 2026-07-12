// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Utilities;

namespace Nncase.IR.Affine;

/// <summary>
/// Builds and validates target-independent affine tensor views.
/// </summary>
public static class BufferViewUtility
{
    /// <summary>
    /// Materializes a full logical tensor descriptor over existing physical storage.
    /// </summary>
    public static TIR.Buffer CreateLogicalBufferView(
        TIR.Buffer source,
        IRType resultType,
        BufferViewTransform transform,
        string name)
    {
        (TensorType ResultTensorType, DistributedType? ResultDistributedType) result = resultType switch
        {
            DistributedType distributed => (distributed.TensorType, distributed),
            TensorType tensor => (tensor, null),
            _ => throw new ArgumentException($"Buffer view result must be tensor-like, got {resultType}.", nameof(resultType)),
        };
        var (resultTensorType, resultDistributedType) = result;
        if (resultTensorType.Shape is not RankedShape resultShape)
        {
            throw new ArgumentException("Buffer view result must have a ranked shape.", nameof(resultType));
        }

        var zeroDomain = Enumerable.Repeat<Dimension>(0L, transform.DomainBounds.Count).ToArray();
        var resultOrigin = transform.ResultMap.Apply(zeroDomain, zeroDomain)
            .Select(range => range.Start.Simplify())
            .ToArray();
        if (resultOrigin.Any(offset => !offset.Equals(Dimension.Zero)))
        {
            throw new NotSupportedException(
                $"Logical buffer view result must cover the origin, got [{string.Join(", ", resultOrigin.Select(offset => offset.ToString()))}].");
        }

        var sourceOrigin = transform.SourceMap.Apply(zeroDomain, zeroDomain)
            .Select(range => range.Start.Simplify())
            .ToArray();
        if (sourceOrigin.Length != source.Rank)
        {
            throw new InvalidOperationException(
                $"Logical buffer view source rank mismatch: map={sourceOrigin.Length}, buffer={source.Rank}.");
        }

        var resultStrides = CreateBufferViewStrides(source, resultTensorType, transform);
        var byteOffset = (TensorUtilities.GetLinearOffset(source.Strides, sourceOrigin) * source.ElemType.SizeInBytes).Simplify();
        var byteSize = (source.MemSpan.Size - byteOffset).Simplify();
        return TIR.T.CreateBufferView(
            source,
            resultTensorType.DType,
            resultShape.Dimensions,
            resultStrides,
            byteOffset,
            byteSize,
            resultDistributedType,
            name);
    }

    /// <summary>
    /// Derives element strides for a typed buffer alias over existing storage.
    /// </summary>
    public static Dimension[] CreateBufferViewStrides(TIR.Buffer source, TensorType resultType, BufferViewTransform transform)
    {
        var sourceDefaultStrides = TensorUtilities.GetDefaultStrides(source.Dimensions);
        var sourceDenseStrides = GetDenseStrides(source.Dimensions);
        var prefixRank = 0;
        var comparableRank = System.Math.Min(transform.SourceMap.Results.Length, transform.ResultMap.Results.Length);
        while (prefixRank < comparableRank && transform.SourceMap.Results[prefixRank].Equals(transform.ResultMap.Results[prefixRank]))
        {
            prefixRank++;
        }

        for (var axis = prefixRank; axis < source.Rank; axis++)
        {
            if (!source.Strides[axis].Equals(sourceDefaultStrides[axis]) &&
                !source.Strides[axis].Equals(sourceDenseStrides[axis]) &&
                !IsDegenerateSourceDimension(source, axis))
            {
                throw new NotSupportedException(
                    $"Buffer view cannot reshape non-contiguous source suffix at axis {axis}: stride={source.Strides[axis]}, " +
                    $"expected={sourceDefaultStrides[axis]} or dense stride {sourceDenseStrides[axis]}.");
            }
        }

        var resultDimensions = ((RankedShape)resultType.Shape).Dimensions.ToArray();
        var resultStrides = TensorUtilities.GetDefaultStrides(resultDimensions);
        var sharedPrefixRank = System.Math.Min(prefixRank, System.Math.Min(source.Rank, resultDimensions.Length));
        for (var axis = 0; axis < sharedPrefixRank; axis++)
        {
            var sourceByteStride = source.Strides[axis] * source.ElemType.SizeInBytes;
            if (sourceByteStride is DimConst byteStride && byteStride.Value % resultType.DType.SizeInBytes != 0)
            {
                throw new NotSupportedException(
                    $"Buffer view byte stride {byteStride.Value} at axis {axis} is not aligned to result element size {resultType.DType.SizeInBytes}.");
            }

            resultStrides[axis] = (sourceByteStride / resultType.DType.SizeInBytes).Simplify();
        }

        return resultStrides;
    }

    /// <summary>
    /// Computes the byte span covered by a strided logical buffer descriptor.
    /// </summary>
    public static Dimension GetByteSpanSize(
        ReadOnlySpan<Dimension> dimensions,
        ReadOnlySpan<Dimension> strides,
        int elementSizeInBytes)
    {
        if (dimensions.Length != strides.Length)
        {
            throw new ArgumentException(
                $"Buffer span rank mismatch: dimensions={dimensions.Length}, strides={strides.Length}.");
        }

        Dimension spanElements = 1L;
        for (var axis = 0; axis < dimensions.Length; axis++)
        {
            spanElements += ((dimensions[axis] - 1L) * strides[axis]).Simplify();
        }

        Dimension byteSize = (spanElements * elementSizeInBytes).Simplify();
        foreach (var dimension in dimensions)
        {
            byteSize = Dimension.Select(dimension, 0L, 0L, byteSize).Simplify();
        }

        return byteSize;
    }

    /// <summary>
    /// Tries to build a storage-preserving affine transform between two tensor types.
    /// </summary>
    public static bool TryCreate(IRType sourceType, IRType resultType, out BufferViewTransform transform)
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
            transform = BufferViewTransform.Identity(sourceShape);
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
                transform = BufferViewTransform.Identity(sourceShape);
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

    private static bool TryGetFlatToFlatMaps(RankedShape sourceShape, RankedShape resultShape, int prefixRank, int sourceLane, int resultLane, out BufferViewTransform transform)
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

    private static bool TryGetSingletonDimensionMaps(RankedShape sourceShape, RankedShape resultShape, int sourceLane, int resultLane, out BufferViewTransform transform)
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

    private static bool TryGetFlatInputToOutputMajorMaps(RankedShape sourceShape, RankedShape resultShape, int prefixRank, int sourceLane, int resultLane, out BufferViewTransform transform)
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

        if (sourceLane == resultLane)
        {
            var preciseDomains = F.Affine.Domains(resultShape.Rank);
            transform = CreateTransform(
                preciseDomains,
                BuildFlattenedSuffixRanges(preciseDomains, resultShape, prefixRank),
                preciseDomains.Select(domain => new AffineRange(domain.Offset, domain.Extent)).ToArray(),
                resultShape.Dimensions);
            return true;
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

    private static AffineRange[] BuildFlattenedSuffixRanges(AffineDomain[] domains, RankedShape expandedShape, int prefixRank)
    {
        var ranges = new AffineRange[prefixRank + 1];
        for (var axis = 0; axis < prefixRank; axis++)
        {
            ranges[axis] = new AffineRange(domains[axis].Offset, domains[axis].Extent);
        }

        AffineExpr offset = 0;
        AffineExpr extent = 1;
        for (var axis = prefixRank; axis < expandedShape.Rank; axis++)
        {
            var stride = ProductFixedSuffix(expandedShape, axis + 1);
            offset += domains[axis].Offset * stride;
            extent += (domains[axis].Extent - 1) * stride;
        }

        ranges[prefixRank] = new AffineRange(offset, extent);
        return ranges;
    }

    private static bool TryGetInputMajorToFlatOutputMaps(RankedShape sourceShape, RankedShape resultShape, int prefixRank, int sourceLane, int resultLane, out BufferViewTransform transform)
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

    private static bool TryGetPrefixFullTileMaps(RankedShape sourceShape, RankedShape resultShape, int prefixRank, int sourceLane, int resultLane, out BufferViewTransform transform)
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

    private static BufferViewTransform CreateTransform(AffineDomain[] domains, AffineRange[] sourceRanges, AffineRange[] resultRanges, ReadOnlySpan<Dimension> domainBounds)
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

    private static bool HaveCompatibleDistributedStorage(IRType sourceType, IRType resultType, BufferViewTransform transform)
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
        var resultMapIsIdentity = transform.ResultMap.Equals(AffineMap.Identity(transform.ResultMap.Results.Length));
        var sourceInverse = resultMapIsIdentity ? null : AffineUtility.Inverse(transform.SourceMap, domainBounds);
        var resultInverse = resultMapIsIdentity ? null : AffineUtility.Inverse(transform.ResultMap, domainBounds);
        var hierarchy = source.Placement.Hierarchy.ToArray();
        var shardCount = checked((int)TensorUtilities.GetProduct(hierarchy));
        for (var linearIndex = 0; linearIndex < shardCount; linearIndex++)
        {
            var shardIndex = DistributedUtility.GetUnraveledIndex(linearIndex, hierarchy);
            var (sourceOffset, sourceShape) = DistributedUtility.GetLocalOffsetAndShape(source, shardIndex);
            var (resultOffset, resultShape) = DistributedUtility.GetLocalOffsetAndShape(result, shardIndex);
            if (resultMapIsIdentity)
            {
                var mappedSource = transform.SourceMap.Apply(resultOffset, resultShape);
                var expectedSource = sourceOffset.Zip(sourceShape).Select(pair => new TIR.Range(pair.First, pair.Second, 1L)).ToArray();
                if (!AreSameRanges(mappedSource, expectedSource))
                {
                    return false;
                }

                continue;
            }

            var sourceDomain = sourceInverse!.Apply(sourceOffset, sourceShape);
            var resultDomain = resultInverse!.Apply(resultOffset, resultShape);
            if (!AreSameRanges(sourceDomain, resultDomain))
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

    private static Dimension[] GetDenseStrides(ReadOnlySpan<Dimension> dimensions)
    {
        var strides = new Dimension[dimensions.Length];
        Dimension stride = 1;
        for (var axis = dimensions.Length - 1; axis >= 0; axis--)
        {
            strides[axis] = stride;
            stride = (stride * dimensions[axis]).Simplify();
        }

        return strides;
    }

    private static bool HaveSameShape(RankedShape lhs, RankedShape rhs)
        => lhs.Rank == rhs.Rank && lhs.Dimensions.ToArray().Zip(rhs.Dimensions.ToArray()).All(pair => IsSameDimension(pair.First, pair.Second));

    private static bool IsUnitDimension(Dimension dimension)
        => dimension is DimConst { Value: 1 };

    private static bool IsDegenerateDimension(Dimension dimension)
        => IsUnitDimension(dimension) ||
           (dimension.Metadata.Range is { } range && range.Min >= 0 && range.Max <= 1);

    private static bool IsDegenerateSourceDimension(TIR.Buffer source, int axis)
    {
        if (IsDegenerateDimension(source.Dimensions[axis]))
        {
            return true;
        }

        if (source.DistributedType is not { } distributedType ||
            DistributedUtility.GetDividedTensorType(distributedType).Shape is not RankedShape localShape ||
            localShape.Rank != source.Rank)
        {
            return false;
        }

        return IsDegenerateDimension(localShape[axis]);
    }
}
