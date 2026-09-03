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
    /// Tries to create a canonical-global descriptor over a compact local shard
    /// whose physical allocation is provably the complete canonical tensor.
    /// </summary>
    public static bool TryCreateCanonicalGlobalReadAlias(
        TIR.Buffer source,
        string name,
        out TIR.Buffer alias,
        out string reason)
    {
        alias = source;
        reason = string.Empty;
        if (source.DistributedType is not { Partial: null } distributedType)
        {
            reason = "the source is not a non-partial distributed buffer";
            return false;
        }

        if (source.DistributedStorageKind == TIR.DistributedBufferStorageKind.CanonicalGlobal)
        {
            return true;
        }

        if (source.DistributedStorageKind != TIR.DistributedBufferStorageKind.CompactLocal)
        {
            reason = $"the source storage kind is {source.DistributedStorageKind}, not CompactLocal";
            return false;
        }

        if (source.MemSpan.Buffer.Location is not (
            TIR.MemoryLocation.Input or
            TIR.MemoryLocation.Rdata or
            TIR.MemoryLocation.ChipLocalRdata))
        {
            reason = $"the source backing {source.MemSpan.Buffer.Location} is not a canonical readonly allocation";
            return false;
        }

        var shardCoordinates = Enumerable.Range(0, distributedType.Placement.Rank)
            .Select(axis => (Dimension)new DimVar($"__shard_coord_{axis}"))
            .ToArray();
        var descriptor = DistributedUtility.GetLocalShardDescriptor(
            distributedType,
            shardCoordinates,
            DistributedUtility.DivideFlags.MaxShape);
        if (!descriptor.TryGetContiguousRegion(out _, out _))
        {
            reason = "the local shard is not one contiguous canonical tensor region";
            return false;
        }

        var localShape = descriptor.LocalCapacityShape.Dimensions.ToArray();
        if (!source.Dimensions.SequenceEqual(localShape))
        {
            reason = $"the source shape [{string.Join(",", source.Dimensions.ToArray().Select(dim => dim.ToString()))}] " +
                $"does not match local capacity [{string.Join(",", localShape.Select(dim => dim.ToString()))}]";
            return false;
        }

        var (globalByteSize, globalStrides) = TensorUtilities.GetTensorMaxSizeAndStridesExpr(
            distributedType.TensorType,
            null);
        var localByteSize = GetByteSpanSize(
            source.Dimensions,
            source.Strides,
            source.ElemType.SizeInBytes).Simplify();
        if (!source.MemSpan.Size.Simplify().Equals(localByteSize))
        {
            reason = $"the local span contains {source.MemSpan.Size.Simplify()} bytes, but its descriptor " +
                $"requires {localByteSize} bytes";
            return false;
        }

        if (!source.MemSpan.Buffer.Size.Simplify().Equals(globalByteSize.Simplify()))
        {
            reason = $"the physical backing contains {source.MemSpan.Buffer.Size.Simplify()} bytes, but the " +
                $"canonical tensor requires {globalByteSize.Simplify()} bytes";
            return false;
        }

        alias = source.With(
            name: name,
            memSpan: new TIR.MemSpan(source.MemSpan.Buffer, Dimension.Zero, globalByteSize),
            dimensions: localShape,
            strides: globalStrides,
            distributedStorageKind: TIR.DistributedBufferStorageKind.CanonicalGlobal);
        return true;
    }

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

        var resultStrides = CreateBufferViewStrides(
            source,
            resultTensorType,
            resultDistributedType,
            transform);
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
        => CreateBufferViewStrides(
            source.ElemType,
            source.Dimensions,
            source.Strides,
            source.DistributedType,
            source.DistributedStorageKind,
            resultType,
            null,
            transform);

    /// <summary>
    /// Derives element strides for a typed buffer alias from an explicit source
    /// layout descriptor.
    /// </summary>
    public static Dimension[] CreateBufferViewStrides(
        DataType sourceElemType,
        ReadOnlySpan<Dimension> sourceDimensions,
        ReadOnlySpan<Dimension> sourceStrides,
        DistributedType? sourceDistributedType,
        TensorType resultType,
        BufferViewTransform transform)
        => CreateBufferViewStrides(
            sourceElemType,
            sourceDimensions,
            sourceStrides,
            sourceDistributedType,
            TIR.DistributedBufferStorageKind.CompactLocal,
            resultType,
            null,
            transform);

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
            TryGetPrefixFullTileMaps(sourceShape, resultShape, prefixRank, sourceLanes, resultLanes, out transform);
        return created && HaveCompatibleDistributedStorage(sourceType, resultType, transform);
    }

    private static Dimension[] CreateBufferViewStrides(
        TIR.Buffer source,
        TensorType resultType,
        DistributedType? resultDistributedType,
        BufferViewTransform transform)
        => CreateBufferViewStrides(
            source.ElemType,
            source.Dimensions,
            source.Strides,
            source.DistributedType,
            source.DistributedStorageKind,
            resultType,
            resultDistributedType,
            transform);

    private static Dimension[] CreateBufferViewStrides(
        DataType sourceElemType,
        ReadOnlySpan<Dimension> sourceDimensions,
        ReadOnlySpan<Dimension> sourceStrides,
        DistributedType? sourceDistributedType,
        TIR.DistributedBufferStorageKind sourceStorageKind,
        TensorType resultType,
        DistributedType? resultDistributedType,
        BufferViewTransform transform)
    {
        if (sourceDimensions.Length != sourceStrides.Length)
        {
            throw new ArgumentException(
                $"Buffer view source rank mismatch: dimensions={sourceDimensions.Length}, strides={sourceStrides.Length}.");
        }

        var sourceStorageDimensions = sourceStorageKind == TIR.DistributedBufferStorageKind.CompactLocal &&
            sourceDistributedType is not null
                ? ((RankedShape)DistributedUtility.GetDividedTensorType(sourceDistributedType).Shape).Dimensions.ToArray()
                : sourceDimensions.ToArray();
        var sourceLogicalDimensions = sourceDistributedType?.TensorType.Shape is RankedShape sourceLogicalShape
            ? sourceLogicalShape.Dimensions.ToArray()
            : sourceDimensions.ToArray();
        var sourceDefaultStrides = TensorUtilities.GetDefaultStrides(sourceStorageDimensions);
        var sourceDenseStrides = GetDenseStrides(sourceStorageDimensions);
        var resultLogicalDimensions = ((RankedShape)resultType.Shape).Dimensions.ToArray();
        var resultStorageDimensions = sourceStorageKind == TIR.DistributedBufferStorageKind.CompactLocal &&
            resultDistributedType is not null
                ? ((RankedShape)DistributedUtility.GetDividedTensorType(resultDistributedType).Shape).Dimensions.ToArray()
                : resultLogicalDimensions;
        var resultStrides = TensorUtilities.GetDefaultStrides(resultStorageDimensions);
        if (TryCreateProjectedViewStrides(
                sourceElemType,
                sourceStrides,
                resultType.DType,
                transform,
                resultStrides,
                out var projectedStrides))
        {
            return projectedStrides;
        }

        var prefixRank = 0;
        var comparableRank = System.Math.Min(
            System.Math.Min(transform.SourceMap.Results.Length, transform.ResultMap.Results.Length),
            System.Math.Min(sourceLogicalDimensions.Length, resultLogicalDimensions.Length));
        while (prefixRank < comparableRank &&
               sourceLogicalDimensions[prefixRank].Equals(resultLogicalDimensions[prefixRank]) &&
               transform.SourceMap.Results[prefixRank].Equals(transform.ResultMap.Results[prefixRank]))
        {
            prefixRank++;
        }

        for (var axis = prefixRank; axis < sourceDimensions.Length; axis++)
        {
            if (!sourceStrides[axis].Equals(sourceDefaultStrides[axis]) &&
                !sourceStrides[axis].Equals(sourceDenseStrides[axis]) &&
                !IsDegenerateSourceDimension(sourceStorageDimensions, sourceDistributedType, axis))
            {
                throw new NotSupportedException(
                    $"Buffer view cannot reshape non-contiguous source suffix at axis {axis}: stride={sourceStrides[axis]}, " +
                    $"expected={sourceDefaultStrides[axis]} or dense stride {sourceDenseStrides[axis]}.");
            }
        }

        var sharedPrefixRank = System.Math.Min(prefixRank, System.Math.Min(sourceDimensions.Length, resultLogicalDimensions.Length));
        for (var axis = 0; axis < sharedPrefixRank; axis++)
        {
            var sourceByteStride = sourceStrides[axis] * sourceElemType.SizeInBytes;
            if (sourceByteStride is DimConst byteStride && byteStride.Value % resultType.DType.SizeInBytes != 0)
            {
                throw new NotSupportedException(
                    $"Buffer view byte stride {byteStride.Value} at axis {axis} is not aligned to result element size {resultType.DType.SizeInBytes}.");
            }

            resultStrides[axis] = (sourceByteStride / resultType.DType.SizeInBytes).Simplify();
        }

        return resultStrides;
    }

    private static bool TryCreateProjectedViewStrides(
        DataType sourceElemType,
        ReadOnlySpan<Dimension> sourceStrides,
        DataType resultElemType,
        BufferViewTransform transform,
        ReadOnlySpan<Dimension> defaultResultStrides,
        out Dimension[] resultStrides)
    {
        resultStrides = Array.Empty<Dimension>();
        if (sourceStrides.Length != transform.SourceMap.Results.Length ||
            defaultResultStrides.Length != transform.ResultMap.Results.Length ||
            !TryGetProjectedAxes(transform.SourceMap, out var sourceAxes) ||
            !TryGetProjectedAxes(transform.ResultMap, out var resultAxes))
        {
            return false;
        }

        resultStrides = defaultResultStrides.ToArray();
        for (var domain = 0; domain < sourceAxes.Length; domain++)
        {
            var sourceByteStride = sourceStrides[sourceAxes[domain]] * sourceElemType.SizeInBytes;
            if (sourceByteStride is DimConst byteStride && byteStride.Value % resultElemType.SizeInBytes != 0)
            {
                throw new NotSupportedException(
                    $"Buffer view projected byte stride {byteStride.Value} from source axis {sourceAxes[domain]} " +
                    $"to result axis {resultAxes[domain]} is not aligned to result element size {resultElemType.SizeInBytes}.");
            }

            resultStrides[resultAxes[domain]] = (sourceByteStride / resultElemType.SizeInBytes).Simplify();
        }

        return true;
    }

    private static bool TryGetProjectedAxes(AffineMap map, out int[] axes)
    {
        axes = Enumerable.Repeat(-1, map.Domains.Length).ToArray();
        for (var axis = 0; axis < map.Results.Length; axis++)
        {
            switch (map.Results[axis])
            {
                case { Offset: AffineDim offset, Extent: AffineExtent extent }
                    when offset.Position == extent.Position &&
                         (uint)offset.Position < (uint)axes.Length &&
                         axes[offset.Position] < 0:
                    axes[offset.Position] = axis;
                    break;
                case
                {
                    Offset: AffineConstant { Value: 0 },
                    Extent: AffineConstant { Value: 1 },
                }:
                    break;
                default:
                    axes = Array.Empty<int>();
                    return false;
            }
        }

        if (axes.Any(axis => axis < 0))
        {
            axes = Array.Empty<int>();
            return false;
        }

        return true;
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
            policy is SBPSplit split && split.HierarchyAxes.Any(axis => axis >= prefixRank));

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
                split.HierarchyAxes.All(axis => axis >= 0 && axis < type.Placement.Rank));

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

        if (HaveEquivalentSingletonProjectedStorage(source, result, transform))
        {
            return true;
        }

        if (HaveEquivalentInnermostLinearizedStorage(source, result, transform))
        {
            return true;
        }

        if (HaveEquivalentFlattenedBlockCyclicStorage(source, result))
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
            var sourceDescriptor = DistributedUtility.GetLocalShardDescriptor(source, shardIndex);
            var resultDescriptor = DistributedUtility.GetLocalShardDescriptor(result, shardIndex);
            if (!sourceDescriptor.TryGetContiguousRegion(out var sourceOffset, out var sourceShape) ||
                !resultDescriptor.TryGetContiguousRegion(out var resultOffset, out var resultShape))
            {
                return false;
            }

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

    private static bool HaveEquivalentFlattenedBlockCyclicStorage(
        DistributedType source,
        DistributedType result)
    {
        if (!TryGetFlattenedBlockCyclicStages(source, out var sourceStages, out var sourceBytes) ||
            !TryGetFlattenedBlockCyclicStages(result, out var resultStages, out var resultBytes) ||
            sourceBytes != resultBytes ||
            sourceStages.Length != resultStages.Length)
        {
            return false;
        }

        return sourceStages.Zip(resultStages).All(pair => pair.First == pair.Second);
    }

    private static bool TryGetFlattenedBlockCyclicStages(
        DistributedType type,
        out SplitStage[] flattenedStages,
        out long tensorBytes)
    {
        flattenedStages = Array.Empty<SplitStage>();
        tensorBytes = 0;
        if (type.TensorType.Shape is not RankedShape shape ||
            shape.Dimensions.ToArray().Any(dimension => !dimension.IsFixed))
        {
            return false;
        }

        var extents = shape.Dimensions.ToArray().Select(dimension => dimension.FixedValue).ToArray();
        tensorBytes = checked(TensorUtilities.GetProduct(extents) * type.TensorType.DType.SizeInBytes);
        var stages = new List<SplitStage>();
        for (var axis = 0; axis < extents.Length; axis++)
        {
            if (type.AxisPolicies[axis] is SBPBroadCast)
            {
                continue;
            }

            if (type.AxisPolicies[axis] is not SBPSplit split ||
                split.Stages.Any(stage => stage.Distribution is not BlockCyclicSplit))
            {
                return false;
            }

            var parentExtent = extents[axis];
            foreach (var stage in split.Stages)
            {
                var blockSize = ((BlockCyclicSplit)stage.Distribution).BlockSize;
                var shardCount = stage.HierarchyAxes.Aggregate(
                    1L,
                    (product, hierarchyAxis) => checked(product * type.Placement.Hierarchy[hierarchyAxis]));
                var period = checked(shardCount * blockSize);
                if (parentExtent % period != 0)
                {
                    return false;
                }

                parentExtent /= shardCount;
            }

            var trailingBytes = (long)type.TensorType.DType.SizeInBytes;
            for (var trailingAxis = axis + 1; trailingAxis < extents.Length; trailingAxis++)
            {
                trailingBytes = checked(trailingBytes * extents[trailingAxis]);
            }

            if (!DistributedUtility.TryScaleSplitUnits(split, trailingBytes, 1, out var scaledSplit))
            {
                return false;
            }

            stages.AddRange(scaledSplit.Stages);
        }

        flattenedStages = stages.ToArray();
        return true;
    }

    private static bool HaveEquivalentInnermostLinearizedStorage(
        DistributedType source,
        DistributedType result,
        BufferViewTransform transform)
    {
        if (source.TensorType.Shape is not RankedShape sourceShape ||
            result.TensorType.Shape is not RankedShape resultShape ||
            sourceShape.Rank == 0 ||
            sourceShape.Rank != resultShape.Rank)
        {
            return false;
        }

        var innermostAxis = sourceShape.Rank - 1;
        if (GetCommonPrefixRank(sourceShape, resultShape) != innermostAxis)
        {
            return false;
        }

        var storageUnitBytes = GreatestCommonDivisor(
            source.TensorType.DType.SizeInBytes,
            result.TensorType.DType.SizeInBytes);
        var sourceLanes = source.TensorType.DType.SizeInBytes / storageUnitBytes;
        var resultLanes = result.TensorType.DType.SizeInBytes / storageUnitBytes;
        if (!TryGetFlatToFlatMaps(
                sourceShape,
                resultShape,
                innermostAxis,
                sourceLanes,
                resultLanes,
                out var expectedTransform) ||
            transform != expectedTransform)
        {
            return false;
        }

        for (var axis = 0; axis < innermostAxis; axis++)
        {
            if (!DistributedUtility.IsSamePolicy(
                    source.AxisPolicies[axis],
                    result.AxisPolicies[axis]))
            {
                return false;
            }
        }

        var sourcePolicy = source.AxisPolicies[innermostAxis];
        var resultPolicy = result.AxisPolicies[innermostAxis];
        if (sourcePolicy is SBPBroadCast && resultPolicy is SBPBroadCast)
        {
            return true;
        }

        return sourcePolicy is SBPSplit sourceSplit &&
            resultPolicy is SBPSplit resultSplit &&
            DistributedUtility.TryScaleSplitUnits(
                sourceSplit,
                source.TensorType.DType.SizeInBytes,
                result.TensorType.DType.SizeInBytes,
                out var scaledSplit) &&
            DistributedUtility.IsSamePolicy(scaledSplit, resultSplit);
    }

    private static bool HaveEquivalentSingletonProjectedStorage(
        DistributedType source,
        DistributedType result,
        BufferViewTransform transform)
    {
        if (source.TensorType.Shape is not RankedShape sourceShape ||
            result.TensorType.Shape is not RankedShape resultShape ||
            source.TensorType.DType.SizeInBytes != result.TensorType.DType.SizeInBytes ||
            !TryGetSingletonDimensionMaps(sourceShape, resultShape, 1, 1, out var expectedTransform) ||
            transform != expectedTransform)
        {
            return false;
        }

        var sourceAxis = 0;
        var resultAxis = 0;
        while (sourceAxis < sourceShape.Rank || resultAxis < resultShape.Rank)
        {
            while (sourceAxis < sourceShape.Rank && IsUnitDimension(sourceShape[sourceAxis]))
            {
                if (source.AxisPolicies[sourceAxis++] is not SBPBroadCast)
                {
                    return false;
                }
            }

            while (resultAxis < resultShape.Rank && IsUnitDimension(resultShape[resultAxis]))
            {
                if (result.AxisPolicies[resultAxis++] is not SBPBroadCast)
                {
                    return false;
                }
            }

            if (sourceAxis == sourceShape.Rank || resultAxis == resultShape.Rank)
            {
                return sourceAxis == sourceShape.Rank && resultAxis == resultShape.Rank;
            }

            if (!DistributedUtility.IsSamePolicy(
                    source.AxisPolicies[sourceAxis++],
                    result.AxisPolicies[resultAxis++]))
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

    private static bool IsDegenerateSourceDimension(
        ReadOnlySpan<Dimension> sourceDimensions,
        DistributedType? sourceDistributedType,
        int axis)
    {
        if (IsDegenerateDimension(sourceDimensions[axis]))
        {
            return true;
        }

        if (sourceDistributedType is null ||
            DistributedUtility.GetDividedTensorType(sourceDistributedType).Shape is not RankedShape localShape ||
            localShape.Rank != sourceDimensions.Length)
        {
            return false;
        }

        return IsDegenerateDimension(localShape[axis]);
    }
}
