// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.NN;

namespace Nncase.Targets;

internal sealed record PagedAttentionCachePhysicalLayout(
    int LaneCount,
    PagedKVCacheDimKind? VectorizedDim,
    int HeadDimBlocks,
    int[] TailShape,
    int SectionElements,
    int LayerStride,
    int HeadStride,
    int DimBlockStride,
    int BlockOffsetStride)
{
    public bool HasContiguousHeadDimension(int laneCount)
        => VectorizedDim == PagedKVCacheDimKind.HeadDim &&
            LaneCount == laneCount &&
            DimBlockStride == 1;
}

internal static class PagedAttentionCacheLayoutUtility
{
    public static PagedAttentionCachePhysicalLayout Analyze(
        IPagedAttentionConfig config,
        AttentionCacheKind kind,
        string context)
    {
        var cacheLayout = config.GetCacheLayout(kind).ToArray();
        var expectedLayout = Enum.GetValues<PagedKVCacheDimKind>()
            .OrderBy(static dim => (int)dim)
            .ToArray();
        if (cacheLayout.Length != expectedLayout.Length ||
            !cacheLayout.OrderBy(static dim => (int)dim).SequenceEqual(expectedLayout))
        {
            throw new NotSupportedException(
                $"{context} {kind} cache layout must be a permutation of " +
                $"[{string.Join(", ", expectedLayout)}], got " +
                $"[{string.Join(", ", cacheLayout)}].");
        }

        var vectorizedAxes = config.GetVectorizedAxes(kind).ToArray();
        var lanes = config.GetLanes(kind).ToArray();
        int laneCount;
        PagedKVCacheDimKind? vectorizedDim;
        if (vectorizedAxes.Length == 0 && lanes.Length == 0)
        {
            laneCount = 1;
            vectorizedDim = null;
        }
        else if (vectorizedAxes.Length == 1 && lanes.Length == 1 &&
            vectorizedAxes[0] is PagedKVCacheDimKind.HeadDim or PagedKVCacheDimKind.BlockSize)
        {
            laneCount = checked((int)lanes[0]);
            vectorizedDim = vectorizedAxes[0];
        }
        else
        {
            throw new NotSupportedException(
                $"{context} supports only {kind} HeadDim or BlockSize " +
                "vectorization with zero or one lane value.");
        }

        if (laneCount <= 0)
        {
            throw new NotSupportedException(
                $"{context} requires {kind} lane to be positive, got {laneCount}.");
        }

        if (vectorizedDim == PagedKVCacheDimKind.HeadDim && config.HeadDim % laneCount != 0)
        {
            throw new NotSupportedException(
                $"{context} requires {kind} head_dim divisible by lane, " +
                $"got head_dim={config.HeadDim}, lane={laneCount}.");
        }

        if (vectorizedDim == PagedKVCacheDimKind.BlockSize && config.BlockSize % laneCount != 0)
        {
            throw new NotSupportedException(
                $"{context} requires {kind} block_size divisible by lane, " +
                $"got block_size={config.BlockSize}, lane={laneCount}.");
        }

        var headDimBlocks = vectorizedDim == PagedKVCacheDimKind.HeadDim
            ? checked(config.HeadDim / laneCount)
            : config.HeadDim;
        var tailDims = new List<int>();
        var tailDimKinds = new List<PagedKVCacheDimKind>();
        foreach (var dimKind in cacheLayout)
        {
            if (dimKind is PagedKVCacheDimKind.NumBlocks or PagedKVCacheDimKind.KV)
            {
                continue;
            }

            tailDimKinds.Add(dimKind);
            var dimExtent = dimKind switch
            {
                PagedKVCacheDimKind.NumLayers => checked((int)config.NumLayers),
                PagedKVCacheDimKind.BlockSize => checked((int)config.BlockSize),
                PagedKVCacheDimKind.NumKVHeads => checked((int)config.NumKVHeads),
                PagedKVCacheDimKind.HeadDim => checked((int)config.HeadDim),
                _ => throw new NotSupportedException(
                    $"{context} does not support {kind} cache dimension {dimKind}."),
            };
            if (dimKind == vectorizedDim)
            {
                dimExtent = checked(dimExtent / laneCount);
            }

            tailDims.Add(dimExtent);
        }

        if (tailDims.Count != 4)
        {
            throw new NotSupportedException(
                $"{context} expects {kind} cache layout to contain exactly " +
                "NumLayers, NumKVHeads, HeadDim, and BlockSize after removing NumBlocks/KV.");
        }

        var strides = ComputeContiguousStrides(tailDims);
        int GetStride(PagedKVCacheDimKind dimKind)
        {
            var index = tailDimKinds.IndexOf(dimKind);
            if (index < 0)
            {
                throw new NotSupportedException(
                    $"{context} {kind} cache layout is missing {dimKind}.");
            }

            return strides[index];
        }

        var sectionVectorElements = tailDims.Aggregate(
            1,
            static (acc, dim) => checked(acc * dim));
        var sectionElements = checked(sectionVectorElements * laneCount);
        return new(
            laneCount,
            vectorizedDim,
            headDimBlocks,
            tailDims.ToArray(),
            sectionElements,
            GetStride(PagedKVCacheDimKind.NumLayers),
            GetStride(PagedKVCacheDimKind.NumKVHeads),
            GetStride(PagedKVCacheDimKind.HeadDim),
            GetStride(PagedKVCacheDimKind.BlockSize));
    }

    private static int[] ComputeContiguousStrides(IReadOnlyList<int> shape)
    {
        var strides = new int[shape.Count];
        var stride = 1;
        for (int i = shape.Count - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride = checked(stride * shape[i]);
        }

        return strides;
    }
}
