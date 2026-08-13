// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Utilities;

namespace Nncase.CostModel;

/// <summary>
/// Target execution strategy for one distributed paged-attention call.
/// </summary>
public enum PagedAttentionExecutionKind
{
    Direct,
    SplitKV,
}

/// <summary>
/// Optional target-options contract for paged-attention execution planning.
/// </summary>
public interface IPagedAttentionExecutionPlanProvider
{
    PagedAttentionExecutionPlan GetPagedAttentionExecutionPlan(
        PagedAttentionExecutionPlanQuery query);
}

/// <summary>
/// A deterministic paged-attention execution decision shared by cost evaluation
/// and semantic TIR selection.
/// </summary>
public sealed record PagedAttentionExecutionPlan(
    PagedAttentionExecutionKind Kind,
    int SplitHierarchyAxis,
    int SplitCount)
{
    public static PagedAttentionExecutionPlan Direct { get; } = new(
        PagedAttentionExecutionKind.Direct,
        -1,
        1);

    /// <summary>
    /// Validates this decision against the query that it was created for.
    /// </summary>
    public void Validate(PagedAttentionExecutionPlanQuery query)
    {
        ArgumentNullException.ThrowIfNull(query);
        if (Kind == PagedAttentionExecutionKind.Direct)
        {
            if (SplitHierarchyAxis != -1 || SplitCount != 1)
            {
                throw new InvalidOperationException(
                    "A direct PagedAttention plan must use split axis -1 and split count 1.");
            }

            return;
        }

        if (Kind != PagedAttentionExecutionKind.SplitKV)
        {
            throw new InvalidOperationException(
                $"Unsupported PagedAttention execution kind {Kind}.");
        }

        var placement = query.QueryType.Placement;
        if (SplitHierarchyAxis < 0 || SplitHierarchyAxis >= placement.Rank)
        {
            throw new InvalidOperationException(
                $"Split-KV PagedAttention hierarchy axis {SplitHierarchyAxis} is outside placement rank {placement.Rank}.");
        }

        if (!placement.IsPhysicalBlockAxis(SplitHierarchyAxis))
        {
            throw new InvalidOperationException(
                $"Split-KV PagedAttention hierarchy axis {SplitHierarchyAxis} is not a physical block axis.");
        }

        var hierarchyExtent = placement.Hierarchy[SplitHierarchyAxis];
        if (SplitCount <= 1 || SplitCount > hierarchyExtent)
        {
            throw new InvalidOperationException(
                $"Split-KV PagedAttention split count {SplitCount} must be in [2, {hierarchyExtent}].");
        }

        if (query.UsesHierarchyAxis(SplitHierarchyAxis))
        {
            throw new InvalidOperationException(
                $"Split-KV PagedAttention hierarchy axis {SplitHierarchyAxis} is already used by the query SBP policy.");
        }
    }
}

/// <summary>
/// Target-independent facts used to choose a paged-attention execution plan.
/// Sizes describe one local query shard, except <see cref="ContextLength"/>,
/// which is the global KV sequence bound.
/// </summary>
public sealed record PagedAttentionExecutionPlanQuery(
    DistributedType QueryType,
    long QuerySequenceLength,
    long LocalQueryHeads,
    long LocalKVHeads,
    long HeadDimension,
    long ContextLength,
    DataType QueryElementType,
    DataType KVElementType)
{
    public int QueryElementSizeBytes => QueryElementType.SizeInBytes;

    public int KVElementSizeBytes => KVElementType.SizeInBytes;

    public double LocalQueryScalarElements =>
        QuerySequenceLength * (double)LocalQueryHeads * HeadDimension;

    public double KVScalarElements =>
        2.0 * QuerySequenceLength * LocalKVHeads * ContextLength * HeadDimension;

    public double ComputeWork =>
        QuerySequenceLength * (double)LocalQueryHeads * ContextLength * ((2.0 * HeadDimension) + 8.0);

    public double PartialStateScalarElements =>
        QuerySequenceLength * (double)LocalQueryHeads * (HeadDimension + 2.0);

    public bool UsesHierarchyAxis(int hierarchyAxis)
        => QueryType.AxisPolicies.Any(policy => policy switch
        {
            SBPSplit split => split.HierarchyAxes.Contains(hierarchyAxis),
            SBPPartial partial => partial.Axes.Contains(hierarchyAxis),
            _ => false,
        });

    public static bool TryCreate(
        IRType queryType,
        IRType extraType,
        TensorType kvCachesType,
        IRArray<AttentionDimKind> layout,
        int hiddenSize,
        out PagedAttentionExecutionPlanQuery query)
    {
        query = null!;
        if (queryType is not DistributedType distributedQuery ||
            kvCachesType.DType is not ReferenceType
            {
                ElemType: PagedAttentionKVCacheType
                {
                    Config: IPagedAttentionConfig config,
                },
            })
        {
            return false;
        }

        var seqAxis = layout.IndexOf(AttentionDimKind.Seq);
        var headAxis = layout.IndexOf(AttentionDimKind.Head);
        var dimAxis = layout.IndexOf(AttentionDimKind.Dim);
        if (seqAxis < 0 || headAxis < 0 || dimAxis < 0 ||
            !TryGetLocalMaxShape(distributedQuery, out var localShape) ||
            localShape.Length != layout.Count)
        {
            return false;
        }

        var laneCount = GetVectorLaneCount(distributedQuery.TensorType.DType);
        var querySequenceLength = Math.Max(1, localShape[seqAxis]);
        var localQueryHeads = Math.Max(1, localShape[headAxis]);
        var headDimension = Math.Max(1, localShape[dimAxis] * laneCount);
        if (config.HeadDim <= 0 || hiddenSize <= 0 || hiddenSize % config.HeadDim != 0 ||
            config.NumKVHeads <= 0)
        {
            return false;
        }

        var globalQueryHeads = hiddenSize / config.HeadDim;
        if (globalQueryHeads % config.NumKVHeads != 0 ||
            !TryGetMaxLocalKVHeads(
                distributedQuery,
                headAxis,
                globalQueryHeads,
                config.NumKVHeads,
                out var localKVHeads))
        {
            return false;
        }

        var contextLength = Math.Max(
            querySequenceLength,
            EstimateGlobalContextLength(extraType, config, hiddenSize, querySequenceLength));
        query = new(
            distributedQuery,
            querySequenceLength,
            localQueryHeads,
            localKVHeads,
            headDimension,
            contextLength,
            GetScalarDataType(distributedQuery.TensorType.DType),
            config.KVPrimType);
        return true;
    }

    private static bool TryGetMaxLocalKVHeads(
        DistributedType queryType,
        int headAxis,
        int globalQueryHeads,
        int globalKVHeads,
        out long maxLocalKVHeads)
    {
        maxLocalKVHeads = 0;
        var queryHeadsPerKVHead = globalQueryHeads / globalKVHeads;
        var hierarchy = queryType.Placement.Hierarchy.ToArray();
        var shardCount = hierarchy.Aggregate(
            1,
            static (count, extent) => checked(count * Math.Max(1, extent)));
        for (int linearIndex = 0; linearIndex < shardCount; linearIndex++)
        {
            var shardIndex = DistributedUtility.GetUnraveledIndex(linearIndex, hierarchy);
            var descriptor = DistributedUtility.GetLocalShardDescriptor(
                queryType,
                shardIndex,
                DistributedUtility.DivideFlags.MaxShape);
            var localHeadAxis = descriptor.Axes[headAxis];
            if (!localHeadAxis.ActiveExtent.IsFixed)
            {
                return false;
            }

            var localHeadCount = Math.Max(0, localHeadAxis.ActiveExtent.FixedValue);
            if (localHeadCount == 0)
            {
                continue;
            }

            var localKVHeads = new HashSet<long>();
            for (long localHead = 0; localHead < localHeadCount; localHead++)
            {
                var globalHead = localHeadAxis.MapLocalToGlobal(localHead);
                if (!globalHead.IsFixed)
                {
                    return false;
                }

                if ((ulong)globalHead.FixedValue < (ulong)globalQueryHeads)
                {
                    localKVHeads.Add(globalHead.FixedValue / queryHeadsPerKVHead);
                }
            }

            maxLocalKVHeads = Math.Max(maxLocalKVHeads, localKVHeads.Count);
        }

        return maxLocalKVHeads > 0;
    }

    private static bool TryGetLocalMaxShape(DistributedType type, out long[] shape)
    {
        var localType = DistributedUtility.GetDividedTensorType(
            type,
            DistributedUtility.DivideFlags.MaxShape);
        if (CompilerServices.TryGetMaxShape(localType.Shape, out var maxShape) &&
            maxShape is not null)
        {
            shape = maxShape;
            return true;
        }

        shape = Array.Empty<long>();
        return false;
    }

    private static long EstimateGlobalContextLength(
        IRType extraType,
        IPagedAttentionConfig config,
        int hiddenSize,
        long fallback)
    {
        // Extra is a distributed workspace, but its global capacity encodes the
        // global context bound. Dividing it by the placement would incorrectly
        // shrink the KV sequence seen by every candidate.
        var tensorType = extraType switch
        {
            DistributedType distributed => distributed.TensorType,
            TensorType tensor => tensor,
            _ => null,
        };
        if (tensorType is null ||
            !CompilerServices.TryGetMaxShape(tensorType.Shape, out var shape) ||
            shape.Length == 0)
        {
            return fallback;
        }

        var elements = shape.Aggregate(
            1.0,
            static (acc, dimension) => acc * Math.Max(1, dimension));
        var numHeads = Math.Max(1.0, hiddenSize / (double)Math.Max(1, config.HeadDim));
        var denominator = Math.Max(1.0, numHeads * Math.Max(1, config.KVPrimType.SizeInBytes));
        var quadratic = elements / denominator;
        if (!double.IsFinite(quadratic) || quadratic <= 0)
        {
            return fallback;
        }

        // The importer reserves dtype_bytes * num_heads * L * (L + 1).
        var estimated = (Math.Sqrt((4.0 * quadratic) + 1.0) - 1.0) / 2.0;
        return double.IsFinite(estimated) && estimated > 0
            ? checked((long)Math.Ceiling(estimated))
            : fallback;
    }

    private static int GetVectorLaneCount(DataType dataType)
        => dataType is VectorType vector
            ? vector.Lanes.Aggregate(1, static (acc, lane) => checked(acc * lane)) *
                GetVectorLaneCount(vector.ElemType)
            : 1;

    private static DataType GetScalarDataType(DataType dataType)
        => dataType is VectorType vector
            ? GetScalarDataType(vector.ElemType)
            : dataType;
}
