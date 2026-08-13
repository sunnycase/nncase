// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.Utilities;

/// <summary>
/// Describes one split stage after it has been bound to a placement and a shard coordinate.
/// </summary>
public sealed record LocalShardStageDescriptor(
    SplitStage Stage,
    Dimension ParentExtent,
    Dimension LocalCapacity,
    Dimension ActiveExtent,
    Dimension LinearShardIndex,
    int ShardCount)
{
    /// <summary>
    /// Maps a coordinate in this stage's local domain to its parent domain.
    /// </summary>
    public Dimension MapLocalToParent(Dimension localCoordinate)
        => Stage.Distribution switch
        {
            ContiguousSplit =>
                (LinearShardIndex * LocalCapacity) + localCoordinate,
            BlockCyclicSplit blockCyclic =>
                ((localCoordinate / blockCyclic.BlockSize) * (ShardCount * blockCyclic.BlockSize)) +
                (LinearShardIndex * blockCyclic.BlockSize) +
                (localCoordinate % blockCyclic.BlockSize),
            _ => throw new NotSupportedException(
                $"Unsupported split distribution {Stage.Distribution.GetType().Name}."),
        };

    internal static Dimension GetLocalCapacity(
        Dimension parentExtent,
        int shardCount,
        SplitDistribution distribution,
        DistributedUtility.DivideFlags divideFlags)
        => distribution switch
        {
            ContiguousSplit contiguous => contiguous.Granularity is { } granularity
                ? divideFlags.HasFlag(DistributedUtility.DivideFlags.MaxShape)
                    ? DistributedUtility.GetMaxDimension(granularity)
                    : granularity
                : divideFlags.HasFlag(DistributedUtility.DivideFlags.FloorDiv)
                    ? parentExtent / shardCount
                    : Dimension.CeilDiv(parentExtent, shardCount),
            BlockCyclicSplit blockCyclic => divideFlags.HasFlag(DistributedUtility.DivideFlags.FloorDiv)
                ? (parentExtent / (shardCount * blockCyclic.BlockSize)) * blockCyclic.BlockSize
                : Dimension.CeilDiv(parentExtent, shardCount * blockCyclic.BlockSize) * blockCyclic.BlockSize,
            _ => throw new NotSupportedException(
                $"Unsupported split distribution {distribution.GetType().Name}."),
        };

    internal static Dimension GetActiveExtent(
        Dimension parentExtent,
        Dimension localCapacity,
        Dimension linearShardIndex,
        int shardCount,
        SplitDistribution distribution)
        => distribution switch
        {
            _ when IsUniformCapacity(parentExtent, localCapacity, shardCount) => localCapacity,
            ContiguousSplit => Dimension.Max(
                0,
                Dimension.Min(localCapacity, parentExtent - (linearShardIndex * localCapacity))),
            BlockCyclicSplit blockCyclic =>
                ((parentExtent / (shardCount * blockCyclic.BlockSize)) * blockCyclic.BlockSize) +
                Dimension.Clamp(
                    (parentExtent % (shardCount * blockCyclic.BlockSize)) -
                    (linearShardIndex * blockCyclic.BlockSize),
                    0,
                    blockCyclic.BlockSize),
            _ => throw new NotSupportedException(
                $"Unsupported split distribution {distribution.GetType().Name}."),
        };

    private static bool IsUniformCapacity(
        Dimension parentExtent,
        Dimension localCapacity,
        int shardCount)
        => parentExtent.IsFixed &&
           localCapacity.IsFixed &&
           parentExtent.FixedValue % shardCount == 0 &&
           parentExtent.FixedValue / shardCount == localCapacity.FixedValue;
}

/// <summary>
/// Describes one tensor axis in a local shard.
/// </summary>
public sealed record LocalShardAxisDescriptor(
    Dimension GlobalExtent,
    Dimension LocalCapacity,
    Dimension ActiveExtent,
    IRArray<LocalShardStageDescriptor> Stages)
{
    public bool IsContiguous => Stages.All(stage => stage.Stage.Distribution is ContiguousSplit);

    /// <summary>
    /// Maps a dense local coordinate to the corresponding global tensor coordinate.
    /// </summary>
    public Dimension MapLocalToGlobal(Dimension localCoordinate)
    {
        var coordinate = localCoordinate;
        for (var stageIndex = Stages.Count - 1; stageIndex >= 0; stageIndex--)
        {
            coordinate = Stages[stageIndex].MapLocalToParent(coordinate);
        }

        return coordinate.Simplify();
    }
}

/// <summary>
/// Canonical local-shard description for a distributed tensor.
/// </summary>
public sealed record LocalShardDescriptor(
    DistributedType DistributedType,
    IRArray<LocalShardAxisDescriptor> Axes)
{
    public RankedShape LocalCapacityShape => new(Axes.Select(axis => axis.LocalCapacity).ToArray());

    public RankedShape ActiveShape => new(Axes.Select(axis => axis.ActiveExtent).ToArray());

    public bool IsContiguous => Axes.All(axis => axis.IsContiguous);

    /// <summary>
    /// Gets a rectangular global region when every split stage is contiguous.
    /// </summary>
    public bool TryGetContiguousRegion(out Dimension[] offset, out Dimension[] shape)
    {
        if (!IsContiguous)
        {
            offset = Array.Empty<Dimension>();
            shape = Array.Empty<Dimension>();
            return false;
        }

        offset = Axes.Select(axis => axis.MapLocalToGlobal(0)).ToArray();
        shape = Axes.Select(axis => axis.ActiveExtent).ToArray();
        return true;
    }
}
