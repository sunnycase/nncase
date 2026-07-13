// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Nncase.IR.Affine;

namespace Nncase.Schedule.TileGraph;

/// <summary>
/// Stable identity of one producer-consumer use in a tile region.
/// </summary>
public readonly record struct TileUseId(
    int ProducerOpId,
    int ProducerOutputIndex,
    int ConsumerOpId,
    int ConsumerAccessIndex)
{
    public override string ToString()
        => $"Op{ProducerOpId}.out{ProducerOutputIndex}->Op{ConsumerOpId}.in{ConsumerAccessIndex}";
}

/// <summary>
/// One immutable use in a maximal tile region.
/// </summary>
public sealed record TileUse(
    TileUseId Id,
    long MaximumBytes,
    bool IsLocallyConnectable,
    bool IsAliasView);

/// <summary>
/// Kind of value placement selected for one tile use.
/// </summary>
public enum TileUsePlacementKind
{
    /// <summary>
    /// The producer result is materialized in the root memory space.
    /// </summary>
    RootStorage,

    /// <summary>
    /// The producer result is materialized in a target tiling memory space.
    /// </summary>
    LocalStorage,

    /// <summary>
    /// The producer result is a zero-copy logical view of existing storage.
    /// </summary>
    AliasView,
}

/// <summary>
/// Physical storage or zero-copy view placement selected for one tile use.
/// Level is -1 for the root scope and otherwise indexes the target tiling
/// memory hierarchy.
/// </summary>
public readonly record struct TileUsePlacement(TileUsePlacementKind Kind, int Level)
{
    public static TileUsePlacement RootStorage { get; } = new(TileUsePlacementKind.RootStorage, -1);

    public static TileUsePlacement LocalStorage(int level)
    {
        if (level < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(level), level, "A local storage level must be non-negative.");
        }

        return new(TileUsePlacementKind.LocalStorage, level);
    }

    public static TileUsePlacement AliasView(int level)
    {
        if (level < -1)
        {
            throw new ArgumentOutOfRangeException(nameof(level), level, "An alias view scope must be root (-1) or non-negative.");
        }

        return new(TileUsePlacementKind.AliasView, level);
    }
}

/// <summary>
/// Immutable maximal region presented to hierarchical AutoTiling.
/// </summary>
public sealed class TileRegion
{
    private TileRegion(TieredTileGraph baseGraph, ImmutableArray<TileUse> uses)
    {
        BaseGraph = baseGraph;
        Uses = uses;
    }

    internal TieredTileGraph BaseGraph { get; }

    public ImmutableArray<TileUse> Uses { get; }

    public static TileRegion Create(TieredTileGraph baseGraph)
    {
        if (baseGraph.Level != -1)
        {
            throw new ArgumentException("A tile region must be built from a root graph.", nameof(baseGraph));
        }

        var uses = baseGraph.Edges
            .Select(edge => CreateUse(edge.Source, edge.Target, edge.Tag))
            .OrderBy(use => use.Id.ProducerOpId)
            .ThenBy(use => use.Id.ProducerOutputIndex)
            .ThenBy(use => use.Id.ConsumerOpId)
            .ThenBy(use => use.Id.ConsumerAccessIndex)
            .ToImmutableArray();
        var duplicate = uses.GroupBy(use => use.Id).FirstOrDefault(group => group.Count() > 1);
        if (duplicate is not null)
        {
            throw new InvalidOperationException($"Tile region contains duplicate use {duplicate.Key}.");
        }

        return new TileRegion(baseGraph, uses);
    }

    private static TileUse CreateUse(TileGrid producer, TileGrid consumer, int consumerAccessIndex)
    {
        var producerOutputIndex = GraphExtensions.GetProducerOutputIndex(
            consumer.Grid.Accesses[consumerAccessIndex].Value,
            producer);
        var producerWriteAccessIndex = producer.GetWriteAccessIndex(producerOutputIndex);
        var maximumBytes = producer.GetBufferElemSize(producerWriteAccessIndex);
        try
        {
            foreach (var extent in producer.BufferShapes[producerWriteAccessIndex])
            {
                maximumBytes = checked(maximumBytes * extent);
            }
        }
        catch (OverflowException ex)
        {
            throw new InvalidOperationException(
                $"Maximum byte size for Op{producer.OpId}.out{producerOutputIndex} exceeds Int64.",
                ex);
        }

        return new TileUse(
            new TileUseId(producer.RegionOpId, producerOutputIndex, consumer.RegionOpId, consumerAccessIndex),
            maximumBytes,
            GraphExtensions.IsFusionLegal(producer, consumer, consumerAccessIndex),
            producer.TryGetAliasSourceAccess(producerWriteAccessIndex, out _));
    }
}

/// <summary>
/// Immutable per-use storage connection decisions. Level -1 denotes root
/// materialization; non-negative levels index target tiling storage spaces.
/// </summary>
public sealed class TileConnectionPlan : IEquatable<TileConnectionPlan>
{
    private readonly ImmutableSortedDictionary<TileUseId, int> _levels;

    private TileConnectionPlan(ImmutableSortedDictionary<TileUseId, int> levels)
    {
        _levels = levels;
    }

    public IEnumerable<KeyValuePair<TileUseId, int>> Connections => _levels;

    public static TileConnectionPlan CreateRoot(TileRegion region)
    {
        var builder = ImmutableSortedDictionary.CreateBuilder<TileUseId, int>(TileUseIdComparer.Instance);
        foreach (var use in region.Uses)
        {
            builder.Add(use.Id, -1);
        }

        return new TileConnectionPlan(builder.ToImmutable());
    }

    public int GetLevel(TileUseId use)
        => _levels.TryGetValue(use, out var level)
            ? level
            : throw new KeyNotFoundException($"Tile connection plan does not contain use {use}.");

    public TileConnectionPlan Connect(TileUseId use, int level)
    {
        if (level < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(level), level, "A local connection level must be non-negative.");
        }

        if (!_levels.ContainsKey(use))
        {
            throw new KeyNotFoundException($"Tile connection plan does not contain use {use}.");
        }

        return new TileConnectionPlan(_levels.SetItem(use, level));
    }

    internal TileConnectionPlan WithLevel(TileUseId use, int level)
    {
        if (level < -1)
        {
            throw new ArgumentOutOfRangeException(nameof(level), level, "A tile connection level must be root (-1) or non-negative.");
        }

        if (!_levels.ContainsKey(use))
        {
            throw new KeyNotFoundException($"Tile connection plan does not contain use {use}.");
        }

        return new TileConnectionPlan(_levels.SetItem(use, level));
    }

    public bool Equals(TileConnectionPlan? other)
        => other is not null && _levels.SequenceEqual(other._levels);

    public override bool Equals(object? obj) => obj is TileConnectionPlan other && Equals(other);

    public override int GetHashCode()
    {
        HashCode hash = default;
        foreach (var connection in _levels)
        {
            hash.Add(connection.Key);
            hash.Add(connection.Value);
        }

        return hash.ToHashCode();
    }

    public override string ToString()
        => string.Join(",", _levels.Select(item => $"{item.Key}@L{item.Value}"));

    private sealed class TileUseIdComparer : IComparer<TileUseId>
    {
        public static TileUseIdComparer Instance { get; } = new();

        public int Compare(TileUseId x, TileUseId y)
        {
            var result = x.ProducerOpId.CompareTo(y.ProducerOpId);
            if (result != 0)
            {
                return result;
            }

            result = x.ProducerOutputIndex.CompareTo(y.ProducerOutputIndex);
            if (result != 0)
            {
                return result;
            }

            result = x.ConsumerOpId.CompareTo(y.ConsumerOpId);
            return result != 0 ? result : x.ConsumerAccessIndex.CompareTo(y.ConsumerAccessIndex);
        }
    }
}

/// <summary>
/// Fully solved region used by lowering.
/// </summary>
public sealed record TileExecutionPlan(
    TileConnectionPlan Connections,
    ImmutableDictionary<TileUseId, TileUsePlacement> Placements,
    TieredTileGraph ScheduleGraph,
    Dictionary<BufferIdentity, IR.Expr> ArgumentMemo,
    long ObjectiveValue);
