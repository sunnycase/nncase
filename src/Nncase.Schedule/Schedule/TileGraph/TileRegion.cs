// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using Nncase.IR;
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
/// Stable identity of one value produced in a tile region.
/// </summary>
public readonly record struct TileValueId(int ProducerOpId, int ProducerOutputIndex)
{
    public override string ToString() => $"Op{ProducerOpId}.out{ProducerOutputIndex}";
}

/// <summary>
/// Stable identity of one lexical tile scope in the original maximal region.
/// The anchor operation is the operation whose initial hierarchy owns the
/// scope. Fused scopes retain the consumer anchor.
/// </summary>
public readonly record struct TileScopeId(int AnchorOpId, int Level)
{
    public override string ToString() => $"Op{AnchorOpId}@L{Level}";
}

/// <summary>
/// One immutable use in a maximal tile region.
/// </summary>
public sealed record TileUse(
    TileUseId Id,
    long MaximumBytes,
    MemoryAccessScope RequiredMemoryScope,
    bool IsAliasView)
{
    /// <summary>
    /// Returns whether this use may be composed at the requested hierarchy
    /// level. Chip-visible uses are phase-fused only at the outermost block
    /// scope; their root materialization remains intact.
    /// </summary>
    public bool CanFuseAtLevel(int level, int levelCount)
    {
        if ((uint)level >= (uint)levelCount)
        {
            return false;
        }

        return RequiredMemoryScope != MemoryAccessScope.Chip || level == levelCount - 1;
    }
}

/// <summary>
/// One original lexical scope available to structural scheduling.
/// </summary>
public sealed record TileScope(TileScopeId Id, int Rank);

/// <summary>
/// Immutable maximal region presented to hierarchical AutoTiling.
/// </summary>
public sealed class TileRegion
{
    private readonly TieredTileGraph _baseGraph;

    private TileRegion(
        TieredTileGraph baseGraph,
        ImmutableArray<TileUse> uses,
        ImmutableArray<TileScope> scopes)
    {
        _baseGraph = baseGraph;
        Uses = uses;
        Scopes = scopes;
    }

    public ImmutableArray<TileUse> Uses { get; }

    public ImmutableArray<TileScope> Scopes { get; }

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

        var scopes = new List<TileScope>();
        CollectScopes(baseGraph, scopes);
        var orderedScopes = scopes
            .OrderBy(scope => scope.Id.AnchorOpId)
            .ThenBy(scope => scope.Id.Level)
            .ToImmutableArray();
        var duplicateScope = orderedScopes.GroupBy(scope => scope.Id).FirstOrDefault(group => group.Count() > 1);
        if (duplicateScope is not null)
        {
            throw new InvalidOperationException($"Tile region contains duplicate lexical scope {duplicateScope.Key}.");
        }

        return new TileRegion(baseGraph, uses, orderedScopes);
    }

    internal TieredTileGraph GetBaseGraph() => _baseGraph;

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
            GraphExtensions.GetRequiredFusionScope(producer, consumer, consumerAccessIndex),
            producer.TryGetAliasSourceAccess(producerWriteAccessIndex, out _));
    }

    private static void CollectScopes(TieredTileGraph graph, ICollection<TileScope> scopes)
    {
        foreach (var child in graph.Clusters.OfType<TieredTileGraph>())
        {
            scopes.Add(new TileScope(new TileScopeId(child.OpId, child.Level), child.DomainBoundExprs.Length));
            CollectScopes(child, scopes);
        }
    }
}

/// <summary>
/// Immutable structural schedule. Fusion levels and lexical loop orders are
/// structural decisions; physical buffer materialization is solved separately.
/// </summary>
public sealed class TileStructuralSchedule : IEquatable<TileStructuralSchedule>
{
    private readonly ImmutableSortedDictionary<TileUseId, int> _fusionLevels;
    private readonly ImmutableSortedDictionary<TileScopeId, ImmutableArray<int>> _loopOrders;

    private TileStructuralSchedule(
        ImmutableSortedDictionary<TileUseId, int> fusionLevels,
        ImmutableSortedDictionary<TileScopeId, ImmutableArray<int>> loopOrders)
    {
        _fusionLevels = fusionLevels;
        _loopOrders = loopOrders;
    }

    public IEnumerable<KeyValuePair<TileUseId, int>> FusionLevels => _fusionLevels;

    public IEnumerable<KeyValuePair<TileScopeId, ImmutableArray<int>>> LoopOrders => _loopOrders;

    public static TileStructuralSchedule Create(TileRegion region)
    {
        var fusionLevels = ImmutableSortedDictionary.CreateBuilder<TileUseId, int>(TileUseIdComparer.Instance);
        foreach (var use in region.Uses)
        {
            fusionLevels.Add(use.Id, -1);
        }

        var loopOrders = ImmutableSortedDictionary.CreateBuilder<TileScopeId, ImmutableArray<int>>(TileScopeIdComparer.Instance);
        foreach (var scope in region.Scopes)
        {
            loopOrders.Add(scope.Id, ImmutableArray.CreateRange(Enumerable.Range(0, scope.Rank)));
        }

        return new TileStructuralSchedule(fusionLevels.ToImmutable(), loopOrders.ToImmutable());
    }

    public int GetFusionLevel(TileUseId use)
        => _fusionLevels.TryGetValue(use, out var level)
            ? level
            : throw new KeyNotFoundException($"Tile structural schedule does not contain use {use}.");

    public ImmutableArray<int> GetLoopOrder(TileScopeId scope)
        => _loopOrders.TryGetValue(scope, out var order)
            ? order
            : throw new KeyNotFoundException($"Tile structural schedule does not contain scope {scope}.");

    public TileStructuralSchedule WithFusionLevel(TileUseId use, int level)
    {
        if (level < -1)
        {
            throw new ArgumentOutOfRangeException(nameof(level), level, "A fusion level must be root (-1) or non-negative.");
        }

        if (!_fusionLevels.ContainsKey(use))
        {
            throw new KeyNotFoundException($"Tile structural schedule does not contain use {use}.");
        }

        return new TileStructuralSchedule(_fusionLevels.SetItem(use, level), _loopOrders);
    }

    public TileStructuralSchedule WithLoopOrder(TileScopeId scope, IEnumerable<int> loopOrder)
    {
        if (!_loopOrders.TryGetValue(scope, out var existing))
        {
            throw new KeyNotFoundException($"Tile structural schedule does not contain scope {scope}.");
        }

        var order = ImmutableArray.CreateRange(loopOrder);
        if (order.Length != existing.Length || !order.Order().SequenceEqual(Enumerable.Range(0, order.Length)))
        {
            throw new ArgumentException(
                $"Loop order for {scope} must be a permutation of [0, {existing.Length}), got [{string.Join(", ", order)}].",
                nameof(loopOrder));
        }

        return new TileStructuralSchedule(_fusionLevels, _loopOrders.SetItem(scope, order));
    }

    public bool Equals(TileStructuralSchedule? other)
        => other is not null &&
            _fusionLevels.SequenceEqual(other._fusionLevels) &&
            _loopOrders.Count == other._loopOrders.Count &&
            _loopOrders.All(item => other._loopOrders.TryGetValue(item.Key, out var order) && item.Value.SequenceEqual(order));

    public override bool Equals(object? obj) => obj is TileStructuralSchedule other && Equals(other);

    public override int GetHashCode()
    {
        HashCode hash = default;
        foreach (var connection in _fusionLevels)
        {
            hash.Add(connection.Key);
            hash.Add(connection.Value);
        }

        foreach (var (scope, order) in _loopOrders)
        {
            hash.Add(scope);
            foreach (var axis in order)
            {
                hash.Add(axis);
            }
        }

        return hash.ToHashCode();
    }

    public override string ToString()
        => $"fusion=[{string.Join(",", _fusionLevels.Select(item => $"{item.Key}@L{item.Value}"))}], " +
            $"loops=[{string.Join(",", _loopOrders.Select(item => $"{item.Key}:[{string.Join("-", item.Value)}]"))}]";

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

    private sealed class TileScopeIdComparer : IComparer<TileScopeId>
    {
        public static TileScopeIdComparer Instance { get; } = new();

        public int Compare(TileScopeId x, TileScopeId y)
        {
            var result = x.AnchorOpId.CompareTo(y.AnchorOpId);
            return result != 0 ? result : x.Level.CompareTo(y.Level);
        }
    }
}

/// <summary>
/// One value placement selected by the exact tile solver.
/// </summary>
public abstract record TileMaterialization(
    TileValueId Value,
    TileScopeId CreationScope,
    int LoopEntry,
    ImmutableArray<TileUseId> Uses);

/// <summary>
/// A value backed by physical storage in a target memory space.
/// </summary>
public sealed record TileStorageMaterialization(
    TileValueId Value,
    TileScopeId CreationScope,
    int LoopEntry,
    ImmutableArray<TileUseId> Uses,
    TargetMemorySpaceId StorageSpace)
    : TileMaterialization(Value, CreationScope, LoopEntry, Uses);

/// <summary>
/// A logical buffer descriptor backed by another value's storage.
/// </summary>
public sealed record TileAliasMaterialization(
    TileValueId Value,
    TileScopeId CreationScope,
    int LoopEntry,
    ImmutableArray<TileUseId> Uses)
    : TileMaterialization(Value, CreationScope, LoopEntry, Uses);

/// <summary>
/// A caller-allocated root buffer retained across sequential phases in one
/// scheduled region. The value is not placed in a block-local tiling space.
/// </summary>
public sealed record TileRootMaterialization(
    TileValueId Value,
    TileScopeId CreationScope,
    int LoopEntry,
    ImmutableArray<TileUseId> Uses,
    MemoryAccessScope RequiredMemoryScope)
    : TileMaterialization(Value, CreationScope, LoopEntry, Uses);

/// <summary>
/// Fully solved region used by lowering.
/// </summary>
public sealed record TileExecutionPlan(
    TileStructuralSchedule Structure,
    ImmutableArray<TileMaterialization> Materializations,
    TieredTileGraph ScheduleGraph,
    Dictionary<BufferIdentity, IR.Expr> ArgumentMemo,
    long ObjectiveValue);
