// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Diagnostics.CodeAnalysis;
using Google.OrTools.ConstraintSolver;
using Nncase.IR;
using Nncase.IR.Affine;

namespace Nncase.Schedule.TileGraph;

public enum BufferEndpoint
{
    Input,
    Output,
}

/// <summary>
/// Inclusive execution phases during which a tile buffer must remain live.
/// Reads and writes performed by the same operation share a phase, so their
/// storage overlaps in time unless the IR declares an explicit alias.
/// </summary>
public readonly record struct TileLifetime
{
    public TileLifetime(int firstPhase, int lastPhase)
    {
        if (firstPhase < 0 || lastPhase < firstPhase)
        {
            throw new ArgumentOutOfRangeException(
                nameof(lastPhase),
                $"A tile lifetime must be a non-empty inclusive phase range, got [{firstPhase}, {lastPhase}].");
        }

        FirstPhase = firstPhase;
        LastPhase = lastPhase;
    }

    public int FirstPhase { get; }

    public int LastPhase { get; }

    public int PhaseCount => checked(LastPhase - FirstPhase + 1);

    public bool Overlaps(TileLifetime other)
        => FirstPhase <= other.LastPhase && other.FirstPhase <= LastPhase;

    public TileLifetime Union(TileLifetime other)
        => new(Math.Min(FirstPhase, other.FirstPhase), Math.Max(LastPhase, other.LastPhase));
}

public sealed record BufferIdentity(TileGrid Node, int Index, BufferEndpoint Endpoint)
{
    public GridAccess Access => Node.Grid.Accesses[Index];

    public bool IsOutput => Endpoint == BufferEndpoint.Output;

    public int OutputIndex => IsOutput ? Node.WriteAccessIndices.IndexOf(Index) : -1;

    public bool IsOutputLiveOut => IsOutput && Node.Attribute.HasFlag(TileGridAttribute.LiveOut);

    public override string ToString() => TileSemanticNaming.GetBufferEndpointName(this);
}

/// <summary>
/// The placement length is domain rank + 1. Entry 0 is outside every
/// lexical loop; entry n is after the first n loops in the scope's LoopOrder.
/// </summary>
/// <param name="Lifetimes">This buffer's inclusive lifetime for each creation loop.</param>
/// <param name="Map">this buffers access map.</param>
/// <param name="Places">
/// Places[lexical loop entry][storage level]. For a creation scope at level
/// cl, storage level sl is legal when 0 &lt;= sl &lt;= cl. Axis identities remain
/// canonical in affine maps; only the entry order follows LoopOrder.</param>
/// <param name="Shapes">the buffer shape according to the placement.</param>
/// <param name="Sizes">the buffer size according to the placement.</param>
/// <param name="Trips">related loop trips at current domain.</param>
/// <param name="Mask">The lexical loop-position mask that affects this buffer.</param>
public sealed record TileNodeBufferInfo<T>(TileLifetime[] Lifetimes, AffineMap Map, T[][] Places, T[][] Shapes, T[] Sizes, T[] Trips, LoopMask Mask)
{
    public int GetLastRelatedPos()
    {
        var lastLoop = Mask.LastRelated(Places.Length - 1);
        return lastLoop + 1;
    }
}

/// <summary>
/// Solver information for one lexical tile scope.
/// </summary>
/// <param name="TripCounts">Forward trip products indexed by lexical loop entry.</param>
/// <param name="BackWardExtents">Backward extents indexed by lexical loop entry, with each extent vector indexed by canonical axis.</param>
/// <param name="DefUseMap">key is def, value is use.</param>
/// <param name="BufferInfoMap">buffer info memo.</param>
public sealed record TileNodeInfo<T>(T[] TripCounts, T[][] BackWardExtents, BiDictionary<BufferIdentity, BufferIdentity> DefUseMap, Dictionary<BufferIdentity, TileNodeBufferInfo<T>> BufferInfoMap)
{
    public BufferIdentity GetByChildBuffer(BufferIdentity sinkBid)
        => TryGetByChildBuffer(sinkBid, out var sourceBid)
            ? sourceBid
            : throw new KeyNotFoundException(sinkBid.ToString());

    public bool TryGetByChildBuffer(BufferIdentity sinkBid, [MaybeNullWhen(false)] out BufferIdentity sourceBid)
    {
        if (DefUseMap.TryGetByValue(sinkBid, out sourceBid))
        {
            return true;
        }

        if (BufferInfoMap.ContainsKey(sinkBid))
        {
            sourceBid = sinkBid;
            return true;
        }

        sourceBid = null!;
        return false;
    }
}

/// <summary>
/// domain infomation.
/// </summary>
/// <param name="TileVars">loop trip vars length = domainRank.</param>
/// <param name="ForwardExtents">forward extents.</param>
/// <param name="DimsMap"> key is current dim, value is partent dim. </param>
/// NOTE dimsMap should be removed when using isl.
public sealed record DomainInfo<T>(T[] TileVars, T[] ForwardExtents, Dictionary<int, int> DimsMap)
{
}

/// <summary>
/// op node info.
/// </summary>
/// <param name="Maps">current node's domain accesses the buffer. it means applyed by this op's domain relation.</param>
/// <param name="Shapes">each buffer's shape expr. </param>
/// <param name="Sizes">each buffer's size.</param>
public sealed record OpNodeInfo<T>(AffineMap[] Maps, T[][] Shapes, T[] Sizes)
{
}

public class BiDictionary<TKey, TValue> : IEnumerable<KeyValuePair<TKey, TValue>>, IEnumerable
    where TKey : notnull
    where TValue : notnull
{
    private readonly Dictionary<TKey, HashSet<TValue>> _forward = new();

    private readonly Dictionary<TValue, TKey> _reverse = new();

    public bool TryGetByKey(TKey key, [MaybeNullWhen(false)] out HashSet<TValue> value) => _forward.TryGetValue(key, out value);

    public bool TryGetByValue(TValue value, [MaybeNullWhen(false)] out TKey key) => _reverse.TryGetValue(value, out key);

    public bool Add(TKey key, TValue value)
    {
        if (!_forward.TryGetValue(key, out var values))
        {
            values = new() { };
            _forward[key] = values;
        }

        if (values.Contains(value))
        {
            return false;
        }

        values.Add(value);
        _reverse[value] = key;
        return true;
    }

    public bool ContainsKey(TKey value) => _forward.ContainsKey(value);

    public bool ContainsValue(TValue value) => _reverse.ContainsKey(value);

    public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator()
    {
        foreach (var kv in _forward)
        {
            foreach (var value in kv.Value)
            {
                yield return new KeyValuePair<TKey, TValue>(kv.Key, value);
            }
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
